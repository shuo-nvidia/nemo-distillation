# Distillation配置文件说明

## 文件概览

本目录包含用于On-Policy蒸馏训练的配置文件，支持数学数据集的知识蒸馏。

## 配置文件列表

### 1. `distillation_math_cl.yaml` - DTensor配置文件
- **用途**: 生产环境使用
- **特点**: 使用DTensor配置，完整的配置，包含所有必要参数
- **模型**: 学生模型(Qwen2.5-7B) + 教师模型(Qwen2.5-32B)
- **策略**: 使用DTensor配置，参考DPO和GRPO的实现方式
- **关键特性**: teacher和student模型共享相同的计算资源，避免重复创建worker

## 配置结构说明

### 通用配置部分
```yaml
# 蒸馏算法配置
distillation:
  lambda: 0.9  # 学生自生成数据占比
  kl_type: "forward"  # KL散度类型
  max_length: 2048  # 生成序列最大长度
  # ... 其他参数

# 数据配置
data:
  dataset_name: "math_cl"  # 使用pe-nlp/math-cl数据集
  prompt_template: "math_distillation"
  # ... 其他参数

# 集群配置
cluster:
  enabled: true
  num_nodes: 1
  gpus_per_node: 8
  # ... 其他参数
```

### 模型配置部分
每个模型（学生/教师）都包含以下配置：

#### 基础配置
- `model_name`: 模型名称
- `tokenizer`: 分词器配置
- `max_total_sequence_length`: 最大序列长度
- `precision`: 精度设置
- `train_global_batch_size`: 全局批次大小

#### DTensor配置
- `dtensor_cfg.enabled: true`: 启用DTensor后端
- `tensor_parallel_size: 1`: 张量并行大小（保持为1避免过多worker）
- `context_parallel_size: 1`: 上下文并行大小（保持为1避免过多worker）
- `sequence_parallel: false`: 禁用序列并行
- `activation_checkpointing: false`: 禁用激活检查点

#### 批处理配置
- `dynamic_batching.enabled: false`: 禁用动态批处理
- `sequence_packing.enabled: false`: 禁用序列打包

## 关键配置说明

### 1. 为什么使用DTensor而不是Megatron？
- **DTensor**: 更灵活的分布式策略，支持动态sharding
- **Megatron**: 更复杂的配置，需要更多的worker管理

### 2. 如何避免多重环境创建问题？
- **统一名称前缀**: 学生和教师模型使用相同的`name_prefix: "distillation_policy"`
- **共享集群**: 两个模型使用完全相同的cluster实例
- **相同资源配置**: teacher和student使用完全相同的DTensor配置
- **环境变量控制**: 设置`NRL_SKIP_VENV_CREATION=true`减少虚拟环境创建

### 3. 批次大小计算
```yaml
train_mb_tokens: 16384  # 8 * 2048
# 8 = train_global_batch_size
# 2048 = max_total_sequence_length
```

### 4. 环境变量设置
在`nemo_rl/algorithms/distillation.py`中设置了：
```python
os.environ["NRL_SKIP_VENV_CREATION"] = "true"
os.environ["NRL_REUSE_EXISTING_VENV"] = "true"
```

## 使用方法

### 标准配置
```bash
uv run examples/run_distillation.py --config examples/configs/distillation_math_cl.yaml
```

## 配置优化建议

### 1. 避免重复字段
- teacher和student模型使用完全相同的DTensor配置
- 确保所有资源配置参数保持一致

### 2. 资源优化
- 根据GPU数量调整批次大小
- 使用梯度累积来模拟更大的批次
- 保持`tensor_parallel_size=1`和`context_parallel_size=1`

### 3. 性能优化
- 启用激活检查点（`activation_checkpointing: true`）如果需要
- 调整序列长度以适应内存

## 故障排除

### 常见问题
1. **Ray集群错误**: 确保`cluster.enabled: true`
2. **虚拟环境创建过多**: 检查环境变量设置和DTensor配置
3. **内存不足**: 减少批次大小或序列长度
4. **多重worker创建**: 确保teacher和student使用相同的资源配置

### 调试技巧
1. 使用较小的数据集进行测试
2. 检查日志中的错误信息
3. 验证配置文件语法
4. 确认两个模型的DTensor配置完全一致

## 相关文件

- `examples/run_distillation.py`: 蒸馏训练脚本
- `nemo_rl/algorithms/distillation.py`: 蒸馏算法实现
- `examples/prompts/math_distillation.txt`: 数学蒸馏提示模板

## 更新日志

- **v1.0**: 初始版本，支持基本的数学蒸馏
- **v1.1**: 优化Ray集群配置，减少worker创建
- **v1.2**: 清理重复字段，提高配置文件可维护性
- **v2.0**: 改用DTensor配置，确保teacher和student共享计算资源
