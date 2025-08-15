# On-Policy蒸馏功能指南

## 概述

On-Policy蒸馏是NeMo-RL框架中实现的一种知识蒸馏技术，通过让学生模型在训练过程中动态生成自监督数据（on-policy数据），结合固定数据集混合训练，解决传统蒸馏中"训练分布与推理分布不匹配"的问题。

## 核心特性

### 1. On-Policy动态数据生成
- 训练过程中，学生模型在每个训练步骤实时生成输出序列
- 生成的数据与当前学生模型的参数状态强关联
- 确保训练数据分布与推理时学生自主生成的序列分布一致

### 2. 混合数据训练策略
- 训练数据由两部分构成，通过超参数λ控制比例
- 固定数据：来自原始任务数据集（如真实标签、教师预生成序列）
- 学生自生成数据：学生模型对输入样本实时生成的输出序列
- 每个训练步骤随机选择数据来源（以λ概率选择学生生成数据，1-λ概率选择固定数据）

### 3. 损失函数设计
- 采用KL散度衡量教师与学生在token级别概率分布的差异
- 支持两种模式：
  - 前向KL（Forward KL）：学生分布逼近教师分布，注重覆盖教师所有可能输出
  - 反向KL（Reverse KL）：学生分布聚焦教师高概率输出，注重核心模式学习

## 架构设计

### 参考GRPO实现
我们的distillation功能完全参考了GRPO的实现架构，确保：

1. **多模型计算资源调度**：与GRPO保持一致
2. **生成接口管理**：支持vLLM和Megatron后端
3. **集群管理**：支持colocated和分离式推理
4. **权重同步**：完整的refit机制

### 核心组件

- **DistillationAlgorithm**: 管理整体训练循环和师生模型交互
- **DistillationLossFn**: 实现KL散度损失计算
- **Dual Policy Management**: 独立的学生和教师策略管理

## 配置文件示例

```yaml
# 蒸馏算法配置
distillation:
  # 教师模型路径
  teacher_model_path: "Qwen/Qwen2.5-32B-Instruct"
  
  # 蒸馏策略参数
  lambda_: 0.5  # 学生自生成数据占比
  kl_type: "forward"  # KL散度类型
  generate_strategy:  # 学生生成参数
    max_length: 2048
    temperature: 0.1
    decoding_method: "greedy"
  
  # 训练配置
  max_steps: 1000
  eval_steps: 100
  save_steps: 500
  logging_steps: 10

# 策略配置
policy:
  model_name: "Qwen/Qwen2.5-1.5B-Instruct"  # 学生模型
  generation:
    backend: "megatron"  # 或 "vllm"
    colocated:
      enabled: true
```

## 使用方法

### 1. 基本训练命令

```bash
uv run python examples/run_distillation.py --config examples/configs/distillation_math_cl.yaml
```

### 2. 自定义配置

```bash
uv run python examples/run_distillation.py \
  --config examples/configs/distillation_math_cl.yaml \
  distillation.lambda_=0.7 \
  distillation.kl_type=reverse \
  policy.generation.backend=vllm
```

### 3. 运行测试

```bash
bash tests/functional/distillation.sh
```

## 数据集支持

### Math-CL数据集
我们提供了对`pe-nlp/math-cl`数据集的完整支持：

```python
from nemo_rl.data.hf_datasets.math_cl import MathCLDataset

# 加载数据集
dataset = MathCLDataset(
    split="train",
    seed=42,
    max_samples=1000
)

# 获取提示模板
prompt_template = dataset.get_prompt_template()
```

### 自定义数据集
可以通过继承`HFDatasetInterface`来支持其他数据集。

## 训练流程

### 1. 初始化阶段
- 加载学生和教师模型
- 设置计算集群
- 初始化生成接口

### 2. 训练循环
```
for step in range(max_steps):
    # 1. 准备批次数据
    prompts = sample_prompts(batch_size)
    
    # 2. 学生模型生成响应（on-policy数据生成）
    generated_sequences = student.generate(prompts)
    
    # 3. 计算logits
    student_logits = student_policy(generated_sequences)
    teacher_logits = teacher_policy(generated_sequences)
    
    # 4. 计算蒸馏损失
    loss = kl_div(student_logits, teacher_logits)
    
    # 5. 反向传播和优化
    loss.backward()
    optimizer.step()
```

### 3. 验证和检查点
- 定期运行验证
- 保存模型检查点
- 记录训练指标

## 性能优化

### 1. 集群配置
- **Colocated模式**：训练和推理共享GPU资源，适合单节点
- **分离模式**：训练和推理使用独立资源，适合多节点

### 2. 生成后端选择
- **Megatron后端**：适合小规模模型，资源占用少
- **vLLM后端**：适合大规模模型，生成速度快

### 3. 批处理优化
- 动态批处理
- 序列打包
- 梯度累积

## 故障排除

### 常见问题

1. **内存不足**
   - 减少batch_size
   - 启用gradient_checkpointing
   - 使用CPU offload

2. **生成失败**
   - 检查generation配置
   - 验证模型权重加载
   - 检查集群状态

3. **训练不稳定**
   - 调整学习率
   - 检查损失函数参数
   - 验证数据质量

### 调试技巧

- 启用详细日志
- 使用小规模数据集测试
- 检查GPU内存使用情况

## 扩展开发

### 自定义损失函数

```python
from nemo_rl.algorithms.loss_functions import DistillationLossFn

class CustomDistillationLoss(DistillationLossFn):
    def __call__(self, student_logits, data, ...):
        # 实现自定义损失逻辑
        pass
```

### 自定义数据集

```python
from nemo_rl.data.hf_datasets.interfaces import HFDatasetInterface

class CustomDataset(HFDatasetInterface):
    def __init__(self, ...):
        # 实现数据集加载逻辑
        pass
```

## 总结

On-Policy蒸馏功能为NeMo-RL框架提供了强大的知识迁移能力，通过参考GRPO的成熟架构，确保了系统的稳定性和可扩展性。该功能特别适合需要将大模型知识高效迁移到小模型的场景，如模型压缩、边缘部署等应用。
