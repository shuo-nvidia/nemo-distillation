# NeMo-RL 蒸馏训练运行脚本使用说明

## 概述

为了方便运行蒸馏训练，我们提供了多个平台的运行脚本，让您不再需要输入长串命令。

## 可用的脚本

### 1. Bash脚本 (Linux/macOS/WSL)
- **文件名**: `run_distillation.sh`
- **权限**: 需要执行权限
- **使用方法**: `./run_distillation.sh`

### 2. Windows批处理文件
- **文件名**: `run_distillation.bat`
- **使用方法**: `run_distillation.bat`

### 3. PowerShell脚本
- **文件名**: `run_distillation.ps1`
- **使用方法**: `.\run_distillation.ps1`

## 快速开始

### Linux/macOS/WSL 用户

1. **给脚本添加执行权限**:
   ```bash
   chmod +x run_distillation.sh
   ```

2. **运行默认配置**:
   ```bash
   ./run_distillation.sh
   ```

3. **使用指定配置**:
   ```bash
   ./run_distillation.sh examples/configs/distillation.yaml
   ```

4. **查看帮助**:
   ```bash
   ./run_distillation.sh --help
   ```

### Windows 用户

#### 使用批处理文件
1. **双击运行**:
   ```
   run_distillation.bat
   ```

2. **命令行运行**:
   ```cmd
   run_distillation.bat
   ```

3. **使用指定配置**:
   ```cmd
   run_distillation.bat examples\configs\distillation.yaml
   ```

#### 使用PowerShell
1. **运行默认配置**:
   ```powershell
   .\run_distillation.ps1
   ```

2. **使用指定配置**:
   ```powershell
   .\run_distillation.ps1 examples\configs\distillation.yaml
   ```

3. **查看帮助**:
   ```powershell
   .\run_distillation.ps1 -Help
   ```

## 脚本功能特性

### ✅ **自动检查**
- 检查 `uv` 是否安装
- 检查 Python 是否可用
- 检查配置文件是否存在

### ✅ **自动创建目录**
- 创建日志目录 (`logs/distillation`)
- 创建检查点目录 (`checkpoints/distillation`)
- 创建结果目录 (`results/distillation`)

### ✅ **环境变量设置**
- 自动设置 `WANDB_PROJECT=nemo-distillation`
- 支持自定义 `WANDB_ENTITY`
- 支持 `CUDA_VISIBLE_DEVICES` 设置

### ✅ **参数传递**
- 支持传递额外的命令行参数
- 自动传递给 Python 脚本

### ✅ **错误处理**
- 详细的错误信息
- 彩色输出提示
- 优雅的错误退出

## 使用示例

### 基本用法
```bash
# 使用默认配置 (distillation_math_cl.yaml)
./run_distillation.sh

# 使用指定配置
./run_distillation.sh examples/configs/distillation.yaml

# 传递额外参数
./run_distillation.sh --wandb_project my_project
```

### 环境变量设置
```bash
# 设置wandb项目名称
export WANDB_PROJECT="my-distillation-project"
./run_distillation.sh

# 设置GPU设备
export CUDA_VISIBLE_DEVICES="0,1"
./run_distillation.sh

# 设置wandb用户名
export WANDB_ENTITY="your_username"
./run_distillation.sh
```

### Windows环境变量设置
```cmd
# 设置wandb项目名称
set WANDB_PROJECT=my-distillation-project
run_distillation.bat

# 设置GPU设备
set CUDA_VISIBLE_DEVICES=0,1
run_distillation.bat
```

## 脚本对比

| 特性 | Bash | Windows批处理 | PowerShell |
|------|------|---------------|------------|
| 跨平台 | ✅ | ❌ | ✅ |
| 颜色输出 | ✅ | ❌ | ✅ |
| 错误处理 | ✅ | ✅ | ✅ |
| 参数解析 | ✅ | ✅ | ✅ |
| 环境变量 | ✅ | ✅ | ✅ |

## 故障排除

### 常见问题

1. **权限被拒绝**
   ```bash
   chmod +x run_distillation.sh
   ```

2. **脚本无法执行**
   - 确保脚本有执行权限
   - 检查脚本文件是否损坏

3. **uv命令未找到**
   - 安装 uv: https://docs.astral.sh/uv/getting-started/installation/
   - 确保 uv 在 PATH 中

4. **配置文件不存在**
   - 检查配置文件路径是否正确
   - 确保配置文件存在

### 调试模式

如果需要调试，可以手动运行命令：
```bash
uv run python examples/run_distillation.py --config examples/configs/distillation_math_cl.yaml
```

## 自定义配置

### 修改默认配置
编辑脚本文件中的 `DEFAULT_CONFIG` 变量：
```bash
# 在 run_distillation.sh 中
DEFAULT_CONFIG="examples/configs/your_config.yaml"
```

### 添加新的环境变量
在脚本的 `setup_environment` 函数中添加：
```bash
if [[ -z "$YOUR_VAR" ]]; then
    export YOUR_VAR="default_value"
fi
```

## 总结

现在您可以使用简单的命令来运行蒸馏训练：

- **Linux/macOS/WSL**: `./run_distillation.sh`
- **Windows**: `run_distillation.bat` 或 `.\run_distillation.ps1`

不再需要输入长串的 `uv run python examples/run_distillation.py --config examples/configs/distillation_math_cl.yaml` 命令！

脚本会自动处理所有准备工作，让您专注于训练本身。
