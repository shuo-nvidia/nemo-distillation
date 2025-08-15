#!/bin/bash

# Distillation功能集成测试脚本
# 测试on-policy蒸馏功能的基本功能

set -e

echo "🧪 Starting Distillation Integration Tests..."

# 设置环境变量
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 测试配置文件加载
echo "📋 Testing configuration loading..."
python -c "
from nemo_rl.algorithms.distillation import MasterConfig
from nemo_rl.utils.config import load_config
config = load_config('examples/configs/distillation_math_cl.yaml')
print('✓ Configuration loaded successfully')
print(f'  - Student model: {config[\"policy\"][\"model_name\"]}')
print(f'  - Teacher model: {config[\"distillation\"][\"teacher_model_path\"]}')
print(f'  - Dataset: {config[\"data\"][\"dataset_name\"]}')
"

# 测试数据集加载
echo "📊 Testing dataset loading..."
python -c "
from nemo_rl.data.hf_datasets.math_cl import MathCLDataset
dataset = MathCLDataset(max_samples=10)
print(f'✓ Math-CL dataset loaded: {len(dataset)} samples')
print(f'  - Train samples: {len(dataset.formatted_ds[\"train\"]) if dataset.formatted_ds[\"train\"] else 0}')
print(f'  - Validation samples: {len(dataset.formatted_ds[\"validation\"]) if dataset.formatted_ds[\"validation\"] else 0}')
"

# 测试算法模块导入
echo "🔧 Testing algorithm module imports..."
python -c "
from nemo_rl.algorithms.distillation import setup, distillation_train, MasterConfig
from nemo_rl.algorithms.loss_functions import DistillationLossFn
print('✓ All distillation modules imported successfully')
"

echo "✅ Distillation integration tests completed successfully!"
echo ""
echo "📝 Next steps:"
echo "  1. Run: uv run python examples/run_distillation.py --config examples/configs/distillation_math_cl.yaml"
echo "  2. Check logs in logs/distillation/"
echo "  3. Monitor training progress and checkpoints"
