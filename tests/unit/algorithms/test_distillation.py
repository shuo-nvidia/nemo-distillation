# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
On-Policy蒸馏算法的单元测试
"""

import pytest
import torch
from unittest.mock import Mock, MagicMock

# 添加项目根目录到Python路径
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from nemo_rl.algorithms.distillation import (
    DistillationConfig,
    DistillationSaveState,
    create_response_mask,
    run_validation,
)
from nemo_rl.algorithms.loss_functions import (
    DistillationLossConfig,
    DistillationLossDataDict,
    DistillationLossFn,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


class TestDistillationConfig:
    """测试蒸馏配置类"""
    
    def test_distillation_config_creation(self):
        """测试蒸馏配置创建"""
        config = DistillationConfig(
            lambda=0.5,
            kl_type="forward",
            mixed_kl_weight=0.5,
            max_length=2048,
            temperature=0.1,
            decoding_method="greedy",
            max_num_epochs=1,
            max_num_steps=100,
            val_period=10,
            val_batches=8,
            val_global_batch_size=32,
            val_micro_batch_size=1,
            val_at_start=True,
            seed=42,
        )
        
        assert config["lambda"] == 0.5
        assert config["kl_type"] == "forward"
        assert config["max_length"] == 2048
        assert config["temperature"] == 0.1
        assert config["seed"] == 42


class TestDistillationSaveState:
    """测试蒸馏保存状态类"""
    
    def test_distillation_save_state_creation(self):
        """测试蒸馏保存状态创建"""
        state = DistillationSaveState(
            step=10,
            val_loss=0.5,
            consumed_samples=1000,
        )
        
        assert state["step"] == 10
        assert state["val_loss"] == 0.5
        assert state["consumed_samples"] == 1000
    
    def test_distillation_save_state_optional_fields(self):
        """测试蒸馏保存状态的可选字段"""
        state = DistillationSaveState(
            step=0,
            consumed_samples=0,
        )
        
        assert state["step"] == 0
        assert state["consumed_samples"] == 0
        assert "val_loss" not in state


class TestDistillationLossFn:
    """测试蒸馏损失函数"""
    
    def setup_method(self):
        """设置测试方法"""
        self.config = DistillationLossConfig(
            kl_type="forward",
            mixed_kl_weight=0.5,
            temperature=1.0,
        )
        self.loss_fn = DistillationLossFn(self.config)
    
    def test_forward_kl_loss(self):
        """测试前向KL损失"""
        # 创建测试数据
        batch_size, seq_len, vocab_size = 2, 5, 10
        
        student_logits = torch.randn(batch_size, seq_len, vocab_size)
        teacher_logits = torch.randn(batch_size, seq_len, vocab_size)
        token_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        
        data = BatchedDataDict[DistillationLossDataDict]({
            "student_logits": student_logits,
            "teacher_logits": teacher_logits,
            "token_mask": token_mask,
            "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len)),
        })
        
        # 计算损失
        result = self.loss_fn(data)
        
        assert "loss" in result
        assert "kl_loss" in result
        assert "kl_type" in result
        assert result["kl_type"] == "forward"
        assert isinstance(result["loss"], torch.Tensor)
        assert result["loss"].item() > 0  # 损失应该为正数
    
    def test_reverse_kl_loss(self):
        """测试反向KL损失"""
        self.config["kl_type"] = "reverse"
        loss_fn = DistillationLossFn(self.config)
        
        batch_size, seq_len, vocab_size = 2, 5, 10
        
        student_logits = torch.randn(batch_size, seq_len, vocab_size)
        teacher_logits = torch.randn(batch_size, seq_len, vocab_size)
        token_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        
        data = BatchedDataDict[DistillationLossDataDict]({
            "student_logits": student_logits,
            "teacher_logits": teacher_logits,
            "token_mask": token_mask,
            "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len)),
        })
        
        result = loss_fn(data)
        
        assert result["kl_type"] == "reverse"
        assert isinstance(result["loss"], torch.Tensor)
    
    def test_mixed_kl_loss(self):
        """测试混合KL损失"""
        self.config["kl_type"] = "mixed"
        loss_fn = DistillationLossFn(self.config)
        
        batch_size, seq_len, vocab_size = 2, 5, 10
        
        student_logits = torch.randn(batch_size, seq_len, vocab_size)
        teacher_logits = torch.randn(batch_size, seq_len, vocab_size)
        token_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        
        data = BatchedDataDict[DistillationLossDataDict]({
            "student_logits": student_logits,
            "teacher_logits": teacher_logits,
            "token_mask": token_mask,
            "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len)),
        })
        
        result = loss_fn(data)
        
        assert result["kl_type"] == "mixed"
        assert isinstance(result["loss"], torch.Tensor)
    
    def test_invalid_kl_type(self):
        """测试无效的KL类型"""
        with pytest.raises(ValueError, match="Invalid kl_type"):
            invalid_config = DistillationLossConfig(
                kl_type="invalid",
                mixed_kl_weight=0.5,
                temperature=1.0,
            )
            DistillationLossFn(invalid_config)
    
    def test_temperature_scaling(self):
        """测试温度缩放"""
        self.config["temperature"] = 2.0
        loss_fn = DistillationLossFn(self.config)
        
        batch_size, seq_len, vocab_size = 2, 5, 10
        
        student_logits = torch.randn(batch_size, seq_len, vocab_size)
        teacher_logits = torch.randn(batch_size, seq_len, vocab_size)
        token_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        
        data = BatchedDataDict[DistillationLossDataDict]({
            "student_logits": student_logits,
            "teacher_logits": teacher_logits,
            "token_mask": token_mask,
            "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len)),
        })
        
        result = loss_fn(data)
        assert isinstance(result["loss"], torch.Tensor)


class TestDistillationUtils:
    """测试蒸馏工具函数"""
    
    def test_create_response_mask(self):
        """测试创建响应掩码"""
        batch_size, seq_len = 3, 10
        sequences = torch.randint(0, 1000, (batch_size, seq_len))
        prompt_lengths = [3, 5, 7]
        
        mask = create_response_mask(sequences, prompt_lengths)
        
        assert mask.shape == (batch_size, seq_len)
        assert mask.dtype == torch.bool
        
        # 检查掩码是否正确
        for i, prompt_len in enumerate(prompt_lengths):
            # prompt部分应该是False
            assert not mask[i, :prompt_len].any()
            # response部分应该是True
            if prompt_len < seq_len:
                assert mask[i, prompt_len:].all()
    
    def test_create_response_mask_edge_cases(self):
        """测试创建响应掩码的边缘情况"""
        # 空序列
        sequences = torch.empty(0, 0)
        prompt_lengths = []
        mask = create_response_mask(sequences, prompt_lengths)
        assert mask.shape == (0, 0)
        
        # 所有prompt长度都等于序列长度
        batch_size, seq_len = 2, 5
        sequences = torch.randint(0, 1000, (batch_size, seq_len))
        prompt_lengths = [seq_len, seq_len]
        
        mask = create_response_mask(sequences, prompt_lengths)
        assert not mask.any()  # 所有token都是prompt
    
    def test_run_validation_mock(self):
        """测试验证函数（使用mock）"""
        # 创建mock对象
        student_policy = Mock()
        teacher_policy = Mock()
        val_dataloader = Mock()
        tokenizer = Mock()
        loss_fn = Mock()
        master_config = {"distillation": {"val_batches": 2}}
        
        # 设置mock返回值
        student_policy.get_forward_logits.return_value = torch.randn(2, 5, 1000)
        teacher_policy.get_forward_logits.return_value = torch.randn(2, 5, 1000)
        loss_fn.return_value = {"loss": torch.tensor(0.5)}
        
        # 模拟数据加载器
        val_dataloader.__iter__ = lambda self: iter([
            {"input_ids": torch.randint(0, 1000, (2, 5))},
            {"input_ids": torch.randint(0, 1000, (2, 5))},
        ])
        
        # 运行验证
        val_loss = run_validation(
            student_policy, teacher_policy, val_dataloader,
            tokenizer, loss_fn, master_config
        )
        
        assert isinstance(val_loss, float)
        assert val_loss > 0
        
        # 验证调用次数
        assert student_policy.eval.call_count == 1
        assert teacher_policy.eval.call_count == 1
        assert student_policy.train.call_count == 1
        assert teacher_policy.train.call_count == 1


if __name__ == "__main__":
    pytest.main([__file__])
