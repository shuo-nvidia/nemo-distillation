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
# See the License for the specific language governing permissions and limitations.

"""
Math-CL数据集支持
将pe-nlp/math-cl数据集转换为hugging face格式，适配NeMo-RL框架
"""

import os
from typing import Any, Optional

from datasets import Dataset, load_dataset

from nemo_rl.data.interfaces import TaskDataSpec


def format_math(data: dict[str, str | float | int]) -> dict[str, list[Any] | str]:
    """将Math-CL数据格式化为标准消息格式"""
    return {
        "messages": [
            {
                "role": "user",
                "content": data["problem"],
            },
            {
                "role": "assistant",
                "content": data["ground_truth_answer"],
            },
        ],
        # For v0.1 release, nemo rl datasets require a task_name key such that user can map a task processor per unique task.
        "task_name": "math",
    }


def prepare_math_cl_dataset(
    split: str = "train",
    seed: int = 42,
    test_size: float = 0.1,
    max_samples: Optional[int] = None,
) -> dict[str, Dataset | None]:
    """加载并分割Math-CL数据集为训练和验证集"""
    # 加载原始数据集
    original_ds = load_dataset("pe-nlp/math-cl", split=split)
    
    # 如果指定了最大样本数，则限制数据集大小
    if max_samples is not None:
        original_ds = original_ds.select(range(min(max_samples, len(original_ds))))
    
    # 使用HF的train_test_split分割数据集
    split_ds = original_ds.train_test_split(test_size=test_size, seed=seed)
    
    # 格式化样本，移除原始列
    train_formatted = split_ds["train"].map(
        format_math,
        remove_columns=split_ds["train"].column_names,
    )
    val_formatted = split_ds["test"].map(
        format_math,
        remove_columns=split_ds["test"].column_names,
    )
    
    return {
        "train": train_formatted,
        "validation": val_formatted,
    }


class MathCLDataset:
    """Math-CL数据集接口
    将pe-nlp/math-cl数据集转换为hugging face格式，支持训练和验证
    """
    
    def __init__(
        self,
        split: Optional[str] = None,
        seed: int = 42,
        max_samples: Optional[int] = None,
        prompt_file: Optional[str] = None,
        system_prompt_file: Optional[str] = None,
    ):
        """初始化Math-CL数据集
        
        Args:
            split: 数据集分割，默认为"train"
            seed: 随机种子，用于可重现的分割
            max_samples: 最大样本数量，用于调试和快速测试
            prompt_file: 提示文件路径，如果为None则使用默认路径
            system_prompt_file: 系统提示文件路径，如果为None则使用默认路径
        """
        self.split = split or "train"
        self.seed = seed
        self.max_samples = max_samples
        self.prompt_file = prompt_file or "examples/prompts/math_cl.txt"
        self.system_prompt_file = system_prompt_file or "examples/prompts/cot.txt"
        self._load_raw_dataset()
        self._convert_to_hf_format()
        print(f"✓ Math-CL dataset loaded: {len(self.formatted_ds[self.split])} samples")
    
    def _load_raw_dataset(self):
        """加载原始数据集"""
        self.raw_dataset = load_dataset("pe-nlp/math-cl", split=self.split)
        if self.max_samples is not None:
            self.raw_dataset = self.raw_dataset.select(range(min(self.max_samples, len(self.raw_dataset))))
        print(f"  ✓ Raw dataset loaded: {len(self.raw_dataset)} samples")
    
    def _convert_to_hf_format(self):
        """转换为Hugging Face格式"""
        # 分割为训练和验证集
        split_dataset = self.raw_dataset.train_test_split(test_size=0.1, seed=self.seed)
        
        # 格式化数据
        train_formatted = split_dataset["train"].map(
            format_math,
            remove_columns=split_dataset["train"].column_names,
        )
        val_formatted = split_dataset["test"].map(
            format_math,
            remove_columns=split_dataset["test"].column_names,
        )
        
        self.formatted_ds = {
            "train": train_formatted,
            "validation": val_formatted,
        }
        
        # 不再在这里创建TaskDataSpec，应该由外部传入
        # self.task_spec = TaskDataSpec(
        #     task_name="math",
        #     prompt_file=self.prompt_file,
        #     system_prompt_file=self.system_prompt_file,
        # )
    
    def __len__(self) -> int:
        """获取数据集长度"""
        return len(self.formatted_ds[self.split])
    
    def __getitem__(self, idx: int) -> dict[str, Any]:
        """获取指定索引的样本"""
        return self.formatted_ds[self.split][idx]
