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

"""Hugging Face datasets for NeMo-RL."""

from .chat_templates import COMMON_CHAT_TEMPLATES
from .deepscaler import DeepScalerDataset
from .dpo import DPODataset
from .helpsteer3 import HelpSteer3Dataset
from .math_cl import MathCLDataset
from .oai_format_dataset import OpenAIFormatDataset
from .oasst import OasstDataset
from .openmathinstruct2 import OpenMathInstruct2Dataset
from .prompt_response_dataset import PromptResponseDataset
from .squad import SquadDataset

__all__ = [
    "COMMON_CHAT_TEMPLATES",
    "DeepScalerDataset",
    "DPODataset",
    "HelpSteer3Dataset",
    "MathCLDataset",
    "OpenAIFormatDataset",
    "OasstDataset",
    "OpenMathInstruct2Dataset",
    "PromptResponseDataset",
    "SquadDataset",
]
