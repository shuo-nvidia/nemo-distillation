#!/usr/bin/env python3
"""
蒸馏训练脚本 - 重构为单一Policy模式（参考GRPO）
使用单一Policy对象，避免Ray命名冲突和资源冲突
"""

import argparse
import os
import pprint
from collections import defaultdict
from typing import Any, Optional

from omegaconf import OmegaConf
from transformers import PreTrainedTokenizerBase

from nemo_rl.algorithms.distillation import MasterConfig, distillation_train, setup
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data import DataConfig
from nemo_rl.data.datasets import AllTaskProcessedDataset
from nemo_rl.data.hf_datasets.openmathinstruct2 import OpenMathInstruct2Dataset
from nemo_rl.data.hf_datasets.deepscaler import DeepScalerDataset
from nemo_rl.data.hf_datasets.math_cl import MathCLDataset
from nemo_rl.data.interfaces import (
    DatumSpec,
    LLMMessageLogType,
    TaskDataProcessFnCallable,
    TaskDataSpec,
)
from nemo_rl.distributed.ray_actor_environment_registry import (
    get_actor_python_env,
)
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.environments.math_environment import MathEnvironment
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils.config import load_config, parse_hydra_overrides
from nemo_rl.utils.logger import get_next_experiment_dir

OmegaConf.register_new_resolver("mul", lambda a, b: a * b)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run distillation training with configuration")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )

    # Parse known args for the script
    args, overrides = parser.parse_known_args()

    return args, overrides


# ===============================================================================
#                             Math Data Processor
# ===============================================================================
TokenizerType = PreTrainedTokenizerBase


# TaskDataProcessFnCallable
def hf_data_processor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer: TokenizerType,
    max_seq_length: int,  # 改为必需参数，与GRPO版本一致
    idx: int,
) -> DatumSpec:
    """处理Hugging Face格式的数据，转换为NeMo-RL格式（与GRPO版本保持一致）"""
    # 安全检查：确保prompt存在
    if task_data_spec.prompt is None:
        print(f"  ❌ TaskDataSpec.prompt is None!")
        print(f"  🔍 task_data_spec.prompt_file: {task_data_spec.prompt_file}")
        print(f"  🔍 Current working directory: {os.getcwd()}")
        if task_data_spec.prompt_file:
            print(f"  🔍 Absolute prompt file path: {os.path.abspath(task_data_spec.prompt_file)}")
            if os.path.exists(task_data_spec.prompt_file):
                print(f"  🔍 Prompt file exists but could not be loaded")
                try:
                    with open(task_data_spec.prompt_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        print(f"  📝 File content (first 100 chars): {content[:100]}...")
                except Exception as e:
                    print(f"  ❌ Failed to read file: {e}")
            else:
                print(f"  🔍 Prompt file does not exist")
        
        raise ValueError(
            f"TaskDataSpec.prompt is None. This usually means the prompt file "
            f"'{task_data_spec.prompt_file}' could not be loaded or is empty. "
            f"Current working directory: {os.getcwd()}, "
            f"Absolute prompt file path: {os.path.abspath(task_data_spec.prompt_file) if task_data_spec.prompt_file else 'None'}"
        )

    # 获取原始消息数据
    messages = datum_dict["messages"]
    problem = messages[0]["content"]
    extra_env_info = {"ground_truth": messages[1]["content"]}

    message_log: LLMMessageLogType = []
    
    # 创建用户消息，使用prompt模板
    try:
        # 使用与GRPO math任务相同的格式化方法
        # math.txt 使用 {} 位置占位符，可以直接用 format(problem)
        formatted_content = task_data_spec.prompt.format(problem)
        '''
        print(f"  🔍 [DEBUG] Using GRPO-style formatting with {{}} placeholder")
        print(f"  🔍 [DEBUG] Prompt template: {task_data_spec.prompt[:100]}...")
        print(f"  🔍 [DEBUG] Problem: {problem[:100]}...")
        print(f"  🔍 [DEBUG] Formatted content: {formatted_content[:100]}...")
        '''
    except Exception as e:
        print(f"  ❌ [DEBUG] Failed to format prompt: {e}")
        '''
        print(f"  🔍 [DEBUG] task_data_spec.prompt type: {type(task_data_spec.prompt)}")
        print(f"  🔍 [DEBUG] task_data_spec.prompt value: {task_data_spec.prompt}")
        print(f"  🔍 [DEBUG] problem type: {type(problem)}")
        print(f"  🔍 [DEBUG] problem value: {problem}")
        print(f"  🔍 [DEBUG] Available placeholders in prompt: {[s for s in task_data_spec.prompt.split('{') if '}' in s]}")
        '''
        raise
    
    user_message = {
        "role": "user",
        "content": formatted_content,
    }
    message: list[str] = tokenizer.apply_chat_template(  # type: ignore
        [user_message],
        tokenize=False,
        add_generation_prompt=True,
        add_special_tokens=False,
    )

    user_message["token_ids"] = tokenizer(
        message,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"][0]
    user_message["content"] = message
    message_log.append(user_message)

    length = sum(len(m["token_ids"]) for m in message_log)

    loss_multiplier = 1.0
    if length > max_seq_length:
        # 修复：与GRPO版本完全一致，正确截断序列
        # 计算每个message可以保留的token数量
        tokens_per_message = max_seq_length // len(message_log)
        for chat_message in message_log:
            # 保留每个message的token，但不超过限制
            chat_message["token_ids"] = chat_message["token_ids"][:tokens_per_message]
        # 重新计算长度
        length = sum(len(m["token_ids"]) for m in message_log)
        loss_multiplier = 0.0

    output: DatumSpec = {
        "message_log": message_log,
        "length": length,
        "extra_env_info": extra_env_info,
        "loss_multiplier": loss_multiplier,
        "idx": idx,
        "task_name": datum_dict["task_name"],
    }
    return output


def setup_data(
    tokenizer: TokenizerType,
    data_config: DataConfig,
    env_configs: dict[str, Any],
    seed: int,
) -> tuple[
    AllTaskProcessedDataset,
    Optional[AllTaskProcessedDataset],
    dict[str, EnvironmentInterface],
    dict[str, EnvironmentInterface],
]:
    print("\n▶ Setting up data...")
    
    # 添加详细的路径调试信息
    prompt_file = data_config["prompt_file"]
    system_prompt_file = data_config["system_prompt_file"]
    
    # 转换为绝对路径，确保文件能被找到
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 修复路径拼接逻辑 - 使用更简单的方法
    if not os.path.isabs(prompt_file):
        # 假设prompt_file是相对于examples/prompts目录的
        prompt_file = os.path.join(current_dir, "prompts", prompt_file)
    
    # 只有当system_prompt_file不为None时才进行路径拼接
    if system_prompt_file is not None and not os.path.isabs(system_prompt_file):
        system_prompt_file = os.path.join(current_dir, "prompts", system_prompt_file)
    
    print(f"  📁 Prompt file path: {prompt_file}")
    print(f"  📁 System prompt file path: {system_prompt_file}")
    
    # 检查文件是否存在
    '''
    print(f"  🔍 Checking prompt file: {prompt_file}")
    print(f"  🔍 Current working directory: {os.getcwd()}")
    print(f"  🔍 Absolute prompt file path: {os.path.abspath(prompt_file)}")
    '''
    if os.path.exists(prompt_file):
        print(f"  ✅ Prompt file exists: {prompt_file}")
        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                prompt_content = f.read()
                print(f"  📝 Prompt file content (first 100 chars): {prompt_content[:100]}...")
                print(f"  📝 Prompt file length: {len(prompt_content)} characters")
        except Exception as e:
            print(f"  ❌ Failed to read prompt file: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"  ❌ Prompt file does not exist: {prompt_file}")
        # 尝试列出目录内容
        prompt_dir = os.path.dirname(prompt_file)
        if os.path.exists(prompt_dir):
            print(f"  🔍 Prompt directory exists: {prompt_dir}")
            try:
                files = os.listdir(prompt_dir)
                print(f"  🔍 Files in prompt directory: {files}")
            except Exception as e:
                print(f"  ❌ Failed to list directory: {e}")
        else:
            print(f"  ❌ Prompt directory does not exist: {prompt_dir}")
    
    if system_prompt_file is not None and os.path.exists(system_prompt_file):
        print(f"  ✅ System prompt file exists: {system_prompt_file}")
    elif system_prompt_file is None:
        print(f"  ℹ️ System prompt file is None (not configured)")
    else:
        print(f"  ❌ System prompt file does not exist: {system_prompt_file}")
    
    # 创建TaskDataSpec，并处理可能的异常
    try:
        math_task_spec = TaskDataSpec(
            task_name="math",
            prompt_file=prompt_file,
            system_prompt_file=system_prompt_file,
        )
        print(f"  ✅ TaskDataSpec created successfully")
    except Exception as e:
        print(f"  ❌ Failed to create TaskDataSpec: {e}")
        print(f"  🔍 Attempting to create TaskDataSpec with manual prompt loading...")
        
        # 手动加载prompt文件
        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                manual_prompt = f.read()
                print(f"  📝 Manually loaded prompt (first 100 chars): {manual_prompt[:100]}...")
                
                # 创建TaskDataSpec，不指定prompt_file，避免自动加载
                math_task_spec = TaskDataSpec(
                    task_name="math",
                    prompt_file=None,
                    system_prompt_file=None,
                )
                # 手动设置prompt
                math_task_spec.prompt = manual_prompt
                math_task_spec.system_prompt = None
                print(f"  ✅ TaskDataSpec created with manual prompt loading")
                
        except Exception as e2:
            print(f"  ❌ Manual prompt loading also failed: {e2}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Failed to create TaskDataSpec: {e}")
    
    # 检查TaskDataSpec是否正确初始化
    '''
    print(f"  🔍 TaskDataSpec.prompt is None: {math_task_spec.prompt is None}")
    print(f"  🔍 TaskDataSpec.system_prompt is None: {math_task_spec.system_prompt is None}")
    print(f"  🔍 TaskDataSpec.prompt_file: {math_task_spec.prompt_file}")
    print(f"  🔍 TaskDataSpec.system_prompt_file: {math_task_spec.system_prompt_file}")
    '''
    if math_task_spec.prompt is not None:
        print(f"  📝 TaskDataSpec.prompt (first 100 chars): {math_task_spec.prompt[:100]}...")
        print(f"  📝 TaskDataSpec.prompt length: {len(math_task_spec.prompt)} characters")
    else:
        print(f"  ⚠️ TaskDataSpec.prompt is None - this will cause errors!")
        print(f"  🔍 Attempting to manually load prompt file...")
        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                manual_prompt = f.read()
                print(f"  📝 Manually loaded prompt (first 100 chars): {manual_prompt[:100]}...")
                # 手动设置prompt
                math_task_spec.prompt = manual_prompt
                print(f"  ✅ Manually set TaskDataSpec.prompt")
        except Exception as e:
            print(f"  ❌ Failed to manually load prompt: {e}")
            import traceback
            traceback.print_exc()

    # Load dataset using nemo rl datasets (优先使用GRPO的成熟实现)
    if data_config["dataset_name"] == "OpenMathInstruct-2":
        print("Loading nvidia/OpenMathInstruct2Dataset for training and validation")
        data: Any = OpenMathInstruct2Dataset(seed=seed)
    elif data_config["dataset_name"] == "DeepScaler":
        print("Loading agentica-org/DeepScaleR-Preview-Dataset for training and validation")
        data: Any = DeepScalerDataset(seed=seed)
    elif data_config["dataset_name"] == "pe-nlp/math-cl":
        print("Loading pe-nlp/math-cl dataset for training and validation")
        data: Any = MathCLDataset(
            seed=seed,
            prompt_file=data_config["prompt_file"],
            system_prompt_file=data_config["system_prompt_file"],
        )
    else:
        raise ValueError(f"No processor for dataset {data_config['dataset_name']}.")

    task_data_processors: dict[str, tuple[TaskDataSpec, TaskDataProcessFnCallable]] = {}
    # 为所有任务设置默认处理器，但不使用lambda函数
    task_data_processors["math"] = (math_task_spec, hf_data_processor)
    
    # 添加调试信息，验证TaskDataSpec的状态
    '''
    print(f"  🔍 TaskDataSpec.prompt is None: {math_task_spec.prompt is None}")
    print(f"  🔍 TaskDataSpec.prompt_file: {math_task_spec.prompt_file}")
    print(f"  🔍 TaskDataSpec.prompt length: {len(math_task_spec.prompt) if math_task_spec.prompt else 'None'}")
    print(f"  🔍 task_data_processors['math'][0].prompt is None: {task_data_processors['math'][0].prompt is None}")
    print(f"  🔍 task_data_processors['math'][0] is math_task_spec: {task_data_processors['math'][0] is math_task_spec}")
    '''
    math_env = MathEnvironment.options(  # type: ignore # it's wrapped with ray.remote
        runtime_env={
            "py_executable": get_actor_python_env(
                "nemo_rl.environments.math_environment.MathEnvironment"
            ),
            "env_vars": dict(os.environ),  # Pass thru all user environment variables
        }
    ).remote(env_configs["math"])
    dataset = AllTaskProcessedDataset(
        data.formatted_ds["train"],
        tokenizer,
        math_task_spec,
        task_data_processors,
        max_seq_length=data_config["max_input_seq_length"],
    )
    '''
    # 添加调试信息
    print(f"  🔍 Dataset length: {len(dataset)}")
    print(f"  🔍 First datum keys: {list(data.formatted_ds['train'][0].keys())}")
    print(f"  🔍 TaskDataSpec.prompt length: {len(math_task_spec.prompt) if math_task_spec.prompt else 'None'}")
    '''
    # 测试第一个数据项的处理
    try:
        first_datum = dataset[0]
        print(f"  ✅ First datum processed successfully: {first_datum.keys()}")
    except Exception as e:
        print(f"  ❌ Failed to process first datum: {e}")
        import traceback
        traceback.print_exc()

    val_dataset: Optional[AllTaskProcessedDataset] = None
    if data.formatted_ds["validation"]:
        val_dataset = AllTaskProcessedDataset(
            data.formatted_ds["validation"],
            tokenizer,
            math_task_spec,
            task_data_processors,
            max_seq_length=data_config["max_input_seq_length"],
        )
    else:
        val_dataset = None

    task_to_env: dict[str, EnvironmentInterface] = defaultdict(lambda: math_env)
    task_to_env["math"] = math_env
    return dataset, val_dataset, task_to_env, task_to_env


def main() -> None:
    """Main entry point."""
    # Parse arguments
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__), "configs", "distillation_math_cl.yaml"
        )

    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")

    if overrides:
        print(f"Overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)

    config: MasterConfig = OmegaConf.to_container(config, resolve=True)
    print("Applied CLI overrides")

    # Print config
    print("Final config:")
    pprint.pprint(config)

    # Get the next experiment directory with incremented ID
    config["logger"]["log_dir"] = get_next_experiment_dir(config["logger"]["log_dir"])
    print(f"📊 Using log directory: {config['logger']['log_dir']}")
    if config["checkpointing"]["enabled"]:
        print(
            f"📊 Using checkpoint directory: {config['checkpointing']['checkpoint_dir']}"
        )

    init_ray()

    # setup tokenizer
    tokenizer = get_tokenizer(config["policy"]["tokenizer"])
    
    # 确保生成配置正确（与GRPO保持一致）
    if config["policy"]["generation"] is not None:
        from nemo_rl.models.generation import configure_generation_config
        config["policy"]["generation"] = configure_generation_config(
            config["policy"]["generation"], tokenizer
        )
        print(f"  ✅ Generation config configured with tokenizer settings")
        print(f"  🔍 pad_token_id: {config['policy']['generation'].get('pad_token_id', 'Not set')}")
        print(f"  🔍 stop_token_ids: {config['policy']['generation'].get('stop_token_ids', 'Not set')}")
    else:
        print(f"  ⚠️ No generation config found, this may cause issues")
    
    # setup data
    (
        dataset,
        val_dataset,
        task_to_env,
        val_task_to_env,
    ) = setup_data(tokenizer, config["data"], config["env"], 42)  # 使用固定种子

    (
        student_policy,
        student_generation,
        cluster,
        dataloader,
        val_dataloader,
        tokenizer,  # 添加tokenizer
        loss_fn,
        logger,
        checkpointer,
        distillation_state,
        master_config,
    ) = setup(config, tokenizer, dataset, val_dataset)

    distillation_train(
        student_policy,
        student_generation,
        cluster,
        dataloader,
        val_dataloader,
        tokenizer,  # 传递tokenizer参数
        loss_fn,
        logger,
        checkpointer,
        distillation_state,
        master_config,
    )


if __name__ == "__main__":
    main()
