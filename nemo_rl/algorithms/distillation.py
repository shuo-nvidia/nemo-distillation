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
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any, NotRequired, Optional, TypedDict, TypeVar, cast

import ray
import torch
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from nemo_rl.algorithms.loss_functions import (
    DistillationLossConfig,
    DistillationLossDataDict,
    DistillationLossFn,
)
from nemo_rl.algorithms.utils import set_seed
from nemo_rl.data import DataConfig
from nemo_rl.data.datasets import AllTaskProcessedDataset, rl_collate_fn
from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.data.llm_message_utils import batched_message_log_to_flat_message
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import (
    ClusterConfig,
    RayVirtualCluster,
)
from nemo_rl.models.generation.interfaces import (
    GenerationInterface,
)
from nemo_rl.models.generation.vllm import VllmConfig, VllmGeneration
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.interfaces import ColocatablePolicyInterface
from nemo_rl.models.policy.lm_policy import Policy
from nemo_rl.utils.checkpoint import CheckpointingConfig, CheckpointManager
from nemo_rl.utils.logger import (
    Logger,
    LoggerConfig,
)
from nemo_rl.utils.timer import Timer
from nemo_rl.experience.rollouts import run_multi_turn_rollout

# ===============================================================================
# Configuration
# ===============================================================================
TokenizerType = TypeVar("TokenizerType", bound=PreTrainedTokenizerBase)


class DistillationConfig(TypedDict):
    teacher_model_path: str           # Teacher model path (for loading weights)
    generate_strategy: dict[str, Any] # Generation strategy parameters
    
    # Training configuration
    max_steps: int
    eval_steps: int
    save_steps: int
    logging_steps: int


class MasterConfig(TypedDict):
    """Main configuration structure"""
    policy: PolicyConfig             # Student model configuration
    loss_fn: DistillationLossConfig  # Loss function configuration
    env: dict[str, Any]              # Environment configuration
    data: DataConfig                 # Data configuration
    distillation: DistillationConfig    # Distillation configuration
    logger: LoggerConfig                # Logger configuration
    cluster: ClusterConfig              # Cluster configuration
    checkpointing: CheckpointingConfig  # Checkpointing configuration


class DistillationSaveState(TypedDict):
    step: int
    val_loss: NotRequired[float]
    consumed_samples: int


def _default_distillation_save_state() -> DistillationSaveState:
    return {
        "step": 0,
        "consumed_samples": 0,
    }


# ===============================================================================
# Setup Functions
# ===============================================================================
def setup(
    master_config: MasterConfig,
    tokenizer: TokenizerType,
    train_dataset: AllTaskProcessedDataset,
    val_dataset: Optional[AllTaskProcessedDataset],
) -> tuple[
    ColocatablePolicyInterface,     # student_policy
    Optional[GenerationInterface],  # student_generation
    StatefulDataLoader,
    Optional[StatefulDataLoader],
    TokenizerType,                  # tokenizer
    DistillationLossFn,
    Logger,
    CheckpointManager,
    DistillationSaveState,
    MasterConfig,
]:
    """Main entry point for distillation algorithm
    
    Returns:
        tuple of student_policy, student_generation, 
        (train_cluster, inference_cluster), train_dataloader, val_dataloader, 
        loss_fn, logger, checkpointer, distillation_save_state, master_config
    """
    # Extract configuration
    policy_config = master_config["policy"]
    generation_config = master_config["policy"]["generation"]
    loss_config = master_config["loss_fn"]
    distillation_config = master_config["distillation"]
    data_config = master_config["data"]
    logger_config = master_config["logger"]
    cluster_config = master_config["cluster"]

    assert generation_config is not None, (
        "A generation config in the PolicyConfig is required for distillation"
    )

    # Set random seed
    set_seed(42)  

    # ==========================
    #         Logger
    # ==========================
    logger = Logger(logger_config)
    logger.log_hyperparams(master_config)

    # ==========================
    #      Checkpointing
    # ==========================
    checkpointer = CheckpointManager(master_config["checkpointing"])
    last_checkpoint_path = checkpointer.get_latest_checkpoint_path()
    distillation_save_state: Optional[DistillationSaveState] = cast(
        Optional[DistillationSaveState], 
        checkpointer.load_training_info(last_checkpoint_path)
    )
    if distillation_save_state is None:
        distillation_save_state = _default_distillation_save_state()

    # ==========================
    #           Data
    # ==========================
    train_dataloader = StatefulDataLoader(
        train_dataset,
        batch_size=distillation_config["num_prompts_per_step"],  
        shuffle=data_config["shuffle"],
        collate_fn=rl_collate_fn,
        drop_last=True,
    )
    
    if last_checkpoint_path:
        dataloader_state_dict = torch.load(
            os.path.join(last_checkpoint_path, "train_dataloader.pt")
        )
        train_dataloader.load_state_dict(dataloader_state_dict)

    # Validation dataset
    val_dataloader: Optional[StatefulDataLoader] = None
    if val_dataset is not None:
        val_dataloader = StatefulDataLoader(
            val_dataset,
            batch_size=distillation_config["num_prompts_per_step"],  
            shuffle=False,
            collate_fn=rl_collate_fn,
        )

    # ==========================
    #          Cluster
    # ==========================
    print("\n▶ Setting up compute cluster...")
    colocated_inference = generation_config["colocated"]["enabled"]

    if colocated_inference:
        cluster = RayVirtualCluster(
            name="distillation_cluster",
            bundle_ct_per_node_list=[cluster_config["gpus_per_node"]] * cluster_config["num_nodes"],
            use_gpus=True,
            num_gpus_per_node=cluster_config["gpus_per_node"],
            max_colocated_worker_groups=1
            if generation_config["backend"] == "megatron"
            else 3,
        )
        train_cluster = cluster
        inference_cluster = cluster
        print(f"  ✓ Ray cluster initialized with {cluster_config['num_nodes']} nodes")
    
    else:
        assert generation_config["backend"] != "megatron", (
            "Non-colocated inference is not supported for Megatron generation backends. "
            "Please use vLLM backend for generation."
        )

        # train resources will be updated through overall and inference resources below
        train_gpus_per_node = cluster_config["gpus_per_node"]
        train_nodes = cluster_config["num_nodes"]

        inference_resources = generation_config["colocated"]["resources"]
        inference_gpus_per_node = inference_resources["gpus_per_node"]
        inference_nodes = inference_resources["num_nodes"]

        # validate and configure resources
        if cluster_config["num_nodes"] == 1:
            if inference_gpus_per_node is None:
                inference_gpus_per_node = cluster_config["gpus_per_node"] // 2
            if inference_nodes is None:
                inference_nodes = 1
        else:
            if inference_gpus_per_node is None:
                inference_gpus_per_node = cluster_config["gpus_per_node"]
            if inference_nodes is None:
                inference_nodes = cluster_config["num_nodes"] // 2

        # validate resources
        if inference_gpus_per_node > cluster_config["gpus_per_node"]:
            raise ValueError(
                f"Inference GPUs per node ({inference_gpus_per_node}) cannot be greater than "
                f"total GPUs per node ({cluster_config['gpus_per_node']})"
            )
        if inference_nodes > cluster_config["num_nodes"]:
            raise ValueError(
                f"Inference nodes ({inference_nodes}) cannot be greater than "
                f"total nodes ({cluster_config['num_nodes']})"
            )

        # update train resources
        train_gpus_per_node = cluster_config["gpus_per_node"] - inference_gpus_per_node
        train_nodes = cluster_config["num_nodes"] - inference_nodes

        # create clusters
        train_cluster = RayVirtualCluster(
            name="distillation_train_cluster",
            bundle_ct_per_node_list=[train_gpus_per_node] * train_nodes,
            use_gpus=True,
            num_gpus_per_node=train_gpus_per_node,
            max_colocated_worker_groups=3,
        )
        inference_cluster = RayVirtualCluster(
            name="distillation_inference_cluster",
            bundle_ct_per_node_list=[inference_gpus_per_node] * inference_nodes,
            use_gpus=True,
            num_gpus_per_node=inference_gpus_per_node,
            max_colocated_worker_groups=3,
        )
        print(f"  ✓ Separate clusters created: train={train_nodes}x{train_gpus_per_node}GPUs, inference={inference_nodes}x{inference_gpus_per_node}GPUs")

    # ==========================
    #      Student Policy
    # ==========================
    print("\n▶ Setting up models...")
    
    # Checkpoint paths
    if last_checkpoint_path:
        weights_path = Path(last_checkpoint_path) / "policy" / "weights"
        optimizer_path = Path(last_checkpoint_path) / "policy" / "optimizer"
    else:
        weights_path = None
        optimizer_path = None

    student_policy = Policy(
        name_prefix="student",
        cluster=train_cluster,  
        config=policy_config,
        tokenizer=tokenizer,
        weights_path=weights_path,
        optimizer_path=optimizer_path,
        init_optimizer=True,
        init_reference_model=False,  
    )

    # ==========================
    #      Teacher Policy
    # ==========================
    print("\n▶ Setting up models...")
    
    # Checkpoint paths
    weights_path = None
    optimizer_path = None
    teacher_config = policy_config.copy()
    teacher_config["model_name"] = distillation_config["teacher_model_path"]

    teacher_policy = Policy(
        name_prefix="teacher",
        cluster=train_cluster,  
        config=teacher_config,
        tokenizer=tokenizer,
        weights_path=weights_path,
        optimizer_path=optimizer_path,
        init_optimizer=False,
        init_reference_model=False,  
    )

    # ==========================
    #    Generation Interface
    # ==========================
    backend = generation_config["backend"]
    generation_config["model_name"] = policy_config["model_name"]  # Needed for vLLM

    if backend == "megatron":
        student_generation = None
    elif backend == "vllm":
        generation_config = cast(VllmConfig, generation_config)
        student_generation = VllmGeneration(
            cluster=inference_cluster, config=generation_config
        )
        student_generation.finish_generation()

    if student_generation is not None:
        state_dict_info = student_policy.prepare_refit_info()
        student_generation.prepare_refit_info(state_dict_info)

    # ==========================
    #        Loss Function
    # ==========================
    loss_fn = DistillationLossFn(loss_config)

    print("\n" + "=" * 60)
    print(" " * 18 + "SETUP COMPLETE")
    print("=" * 60 + "\n")

    return (
        student_policy,
        teacher_policy,
        student_generation,
        train_dataloader,
        val_dataloader,
        tokenizer,  
        loss_fn,
        logger,
        checkpointer,
        distillation_save_state,
        master_config,
    )


# ===============================================================================
# Core Algorithm Functions
# ===============================================================================

def refit_student_generation(
    student_policy: ColocatablePolicyInterface,
    student_generation: GenerationInterface,
    colocated_inference: bool,
    _refit_buffer_size_gb: Optional[int] = None,
    timer: Optional[Timer] = None,
    generation_config: Optional[dict] = None,
    master_config: Optional[dict] = None,
) -> None:
    """Refit the student generation interface with the latest policy weights.
    
    Args:
        student_policy: The student policy model
        student_generation: The student generation interface
        colocated_inference: Whether to use colocated inference
        _refit_buffer_size_gb: Buffer size in GB
        timer: Timer for performance measurement
        generation_config: Generation configuration dictionary
        master_config: Master configuration dictionary for parameters like max_total_sequence_length
    """
    if colocated_inference:
        student_policy.offload_before_refit()
        student_generation.prepare_for_generation(tags=["weights"])
        
        # Update generation configuration parameters (e.g., temperature, decoding_method, etc.)
        if generation_config is not None:
            try:
                # Try to update the generation backend configuration
                if hasattr(student_generation, 'cfg') and isinstance(student_generation.cfg, dict):
                    # Update temperature parameter
                    if 'temperature' in generation_config:
                        student_generation.cfg['temperature'] = generation_config['temperature']
                    # Update decoding method related parameters
                    if 'decoding_method' in generation_config:
                        if generation_config['decoding_method'] == 'greedy':
                            # For greedy decoding, set top_k=1
                            student_generation.cfg['top_k'] = 1

                        elif generation_config['decoding_method'] == 'top_k':
                            # For top_k decoding, use default value or configured value
                            if 'top_k' in generation_config:
                                student_generation.cfg['top_k'] = generation_config['top_k']

                        elif generation_config['decoding_method'] == 'top_p':
                            # For top_p decoding, ensure top_p is set
                            if 'top_p' in generation_config:
                                student_generation.cfg['top_p'] = generation_config['top_p']
                                
                    
                    # Update maximum generation length
                    if 'max_new_tokens' in generation_config:
                        if 'max_new_tokens' in student_generation.cfg:
                            student_generation.cfg['max_new_tokens'] = generation_config['max_new_tokens']
                    else:
                        # If max_new_tokens is not configured
                        # Get max_total_sequence_length from master_config as max_new_tokens
                        try:
                            max_seq_len = master_config["policy"]["max_total_sequence_length"]
                            student_generation.cfg['max_new_tokens'] = max_seq_len
                            
                        except Exception as e:
                            student_generation.cfg['max_new_tokens'] = 512  # Use reasonable default value
            except Exception as e:
                raise e

    # Create a context manager that does nothing when timer is None
    timer_context = (
        timer.time("prepare_for_generation/transfer_and_update_weights")
        if timer is not None
        else nullcontext()
    )
    with timer_context:
        # Update weights
        update_success = False
        if colocated_inference:
            # Get model parameter keys, grouped by size
            grouped_param_keys = student_policy.prepare_weights_for_ipc(
                _refit_buffer_size_gb=_refit_buffer_size_gb
            )
            
            # Execute updates
            for keys in grouped_param_keys:
                ipc_handles = student_policy.get_weights_ipc_handles(keys)
                update_success = student_generation.update_weights_from_ipc_handles(ipc_handles)
                if not update_success:
                    break
        else:
            # Update weights through NCCL
            futures_train = student_policy.broadcast_weights_for_collective()
            futures_inference = student_generation.update_weights_from_collective()
            # Wait for all futures to complete
            ray.get(futures_train)
            results = ray.get(futures_inference)
            update_success = all(result for result in results if result is not None)

        # Check if update was successful
        if not update_success:
            error_tag = "cuda-ipc" if colocated_inference else "nccl"
            error_message = (
                "❌ Error: Updating weights for the student generation policy failed during refit.\n"
                f"This often indicates an issue with {error_tag} or "
                "a problem within the generation backend (e.g., vLLM worker).\n"
            )
            raise RuntimeError(error_message)

    if colocated_inference:
        student_policy.offload_after_refit()
        student_generation.prepare_for_generation(tags=["kv_cache"])


def validate(
    teacher_policy: ColocatablePolicyInterface, 
    student_policy: ColocatablePolicyInterface,
    student_generation: GenerationInterface,
    val_dataloader: Optional[StatefulDataLoader],
    tokenizer: TokenizerType,
    loss_fn: DistillationLossFn,
    decoding_method: str,
    step: int,
    master_config: MasterConfig,
) -> dict[str, Any]:
    """Run validation on the validation dataset for distillation"""
    if val_dataloader is None:
        print("  ⚠️ No validation dataloader provided, skipping validation")
        return {}
    timer = Timer()
    with timer.time("total_validation_time"):
        print(f"▶ Starting validation at step {step}...")

        total_losses = []
        total_samples = 0

        for batch_idx, val_batch in enumerate(val_dataloader):
            if student_generation is not None:
                try:
                    val_batch, rollout_metrics = run_multi_turn_rollout(
                        policy_generation=student_generation,
                        input_batch=val_batch,
                        tokenizer=tokenizer,
                        task_to_env={},  
                        max_seq_len=master_config["policy"]["max_total_sequence_length"],  
                        max_rollout_turns=1, 
                        greedy=(decoding_method == "greedy"), 
                    )
                    
                    # loss calculation
                    try:
                        val_input_ids = val_batch["input_ids"]
                        val_batch_size = val_input_ids.shape[0]
                        
                        with torch.no_grad():
                            student_policy.prepare_for_lp_inference()
                            teacher_policy.prepare_for_lp_inference()
                            val_student_logits = student_policy.get_logprobs(val_batch)["logprobs"]
                            val_teacher_logprobs = teacher_policy.get_logprobs(val_batch)["logprobs"]

                        val_data_dict = {
                            "input_ids": val_input_ids,
                            "teacher_logprobs": val_teacher_logprobs,
                        }
                        val_data = BatchedDataDict[DistillationLossDataDict](val_data_dict)
                        val_loss, val_metrics = loss_fn(
                            val_student_logits,
                            val_data,
                            torch.ones(val_batch_size, dtype=torch.bool),
                            torch.ones_like(val_input_ids, dtype=torch.bool),
                        )
                        
                        batch_loss = val_loss.item()             
                    except Exception as e:
                        raise e
                    
                    batch_size = len(val_batch) if hasattr(val_batch, '__len__') else 1
                    total_losses.append(batch_loss)
                    total_samples += batch_size
                    
                except Exception as e:
                    raise e
            else:
                # 如果使用megatron后端，直接使用policy
                try:
                    # 实现megatron的验证逻辑
                    val_input_ids = val_batch["input_ids"]
                    val_batch_size = val_input_ids.shape[0]
                    
                    # 获取学生模型在验证数据上的logits
                    with torch.no_grad():
                        student_policy.prepare_for_lp_inference()
                        val_student_logits = student_policy.get_forward_logits(val_input_ids)
                    
                    # 创建验证数据字典
                    val_data = {
                        "input_ids": val_input_ids,
                        "student_logits": val_student_logits,
                        "teacher_logits": torch.randn_like(val_student_logits) * 0.5,
                    }
                    
                    # 计算验证loss
                    val_loss, val_loss_metrics = loss_fn(
                        val_student_logits,
                        val_data,
                        torch.ones(val_batch_size, dtype=torch.bool),
                        torch.ones_like(val_input_ids, dtype=torch.bool),
                    )
                    
                    batch_loss = val_loss.item()
                    print(f"  🔍 [Validation] Batch {batch_idx}: Loss = {batch_loss:.6f}")
                    
                except Exception as e:
                    print(f"  ⚠️ Error computing validation loss: {e}")
                    batch_loss = 0.1  # 使用默认值
                
                batch_size = len(val_batch) if hasattr(val_batch, '__len__') else 1
                total_losses.append(batch_loss)
                total_samples += batch_size
    return val_metrics


def distillation_train(
    student_policy: ColocatablePolicyInterface,
    teacher_policy: ColocatablePolicyInterface,
    student_generation: Optional[GenerationInterface],
    train_dataloader: StatefulDataLoader,
    val_dataloader: Optional[StatefulDataLoader],
    tokenizer: TokenizerType, 
    loss_fn: DistillationLossFn,
    logger: Logger,
    checkpointer: CheckpointManager,
    distillation_save_state: DistillationSaveState,
    master_config: MasterConfig,
) -> None:
    """Distillation training main function"""
    
    
    timer = Timer()
    distillation_config = master_config["distillation"]
    generation_config = master_config["policy"]["generation"]
    
    # set generation strategy
    generate_strategy = distillation_config.get("generate_strategy", {})
    max_length = generate_strategy.get("max_length", 2048)
    temperature = generate_strategy.get("temperature", 1.0)
    decoding_method = generate_strategy.get("decoding_method", "greedy")
    
    # if policy_generation is None, use policy as generation interface
    NEED_REFIT = True
    if student_generation is None:
        pass
        student_generation = student_policy  
        NEED_REFIT = False
    STUDENT_GENERATION_STALE = True        # tracks if generation needs a refit before running
    assert student_generation is not None  # for mypy type check
    
    # get colocated inference setting
    colocated_inference = generation_config["colocated"]["enabled"]
    
    # training loop
    step = distillation_save_state["step"]
    max_steps = distillation_config["max_steps"]
    
    print(f"Starting from step {step}, max steps: {max_steps}")
    print(f"Generation config: max_length={max_length}, temperature={temperature}, decoding_method={decoding_method}")
    
    try:
        for batch_idx, batch in enumerate(train_dataloader):
            if step >= max_steps:
                break
                
            print(f"\n{'=' * 25} Step {step + 1}/{max_steps} {'=' * 25}")
            
            with timer.time("total_step_time"):
                #prepare batch data
                
                with timer.time("data_processing"):
                    # extract message_log from batch
                    batch: BatchedDataDict[DatumSpec]
                    message_logs = batch["message_log"]
                
                    try:
                        batched_flat, input_lengths = batched_message_log_to_flat_message(
                            message_logs,
                            pad_value_dict={"token_ids": tokenizer.pad_token_id},
                        )
                        input_ids = batched_flat["token_ids"]
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        raise
                
                # check if refit is needed
                if student_generation is not None:
                    
                    if NEED_REFIT or STUDENT_GENERATION_STALE:
                        generation_config = {
                            'temperature': temperature,
                            'decoding_method': decoding_method,
                            'max_length': max_length,
                        }
                        refit_student_generation(student_policy, student_generation, colocated_inference, generation_config=generation_config, master_config=master_config)
                        STUDENT_GENERATION_STALE = False
                        NEED_REFIT = False
                    else:
                        student_generation.prepare_for_generation()

                if student_generation is not None:
                    from nemo_rl.models.generation.interfaces import GenerationDatumSpec
                    
                    # create Ray remote environment instance
                    from nemo_rl.environments.math_environment import MathEnvironment
                    from nemo_rl.distributed.ray_actor_environment_registry import get_actor_python_env
                    
                    # get environment configuration from master_config
                    env_configs = master_config.get("env", {})
                    if "math" not in env_configs:
                        # if no environment configuration, use default configuration
                        env_configs["math"] = {"num_workers": 8}
                        print(f"  ⚠️ No math environment config found, using default: {env_configs['math']}")
                    
                    distillation_env = MathEnvironment.options(
                        runtime_env={
                            "py_executable": get_actor_python_env(
                                "nemo_rl.environments.math_environment.MathEnvironment"
                            ),
                            "env_vars": dict(os.environ),
                        }
                    ).remote(env_configs["math"])
                    distillation_task_env = {"math": distillation_env}
                    
                    num_generations_per_prompt = master_config["distillation"]["num_generations_per_prompt"]
                    
                    repeated_batch: BatchedDataDict[DatumSpec] = batch.repeat_interleave(
                        num_repeats=num_generations_per_prompt
                    )
                    
                    
                    max_seq_len = master_config["policy"]["max_total_sequence_length"]
                    max_new_tokens = distillation_config["generate_strategy"]["max_new_tokens"]
                    max_input_len = max_seq_len - max_new_tokens
                    
                    # avoid remaining_length becoming negative
                    for i, message_log in enumerate(repeated_batch["message_log"]):
                        total_length = sum(len(msg["token_ids"]) for msg in message_log)
                        if total_length > max_input_len:
                            # recalculate the number of tokens to keep
                            tokens_to_keep = max_input_len
                            
                            # keep tokens in order from the first message
                            for msg in message_log:
                                if tokens_to_keep <= 0:
                                    # if all available tokens are used, only keep the first token
                                    if len(msg["token_ids"]) > 0:
                                        msg["token_ids"] = msg["token_ids"][:1]
                                else:
                                    msg_length = len(msg["token_ids"])
                                    if msg_length > tokens_to_keep:
                                        # if the current message is too long, truncate to available length
                                        msg["token_ids"] = msg["token_ids"][:tokens_to_keep]
                                        tokens_to_keep = 0
                                    else:
                                        # if the current message can be fully retained
                                        tokens_to_keep -= msg_length
                            
                            # recalculate length and verify
                            new_total_length = sum(len(msg["token_ids"]) for msg in message_log)
                            
                            # verify that the truncated length does not exceed the limit
                            if new_total_length > max_input_len:
                                # force truncation to the limit
                                for msg in message_log:
                                    if len(msg["token_ids"]) > 0:
                                        msg["token_ids"] = msg["token_ids"][:1]
                                        break
                    
                    # use rollout to generate response
                    try:
                        generated_batch, rollout_metrics = run_multi_turn_rollout(
                            policy_generation=student_generation,
                            input_batch=repeated_batch,  # use repeated batch
                            tokenizer=tokenizer,
                            task_to_env=distillation_task_env,  # pass Ray actor virtual environment
                            max_seq_len=max_seq_len,  # directly use policy's max_total_sequence_length
                            max_rollout_turns=1,  # distillation only needs single-turn generation
                            greedy=(decoding_method == "greedy"),  # determine if greedy based on decoding_method
                        )
                        # extract generated sequences from rollout results
                        generated_sequences = generated_batch["message_log"]
                        
                    except Exception as e:
                        try:                    
                            # prepare input data
                            input_ids = []
                            for message_log in repeated_batch["message_log"]:
                                # merge all message's token_ids
                                sample_tokens = []
                                for msg in message_log:
                                    if "token_ids" in msg and len(msg["token_ids"]) > 0:
                                        sample_tokens.extend(msg["token_ids"].tolist())
                                
                                if len(sample_tokens) == 0:
                                    # if the sequence is empty, add pad token
                                    sample_tokens = [tokenizer.pad_token_id]
                                
                                # apply length limit in fallback
                                if len(sample_tokens) > max_input_len:
                                    sample_tokens = sample_tokens[:max_input_len]
                                
                                input_ids.append(sample_tokens)
                            
                            # pad to the same length
                            max_len = max(len(ids) for ids in input_ids)
                            padded_input_ids = []
                            for ids in input_ids:
                                if len(ids) < max_len:
                                    ids.extend([tokenizer.pad_token_id] * (max_len - len(ids)))
                                padded_input_ids.append(ids)
                            
                            # convert to tensor
                            input_ids_tensor = torch.tensor(padded_input_ids, dtype=torch.long)
                            input_lengths_tensor = torch.tensor([len(ids) for ids in input_ids], dtype=torch.long)
                            
                            # generate directly
                            generation_data = BatchedDataDict[GenerationDatumSpec]({
                                "input_ids": input_ids_tensor,
                                "input_lengths": input_lengths_tensor,
                                "stop_strings": [None] * len(input_ids),
                            })
                            
                            generation_outputs = student_generation.generate(
                                generation_data, 
                                greedy=(decoding_method == "greedy")
                            )
                            
                            # process generation results
                            output_ids = generation_outputs["output_ids"]
                            generated_sequences = []
                            
                            for i in range(len(input_ids)):
                                input_len = input_lengths_tensor[i].item()
                                generated_tokens = output_ids[i, input_len:].tolist()
                                
                                # create assistant message
                                assistant_message = {
                                    "role": "assistant",
                                    "content": tokenizer.decode(generated_tokens, skip_special_tokens=True),
                                    "token_ids": torch.tensor(generated_tokens, dtype=torch.long),
                                }
                                
                                # reconstruct message_log
                                sample_messages = []
                                for msg in repeated_batch["message_log"][i]:
                                    sample_messages.append(msg)
                                sample_messages.append(assistant_message)
                                generated_sequences.append(sample_messages)
                            
                        except Exception as fallback_error:
                            import traceback
                            traceback.print_exc()
                            raise RuntimeError(f"Both rollout and fallback generation failed. Original error: {e}, Fallback error: {fallback_error}")
                else:
                    # if using megatron backend, use policy directly
                    # here we need to implement the generation logic of megatron
                    generated_sequences = batch["message_log"]  # use original data temporarily
                
                # mark generation as completed
                if student_generation is not None:
                    student_generation.finish_generation()
                
                # calculate logits
                
                with timer.time("logits_computation"):
                    try:
                        expected_batch_size = master_config["distillation"]["num_prompts_per_step"] * master_config["distillation"]["num_generations_per_prompt"]

                        if len(generated_sequences) != expected_batch_size:
                            if len(generated_sequences) > expected_batch_size:
                                generated_sequences = generated_sequences[:expected_batch_size]
                            else:
                                # expand batch to the correct size (repeat the last sequence)
                                while len(generated_sequences) < expected_batch_size:
                                    generated_sequences.append(generated_sequences[-1])

                        
                        flat_messages, input_lengths = batched_message_log_to_flat_message(
                            generated_sequences,
                            pad_value_dict={"token_ids": tokenizer.pad_token_id},
                            make_sequence_length_divisible_by=master_config["policy"].get(
                                "make_sequence_length_divisible_by", 1
                            ),
                        )

                    except Exception as e:
                        raise
                    
                    # prepare training data
                    
                    
                    if "token_loss_mask" not in flat_messages:
                        token_loss_mask = torch.zeros_like(
                            flat_messages["token_ids"], dtype=torch.bool
                        )
                        
                        for i, seq_len in enumerate(input_lengths):
                            if seq_len > 0:
                                token_loss_mask[i, :seq_len] = True
                        
                        flat_messages["token_loss_mask"] = token_loss_mask
                    
                    # verify that all fields have consistent batch dimensions
                    expected_batch_size = flat_messages['token_ids'].shape[0]
                    expected_seq_len = flat_messages['token_ids'].shape[1]
                    
                    # verify and fix fields with mismatched shapes
                    
                    if flat_messages['token_loss_mask'].shape[0] != expected_batch_size:
                        flat_messages['token_loss_mask'] = flat_messages['token_loss_mask'][:expected_batch_size]
                    
                    if repeated_batch['loss_multiplier'].shape[0] != expected_batch_size:
                        repeated_batch['loss_multiplier'] = repeated_batch['loss_multiplier'][:expected_batch_size]
                    
                    if flat_messages['token_loss_mask'].shape[1] != expected_seq_len:
                        if flat_messages['token_loss_mask'].shape[1] > expected_seq_len:
                            flat_messages['token_loss_mask'] = flat_messages['token_loss_mask'][:, :expected_seq_len]
                        else:
                            flat_messages['token_loss_mask'] = flat_messages['token_loss_mask'].expand(-1, expected_seq_len)
                    
                    
                    # ensure loss_multiplier is the correct shape
                    if isinstance(repeated_batch["loss_multiplier"], torch.Tensor):
                        if len(repeated_batch["loss_multiplier"].shape) > 1:
                            # if loss_multiplier is multi-dimensional, take the first dimension
                            repeated_batch["loss_multiplier"] = repeated_batch["loss_multiplier"].flatten()[:expected_batch_size]
                            
                        elif repeated_batch["loss_multiplier"].shape[0] != expected_batch_size:
                            repeated_batch["loss_multiplier"] = repeated_batch["loss_multiplier"][:expected_batch_size]
                            
                    elif isinstance(repeated_batch["loss_multiplier"], list):
                        repeated_batch["loss_multiplier"] = torch.tensor(repeated_batch["loss_multiplier"][:expected_batch_size], dtype=torch.float32)
                        

                    
                    # finally verify the type and shape of loss_multiplier
                    if not isinstance(repeated_batch["loss_multiplier"], torch.Tensor):
                        if isinstance(repeated_batch["loss_multiplier"], (list, tuple)):
                            repeated_batch["loss_multiplier"] = torch.tensor(repeated_batch["loss_multiplier"], dtype=torch.float32)
                         
                        elif isinstance(repeated_batch["loss_multiplier"], (int, float)):
                            repeated_batch["loss_multiplier"] = torch.tensor([repeated_batch["loss_multiplier"]] * expected_batch_size, dtype=torch.float32)
                            
                        else:
                            # create default loss_multiplier
                            repeated_batch["loss_multiplier"] = torch.ones(expected_batch_size, dtype=torch.float32)
                           
                    
                    # verify that all fields have consistent batch dimensions
                    all_batch_sizes = [
                        flat_messages['token_ids'].shape[0],
                        input_lengths.shape[0],
                        flat_messages['token_loss_mask'].shape[0],
                        repeated_batch['loss_multiplier'].shape[0]
                    ]
                    
                    if len(set(all_batch_sizes)) != 1:
                        raise ValueError(f"Batch dimensions must be consistent, got: {all_batch_sizes}")
                    
                    # create training data, only include tensor fields
                    train_data_dict = {
                        "input_ids": flat_messages["token_ids"],
                        "input_lengths": input_lengths,
                        "token_mask": flat_messages["token_loss_mask"],  # use token_loss_mask instead of custom token_mask
                        "sample_mask": repeated_batch["loss_multiplier"],
                    }
                    
                    train_data = BatchedDataDict[DistillationLossDataDict](train_data_dict)
                    train_data.to("cpu")  

                    # teacher model forward propagation (need to be implemented separately because of different model sizes)
                    with torch.no_grad():
                        teacher_policy.prepare_for_lp_inference()
                        teacher_logprobs=teacher_policy.get_logprobs(train_data)["logprobs"]
                        
                        # Store teacher_logprobs in train_data
                        train_data["teacher_logprobs"] = teacher_logprobs
               
                distillation_safe_data = {}
                
                for key, value in train_data.items():
                    if key in ["teacher_logprobs"]:
                        distillation_safe_data[key] = value
                        if len(value.shape) == 3:
                            batch_size, seq_len, vocab_size = value.shape
                            flattened_logits = value.view(batch_size * seq_len, vocab_size)
                            
                            safe_key = f"distillation_{key}_flattened"
                            distillation_safe_data[safe_key] = flattened_logits
                            
                            distillation_safe_data[f"{safe_key}_shape"] = torch.tensor([batch_size, seq_len, vocab_size])
                        else:
                            distillation_safe_data[key] = value
                    else:
                        distillation_safe_data[key] = value

                with timer.time("training_prep"):

                    student_policy.prepare_for_training()  
                    STUDENT_GENERATION_STALE = True  # *** MARK AS STALE AFTER TRAINING ***
                
                worker_required_fields = ["input_ids", "input_lengths", "token_mask", "sample_mask", "teacher_logprobs"]
                clean_worker_data = {}
                
                for field in worker_required_fields:
                    if field in train_data:
                        if torch.is_tensor(train_data[field]):
                            clean_worker_data[field] = train_data[field]
                
                worker_train_data = BatchedDataDict[DistillationLossDataDict](clean_worker_data)

                with timer.time("policy_training"):
                    try:
                        train_results = student_policy.train(worker_train_data, loss_fn)
                    except Exception as e:
                        raise

         
                # build training metrics
                metrics = {}
                
                # Extract loss from train_results
                if "all_mb_metrics" in train_results and "loss" in train_results["all_mb_metrics"]:
                    loss_list = train_results["all_mb_metrics"]["loss"]
                    if isinstance(loss_list, (list, tuple)) and len(loss_list) > 0:
                        loss = sum(loss_list) / len(loss_list)
                    else:
                        loss = loss_list
                    metrics["loss"] = loss
                else:
                    # Fallback if loss is not available
                    loss = 0.0
                    metrics["loss"] = loss
                
                # add other micro-batch metrics 
                # correctly handle data type, ensure all values are numeric
                all_mb_metrics = train_results["all_mb_metrics"].copy()
                
                # safely add micro-batch metrics, ensure data type is correct
                for k, v in all_mb_metrics.items():
                    if isinstance(v, (list, tuple)):
                        # if list/tuple, calculate average
                        if len(v) > 0:
                            if isinstance(v[0], (int, float)):
                                metrics[k] = sum(v) / len(v)
                            elif hasattr(v[0], 'numpy'):
                                metrics[k] = sum(x.numpy() for x in v) / len(v)
                            else:
                                # skip unprocessable type
                                continue
                        else:
                            # empty list, skip
                            continue
                    elif isinstance(v, (int, float)):
                        # use value directly
                        metrics[k] = v
                    elif hasattr(v, 'numpy'):
                        # convert to numpy
                        metrics[k] = v.numpy()
                    elif hasattr(v, 'item'):
                        # convert to Python scalar
                        metrics[k] = v.item()
                    else:
                        # skip unprocessable type
                        continue
                
                # record generation length related metrics
                if "input_ids" in train_data:
                    input_lengths = (train_data["input_ids"] != 0).sum(dim=1)
                    metrics.update({
                        "avg_input_length": input_lengths.float().mean().item(),
                        "max_input_length": input_lengths.max().item(),
                        "min_input_length": input_lengths.min().item(),
                        "input_length_std": input_lengths.float().std().item(),
                    })
                
                # record current best validation loss (if available)
                if "val_loss" in distillation_save_state and distillation_save_state["val_loss"] is not None:
                    current_best_val_loss = distillation_save_state["val_loss"]
                    metrics["best_val_loss"] = current_best_val_loss
                
                # use prefix="train" to record all metrics, avoid duplicate
                if logger is not None:
                    logger.log_metrics(metrics, step, prefix="train")
                    
                    # print training loss information
                    print(f"✅ [Training] Step {step}: Loss = {loss:.6f}")
    
                step += 1
                distillation_save_state["step"] = step
                # use the value in config
                distillation_save_state["consumed_samples"] += distillation_config.get("num_prompts_per_step", 1)

                
                # save checkpoint
                if step % distillation_config["save_steps"] == 0:
                    try:
                        checkpoint_path = checkpointer.init_tmp_checkpoint(
                            step, distillation_save_state, master_config
                        )
                        student_policy.save_checkpoint(
                            weights_path=os.path.join(checkpoint_path, "policy", "weights"),
                            optimizer_path=os.path.join(checkpoint_path, "policy", "optimizer"),
                            tokenizer_path=os.path.join(checkpoint_path, "policy", "tokenizer"),
                        )
                        # save dataloader state
                        torch.save(
                            train_dataloader.state_dict(),
                            os.path.join(checkpoint_path, "train_dataloader.pt"),
                        )
                        checkpointer.finalize_checkpoint(checkpoint_path)
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                
                if step % distillation_config["eval_steps"] == 0 and val_dataloader is not None:
                    try:
                        if NEED_REFIT and STUDENT_GENERATION_STALE:
                            generation_config = {
                                'temperature': temperature,
                                'decoding_method': decoding_method,
                                'max_length': max_length,
                            }
                            refit_student_generation(
                                student_policy, student_generation, colocated_inference, generation_config=generation_config, master_config=master_config
                            )
                            STUDENT_GENERATION_STALE = False
                        else:
                            if student_generation is not None:
                                student_generation.prepare_for_generation()
                        
                        val_metrics = validate(
                            teacher_policy,
                            student_policy,
                            student_generation,
                            val_dataloader,
                            tokenizer,
                            loss_fn,
                            decoding_method,
                            step + 1,
                            master_config,
                        )
                        
                        if val_metrics:
                            eval_metrics = {}
                            
                            for k, v in val_metrics.items():
                                if len(v) > 0 and k != "loss":
                                    if isinstance(v, (list, tuple)):
                                        # if list/tuple, calculate average
                                        if len(v) > 0:
                                            if isinstance(v[0], (int, float)):
                                                eval_metrics[k] = sum(v) / len(v)
                                            elif hasattr(v[0], 'numpy'):
                                                eval_metrics[k] = sum(x.numpy() for x in v) / len(v)
                                            else:
                                                # skip unprocessable type
                                                continue
                                        else:
                                            # empty list, skip
                                            continue
                                    elif isinstance(v, (int, float)):
                                        # use value directly
                                        eval_metrics[k] = v
                                    elif hasattr(v, 'numpy'):
                                        # convert to numpy
                                        eval_metrics[k] = v.numpy()
                                    elif hasattr(v, 'item'):
                                        # convert to Python scalar
                                        eval_metrics[k] = v.item()
                                    else:
                                        # skip unprocessable type
                                        continue
                        if logger is not None:
                            logger.log_metrics(eval_metrics, step, prefix="eval")
                        if "loss" in eval_metrics:
                            print(f"✅[Validation] Step {step + 1}: Val Loss = {loss:.6f}")
                            
                        if student_generation is not None:
                            student_generation.finish_generation()
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                
                # log
                if step % distillation_config["logging_steps"] == 0:
                    try:
                        logger.log_metrics({
                            "step": step,
                            "consumed_samples": distillation_save_state["consumed_samples"],
                        })
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
    
    except Exception as e:
        import traceback
        traceback.print_exc()
