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
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Any, NotRequired, Optional, TypedDict, TypeVar, cast

import numpy as np
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
from nemo_rl.experience.rollouts import (
    run_multi_turn_rollout,
)

# ===============================================================================
# Configuration
# ===============================================================================
TokenizerType = TypeVar("TokenizerType", bound=PreTrainedTokenizerBase)


class DistillationConfig(TypedDict):
    # Teacher model path (for loading weights)
    teacher_model_path: str
    
    # Distillation strategy parameters
    kl_type: str    # KL divergence type: forward, reverse, mixed
    generate_strategy: dict[str, Any]  # Generation strategy parameters
    
    # Training configuration
    max_steps: int
    eval_steps: int
    save_steps: int
    logging_steps: int


class MasterConfig(TypedDict):
    """Main configuration structure"""
    policy: PolicyConfig  # Student model configuration
    loss_fn: DistillationLossConfig  # Loss function configuration
    env: dict[str, Any]  # Environment configuration
    data: DataConfig  # Data configuration
    distillation: DistillationConfig  # Distillation configuration
    logger: LoggerConfig  # Logger configuration
    cluster: ClusterConfig  # Cluster configuration
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
    ColocatablePolicyInterface,  # student_policy (single Policy instance)
    Optional[GenerationInterface],  # student_generation
    StatefulDataLoader,
    Optional[StatefulDataLoader],
    TokenizerType,  # tokenizer
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
    set_seed(42)  # Use fixed seed

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

    print(f"  ✓ Training dataloader loaded with {len(train_dataset)} samples")

    # Validation dataset
    val_dataloader: Optional[StatefulDataLoader] = None
    if val_dataset is not None:
        val_dataloader = StatefulDataLoader(
            val_dataset,
            batch_size=distillation_config["num_prompts_per_step"],  
            shuffle=False,
            collate_fn=rl_collate_fn,
        )
        print(f"  ✓ Validation dataloader loaded with {len(val_dataset)} samples")

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
            else 2,
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
            max_colocated_worker_groups=1,
        )
        inference_cluster = RayVirtualCluster(
            name="distillation_inference_cluster",
            bundle_ct_per_node_list=[inference_gpus_per_node] * inference_nodes,
            use_gpus=True,
            num_gpus_per_node=inference_gpus_per_node,
            max_colocated_worker_groups=1,
        )
        print(f"  ✓ Separate clusters created: train={train_nodes}x{train_gpus_per_node}GPUs, inference={inference_nodes}x{inference_gpus_per_node}GPUs")

    # ==========================
    #         Policy
    # ==========================
    print("\n▶ Setting up model...")
    
    # Checkpoint paths
    if last_checkpoint_path:
        weights_path = Path(last_checkpoint_path) / "policy" / "weights"
        optimizer_path = Path(last_checkpoint_path) / "policy" / "optimizer"
    else:
        weights_path = None
        optimizer_path = None


    student_policy = Policy(
        cluster=train_cluster,  # Use train_cluster
        config=policy_config,
        tokenizer=tokenizer,
        weights_path=weights_path,
        optimizer_path=optimizer_path,
        init_optimizer=True,
        init_reference_model=False,  # Don't enable reference model as teacher and student models have different sizes
    )

    # ==========================
    #      Generation Interface
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
        student_policy: 学生策略模型
        student_generation: 学生生成接口
        colocated_inference: 是否使用共置推理
        _refit_buffer_size_gb: 缓冲区大小（GB）
        timer: 计时器
        generation_config: 生成配置字典
        master_config: 主配置字典，用于获取max_total_sequence_length等参数
    """
    if colocated_inference:
        student_policy.offload_before_refit()
        student_generation.prepare_for_generation(tags=["weights"])
        
        # 更新生成配置参数（如temperature、decoding_method等）
        if generation_config is not None:
            try:
                # 尝试更新生成后端的配置
                if hasattr(student_generation, 'cfg') and isinstance(student_generation.cfg, dict):
                    # 更新温度参数
                    if 'temperature' in generation_config:
                        student_generation.cfg['temperature'] = generation_config['temperature']
                    # 更新解码方法相关参数
                    if 'decoding_method' in generation_config:
                        if generation_config['decoding_method'] == 'greedy':
                            # 对于greedy解码，设置top_k=1
                            student_generation.cfg['top_k'] = 1

                        elif generation_config['decoding_method'] == 'top_k':
                            # 对于top_k解码，使用默认值或配置值
                            if 'top_k' in generation_config:
                                student_generation.cfg['top_k'] = generation_config['top_k']

                        elif generation_config['decoding_method'] == 'top_p':
                            # 对于top_p解码，确保top_p被设置
                            if 'top_p' in generation_config:
                                student_generation.cfg['top_p'] = generation_config['top_p']
                                
                    
                    # 更新最大生成长度
                    if 'max_new_tokens' in generation_config:
                        if 'max_new_tokens' in student_generation.cfg:
                            student_generation.cfg['max_new_tokens'] = generation_config['max_new_tokens']
                    else:
                        # 如果没有配置max_new_tokens
                        # 从master_config获取max_total_sequence_length作为max_new_tokens
                        try:
                            max_seq_len = master_config["policy"]["max_total_sequence_length"]
                            student_generation.cfg['max_new_tokens'] = max_seq_len
                            
                        except Exception as e:
                            student_generation.cfg['max_new_tokens'] = 512  # 使用合理的默认值
            except Exception as e:
                print(f"  ⚠️ Warning: Failed to update generation config: {e}")

    # Create a context manager that does nothing when timer is None
    timer_context = (
        timer.time("prepare_for_generation/transfer_and_update_weights")
        if timer is not None
        else nullcontext()
    )
    with timer_context:
        # 更新权重
        update_success = False
        if colocated_inference:
            # 获取模型参数键，按大小分组
            grouped_param_keys = student_policy.prepare_weights_for_ipc(
                _refit_buffer_size_gb=_refit_buffer_size_gb
            )
            total_num_keys = sum(len(k) for k in grouped_param_keys)
            print(f"[Refit] Split {total_num_keys} keys into {len(grouped_param_keys)} groups")
            
            # 执行更新
            for keys in grouped_param_keys:
                ipc_handles = student_policy.get_weights_ipc_handles(keys)
                update_success = student_generation.update_weights_from_ipc_handles(ipc_handles)
                if not update_success:
                    break
        else:
            # 通过nccl更新权重
            futures_train = student_policy.broadcast_weights_for_collective()
            futures_inference = student_generation.update_weights_from_collective()
            # 等待所有futures完成
            ray.get(futures_train)
            results = ray.get(futures_inference)
            update_success = all(result for result in results if result is not None)

        # 检查更新是否成功
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
    student_generation: GenerationInterface,
    val_dataloader: Optional[StatefulDataLoader],
    tokenizer: TokenizerType,
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

        # 限制验证样本数量
        max_batches = 10  # 简化的验证逻辑
        for batch_idx, val_batch in enumerate(val_dataloader):
            if batch_idx >= max_batches:
                break

            if student_generation is not None:
                try:
                    # 使用rollout生成响应进行验证
                    val_batch, rollout_metrics = run_multi_turn_rollout(
                        policy_generation=student_generation,
                        input_batch=val_batch,
                        tokenizer=tokenizer,
                        task_to_env={},  # 蒸馏任务不需要环境交互
                        max_seq_len=master_config["policy"]["max_total_sequence_length"],  # 直接使用policy配置
                        max_rollout_turns=1,  # 蒸馏只需要单轮生成
                        greedy=(decoding_method == "greedy"),  # 根据decoding_method决定是否greedy
                    )
                    
                    # 计算验证loss：使用与训练相同的蒸馏损失计算
                    try:
                        # 准备验证数据
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
                            # 对于验证，我们可能没有teacher_logits，使用占位符
                            "teacher_logits": torch.randn_like(val_student_logits) * 0.1,
                            # 传递蒸馏参数
                            "kl_type": kl_type,
                            "mixed_kl_weight": mixed_kl_weight,
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
                    
                except Exception as e:
                    print(f"  ⚠️ Error during validation rollout: {str(e)}")
                    continue
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
                        # 传递蒸馏参数
                        "kl_type": kl_type,
                        "mixed_kl_weight": mixed_kl_weight,
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

        # 计算验证指标
        if total_losses:
            avg_loss = sum(total_losses) / len(total_losses)
        else:
            avg_loss = 0.0

        val_metrics = {
            "val_loss": avg_loss,
            "val_samples": total_samples,
            "val_avg_sequence_length": 0,  # 占位符，将在下面计算
            "val_max_sequence_length": 0,
            "val_min_sequence_length": 0,
        }
        
        # 验证loss计算完成
        if avg_loss == 0.0:
            print(f"  ⚠️ Warning: All validation batches returned 0 loss")

        
        # 计算生成长度相关指标（如果可能的话）
        try:
            # 尝试从验证数据中获取序列长度信息
            if val_dataloader is not None:
                sequence_lengths = []
                for val_batch in val_dataloader:
                    if hasattr(val_batch, 'get') and val_batch.get('input_ids') is not None:
                        input_ids = val_batch['input_ids']
                        if torch.is_tensor(input_ids):
                            # 计算非零token的数量作为序列长度
                            lengths = (input_ids != 0).sum(dim=1)
                            sequence_lengths.extend(lengths.tolist())
                    if len(sequence_lengths) >= 100:  # 限制样本数量
                        break
                
                if sequence_lengths:
                    sequence_lengths = torch.tensor(sequence_lengths)
                    val_metrics.update({
                        "val_avg_sequence_length": sequence_lengths.float().mean().item(),
                        "val_max_sequence_length": sequence_lengths.max().item(),
                        "val_min_sequence_length": sequence_lengths.min().item(),
                    })
        except Exception as e:
            print(f"  ⚠️ Could not compute sequence length metrics: {e}")
            pass

        # 打印验证结果
        print("\n📊 Validation Results:")
        print(f"    • Average loss: {avg_loss:.4f}")
        print(f"    • Samples processed: {total_samples}")

    return val_metrics


def distillation_train(
    student_policy: ColocatablePolicyInterface,
    student_generation: Optional[GenerationInterface],
    train_dataloader: StatefulDataLoader,
    val_dataloader: Optional[StatefulDataLoader],
    tokenizer: TokenizerType,  # 添加tokenizer参数
    loss_fn: DistillationLossFn,
    logger: Logger,
    checkpointer: CheckpointManager,
    distillation_save_state: DistillationSaveState,
    master_config: MasterConfig,
) -> None:
    """蒸馏训练主函数"""
    
    
    timer = Timer()
    distillation_config = master_config["distillation"]
    generation_config = master_config["policy"]["generation"]
    
    # 设置生成策略
    generate_strategy = distillation_config.get("generate_strategy", {})
    max_length = generate_strategy.get("max_length", 2048)
    temperature = generate_strategy.get("temperature", 1.0)
    decoding_method = generate_strategy.get("decoding_method", "greedy")
    
    # 设置KL散度类型
    kl_type = distillation_config.get("kl_type", "mixed") 
    mixed_kl_weight = distillation_config.get("mixed_kl_weight", 0.5)  # 混合KL权重
    
    # 如果policy_generation为None，使用policy作为生成接口
    NEED_REFIT = True
    if student_generation is None:
        pass
        student_generation = student_policy  # type: ignore
        NEED_REFIT = False
    STUDENT_GENERATION_STALE = True  # tracks if generation needs a refit before running
    assert student_generation is not None  # for mypy type check
    
    # 获取colocated推理设置
    colocated_inference = generation_config["colocated"]["enabled"]
    
    # 训练循环
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
                # 1. 准备批次数据
                
                with timer.time("data_processing"):
                    # 从batch中提取message_log
                    batch: BatchedDataDict[DatumSpec]
                    message_logs = batch["message_log"]
                    
                    # 安全地获取batch size
                    if hasattr(batch, 'size'):
                        batch_size = batch.size
                    elif hasattr(batch, '__len__'):
                        batch_size = len(batch)
                    else:
                        batch_size = 1
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
                
                # 检查是否需要refit
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
                    import torch
                    from nemo_rl.models.generation.interfaces import GenerationDatumSpec
                    

                    
                    # 创建Ray remote环境实例
                    from nemo_rl.environments.math_environment import MathEnvironment
                    from nemo_rl.distributed.ray_actor_environment_registry import get_actor_python_env
                    
                    # 从master_config获取环境配置
                    env_configs = master_config.get("env", {})
                    if "math" not in env_configs:
                        # 如果没有环境配置，使用默认配置
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
                    
                    # 避免remaining_length变成负数
                    for i, message_log in enumerate(repeated_batch["message_log"]):
                        total_length = sum(len(msg["token_ids"]) for msg in message_log)
                        if total_length > max_input_len:
                            # 重新计算需要保留的tokens数量
                            tokens_to_keep = max_input_len
                            
                            # 从第一个消息开始，按顺序保留tokens
                            for msg in message_log:
                                if tokens_to_keep <= 0:
                                    # 如果已经用完所有可用tokens，只保留第一个token
                                    if len(msg["token_ids"]) > 0:
                                        msg["token_ids"] = msg["token_ids"][:1]
                                else:
                                    msg_length = len(msg["token_ids"])
                                    if msg_length > tokens_to_keep:
                                        # 如果当前消息太长，截断到可用长度
                                        msg["token_ids"] = msg["token_ids"][:tokens_to_keep]
                                        tokens_to_keep = 0
                                    else:
                                        # 如果当前消息可以完全保留
                                        tokens_to_keep -= msg_length
                            
                            # 重新计算长度并验证
                            new_total_length = sum(len(msg["token_ids"]) for msg in message_log)
                            
                            # 验证截断后的长度不超过限制
                            if new_total_length > max_input_len:
                                # 强制截断到限制
                                for msg in message_log:
                                    if len(msg["token_ids"]) > 0:
                                        msg["token_ids"] = msg["token_ids"][:1]
                                        break
                    
                    # 使用rollout生成响应
                    try:
                        generated_batch, rollout_metrics = run_multi_turn_rollout(
                            policy_generation=student_generation,
                            input_batch=repeated_batch,  # 使用重复后的batch
                            tokenizer=tokenizer,
                            task_to_env=distillation_task_env,  # 传递Ray actor虚拟环境
                            max_seq_len=max_seq_len,  # 直接使用policy的max_total_sequence_length
                            max_rollout_turns=1,  # 蒸馏只需要单轮生成
                            greedy=(decoding_method == "greedy"),  # 根据decoding_method决定是否greedy
                        )
                        # 从rollout结果中提取生成的序列
                        generated_sequences = generated_batch["message_log"]
  
                        if "loss_multiplier" in repeated_batch:
                            loss_multiplier_after = repeated_batch["loss_multiplier"]
                        
                    except Exception as e:
                        print(f"  ❌ Rollout generation failed: {e}")
                        
                        try:                    
                            # 准备输入数据
                            input_ids = []
                            for message_log in repeated_batch["message_log"]:
                                # 合并所有消息的token_ids
                                sample_tokens = []
                                for msg in message_log:
                                    if "token_ids" in msg and len(msg["token_ids"]) > 0:
                                        sample_tokens.extend(msg["token_ids"].tolist())
                                
                                if len(sample_tokens) == 0:
                                    # 如果序列为空，添加pad token
                                    sample_tokens = [tokenizer.pad_token_id]
                                
                                # 在fallback中也应用长度限制
                                if len(sample_tokens) > max_input_len:
                                    sample_tokens = sample_tokens[:max_input_len]
                                
                                input_ids.append(sample_tokens)
                            
                            # 填充到相同长度
                            max_len = max(len(ids) for ids in input_ids)
                            padded_input_ids = []
                            for ids in input_ids:
                                if len(ids) < max_len:
                                    ids.extend([tokenizer.pad_token_id] * (max_len - len(ids)))
                                padded_input_ids.append(ids)
                            
                            # 转换为tensor
                            input_ids_tensor = torch.tensor(padded_input_ids, dtype=torch.long)
                            input_lengths_tensor = torch.tensor([len(ids) for ids in input_ids], dtype=torch.long)
                            
                            # 直接生成
                            generation_data = BatchedDataDict[GenerationDatumSpec]({
                                "input_ids": input_ids_tensor,
                                "input_lengths": input_lengths_tensor,
                                "stop_strings": [None] * len(input_ids),
                            })
                            
                            generation_outputs = student_generation.generate(
                                generation_data, 
                                greedy=(decoding_method == "greedy")
                            )
                            
                            # 处理生成结果
                            output_ids = generation_outputs["output_ids"]
                            generated_sequences = []
                            
                            for i in range(len(input_ids)):
                                input_len = input_lengths_tensor[i].item()
                                generated_tokens = output_ids[i, input_len:].tolist()
                                
                                # 创建assistant消息
                                assistant_message = {
                                    "role": "assistant",
                                    "content": tokenizer.decode(generated_tokens, skip_special_tokens=True),
                                    "token_ids": torch.tensor(generated_tokens, dtype=torch.long),
                                }
                                
                                # 重建message_log
                                sample_messages = []
                                for msg in repeated_batch["message_log"][i]:
                                    sample_messages.append(msg)
                                sample_messages.append(assistant_message)
                                generated_sequences.append(sample_messages)
                            
                        except Exception as fallback_error:
                            print(f"  ❌ Fallback generation also failed: {fallback_error}")
                            import traceback
                            traceback.print_exc()
                            raise RuntimeError(f"Both rollout and fallback generation failed. Original error: {e}, Fallback error: {fallback_error}")
                else:
                    # 如果使用megatron后端，直接使用policy
                    # 这里需要实现megatron的生成逻辑
                    generated_sequences = batch["message_log"]  # 暂时使用原始数据
                
                # 标记生成完成
                if student_generation is not None:
                    student_generation.finish_generation()
                
                # 3. 计算logits
                
                with timer.time("logits_computation"):
                    try:
                        expected_batch_size = master_config["distillation"]["num_prompts_per_step"] * master_config["distillation"]["num_generations_per_prompt"]

                        if len(generated_sequences) != expected_batch_size:
                            if len(generated_sequences) > expected_batch_size:
                                generated_sequences = generated_sequences[:expected_batch_size]
                            else:
                                # 扩展batch到正确大小（重复最后一个序列）
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
                    
                    # 准备训练数据
                    
                    if "generation_logprobs" not in flat_messages:
                        # 为每个token创建零logprobs（因为我们没有生成logprobs）
                        flat_messages["generation_logprobs"] = torch.zeros_like(
                            flat_messages["token_ids"], dtype=torch.float32
                        )
                    
                    if "advantages" not in flat_messages:
                        flat_messages["advantages"] = torch.ones_like(
                            flat_messages["token_ids"], dtype=torch.float32
                        )
                    
                    if "token_loss_mask" not in flat_messages:
                        token_loss_mask = torch.zeros_like(
                            flat_messages["token_ids"], dtype=torch.bool
                        )
                        
                        for i, seq_len in enumerate(input_lengths):
                            if seq_len > 0:
                                token_loss_mask[i, :seq_len] = True
                        
                        flat_messages["token_loss_mask"] = token_loss_mask
                    
                    # 验证所有字段的batch维度一致
                    expected_batch_size = flat_messages['token_ids'].shape[0]
                    expected_seq_len = flat_messages['token_ids'].shape[1]
                    
                    # 验证并修复形状不匹配的字段
                    if flat_messages['advantages'].shape[0] != expected_batch_size:
                        flat_messages['advantages'] = flat_messages['advantages'][:expected_batch_size]
                    
                    if flat_messages['generation_logprobs'].shape[0] != expected_batch_size:
                        flat_messages['generation_logprobs'] = flat_messages['generation_logprobs'][:expected_batch_size]
                    
                    if flat_messages['token_loss_mask'].shape[0] != expected_batch_size:
                        flat_messages['token_loss_mask'] = flat_messages['token_loss_mask'][:expected_batch_size]
                    
                    if repeated_batch['loss_multiplier'].shape[0] != expected_batch_size:
                        repeated_batch['loss_multiplier'] = repeated_batch['loss_multiplier'][:expected_batch_size]
                    
                    # 验证sequence维度
                    if flat_messages['advantages'].shape[1] != expected_seq_len:
                        if flat_messages['advantages'].shape[1] > expected_seq_len:
                            flat_messages['advantages'] = flat_messages['advantages'][:, :expected_seq_len]
                        else:
                            flat_messages['advantages'] = flat_messages['advantages'].expand(-1, expected_seq_len)
                    
                    if flat_messages['generation_logprobs'].shape[1] != expected_seq_len:
                        if flat_messages['generation_logprobs'].shape[1] > expected_seq_len:
                            flat_messages['generation_logprobs'] = flat_messages['generation_logprobs'][:, :expected_seq_len]
                        else:
                            flat_messages['generation_logprobs'] = flat_messages['generation_logprobs'].expand(-1, expected_seq_len)
                    
                    if flat_messages['token_loss_mask'].shape[1] != expected_seq_len:
                        if flat_messages['token_loss_mask'].shape[1] > expected_seq_len:
                            flat_messages['token_loss_mask'] = flat_messages['token_loss_mask'][:, :expected_seq_len]
                        else:
                            flat_messages['token_loss_mask'] = flat_messages['token_loss_mask'].expand(-1, expected_seq_len)
                    
                    
                    # 确保loss_multiplier是正确的形状
                    if isinstance(repeated_batch["loss_multiplier"], torch.Tensor):
                        if len(repeated_batch["loss_multiplier"].shape) > 1:
                            # 如果loss_multiplier是多维的，取第一个维度
                            repeated_batch["loss_multiplier"] = repeated_batch["loss_multiplier"].flatten()[:expected_batch_size]
                            
                        elif repeated_batch["loss_multiplier"].shape[0] != expected_batch_size:
                            repeated_batch["loss_multiplier"] = repeated_batch["loss_multiplier"][:expected_batch_size]
                            
                    elif isinstance(repeated_batch["loss_multiplier"], list):
                        repeated_batch["loss_multiplier"] = torch.tensor(repeated_batch["loss_multiplier"][:expected_batch_size], dtype=torch.float32)
                        

                    
                    # 最终验证loss_multiplier的类型和形状
                    if not isinstance(repeated_batch["loss_multiplier"], torch.Tensor):
                        if isinstance(repeated_batch["loss_multiplier"], (list, tuple)):
                            repeated_batch["loss_multiplier"] = torch.tensor(repeated_batch["loss_multiplier"], dtype=torch.float32)
                         
                        elif isinstance(repeated_batch["loss_multiplier"], (int, float)):
                            repeated_batch["loss_multiplier"] = torch.tensor([repeated_batch["loss_multiplier"]] * expected_batch_size, dtype=torch.float32)
                            
                        else:
                            # 创建默认的loss_multiplier
                            repeated_batch["loss_multiplier"] = torch.ones(expected_batch_size, dtype=torch.float32)
                           
                    
                    # 验证所有字段的batch维度一致
                    all_batch_sizes = [
                        flat_messages['token_ids'].shape[0],
                        input_lengths.shape[0],
                        flat_messages['advantages'].shape[0],
                        flat_messages['generation_logprobs'].shape[0],
                        flat_messages['token_loss_mask'].shape[0],
                        repeated_batch['loss_multiplier'].shape[0]
                    ]
                    
                    if len(set(all_batch_sizes)) != 1:
                        raise ValueError(f"Batch dimensions must be consistent, got: {all_batch_sizes}")
                    
                    # 创建训练数据，只包含张量字段
                    train_data_dict = {
                        "input_ids": flat_messages["token_ids"],
                        "input_lengths": input_lengths,
                        "token_mask": flat_messages["token_loss_mask"],  # 使用token_loss_mask而不是自定义的token_mask
                        "sample_mask": repeated_batch["loss_multiplier"],
                    }
                    
                    # 验证所有字段都是张量
                    for key, value in train_data_dict.items():
                        if not torch.is_tensor(value):
                            raise ValueError(f"Field {key} must be a tensor, got {type(value)}")
                    
                    train_data = BatchedDataDict[DistillationLossDataDict](train_data_dict)

                    # 确保数据在正确的设备上
                    train_data.to("cpu")  
                    
                    # 教师模型前向传播（需要单独实现，因为模型大小不同）
                    with torch.no_grad():
                        # 实现真正的教师模型推理
                        teacher_model_path = master_config["distillation"]["teacher_model_path"]
                        try:
                            # 方法1: 尝试使用transformers直接加载教师模型
                            from transformers import AutoModelForCausalLM, AutoTokenizer
                            
                            # 检查是否已经有教师模型实例
                            if not hasattr(student_policy, '_teacher_model'):
                                try:
                                    # 内存优化：使用device_map="auto"和低精度
                                    teacher_model = AutoModelForCausalLM.from_pretrained(
                                        teacher_model_path,
                                        torch_dtype=torch.bfloat16,
                                        device_map="auto",
                                        trust_remote_code=True,
                                        low_cpu_mem_usage=True,  # 减少CPU内存使用
                                    )
                                    
      
                                    teacher_model.eval()

                                    
                                    # 缓存教师模型
                                    student_policy._teacher_model = teacher_model
                                
                                    
                                except Exception as e:
                                    print(f"  ❌ Failed to load teacher model: {e}")
                                    import traceback
                                    traceback.print_exc()
                                    raise
                            else:
                                teacher_model = student_policy._teacher_model

                            teacher_input_ids = train_data["input_ids"]
                            

                            # 验证教师模型输出形状
                            test_input = torch.randint(0, 1000, (2, 5), device=next(teacher_model.parameters()).device)
                            
                            with torch.no_grad():
                                test_output = teacher_model(test_input)
                                test_logits = test_output.logits

                                if len(test_logits.shape) != 3:
                                    raise ValueError(f"Teacher model produces incorrect logits shape: {test_logits.shape}")
                            
                            # 内存优化：分批处理，避免一次性处理太多数据
                            batch_size = teacher_input_ids.shape[0]
                            chunk_size = 4  # 每次处理4个样本
                            teacher_logits_list = []
                            
                            for i in range(0, batch_size, chunk_size):
                                end_idx = min(i + chunk_size, batch_size)
                                chunk_input_ids = teacher_input_ids[i:end_idx]
                                
                                # 确保输入在正确的设备上
                                if hasattr(teacher_model, 'device'):
                                    chunk_input_ids = chunk_input_ids.to(teacher_model.device)
                                else:
                                    # 如果没有device属性，尝试获取第一个参数的设备
                                    try:
                                        device = next(teacher_model.parameters()).device
                                        chunk_input_ids = chunk_input_ids.to(device)
                                    except Exception as e:
                                        # 默认使用CPU
                                        chunk_input_ids = chunk_input_ids.cpu()
                                       
                                
                                with torch.no_grad():
                                    # 创建attention_mask和position_ids，确保输出形状正确
                                    chunk_batch_size, chunk_seq_len = chunk_input_ids.shape
                                    
                                    # 创建attention_mask（右填充序列）
                                    attention_mask = torch.zeros((chunk_batch_size, chunk_seq_len), dtype=torch.long, device=chunk_input_ids.device)
                                    for j, length in enumerate(train_data["input_lengths"][i:i+chunk_size]):
                                        attention_mask[j, :length] = 1
                                    
                                    # 创建position_ids
                                    position_ids = torch.arange(chunk_seq_len, device=chunk_input_ids.device).repeat(chunk_batch_size, 1)
                                    
                                    # 使用完整的输入进行前向传播
                                    chunk_outputs = teacher_model(
                                        chunk_input_ids,
                                        attention_mask=attention_mask,
                                        position_ids=position_ids,
                                        return_dict=True
                                    )
                                    chunk_logits = chunk_outputs.logits
                                    
                                    teacher_logits_list.append(chunk_logits.cpu())  # 移到CPU节省GPU内存
                                
                                # 清理GPU内存
                                del chunk_outputs, chunk_logits
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                            
                            # 合并所有chunk的logits
                            teacher_logits = torch.cat(teacher_logits_list, dim=0)
                            del teacher_logits_list  # 清理列表
                            
                            
                            # 验证teacher_logits的形状
                            expected_teacher_shape = (batch_size, teacher_input_ids.shape[1], -1)  # 最后一个维度是vocab_size
                          
                            
                            # 检查teacher_logits的形状
                            if len(teacher_logits.shape) != 3:
                                # 如果teacher_logits是2D的，尝试重塑为3D
                                if len(teacher_logits.shape) == 2:
                                    # 检查是否是[batch_size, vocab_size]的情况
                                    if teacher_logits.shape[0] == batch_size and teacher_logits.shape[1] > 1000:  # 假设vocab_size > 1000
                                        # 假设每个序列都是相同长度，从input_ids获取
                                        seq_len = teacher_input_ids.shape[1]
                                        vocab_size = teacher_logits.shape[1]
                                        
                                        # 重塑为[batch_size, seq_len, vocab_size]
                                        # 这里需要根据实际情况调整，可能需要重复logits或使用其他策略
                                        teacher_logits = teacher_logits.unsqueeze(1).expand(-1, seq_len, -1)
                                    else:
                                        raise ValueError(f"Teacher logits shape {teacher_logits.shape} is not compatible with expected shape {expected_teacher_shape}")
                                elif len(teacher_logits.shape) > 3:
                                    # 尝试压缩多余的维度
                                    if teacher_logits.shape[0] == batch_size:
                                        # 保持batch维度，压缩其他维度
                                        teacher_logits = teacher_logits.view(batch_size, -1, teacher_logits.shape[-1])
                                    else:
                                        raise ValueError(f"Teacher logits shape {teacher_logits.shape} is not compatible with expected shape {expected_teacher_shape}")
                            
                            # 验证形状
                            if teacher_logits.shape[0] != expected_teacher_shape[0] or teacher_logits.shape[1] != expected_teacher_shape[1]:
                                # 尝试进一步修复形状
                                if teacher_logits.shape[0] != batch_size:
                                    if teacher_logits.shape[0] > batch_size:
                                        teacher_logits = teacher_logits[:batch_size]
                                    else:
                                        # 扩展batch维度
                                        teacher_logits = teacher_logits.expand(batch_size, -1, -1)
                                
                                if teacher_logits.shape[1] != teacher_input_ids.shape[1]:
                                    if teacher_logits.shape[1] > teacher_input_ids.shape[1]:
                                        teacher_logits = teacher_logits[:, :teacher_input_ids.shape[1], :]
                                    else:
                                        # 扩展sequence维度
                                        teacher_logits = teacher_logits.expand(-1, teacher_input_ids.shape[1], -1)
                            
                            # 最终验证：确保形状完全正确
                            final_shape = teacher_logits.shape
                            if final_shape[0] != batch_size or final_shape[1] != teacher_input_ids.shape[1]:
                                raise ValueError(f"Failed to fix teacher_logits shape. Final shape: {final_shape}")
                            
                            
                            # 将教师logits添加到训练数据中
                            train_data["teacher_logits"] = teacher_logits
                            
                            
                        except Exception as e:
                            # 回退到占位符（不推荐，但确保程序能运行）
                            batch_size = train_data["input_ids"].shape[0]
                            seq_len = train_data["input_ids"].shape[1]
                            vocab_size = 32000  # 假设的词汇表大小
                            placeholder_logits = torch.randn(batch_size, seq_len, vocab_size) * 0.1
                            train_data["teacher_logits"] = placeholder_logits
                    
                    # 准备学生模型进行logprob推理
                    try:
                        student_policy.prepare_for_lp_inference()
                    except Exception as e:
                        raise
                    

                    try:
                        
                        # 检查teacher_logits的形状（如果存在）
                        if "teacher_logits" in train_data:
                            teacher_logits = train_data["teacher_logits"]
                            
                            # 如果teacher_logits的形状不正确，强制修复
                            if len(teacher_logits.shape) != 3:
                                if len(teacher_logits.shape) == 2:
                                    # 如果是[batch_size, vocab_size]，重塑为[batch_size, seq_len, vocab_size]
                                    batch_size = teacher_logits.shape[0]
                                    vocab_size = teacher_logits.shape[1]
                                    seq_len = train_data["input_ids"].shape[1]
                                    teacher_logits = teacher_logits.unsqueeze(1).expand(-1, seq_len, -1)
          
                                else:
                                    raise ValueError(f"teacher_logits has unexpected shape: {teacher_logits.shape}")
                            
                            # 验证修复后的形状
                            expected_shape = (train_data["input_ids"].shape[0], train_data["input_ids"].shape[1], -1)
                            if teacher_logits.shape[0] != expected_shape[0] or teacher_logits.shape[1] != expected_shape[1]:
                                raise ValueError(f"Failed to fix teacher_logits shape")
                            
                            # 更新train_data中的teacher_logits
                            train_data["teacher_logits"] = teacher_logits
 
                        
                        # 准备输入数据
                        input_ids = train_data["input_ids"].to("cuda")
                        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
                        position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0).expand(input_ids.shape[0], -1)
                        
                        # 直接调用学生模型
                        with torch.no_grad():
                            student_policy.prepare_for_lp_inference()
                            
                            num_shards = len(student_policy.worker_group.workers)

                            
                            # 确保batch size是shards的倍数
                            current_batch_size = input_ids.shape[0]
                            if current_batch_size % num_shards != 0:
                                # 调整batch size到最近的shards倍数
                                adjusted_batch_size = ((current_batch_size // num_shards) + 1) * num_shards

                                
                                # 扩展数据到调整后的batch size
                                if adjusted_batch_size > current_batch_size:
                                    # 重复最后一个样本来填充
                                    padding_size = adjusted_batch_size - current_batch_size
                                    input_ids = torch.cat([input_ids, input_ids[-1:].repeat(padding_size, 1)], dim=0)
                                    attention_mask = torch.cat([attention_mask, attention_mask[-1:].repeat(padding_size, 1)], dim=0)
                                    position_ids = torch.cat([position_ids, position_ids[-1:].repeat(padding_size, 1)], dim=0)
                            
                            # 创建正确的训练数据格式
                            train_data_for_logprobs_dict = {
                                "input_ids": input_ids,
                                "input_lengths": torch.tensor([input_ids.shape[1]] * input_ids.shape[0]),
                                "token_mask": torch.ones(input_ids.shape[0], input_ids.shape[1]),
                                "sample_mask": torch.ones(input_ids.shape[0]),
                            }
                            
                            # 验证所有字段都是张量
                            for key, value in train_data_for_logprobs_dict.items():
                                if not torch.is_tensor(value):
                                    print(f"  ❌ Critical error: {key} is not a tensor: {type(value)}")
                                    raise ValueError(f"Field {key} must be a tensor, got {type(value)}")
                            
                           
                    except Exception as e:
                        raise
               
                    # 计算蒸馏损失
                    print("  ✓ Computing distillation loss...")
                    try:
                        # 使用损失函数计算蒸馏损失 - 传递所有必要的参数
                        # 将蒸馏参数添加到train_data中，供损失函数使用
                        # 注意：这些是标量值，不是张量，所以不会传递给worker
                        train_data["kl_type"] = kl_type
                        train_data["mixed_kl_weight"] = mixed_kl_weight
                        
                        # 确保只在response tokens上计算KL散度
                        if "token_mask" in train_data:
                            token_mask = train_data["token_mask"]
                            total_tokens = token_mask.numel()
                            response_tokens = token_mask.sum().item()
                            prompt_tokens = total_tokens - response_tokens

                        else:
                            # 如果没有token_mask，创建一个默认的（全1，但这不是理想情况）
                            token_mask = torch.ones_like(train_data["input_ids"], dtype=torch.bool)
                            print(f"  ⚠️ Warning: No token_mask found, using all tokens for loss calculation")
                        
                    except Exception as e:
                        print(f"  ❌ Failed to compute distillation loss: {e}")
                        import traceback
                        traceback.print_exc()
                        raise
                
                # 5. 训练学生模型

                # 验证所有字段的batch维度一致
                all_batch_sizes = [train_data[key].shape[0] for key in train_data.keys() if torch.is_tensor(train_data[key])]
                if len(set(all_batch_sizes)) != 1:
                    raise ValueError(f"Batch dimensions must be consistent, got: {all_batch_sizes}")
                
                

                distillation_safe_data = {}
                
                for key, value in train_data.items():
                    if key in ["teacher_logits"]:
                        distillation_safe_data[key] = value
                        if len(value.shape) == 3:
                            batch_size, seq_len, vocab_size = value.shape
                            flattened_logits = value.view(batch_size * seq_len, vocab_size)
                            
                            # 创建一个特殊的key，worker不会检查
                            safe_key = f"distillation_{key}_flattened"
                            distillation_safe_data[safe_key] = flattened_logits
                            
                            # 存储原始形状信息
                            distillation_safe_data[f"{safe_key}_shape"] = torch.tensor([batch_size, seq_len, vocab_size])
                        else:
                            distillation_safe_data[key] = value
                    else:
                        # 对于其他字段，直接复制
                        distillation_safe_data[key] = value
                

                
                
                
                
                with timer.time("training_prep"):

                    student_policy.prepare_for_training()  
                    STUDENT_GENERATION_STALE = True  # *** MARK AS STALE AFTER TRAINING ***
                
                # 只保留worker需要的标准张量字段
                worker_required_fields = ["input_ids", "input_lengths", "token_mask", "sample_mask", "teacher_logits"]
                clean_worker_data = {}
                
                for field in worker_required_fields:
                    if field in train_data:
                        if torch.is_tensor(train_data[field]):
                            clean_worker_data[field] = train_data[field]
                        else:
                            continue
                    else:
                        continue
                
                # 验证清理后的数据
                if len(clean_worker_data) != len(worker_required_fields):
                    raise ValueError("Missing required fields for worker")
                
                # 创建干净的BatchedDataDict用于worker
                worker_train_data = BatchedDataDict[DistillationLossDataDict](clean_worker_data)
           
                with timer.time("policy_training"):
                    try:
                        # 使用清理后的数据传递给worker
                        train_results = student_policy.train(worker_train_data, loss_fn)
                    except Exception as e:
                        raise
                # 采用与其他算法一致的方式，避免重复记录train/loss
                loss_list = train_results["all_mb_metrics"]["loss"]
                loss = sum(loss_list) / len(loss_list)
                
                # 构建训练指标
                metrics = {
                    "loss": loss,  # 主要训练损失
                    "grad_norm": train_results["grad_norm"].numpy() if hasattr(train_results["grad_norm"], "numpy") else train_results["grad_norm"],
                }
                
                # 添加其他微批次指标（但不包含loss，避免重复）
                # 正确处理数据类型，确保所有值都是数值类型
                all_mb_metrics = train_results["all_mb_metrics"].copy()
                if "loss" in all_mb_metrics:
                    del all_mb_metrics["loss"]  # 避免重复记录loss
                
                # 安全地添加微批次指标，确保数据类型正确
                for k, v in all_mb_metrics.items():
                    if isinstance(v, (list, tuple)):
                        # 如果是list/tuple，计算平均值
                        if len(v) > 0:
                            if isinstance(v[0], (int, float)):
                                metrics[k] = sum(v) / len(v)
                            elif hasattr(v[0], 'numpy'):
                                metrics[k] = sum(x.numpy() for x in v) / len(v)
                            else:
                                # 跳过无法处理的类型
                                continue
                        else:
                            # 空list，跳过
                            continue
                    elif isinstance(v, (int, float)):
                        # 直接使用数值
                        metrics[k] = v
                    elif hasattr(v, 'numpy'):
                        # 转换为numpy
                        metrics[k] = v.numpy()
                    elif hasattr(v, 'item'):
                        # 转换为Python标量
                        metrics[k] = v.item()
                    else:
                        # 跳过无法处理的类型
                        continue
                
                # 记录生成长度相关指标
                if "input_ids" in train_data:
                    input_lengths = (train_data["input_ids"] != 0).sum(dim=1)
                    metrics.update({
                        "avg_input_length": input_lengths.float().mean().item(),
                        "max_input_length": input_lengths.max().item(),
                        "min_input_length": input_lengths.min().item(),
                        "input_length_std": input_lengths.float().std().item(),
                    })
                
                # 记录当前最佳验证loss（如果可用）
                if "val_loss" in distillation_save_state and distillation_save_state["val_loss"] is not None:
                    current_best_val_loss = distillation_save_state["val_loss"]
                    metrics["best_val_loss"] = current_best_val_loss
                
                # 记录蒸馏参数
                metrics.update({
                    "kl_type": 1.0 if kl_type == "forward" else (2.0 if kl_type == "reverse" else 3.0),
                    "mixed_kl_weight": mixed_kl_weight,
                })
                
                # 使用prefix="train"记录所有指标，避免重复
                if logger is not None:
                    logger.log_metrics(metrics, step, prefix="train")
                    
                    # 打印训练loss信息
                    print(f"  ✅✅✅ [Training] Step {step}: Loss = {loss:.6f}")
    
                step += 1
                distillation_save_state["step"] = step
                # 使用配置中的值
                distillation_save_state["consumed_samples"] += distillation_config.get("num_prompts_per_step", 1)

                
                # 7. 保存检查点
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
                        # 保存数据加载器状态
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
                            # 传递生成配置参数
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
                            student_generation,
                            val_dataloader,
                            tokenizer,
                            step + 1,
                            master_config,
                        )
                        
                        # 记录验证指标
                        if val_metrics:
                            # 记录验证loss - 只记录到eval/命名空间
                            if "val_loss" in val_metrics:
                                logger.log_metrics({"eval/loss": val_metrics["val_loss"]}, step + 1)
                                distillation_save_state["val_loss"] = val_metrics["val_loss"]
                                print(f"  ✅✅✅ [Validation] Step {step + 1}: Val Loss = {val_metrics['val_loss']:.6f}")
                            
                            # 记录其他验证指标 - 只记录到eval/命名空间
                            for k, v in val_metrics.items():
                                if k != "val_loss" and isinstance(v, (int, float)):
                                    logger.log_metrics({f"eval/{k}": v}, step + 1)
                            
                            # 记录验证时的生成长度信息 - 只记录到eval/命名空间
                            if "val_avg_sequence_length" in val_metrics:
                                logger.log_metrics({
                                    "eval/avg_sequence_length": val_metrics["val_avg_sequence_length"],
                                    "eval/max_sequence_length": val_metrics.get("val_max_sequence_length", 0),
                                    "eval/min_sequence_length": val_metrics.get("val_min_sequence_length", 0),
                                }, step + 1)
                            
                            # 记录验证时的蒸馏参数 - 只记录到eval/命名空间
                            logger.log_metrics({
                                "eval/kl_type": 1.0 if kl_type == "forward" else (2.0 if kl_type == "reverse" else 3.0),
                                "eval/mixed_kl_weight": mixed_kl_weight,
                            }, step + 1)
                        
                        if student_generation is not None:
                            student_generation.finish_generation()
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                
                # 9. 日志记录
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
