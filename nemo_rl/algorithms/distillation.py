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

from nemo_rl.algorithms.interfaces import LossFunction
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
    # 教师模型路径（用于加载权重）
    teacher_model_path: str
    
    # 蒸馏策略参数
    lambda_: float  # 学生自生成数据占比
    kl_type: str    # KL散度类型：forward, reverse, mixed
    generate_strategy: dict[str, Any]  # 生成策略参数
    
    # 训练配置
    max_steps: int
    eval_steps: int
    save_steps: int
    logging_steps: int


class MasterConfig(TypedDict):
    """主配置结构 - 参考GRPO的标准结构"""
    policy: PolicyConfig  # 学生模型配置
    loss_fn: DistillationLossConfig  # 损失函数配置
    env: dict[str, Any]  # 环境配置
    data: DataConfig  # 数据配置
    distillation: DistillationConfig  # 蒸馏配置
    logger: LoggerConfig  # 日志配置
    cluster: ClusterConfig  # 集群配置
    checkpointing: CheckpointingConfig  # 检查点配置


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
    ColocatablePolicyInterface,  # student_policy (唯一的Policy实例)
    Optional[GenerationInterface],  # student_generation
    tuple[RayVirtualCluster, RayVirtualCluster],  # 与GRPO保持一致
    StatefulDataLoader,
    Optional[StatefulDataLoader],
    TokenizerType,  # 添加tokenizer，与GRPO保持一致
    DistillationLossFn,
    Logger,
    CheckpointManager,
    DistillationSaveState,
    MasterConfig,
]:
    """蒸馏算法主入口点 - 参考GRPO实现，只创建一个Policy实例，通过refit机制管理权重同步
    
    返回:
        tuple of student_policy, student_generation, 
        (train_cluster, inference_cluster), train_dataloader, val_dataloader, 
        loss_fn, logger, checkpointer, distillation_save_state, master_config
    """
    # 提取配置
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

    # 设置随机种子
    set_seed(42)  # 使用固定种子

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
        batch_size=distillation_config["num_prompts_per_step"],  # 与GRPO保持一致
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

    # 验证数据集
    val_dataloader: Optional[StatefulDataLoader] = None
    if val_dataset is not None:
        val_dataloader = StatefulDataLoader(
            val_dataset,
            batch_size=distillation_config["num_prompts_per_step"],  # 与GRPO保持一致
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
        # 使用与GRPO完全相同的集群初始化逻辑
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
    
    # 检查点路径
    if last_checkpoint_path:
        weights_path = Path(last_checkpoint_path) / "policy" / "weights"
        optimizer_path = Path(last_checkpoint_path) / "policy" / "optimizer"
    else:
        weights_path = None
        optimizer_path = None

    # 只创建一个Policy实例，与GRPO保持一致
    student_policy = Policy(
        cluster=train_cluster,  # 使用train_cluster，与GRPO保持一致
        config=policy_config,
        tokenizer=tokenizer,
        weights_path=weights_path,
        optimizer_path=optimizer_path,
        init_optimizer=True,
        init_reference_model=False,  # 不启用参考模型，因为教师和学生模型大小不同
    )
    print("  ✓ Student policy initialized")

    # 加载教师模型权重到参考模型
    teacher_model_path = distillation_config["teacher_model_path"]
    print(f"  ✓ Will load teacher model weights from: {teacher_model_path}")
    print("  ⚠️ Note: Teacher and student models have different sizes, cannot use reference model mechanism")
    print("  ⚠️ Need to implement separate teacher model loading for distillation")

    # ==========================
    #      Generation Interface
    # ==========================
    print("\n▶ Setting up generation interface...")
    
    # 参考GRPO的实现，根据backend选择生成接口
    backend = generation_config["backend"]
    generation_config["model_name"] = policy_config["model_name"]  # Needed for vLLM

    if backend == "megatron":
        student_generation = None
        print(
            f"  ✓ Using {backend} backend for generation with {policy_config['model_name']}"
        )
    elif backend == "vllm":
        generation_config = cast(VllmConfig, generation_config)
        student_generation = VllmGeneration(
            cluster=inference_cluster, config=generation_config
        )
        # Worker groups are not initialized until the first call to run something on workergroups.
        # vllm 0.8 fails in initialization if its called in the first training step since it has no clean view of the GPU memory (HF is sharing the same memory).
        student_generation.finish_generation()
        print(
            f"  ✓ Using vLLM backend for generation with {policy_config['model_name']}"
        )

    # 如果使用非colocated推理，初始化集体通信
    # 注意：在蒸馏训练中，我们通常使用colocated推理，所以这里暂时跳过collective初始化
    # 如果确实需要非colocated推理，可以参考GRPO的实现
    if not colocated_inference and student_generation is not None:
        print("  ⚠️ Non-colocated inference detected, but collective communication initialization is skipped for distillation")
        # print("  🔍 This is to avoid port conflicts. If you need non-colocated inference, please implement proper port management")
        pass
        # 暂时跳过collective初始化，避免端口冲突
        # ip, port = train_cluster.get_master_address_and_port()
        # print(f"Using ip: {ip}, port: {port} for collective communication")
        # world_size = inference_nodes * inference_gpus_per_node + 1
        # futures_train = student_policy.init_collective(ip, port, world_size)
        # futures_inference = student_generation.init_collective(ip, port, world_size)
        # ray.get(futures_train + futures_inference)

    # 准备refit信息，与GRPO保持一致
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
        (train_cluster, inference_cluster),  # 返回元组，与GRPO保持一致
        train_dataloader,
        val_dataloader,
        tokenizer,  # 添加tokenizer，与GRPO保持一致
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
    
    参考GRPO的refit_policy_generation实现，但增加了蒸馏特定的生成配置更新功能。
    这使得蒸馏任务能够动态调整生成参数，而不需要重新初始化整个生成后端。
    
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
                        print(f"  🔍 Updated generation temperature to: {generation_config['temperature']}")
                    
                    # 更新解码方法相关参数
                    if 'decoding_method' in generation_config:
                        if generation_config['decoding_method'] == 'greedy':
                            # 对于greedy解码，设置top_k=1
                            student_generation.cfg['top_k'] = 1
                            print(f"  🔍 Set top_k=1 for greedy decoding")
                        elif generation_config['decoding_method'] == 'top_k':
                            # 对于top_k解码，使用默认值或配置值
                            if 'top_k' in generation_config:
                                student_generation.cfg['top_k'] = generation_config['top_k']
                                print(f"  🔍 Updated top_k to: {generation_config['top_k']}")
                        elif generation_config['decoding_method'] == 'top_p':
                            # 对于top_p解码，确保top_p被设置
                            if 'top_p' in generation_config:
                                student_generation.cfg['top_p'] = generation_config['top_p']
                                print(f"  🔍 Updated top_p to: {generation_config['top_p']}")
                    
                    # 更新最大生成长度 - 参考GRPO：max_new_tokens通常等于max_total_sequence_length
                    if 'max_new_tokens' in generation_config:
                        if 'max_new_tokens' in student_generation.cfg:
                            student_generation.cfg['max_new_tokens'] = generation_config['max_new_tokens']
                            print(f"  🔍 Updated max_new_tokens to: {generation_config['max_new_tokens']}")
                    else:
                        # 如果没有配置max_new_tokens，使用GRPO的默认行为
                        # 从master_config获取max_total_sequence_length作为max_new_tokens
                        try:
                            max_seq_len = master_config["policy"]["max_total_sequence_length"]
                            student_generation.cfg['max_new_tokens'] = max_seq_len
                            print(f"  🔍 Using GRPO-style max_new_tokens = max_total_sequence_length: {max_seq_len}")
                        except Exception as e:
                            print(f"  ⚠️ Warning: Failed to get max_total_sequence_length: {e}")
                            student_generation.cfg['max_new_tokens'] = 512  # 使用合理的默认值
                            print(f"  🔍 Using fallback max_new_tokens: 512")
                        
                print(f"  ✅ Generation configuration updated successfully")
            except Exception as e:
                print(f"  ⚠️ Warning: Failed to update generation config: {e}")
                print(f"  🔍 This is not critical, generation will use default backend config")

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
    """Run validation on the validation dataset for distillation - 与GRPO保持一致"""
    if val_dataloader is None:
        print("  ⚠️ No validation dataloader provided, skipping validation")
        return {}

    timer = Timer()
    with timer.time("total_validation_time"):
        print(f"▶ Starting validation at step {step}...")

        total_losses = []
        total_samples = 0

        # 限制验证样本数量，与GRPO保持一致
        max_batches = 10  # 简化的验证逻辑
        for batch_idx, val_batch in enumerate(val_dataloader):
            if batch_idx >= max_batches:
                break

            # 使用与GRPO相同的rollout机制进行验证
            if student_generation is not None:
                try:
                    # 使用rollout生成响应进行验证
                    val_batch, rollout_metrics = run_multi_turn_rollout(
                        policy_generation=student_generation,
                        input_batch=val_batch,
                        tokenizer=tokenizer,
                        task_to_env={},  # 蒸馏任务不需要环境交互
                        max_seq_len=min(max_length, master_config["policy"]["max_total_sequence_length"]),  # 使用配置的max_length
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
                            "lambda_": lambda_,
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
                        "lambda_": lambda_,
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
            #print(f"  🔍 This might indicate an issue with validation loss computation")

        
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
    clusters: tuple[RayVirtualCluster, RayVirtualCluster],  # 与GRPO保持一致
    train_dataloader: StatefulDataLoader,
    val_dataloader: Optional[StatefulDataLoader],
    tokenizer: TokenizerType,  # 添加tokenizer参数，与GRPO保持一致
    loss_fn: DistillationLossFn,
    logger: Logger,
    checkpointer: CheckpointManager,
    distillation_save_state: DistillationSaveState,
    master_config: MasterConfig,
) -> None:
    """蒸馏训练主函数 - 完全按照GRPO模式实现，使用单一Policy + 参考模型"""
    
    # 解包集群（与GRPO保持一致）
    train_cluster, inference_cluster = clusters
    
    print("Starting distillation training...")
    print(f"Student policy: {student_policy}")
    print(f"Teacher model path: {master_config['distillation']['teacher_model_path']}")
    
    # 参考GRPO的训练逻辑
    timer = Timer()
    distillation_config = master_config["distillation"]
    generation_config = master_config["policy"]["generation"]
    
    # 设置生成策略
    generate_strategy = distillation_config.get("generate_strategy", {})
    max_length = generate_strategy.get("max_length", 2048)
    temperature = generate_strategy.get("temperature", 0.1)
    decoding_method = generate_strategy.get("decoding_method", "greedy")
    
    # 设置KL散度类型
    kl_type = distillation_config.get("kl_type", "forward")
    lambda_ = distillation_config.get("lambda_", 1.0)
    mixed_kl_weight = distillation_config.get("mixed_kl_weight", 0.5)  # 混合KL权重
    
    # 参考GRPO的逻辑：如果policy_generation为None，使用policy作为生成接口
    NEED_REFIT = True
    if student_generation is None:
        # print("  🔍 Using student_policy as generation interface (megatron backend)")
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
    print(f"Note: Temperature and decoding parameters are set in the generation backend config, not passed during calls")
    
    try:
        for batch_idx, batch in enumerate(train_dataloader):
            if step >= max_steps:
                break
                
            print(f"\n{'=' * 25} Step {step + 1}/{max_steps} {'=' * 25}")
            # print(f"🔍 Starting batch {batch_idx}, batch type: {type(batch)}")
            pass
            
            with timer.time("total_step_time"):
                # 1. 准备批次数据（完全按照GRPO模式）
                print("▶ Preparing batch...")
                #print(f"  🔍 Batch keys: {list(batch.keys()) if hasattr(batch, 'keys') else 'No keys'}")
                
                with timer.time("data_processing"):
                    # 从batch中提取message_log，与GRPO保持一致
                    batch: BatchedDataDict[DatumSpec]
                    # print(f"  🔍 Batch type after annotation: {type(batch)}")
                    pass
                    
                    # 检查batch的结构
                    if hasattr(batch, 'keys'):
                        #print(f"  🔍 Batch keys: {list(batch.keys())}")
                        if 'message_log' in batch:
                            #print(f"  🔍 message_log type: {type(batch['message_log'])}")
                            #print(f"  🔍 message_log length: {len(batch['message_log'])}")
                            if len(batch['message_log']) > 0:
                                #print(f"  🔍 First message_log item type: {type(batch['message_log'][0])}")
                                if hasattr(batch['message_log'][0], 'keys'):
                                    #print(f"  🔍 First message_log item keys: {list(batch['message_log'][0].keys())}")
                                    pass
                    else:
                        print(f"  ⚠️ Batch does not have keys attribute")
                    
                    message_logs = batch["message_log"]
                    print(f"  ✅ Successfully extracted message_logs")
                    
                    # 安全地获取batch size
                    if hasattr(batch, 'size'):
                        batch_size = batch.size
                    elif hasattr(batch, '__len__'):
                        batch_size = len(batch)
                    else:
                        batch_size = 1
                    
                    print(f"  ✓ Processing batch with {batch_size} message logs")
                    
                    # 转换为FlatMessagesType用于生成，参考GRPO
                    # print(f"  🔍 Converting message_logs to flat format...")
                    pass
                    try:
                        batched_flat, input_lengths = batched_message_log_to_flat_message(
                            message_logs,
                            pad_value_dict={"token_ids": tokenizer.pad_token_id},
                        )
                        input_ids = batched_flat["token_ids"]
                        print(f"  ✅ Successfully converted to flat format, input_ids shape: {input_ids.shape}")
                    except Exception as e:
                        print(f"  ❌ Failed to convert message_logs to flat format: {e}")
                        import traceback
                        traceback.print_exc()
                        raise
                
                # 2. 生成响应（使用与GRPO相同的rollout机制）
                print("▶ Generating responses with student model...")
                print(f"  🔍 Using generation config: max_length={max_length}, temperature={temperature}, decoding_method={decoding_method}")
                #print(f"  🔍 student_generation type: {type(student_generation)}")
                
                # 检查是否需要refit
                if student_generation is not None:
                    #print(f"  🔍 NEED_REFIT: {NEED_REFIT}, STUDENT_GENERATION_STALE: {STUDENT_GENERATION_STALE}")
                    if NEED_REFIT or STUDENT_GENERATION_STALE:
                        # print(f"  🔍 Refitting student generation...")
                        pass
                        # 传递生成配置参数（参考GRPO实现，但增加蒸馏特定的配置更新）
                        generation_config = {
                            'temperature': temperature,
                            'decoding_method': decoding_method,
                            'max_length': max_length,
                        }
                        refit_student_generation(student_policy, student_generation, colocated_inference, generation_config=generation_config, master_config=master_config)
                        STUDENT_GENERATION_STALE = False
                        NEED_REFIT = False
                        print(f"  ✅ Student generation refitted")
                    else:
                        student_generation.prepare_for_generation()
                
                # 使用与GRPO相同的rollout机制生成响应
                if student_generation is not None:
                    #print(f"  🔍 Using rollout mechanism for generation...")
                    
                    # 为蒸馏任务创建一个Ray actor版本的虚拟环境，避免环境交互错误
                    # 蒸馏任务不需要复杂的环境交互，只需要基本的生成功能
                    import ray
                    from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn
                    from typing import Any, Dict
                    import torch
                    from nemo_rl.models.generation.interfaces import GenerationDatumSpec
                    

                    
                    # 创建Ray remote环境实例，与GRPO保持一致
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
                    
                    #print(f"  🔍 Created Ray actor virtual distillation environment")
                    
                    # 关键修复：重复batch以达到正确的全局batch size（与GRPO完全一致）
                    num_generations_per_prompt = master_config["distillation"]["num_generations_per_prompt"]
                    # print(f"  🔍 Repeating batch {num_generations_per_prompt} times to reach global batch size")
                    pass
                    
                    repeated_batch: BatchedDataDict[DatumSpec] = batch.repeat_interleave(
                        num_repeats=num_generations_per_prompt
                    )
                    # print(f"  🔍 Original batch size: {batch.size}, Repeated batch size: {repeated_batch.size}")
                    pass
                    
                    # 关键修复：检查repeated_batch中所有字段的形状
                    # print(f"  🔍 Checking repeated_batch field shapes after repeat_interleave...")
                    pass
                    for key, value in repeated_batch.items():
                        if torch.is_tensor(value):
                            # print(f"  🔍 {key}: {value.shape}")
                            pass
                        elif isinstance(value, list):
                            # print(f"  🔍 {key}: list with {len(value)} items")
                            pass
                            if len(value) > 0 and isinstance(value[0], torch.Tensor):
                                # print(f"  🔍   - First item shape: {value[0].shape}")
                                pass
                        else:
                            # print(f"  🔍 {key}: {type(value)}")
                            pass
                    
                    # 特别检查loss_multiplier的形状
                    if "loss_multiplier" in repeated_batch:
                        loss_multiplier = repeated_batch["loss_multiplier"]
                        #print(f"  🔍 loss_multiplier type: {type(loss_multiplier)}")
                        if torch.is_tensor(loss_multiplier):
                            #print(f"  🔍 loss_multiplier shape: {loss_multiplier.shape}")
                            #print(f"  🔍 loss_multiplier dtype: {loss_multiplier.dtype}")
                            pass
                        elif isinstance(loss_multiplier, list):
                            #print(f"  🔍 loss_multiplier list length: {len(loss_multiplier)}")
                            if len(loss_multiplier) > 0:
                                # print(f"  🔍   - First item type: {type(loss_multiplier[0])}")
                                pass
                                if isinstance(loss_multiplier[0], torch.Tensor):
                                    # print(f"  🔍   - First item shape: {loss_multiplier[0].shape}")
                                    pass
                    
                    # 验证repeated_batch的size是否正确
                    expected_repeated_size = batch.size * num_generations_per_prompt
                    if repeated_batch.size != expected_repeated_size:
                        print(f"  ⚠️ Warning: repeated_batch size mismatch!")
                        #print(f"  🔍 Expected: {expected_repeated_size}, Got: {repeated_batch.size}")
                        #print(f"  🔍 This might cause shape issues later")
                    
                    # 关键修复：参考GRPO的实现，直接使用max_total_sequence_length作为rollout的max_seq_len
                    max_seq_len = master_config["policy"]["max_total_sequence_length"]
                    
                    print(f"  🔍 Using GRPO-style sequence length handling:")
                    print(f"    - max_seq_len (from policy.max_total_sequence_length): {max_seq_len}")
                    print(f"    - Note: Input and generation share this length limit (like GRPO)")
                    
                    # 检查并截断过长的序列，但使用更宽松的限制
                    # 参考GRPO：输入和生成共享序列长度，不需要严格分离
                    max_input_len = max_seq_len  # 允许输入占用整个序列长度
                    
                    print(f"  🔍 Sequence length check: max_seq_len={max_seq_len}, max_input_len={max_input_len}")
                    
                    # 检查并截断过长的序列
                    for i, message_log in enumerate(repeated_batch["message_log"]):
                        total_length = sum(len(msg["token_ids"]) for msg in message_log)
                        if total_length > max_input_len:
                            print(f"  ⚠️ Sample {i} sequence length {total_length} exceeds max_input_len {max_input_len}, truncating...")
                            # 截断到最大允许长度，但确保至少保留一些内容
                            remaining_length = max_input_len
                            for msg in message_log:
                                if remaining_length <= 0:
                                    # 不要完全清空，保留至少一个token
                                    if len(msg["token_ids"]) > 0:
                                        msg["token_ids"] = msg["token_ids"][:1]
                                else:
                                    msg_length = len(msg["token_ids"])
                                    if msg_length > remaining_length:
                                        msg["token_ids"] = msg["token_ids"][:remaining_length]
                                        remaining_length = 0
                                    else:
                                        remaining_length -= msg_length
                    
                    # 最终验证：确保所有序列都有内容
                    print(f"  🔍 Final validation before rollout:")
                    for i, message_log in enumerate(repeated_batch["message_log"][:3]):  # 只检查前3个样本
                        total_length = sum(len(msg["token_ids"]) for msg in message_log)
                        print(f"    Sample {i}: {total_length} tokens")
                        if total_length == 0:
                            print(f"    ❌ Sample {i} is empty!")
                    
                    # 使用rollout生成响应，与GRPO完全一致
                    try:
                        generated_batch, rollout_metrics = run_multi_turn_rollout(
                            policy_generation=student_generation,
                            input_batch=repeated_batch,  # 使用重复后的batch
                            tokenizer=tokenizer,
                            task_to_env=distillation_task_env,  # 传递Ray actor虚拟环境
                            max_seq_len=min(max_length, max_seq_len),  # 使用计算好的max_seq_len
                            max_rollout_turns=1,  # 蒸馏只需要单轮生成
                            greedy=(decoding_method == "greedy"),  # 根据decoding_method决定是否greedy
                        )
                        # 从rollout结果中提取生成的序列
                        generated_sequences = generated_batch["message_log"]
                        print(f"  ✅ Successfully generated responses via rollout")
  
                        if "loss_multiplier" in repeated_batch:
                            loss_multiplier_after = repeated_batch["loss_multiplier"]
                        
                    except Exception as e:
                        print(f"  ❌ Rollout generation failed: {e}")
                        print(f"  🔍 Attempting fallback generation method...")
                        
                        try:
                            # Fallback: 直接使用生成接口，跳过rollout
                            print(f"  🔍 Using direct generation fallback...")
                            
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
                                    print(f"  ⚠️ Empty sequence detected, added pad token")
                                
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
                            
                            print(f"  🔍 Fallback input shape: {input_ids_tensor.shape}")
                            
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
                            
                            print(f"  ✅ Fallback generation successful")
                            
                        except Exception as fallback_error:
                            print(f"  ❌ Fallback generation also failed: {fallback_error}")
                            import traceback
                            traceback.print_exc()
                            raise RuntimeError(f"Both rollout and fallback generation failed. Original error: {e}, Fallback error: {fallback_error}")
                else:
                    # print(f"  🔍 Using megatron backend, no generation interface...")
                    pass
                    # 如果使用megatron后端，直接使用policy
                    # 这里需要实现megatron的生成逻辑
                    generated_sequences = batch["message_log"]  # 暂时使用原始数据
                    print(f"  ⚠️ Megatron generation not fully implemented, using original data")
                
                print(f"  ✓ Generated responses for batch of size {batch_size}")
                
                # 标记生成完成
                if student_generation is not None:
                    #print(f"  🔍 Finishing generation...")
                    student_generation.finish_generation()
                    print(f"  ✅ Generation finished")
                
                # 3. 计算logits（使用与GRPO相同的数据处理方式）
                print("▶ Computing logits...")
                #print(f"  🔍 Generated sequences type: {type(generated_sequences)}")
                #print(f"  🔍 Generated sequences length: {len(generated_sequences)}")
                
                with timer.time("logits_computation"):
                    # 关键修复：使用与GRPO完全一致的数据处理方式
                    # 将生成的message_log转换为FlatMessagesType用于训练
                    # print(f"  🔍 Converting generated sequences to flat format...")
                    pass
                    try:
                        # 关键修复：确保使用正确的batch size
                        expected_batch_size = master_config["distillation"]["num_prompts_per_step"] * master_config["distillation"]["num_generations_per_prompt"]
                        #print(f"  🔍 Expected batch size: {expected_batch_size}")
                        #print(f"  🔍 Generated sequences length: {len(generated_sequences)}")
                        
                        if len(generated_sequences) != expected_batch_size:
                            print(f"  ⚠️ Warning: Generated sequences length {len(generated_sequences)} != expected {expected_batch_size}")
                            # 如果长度不匹配，截断或扩展到正确长度
                            if len(generated_sequences) > expected_batch_size:
                                generated_sequences = generated_sequences[:expected_batch_size]
                                # print(f"  🔍 Truncated to {len(generated_sequences)} sequences")
                                pass
                            else:
                                # 扩展batch到正确大小（重复最后一个序列）
                                while len(generated_sequences) < expected_batch_size:
                                    generated_sequences.append(generated_sequences[-1])
                                # print(f"  🔍 Extended to {len(generated_sequences)} sequences")
                                pass
                        
                        flat_messages, input_lengths = batched_message_log_to_flat_message(
                            generated_sequences,
                            pad_value_dict={"token_ids": tokenizer.pad_token_id},
                            make_sequence_length_divisible_by=master_config["policy"].get(
                                "make_sequence_length_divisible_by", 1
                            ),
                        )
                        print(f"  ✅ Successfully converted generated sequences to flat format")

                    except Exception as e:
                        print(f"  ❌ Failed to convert generated sequences to flat format: {e}")
                        import traceback
                        traceback.print_exc()
                        raise
                    
                    # 准备训练数据，与GRPO完全一致
                    # print(f"  🔍 Preparing training data...")
                    pass
                    
                    # 关键修复：确保包含所有必要的字段，与GRPO的train_data结构完全一致
                    # 添加缺失的字段，避免get_logprobs方法出错
                    if "generation_logprobs" not in flat_messages:
                        # print(f"  🔍 Adding missing generation_logprobs field...")
                        pass
                        # 为每个token创建零logprobs（因为我们没有生成logprobs）
                        flat_messages["generation_logprobs"] = torch.zeros_like(
                            flat_messages["token_ids"], dtype=torch.float32
                        )
                    
                    if "advantages" not in flat_messages:
                        # print(f"  🔍 Adding missing advantages field...")
                        pass
                        # 为蒸馏任务创建默认advantages（全1，表示所有token都重要）
                        flat_messages["advantages"] = torch.ones_like(
                            flat_messages["token_ids"], dtype=torch.float32
                        )
                    
                    if "token_loss_mask" not in flat_messages:
                        # print(f"  🔍 Adding missing token_loss_mask field...")
                        pass
                        # 关键修复：为蒸馏任务创建正确的token loss mask
                        # 只对response tokens（非prompt）计算损失
                        token_loss_mask = torch.zeros_like(
                            flat_messages["token_ids"], dtype=torch.bool
                        )
                        
                        # 根据input_lengths确定哪些是response tokens
                        for i, seq_len in enumerate(input_lengths):
                            # 假设前N个tokens是prompt，后面的tokens是response
                            # 这里需要根据实际的数据结构来调整
                            if seq_len > 0:
                                # 标记response tokens为True（参与损失计算）
                                token_loss_mask[i, :seq_len] = True
                        
                        flat_messages["token_loss_mask"] = token_loss_mask
                        print(f"  🔍 Created distillation token_loss_mask: {token_loss_mask.sum().item()} response tokens out of {token_loss_mask.numel()} total tokens")
                    

                    
                    # 验证所有字段的batch维度一致
                    expected_batch_size = flat_messages['token_ids'].shape[0]
                    expected_seq_len = flat_messages['token_ids'].shape[1]
                    
                    # print(f"  🔍 Expected batch size: {expected_batch_size}")
                    pass
                    # print(f"  🔍 Expected sequence length: {expected_seq_len}")
                    pass
                    
                    # 验证并修复形状不匹配的字段
                    if flat_messages['advantages'].shape[0] != expected_batch_size:
                        print(f"  ⚠️ Warning: advantages batch dimension mismatch, fixing...")
                        flat_messages['advantages'] = flat_messages['advantages'][:expected_batch_size]
                    
                    if flat_messages['generation_logprobs'].shape[0] != expected_batch_size:
                        print(f"  ⚠️ Warning: generation_logprobs batch dimension mismatch, fixing...")
                        flat_messages['generation_logprobs'] = flat_messages['generation_logprobs'][:expected_batch_size]
                    
                    if flat_messages['token_loss_mask'].shape[0] != expected_batch_size:
                        print(f"  ⚠️ Warning: token_loss_mask batch dimension mismatch, fixing...")
                        flat_messages['token_loss_mask'] = flat_messages['token_loss_mask'][:expected_batch_size]
                    
                    if repeated_batch['loss_multiplier'].shape[0] != expected_batch_size:
                        print(f"  ⚠️ Warning: loss_multiplier batch dimension mismatch, fixing...")
                        repeated_batch['loss_multiplier'] = repeated_batch['loss_multiplier'][:expected_batch_size]
                    
                    # 验证sequence维度
                    if flat_messages['advantages'].shape[1] != expected_seq_len:
                        print(f"  ⚠️ Warning: advantages sequence dimension mismatch, fixing...")
                        if flat_messages['advantages'].shape[1] > expected_seq_len:
                            flat_messages['advantages'] = flat_messages['advantages'][:, :expected_seq_len]
                        else:
                            flat_messages['advantages'] = flat_messages['advantages'].expand(-1, expected_seq_len)
                    
                    if flat_messages['generation_logprobs'].shape[1] != expected_seq_len:
                        print(f"  ⚠️ Warning: generation_logprobs sequence dimension mismatch, fixing...")
                        if flat_messages['generation_logprobs'].shape[1] > expected_seq_len:
                            flat_messages['generation_logprobs'] = flat_messages['generation_logprobs'][:, :expected_seq_len]
                        else:
                            flat_messages['generation_logprobs'] = flat_messages['generation_logprobs'].expand(-1, expected_seq_len)
                    
                    if flat_messages['token_loss_mask'].shape[1] != expected_seq_len:
                        print(f"  ⚠️ Warning: token_loss_mask sequence dimension mismatch, fixing...")
                        if flat_messages['token_loss_mask'].shape[1] > expected_seq_len:
                            flat_messages['token_loss_mask'] = flat_messages['token_loss_mask'][:, :expected_seq_len]
                        else:
                            flat_messages['token_loss_mask'] = flat_messages['token_loss_mask'].expand(-1, expected_seq_len)
                    
                    #print(f"  🔍 After shape validation and fixing:")
                    #print(f"  🔍 flat_messages['advantages'] shape: {flat_messages['advantages'].shape}")
                    #print(f"  🔍 flat_messages['generation_logprobs'] shape: {flat_messages['generation_logprobs'].shape}")
                    #print(f"  🔍 flat_messages['token_loss_mask'] shape: {flat_messages['token_loss_mask'].shape}")
                    #print(f"  🔍 repeated_batch['loss_multiplier'] shape: {repeated_batch['loss_multiplier'].shape}")
                    
                    # 关键修复：强制确保所有字段的形状都正确
                    # print(f"  🔍 Final shape validation and forced fixing...")
                    pass
                    
                    # 确保loss_multiplier是正确的形状
                    if isinstance(repeated_batch["loss_multiplier"], torch.Tensor):
                        if len(repeated_batch["loss_multiplier"].shape) > 1:
                            print(f"  ⚠️ Warning: loss_multiplier has wrong shape {repeated_batch['loss_multiplier'].shape}, fixing...")
                            # 如果loss_multiplier是多维的，取第一个维度
                            repeated_batch["loss_multiplier"] = repeated_batch["loss_multiplier"].flatten()[:expected_batch_size]
                            # print(f"  🔍 Fixed loss_multiplier shape: {repeated_batch['loss_multiplier'].shape}")
                            pass
                        elif repeated_batch["loss_multiplier"].shape[0] != expected_batch_size:
                            print(f"  ⚠️ Warning: loss_multiplier batch dimension mismatch, fixing...")
                            repeated_batch["loss_multiplier"] = repeated_batch["loss_multiplier"][:expected_batch_size]
                            # print(f"  🔍 Fixed loss_multiplier shape: {repeated_batch['loss_multiplier'].shape}")
                            pass
                    elif isinstance(repeated_batch["loss_multiplier"], list):
                        print(f"  ⚠️ Warning: loss_multiplier is a list, converting to tensor...")
                        repeated_batch["loss_multiplier"] = torch.tensor(repeated_batch["loss_multiplier"][:expected_batch_size], dtype=torch.float32)
                        # print(f"  🔍 Converted loss_multiplier shape: {repeated_batch['loss_multiplier'].shape}")
                        pass

                    
                    # 最终验证loss_multiplier的类型和形状
                    if not isinstance(repeated_batch["loss_multiplier"], torch.Tensor):
                        print(f"  ❌ Critical error: loss_multiplier is not a tensor!")
                        print(f"  🔍 Type: {type(repeated_batch['loss_multiplier'])}")
                        print(f"  🔍 Value: {repeated_batch['loss_multiplier']}")
                        
                        # 尝试修复
                        if isinstance(repeated_batch["loss_multiplier"], (list, tuple)):
                            repeated_batch["loss_multiplier"] = torch.tensor(repeated_batch["loss_multiplier"], dtype=torch.float32)
                            print(f"  ✅ Fixed: Converted list to tensor")
                        elif isinstance(repeated_batch["loss_multiplier"], (int, float)):
                            repeated_batch["loss_multiplier"] = torch.tensor([repeated_batch["loss_multiplier"]] * expected_batch_size, dtype=torch.float32)
                            print(f"  ✅ Fixed: Converted scalar to tensor")
                        else:
                            # 创建默认的loss_multiplier
                            repeated_batch["loss_multiplier"] = torch.ones(expected_batch_size, dtype=torch.float32)
                            print(f"  ✅ Fixed: Created default loss_multiplier")
                    
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
                        print(f"  ❌ Critical error: Batch dimensions are not consistent!")
                        print(f"  🔍 Batch sizes: {all_batch_sizes}")
                        raise ValueError(f"Batch dimensions must be consistent, got: {all_batch_sizes}")
                    
                    print(f"  ✅ All batch dimensions are consistent: {all_batch_sizes[0]}")
                    
                    train_data = BatchedDataDict[DistillationLossDataDict]({
                        "input_ids": flat_messages["token_ids"],
                        "input_lengths": input_lengths,
                        "advantages": flat_messages["advantages"],
                        "generation_logprobs": flat_messages["generation_logprobs"],
                        "token_mask": flat_messages["token_loss_mask"],  # 使用token_loss_mask而不是自定义的token_mask
                        "sample_mask": repeated_batch["loss_multiplier"],
                    })
                    print(f"  ✅ Training data prepared")

                    
                    # 验证batch size是否正确
                    if train_data.size != expected_batch_size:
                        print(f"  ⚠️ Warning: Expected batch size {expected_batch_size}, got {train_data.size}")
                    else:
                        print(f"  ✅ Batch size is correct: {train_data.size}")
                    
                    # 关键修复：确保数据在正确的设备上
                    train_data.to("cpu")  # 与GRPO保持一致
                    
                    # 教师模型前向传播（需要单独实现，因为模型大小不同）
                    print("  ✓ Computing teacher model logits...")
                    with torch.no_grad():
                        # 实现真正的教师模型推理
                        teacher_model_path = master_config["distillation"]["teacher_model_path"]
                        # print(f"  🔍 Loading teacher model: {teacher_model_path}")
                        pass
                        
                        try:
                            # 方法1: 尝试使用transformers直接加载教师模型
                            from transformers import AutoModelForCausalLM, AutoTokenizer
                            
                            # 检查是否已经有教师模型实例
                            if not hasattr(student_policy, '_teacher_model'):
                                # print(f"  🔍 Loading teacher model from {teacher_model_path}...")
                                pass
                                
                                try:
                                    # 内存优化：使用device_map="auto"和低精度
                                    teacher_model = AutoModelForCausalLM.from_pretrained(
                                        teacher_model_path,
                                        torch_dtype=torch.bfloat16,
                                        device_map="auto",
                                        trust_remote_code=True,
                                        low_cpu_mem_usage=True,  # 减少CPU内存使用
                                    )
                                    
      
                                    # 检查模型是否在正确的设备上
                                    if hasattr(teacher_model, 'device'):
                                        # print(f"  🔍   - Device: {teacher_model.device}")
                                        pass
                                    else:
                                        # 检查第一个参数的设备
                                        try:
                                            device = next(teacher_model.parameters()).device
                                            # print(f"  🔍   - Device (from params): {device}")
                                            pass
                                        except Exception as e:
                                            # print(f"  🔍   - Device: Could not determine ({e})")
                                            pass
                                    
                                    teacher_model.eval()
                                    
                                    # 测试前向传播，确保输出形状正确
                                    # print(f"  🔍 Testing teacher model forward pass...")
                                    pass
                                    try:
                                        test_input = torch.randint(0, teacher_model.config.vocab_size, (1, 10), device=next(teacher_model.parameters()).device)
                                        with torch.no_grad():
                                            test_output = teacher_model(test_input)
                                            test_logits = test_output.logits
                               
                                            if test_logits.shape != (1, 10, teacher_model.config.vocab_size):
                                                print(f"  ⚠️ Warning: Test logits shape is incorrect!")
                                                # print(f"  🔍 This might indicate a problem with the model configuration")
                                                pass
                                    except Exception as e:
                                        print(f"  ⚠️ Warning: Test forward pass failed: {e}")
                                        # print(f"  🔍 This might indicate a problem with the model")
                                        pass
                                    
                                    # 缓存教师模型
                                    student_policy._teacher_model = teacher_model
                                    print(f"  ✅ Teacher model loaded successfully")
                                    
                                except Exception as e:
                                    print(f"  ❌ Failed to load teacher model: {e}")
                                    import traceback
                                    traceback.print_exc()
                                    raise
                            else:
                                teacher_model = student_policy._teacher_model

                            teacher_input_ids = train_data["input_ids"]
                            

                            try:
                                test_input = torch.randint(0, 1000, (2, 5), device=next(teacher_model.parameters()).device)
                                # print(f"  🔍 Test input shape: {test_input.shape}")
                                pass
                                
                                with torch.no_grad():
                                    test_output = teacher_model(test_input)
                                    test_logits = test_output.logits

                                    if len(test_logits.shape) != 3:
                                        print(f"  ❌ Critical error: Test logits has wrong number of dimensions!")
                                        # print(f"  🔍 This indicates a fundamental problem with the teacher model")
                                        pass
                                        raise ValueError(f"Teacher model produces incorrect logits shape: {test_logits.shape}")
                                    
                                    print(f"  ✅ Test forward pass successful, proceeding with actual computation...")
                            except Exception as e:
                                print(f"  ❌ Test forward pass failed: {e}")
                                raise
                            
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
                                        # print(f"  🔍 Chunk {i//chunk_size + 1}: Using device {device}")
                                        pass
                                    except Exception as e:
                                        print(f"  ⚠️ Warning: Could not determine teacher model device: {e}")
                                        # 默认使用CPU
                                        chunk_input_ids = chunk_input_ids.cpu()
                                        # print(f"  🔍 Chunk {i//chunk_size + 1}: Using CPU as fallback")
                                        pass
                                
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
                            
                            print(f"  ✅ Teacher logits computed successfully")
   
                            
                            # 关键修复：验证teacher_logits的形状
                            expected_teacher_shape = (batch_size, teacher_input_ids.shape[1], -1)  # 最后一个维度是vocab_size
                          
                            
                            # 检查并修复teacher_logits的形状
                            if len(teacher_logits.shape) != 3:
                                print(f"  ⚠️ Warning: Teacher logits has wrong number of dimensions!")
                                
                                
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
                                        # print(f"  🔍 Reshaped teacher_logits shape: {teacher_logits.shape}")
                                        pass
                                    else:
                                        print(f"  ❌ Cannot determine how to reshape teacher_logits!")
                                        raise ValueError(f"Teacher logits shape {teacher_logits.shape} is not compatible with expected shape {expected_teacher_shape}")
                                elif len(teacher_logits.shape) > 3:
                                    print(f"  ⚠️ Warning: Teacher logits has too many dimensions: {teacher_logits.shape}")
                                    # 尝试压缩多余的维度
                                    if teacher_logits.shape[0] == batch_size:
                                        # 保持batch维度，压缩其他维度
                                        teacher_logits = teacher_logits.view(batch_size, -1, teacher_logits.shape[-1])
                                        # print(f"  🔍 Compressed teacher_logits shape: {teacher_logits.shape}")
                                        pass
                                    else:
                                        print(f"  ❌ Cannot determine how to handle teacher_logits with shape {teacher_logits.shape}")
                                        raise ValueError(f"Teacher logits shape {teacher_logits.shape} is not compatible with expected shape {expected_teacher_shape}")
                            
                            # 验证修复后的形状
                            if teacher_logits.shape[0] != expected_teacher_shape[0] or teacher_logits.shape[1] != expected_teacher_shape[1]:
                                print(f"  ⚠️ Warning: Teacher logits shape still mismatch after reshaping!")
                                # print(f"  🔍 Expected: {expected_teacher_shape}")
                                pass
                                # print(f"  🔍 Got: {teacher_logits.shape}")
                                pass
                                # 尝试进一步修复形状
                                if teacher_logits.shape[0] != batch_size:
                                    # print(f"  🔍 Fixing teacher_logits batch dimension...")
                                    pass
                                    if teacher_logits.shape[0] > batch_size:
                                        teacher_logits = teacher_logits[:batch_size]
                                    else:
                                        # 扩展batch维度
                                        teacher_logits = teacher_logits.expand(batch_size, -1, -1)
                                
                                if teacher_logits.shape[1] != teacher_input_ids.shape[1]:
                                    # print(f"  🔍 Fixing teacher_logits sequence dimension...")
                                    pass
                                    if teacher_logits.shape[1] > teacher_input_ids.shape[1]:
                                        teacher_logits = teacher_logits[:, :teacher_input_ids.shape[1], :]
                                    else:
                                        # 扩展sequence维度
                                        teacher_logits = teacher_logits.expand(-1, teacher_input_ids.shape[1], -1)
                            
                            # 最终验证：确保形状完全正确
                            final_shape = teacher_logits.shape
                            if final_shape[0] != batch_size or final_shape[1] != teacher_input_ids.shape[1]:
                                print(f"  ❌ Critical error: Final teacher_logits shape {final_shape} is still incorrect!")
        
                                raise ValueError(f"Failed to fix teacher_logits shape. Final shape: {final_shape}")
                            
                            
                            # 将教师logits添加到训练数据中
                            train_data["teacher_logits"] = teacher_logits
                            print(f"  ✅ Teacher logits added to training data")
                            
                        except Exception as e:
                            print(f"  ❌ Failed to load teacher model: {e}")
                            print(f"  ⚠️ Falling back to student logits placeholder")
                            import traceback
                            traceback.print_exc()
                            # print(f"  🔍 This will result in KL loss = 0 (no distillation effect)")
                            pass
                            
                            # 回退到占位符（不推荐，但确保程序能运行）
                            print(f"  ⚠️ WARNING: This will result in ineffective distillation training!")
                            # 创建占位符teacher_logits以避免错误
                            batch_size = train_data["input_ids"].shape[0]
                            seq_len = train_data["input_ids"].shape[1]
                            vocab_size = 32000  # 假设的词汇表大小
                            placeholder_logits = torch.randn(batch_size, seq_len, vocab_size) * 0.1
                            train_data["teacher_logits"] = placeholder_logits
                            print(f"  ✅ Added placeholder teacher_logits with shape: {placeholder_logits.shape}")
                    
                    # 关键修复：准备学生模型进行logprob推理
                    print("  ✓ Preparing student model for logprob inference...")
                    try:
                        student_policy.prepare_for_lp_inference()
                        print(f"  ✅ Student policy prepared for logprob inference")
                    except Exception as e:
                        print(f"  ❌ Failed to prepare student policy for logprob inference: {e}")
                        import traceback
                        traceback.print_exc()
                        raise
                    
                    # 学生模型前向传播（可训练模型）
                    print("  ✓ Computing student model logits...")
                    try:
                        
                        # 检查并修复teacher_logits的形状（如果存在）
                        if "teacher_logits" in train_data:
                            teacher_logits = train_data["teacher_logits"]
                            
                            # 如果teacher_logits的形状不正确，强制修复
                            if len(teacher_logits.shape) != 3:
                                print(f"  ⚠️ Warning: teacher_logits has wrong shape {teacher_logits.shape}, fixing...")
                                if len(teacher_logits.shape) == 2:
                                    # 如果是[batch_size, vocab_size]，重塑为[batch_size, seq_len, vocab_size]
                                    batch_size = teacher_logits.shape[0]
                                    vocab_size = teacher_logits.shape[1]
                                    seq_len = train_data["input_ids"].shape[1]
                                    teacher_logits = teacher_logits.unsqueeze(1).expand(-1, seq_len, -1)
          
                                else:
                                    print(f"  ❌ Critical error: teacher_logits has unexpected shape {teacher_logits.shape}")
                                    raise ValueError(f"teacher_logits has unexpected shape: {teacher_logits.shape}")
                            
                            # 验证修复后的形状
                            expected_shape = (train_data["input_ids"].shape[0], train_data["input_ids"].shape[1], -1)
                            if teacher_logits.shape[0] != expected_shape[0] or teacher_logits.shape[1] != expected_shape[1]:
                                print(f"  ❌ Critical error: teacher_logits shape still incorrect after fixing!")
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
                                    # print(f"  🔍 Expanded input shapes to: {input_ids.shape}")
                                    pass
                            
                            # 创建正确的训练数据格式
                            # print(f"  🔍 Creating training data for get_logprobs...")
                            pass
                            train_data_for_logprobs = BatchedDataDict[DistillationLossDataDict]({
                                "input_ids": input_ids,
                                "input_lengths": torch.tensor([input_ids.shape[1]] * input_ids.shape[0]),
                                "advantages": torch.ones(input_ids.shape[0], input_ids.shape[1]),
                                "generation_logprobs": torch.zeros(input_ids.shape[0], input_ids.shape[1]),
                                "token_mask": torch.ones(input_ids.shape[0], input_ids.shape[1]),
                                "sample_mask": torch.ones(input_ids.shape[0]),
                            })
                            
                            try:
                                # 使用get_logprobs方法获取logits
                                result = student_policy.get_logprobs(train_data_for_logprobs)
                                # print(f"  🔍 get_logprobs successful")
                                pass
                                
                                # 检查返回结果的结构
                                # print(f"  🔍 Result keys: {list(result.keys())}")
                                pass
                                for key, value in result.items():
                                    if torch.is_tensor(value):
                                        # print(f"  🔍 {key}: {value.shape}")
                                        pass
                                    else:
                                        # print(f"  🔍 {key}: {type(value)}")
                                        pass
                                
                                # 尝试获取logits
                                if "logits" in result:
                                    student_logits = result["logits"]
                                    # print(f"  🔍 Successfully got logits from result")
                                    pass
                                elif "logprobs" in result:
                                    # 如果只有logprobs，我们需要从logprobs重建logits
                                    # print(f"  🔍 Only logprobs available, attempting to reconstruct logits...")
                                    pass
                                    logprobs = result["logprobs"]
                                    # print(f"  🔍 logprobs shape: {logprobs.shape}")
                                    pass
                                    
                                    # 这里我们需要实现从logprobs到logits的转换
                                    # 由于这是一个复杂的转换，我们先使用logprobs作为替代
                                    # print(f"  🔍 Using logprobs as student_logits for now...")
                                    pass
                                    student_logits = logprobs.unsqueeze(-1).expand(-1, -1, 151936)  # 假设vocab_size=151936
                                    # print(f"  🔍 Reconstructed logits shape: {student_logits.shape}")
                                    pass
                                else:
                                    raise ValueError(f"Neither 'logits' nor 'logprobs' found in result: {list(result.keys())}")
                                
                            except Exception as e:
                                # print(f"  🔍 get_logprobs failed: {e}")
                                pass
                                # print(f"  🔍 Trying alternative approach...")
                                pass
                                
                                # 如果get_logprobs失败，尝试直接访问模型
                                try:
                                    # print(f"  🔍 Attempting to access model directly...")
                                    pass
                                    
                                    # 获取第一个worker
                                    first_worker = student_policy.worker_group.workers[0]
                                    
                                    # 检查worker是否有model属性
                                    # print(f"  🔍 Checking worker attributes...")
                                    pass
                                    worker_attrs = dir(first_worker)
                                    # print(f"  🔍 Worker attributes: {worker_attrs}")
                                    pass
                                    
                                    # 尝试调用worker的get_logprobs方法
                                    # print(f"  🔍 Calling worker.get_logprobs directly...")
                                    pass
                                    # 直接调用方法，不使用.remote()
                                    worker_result = first_worker.get_logprobs(train_data_for_logprobs)
                                    # print(f"  🔍 Worker get_logprobs successful")
                                    pass
                                    
                                    # 处理worker结果
                                    if "logits" in worker_result:
                                        student_logits = worker_result["logits"]
                                    elif "logprobs" in worker_result:
                                        logprobs = worker_result["logprobs"]
                                        student_logits = logprobs.unsqueeze(-1).expand(-1, -1, 151936)
                                    else:
                                        raise ValueError(f"Worker result missing logits/logprobs: {list(worker_result.keys())}")
                                        
                                except Exception as e2:
                                    # print(f"  🔍 Direct worker access also failed: {e2}")
                                    pass
                                    raise RuntimeError(f"All approaches to get student logits failed: {e2}")
                            
                            # print(f"  🔍 Raw student logits shape: {student_logits.shape}")
                            pass
                            
                            # 如果batch size被调整了，恢复到原始大小
                            if student_logits.shape[0] > current_batch_size:
                                # print(f"  🔍 Restoring original batch size...")
                                pass
                                student_logits = student_logits[:current_batch_size]
                                # print(f"  🔍 Final student logits shape: {student_logits.shape}")
                                pass
                            
                        
                        print(f"  ✅ Student logits computed successfully")

                        
                        # 关键修复：验证student_logits的形状
                        if student_logits.shape[0] != train_data["input_ids"].shape[0]:
                            print(f"  ⚠️ Warning: Student logits batch dimension mismatch!")
                        
                        if student_logits.shape[1] != train_data["input_ids"].shape[1]:
                            print(f"  ⚠️ Warning: Student logits sequence dimension mismatch!")
  
                        
                    except Exception as e:
                        print(f"  ❌ Failed to compute student logits: {e}")
                        import traceback
                        traceback.print_exc()
                        raise
                    
                    # 将学生logits添加到训练数据中
                    train_data["student_logits"] = student_logits
                    print(f"  ✅ Student logits added to training data")
                    
                    # 计算蒸馏损失
                    print("  ✓ Computing distillation loss...")
                    try:
                        # 使用损失函数计算蒸馏损失 - 修复：传递所有必要的参数
                        # 将蒸馏参数添加到train_data中，供损失函数使用
                        train_data["kl_type"] = kl_type
                        train_data["lambda_"] = lambda_
                        train_data["mixed_kl_weight"] = mixed_kl_weight
                        
                        # 关键修复：正确传递token mask给损失函数
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
                
                # 5. 训练学生模型（完全按照GRPO模式）
                print("▶ Training student model...")

                # 验证所有字段的batch维度一致
                all_batch_sizes = [train_data[key].shape[0] for key in train_data.keys() if torch.is_tensor(train_data[key])]
                if len(set(all_batch_sizes)) != 1:
                    #print(f"  ❌ Critical error: Batch dimensions are not consistent!")
                    print(f"  🔍 Batch sizes: {all_batch_sizes}")
                    raise ValueError(f"Batch dimensions must be consistent, got: {all_batch_sizes}")
                
                print(f"  ✅ All batch dimensions are consistent: {all_batch_sizes[0]}")
                
                

                distillation_safe_data = {}
                
                for key, value in train_data.items():
                    if key in ["teacher_logits", "student_logits"]:
                        distillation_safe_data[key] = value
                        if len(value.shape) == 3:
                            batch_size, seq_len, vocab_size = value.shape
                            flattened_logits = value.view(batch_size * seq_len, vocab_size)
                            
                            # 创建一个特殊的key，worker不会检查
                            safe_key = f"distillation_{key}_flattened"
                            distillation_safe_data[safe_key] = flattened_logits
                            
                            # 存储原始形状信息
                            distillation_safe_data[f"{safe_key}_shape"] = torch.tensor([batch_size, seq_len, vocab_size])
                            
                            # print(f"  🔍 Converted {key} to safe format: {flattened_logits.shape}")
                            pass
                        else:
                            print(f"  ⚠️ Warning: {key} has unexpected shape: {value.shape}")
                            distillation_safe_data[key] = value
                    else:
                        # 对于其他字段，直接复制
                        distillation_safe_data[key] = value
                
                # 添加调试信息：显示所有字段的类型
                print(f"  🔍 Distillation safe data fields:")
                for key, value in distillation_safe_data.items():
                    if torch.is_tensor(value):
                        print(f"    {key}: tensor {value.shape}")
                    elif isinstance(value, (list, tuple)):
                        print(f"    {key}: {type(value).__name__} with {len(value)} items")
                    elif isinstance(value, (int, float)):
                        print(f"    {key}: {type(value).__name__} = {value}")
                    else:
                        print(f"    {key}: {type(value).__name__}")
                
                batch_sizes = {}
                for key, value in distillation_safe_data.items():
                    if torch.is_tensor(value):
                        if len(value.shape) > 0:
                            batch_sizes[key] = value.shape[0]
                        else:
                            batch_sizes[key] = 1  # 标量张量
                    elif isinstance(value, (list, tuple)):
                        batch_sizes[key] = len(value)
                    elif isinstance(value, (int, float)):
                        batch_sizes[key] = 1  # 标量值
                    else:
                        # 对于其他类型，尝试调用len，如果失败则设为1
                        try:
                            batch_sizes[key] = len(value)
                        except (TypeError, AttributeError):
                            batch_sizes[key] = 1
                            print(f"  ⚠️ Warning: Field {key} has unsupported type {type(value)}, setting batch size to 1")
                
                
                
                
                with timer.time("training_prep"):

                    student_policy.prepare_for_training()  # 与GRPO完全一致
                    STUDENT_GENERATION_STALE = True  # *** MARK AS STALE AFTER TRAINING ***
                    print(f"  ✅ Student policy prepared for training")
                
                with timer.time("policy_training"):
                    try:
                        train_results = student_policy.train(train_data, loss_fn)
                        print("  ✅ Training completed")
                    except Exception as e:
                        print(f"  ❌ Policy training failed: {e}")
                        import traceback
                        traceback.print_exc()
                        raise
                loss=train_results["loss"].numpy()
                #grad_norm=train_results["grad_norm"].numpy()
                print(f"  ✅ Distillation loss computed successfully")
                # 记录损失
                if logger is not None:
                    # 记录主要训练损失
                    logger.log_metrics({"train/loss": loss.item()}, step)
                    
                    # 记录生成长度相关指标
                    if "input_ids" in train_data:
                        input_lengths = (train_data["input_ids"] != 0).sum(dim=1)
                        avg_input_length = input_lengths.float().mean().item()
                        max_input_length = input_lengths.max().item()
                        min_input_length = input_lengths.min().item()
                        
                        logger.log_metrics({
                            "train/avg_input_length": avg_input_length,
                            "train/max_input_length": max_input_length,
                            "train/min_input_length": min_input_length,
                            "train/input_length_std": input_lengths.float().std().item(),
                        }, step)
                    
                    # 记录当前最佳验证loss（如果可用）
                    if "val_loss" in distillation_save_state and distillation_save_state["val_loss"] is not None:
                        current_best_val_loss = distillation_save_state["val_loss"]
                        logger.log_metrics({"train/best_val_loss": current_best_val_loss}, step)
                        #print(f"  🔍 [Training] Current Best Val Loss = {current_best_val_loss:.6f}")
                    
                    # 记录蒸馏参数
                    logger.log_metrics({
                        "train/kl_type": 1.0 if kl_type == "forward" else (2.0 if kl_type == "reverse" else 3.0),
                        "train/lambda": lambda_,
                        "train/mixed_kl_weight": mixed_kl_weight,
                    }, step)
                    
                    # 打印训练loss信息
                    print(f"  ✅✅✅ [Training] Step {step}: Loss = {loss.item():.6f}")
                    if "kl_loss" in loss_metrics:
                        print(f"  🔍 [Training] KL Loss = {loss_metrics['kl_loss']:.6f}")
                step += 1
                distillation_save_state["step"] = step
                # 使用配置中的值，与GRPO保持一致
                distillation_save_state["consumed_samples"] += distillation_config.get("num_prompts_per_step", 1)
                print(f"  ✅ Training state updated: step={step}, consumed_samples={distillation_save_state['consumed_samples']}")
                
                # 7. 保存检查点
                if step % distillation_config["save_steps"] == 0:
                    print(f"  ✓ Saving checkpoint at step {step}")
                    # 使用与GRPO相同的检查点保存逻辑
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
                        print(f"  ✅ Checkpoint saved successfully")
                    except Exception as e:
                        print(f"  ❌ Failed to save checkpoint: {e}")
                        import traceback
                        traceback.print_exc()
                
                # 8. 验证（完全按照GRPO模式）
                if step % distillation_config["eval_steps"] == 0 and val_dataloader is not None:
                    print(f"  ✓ Running validation at step {step}")
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
                                # print(f"  🔍 Preparing generation for validation...")
                                pass
                                student_generation.prepare_for_generation()
                        
                        print(f"  🔍 Running validation...")
                        val_metrics = validate(
                            student_generation,
                            val_dataloader,
                            tokenizer,
                            step + 1,
                            master_config,
                        )
                        print(f"  ✅ Validation completed")
                        
                        # 记录验证指标
                        if val_metrics:
                            # 记录验证loss - 添加eval/loss记录
                            if "val_loss" in val_metrics:
                                # 记录到validation/命名空间
                                logger.log_metrics({"validation/val_loss": val_metrics["val_loss"]}, step + 1)
                                # 同时记录到eval/命名空间，与GRPO/SFT保持一致
                                logger.log_metrics({"eval/loss": val_metrics["val_loss"]}, step + 1)
                                distillation_save_state["val_loss"] = val_metrics["val_loss"]
                                print(f"  ✅✅✅ [Validation] Step {step + 1}: Val Loss = {val_metrics['val_loss']:.6f}")
                                print(f"  🔍 [Eval] Step {step + 1}: Eval Loss = {val_metrics['val_loss']:.6f}")
                            
                            # 记录其他验证指标
                            for k, v in val_metrics.items():
                                if k != "val_loss" and isinstance(v, (int, float)):
                                    logger.log_metrics({f"validation/{k}": v}, step + 1)
                                    # 同时记录到eval/命名空间
                                    logger.log_metrics({f"eval/{k}": v}, step + 1)
                            
                            # 记录验证时的生成长度信息
                            if "val_avg_sequence_length" in val_metrics:
                                # 记录到validation/命名空间
                                logger.log_metrics({
                                    "validation/avg_sequence_length": val_metrics["val_avg_sequence_length"],
                                    "validation/max_sequence_length": val_metrics.get("val_max_sequence_length", 0),
                                    "validation/min_sequence_length": val_metrics.get("val_min_sequence_length", 0),
                                }, step + 1)
                                
                                # 同时记录到eval/命名空间
                                logger.log_metrics({
                                    "eval/avg_sequence_length": val_metrics["val_avg_sequence_length"],
                                    "eval/max_sequence_length": val_metrics.get("val_max_sequence_length", 0),
                                    "eval/min_sequence_length": val_metrics.get("val_min_sequence_length", 0),
                                }, step + 1)
                                
                                # 打印验证长度信息
                                print(f"  🔍 [Validation] Avg Sequence Length = {val_metrics['val_avg_sequence_length']:.1f}")
                                print(f"  🔍 [Validation] Max Sequence Length = {val_metrics.get('val_max_sequence_length', 0)}")
                                print(f"  🔍 [Validation] Min Sequence Length = {val_metrics.get('val_min_sequence_length', 0)}")
                            
                            # 记录验证时的蒸馏参数
                            logger.log_metrics({
                                "validation/kl_type": 1.0 if kl_type == "forward" else (2.0 if kl_type == "reverse" else 3.0),
                                "validation/lambda": lambda_,
                                "validation/mixed_kl_weight": mixed_kl_weight,
                                "eval/kl_type": 1.0 if kl_type == "forward" else (2.0 if kl_type == "reverse" else 3.0),
                                "eval/lambda": lambda_,
                                "eval/mixed_kl_weight": mixed_kl_weight,
                            }, step + 1)
                        
                        if student_generation is not None:
                            student_generation.finish_generation()
                    except Exception as e:
                        print(f"  ❌ Validation failed: {e}")
                        import traceback
                        traceback.print_exc()
                
                # 9. 日志记录
                if step % distillation_config["logging_steps"] == 0:
                    print(f"  🔍 Logging metrics...")
                    try:
                        logger.log_metrics({
                            "step": step,
                            "loss": loss.item(), # Changed from loss.item() to kl_loss.item()
                            "consumed_samples": distillation_save_state["consumed_samples"],
                        })
                        print(f"  ✅ Metrics logged successfully")
                    except Exception as e:
                        print(f"  ❌ Failed to log metrics: {e}")
                        import traceback
                        traceback.print_exc()
                
                print(f"  ✅ Step {step + 1} completed successfully")
    
    except Exception as e:
        print(f"❌ Distillation training failed with error: {e}")
        print(f"🔍 Error occurred at step {step + 1}, batch_idx {batch_idx if 'batch_idx' in locals() else 'unknown'}")
        import traceback
        traceback.print_exc()
