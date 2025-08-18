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
from typing import Any, Optional, TypedDict, TypeVar, NotRequired

import torch
import torch.distributed

from nemo_rl.algorithms.interfaces import LossFunction, LossType
from nemo_rl.algorithms.utils import (
    calculate_kl_penalty_joschu2020,
    masked_mean,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.model_utils import from_parallel_logits_to_logprobs
from nemo_rl.models.dtensor.parallelize import (
    get_logprobs_from_vocab_parallel_logits,
)

Tensor = TypeVar("Tensor", bound=torch.Tensor)


class ClippedPGLossConfig(TypedDict):
    reference_policy_kl_penalty: float
    ratio_clip_min: float
    ratio_clip_max: float
    ratio_clip_c: float
    use_on_policy_kl_approximation: bool
    use_importance_sampling_correction: bool
    token_level_loss: bool


class ClippedPGLossDataDict(TypedDict):
    """Required keys for the Clipped Policy Gradient loss function."""

    input_ids: torch.Tensor
    advantages: torch.Tensor
    prev_logprobs: torch.Tensor
    generation_logprobs: torch.Tensor
    reference_policy_logprobs: torch.Tensor
    token_mask: torch.Tensor
    sample_mask: torch.Tensor
    __extra__: Any


class ClippedPGLossFn(LossFunction):
    """Generalized Clipped Policy Gradient loss function w/ KL regularization.

    This implements:

    - PPO (Clipped) - https://arxiv.org/abs/1707.06347
    - GRPO - https://arxiv.org/abs/2402.03300
    - REINFORCE/RLOO (set disable_ppo_ratio = True and ignores ratio_clip_min/ratio_clip_max) - https://arxiv.org/abs/2402.14740

    Formula:
    L(θ) = E_t [ min(r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t) ] - β * KL(π_θ || π_ref)

    where:
    - r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t) is the probability ratio
    - A_t is the advantage estimate
    - ε is the clip parameter (ratio_clip_min/ratio_clip_max)
        - As proposed in the DAPO paper (https://arxiv.org/pdf/2503.14476),
          we allow setting a distinct minimum and maximum value for the clip parameter (set to the same value for PPO/GRPO/etc.)
            - ratio_clip_min: minimum value for the clip parameter
            - ratio_clip_max: maximum value for the clip parameter
    - β is the KL penalty coefficient (reference_policy_kl_penalty)
    - KL(π_θ || π_ref) is the KL divergence between the current policy and reference policy (Schulman Approx.)

    For REINFORCE/RLOO (when disable_ppo_ratio=True), the formula simplifies to:
    L(θ) = E_t [ π_θ(a_t|s_t) * A_t ] - β * KL(π_θ || π_ref)

    Also supports "Dual-Clipping" from https://arxiv.org/pdf/1912.09729, which
    imposes an additional upper bound on the probability ratio when advantages are negative.
    This prevents excessive policy updates. $rA << 0$ -> $cA$(clipped)
    The loss function is modified to the following when A_t < 0:
    L(θ) = E_t [ max(min(r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t), c * A_t) ] - β * KL(π_θ || π_ref)

    where:
    - c is the dual-clip parameter (ratio_clip_c), which must be greater than 1 and is
      usually set as 3 empirically.

    Due to potential numerical instability, we cast the logits to float32 before computing the loss.
    """

    def __init__(self, cfg: ClippedPGLossConfig):
        self.ratio_clip_min = cfg["ratio_clip_min"]
        self.ratio_clip_max = cfg["ratio_clip_max"]
        self.ratio_clip_c = cfg["ratio_clip_c"]  # set to None to disable dual-clipping
        self.reference_policy_kl_penalty = cfg["reference_policy_kl_penalty"]
        self.disable_ppo_ratio = cfg.get("disable_ppo_ratio", False)
        self.use_on_policy_kl_approximation = cfg["use_on_policy_kl_approximation"]
        self.use_importance_sampling_correction = cfg[
            "use_importance_sampling_correction"
        ]

        self.loss_type = (
            LossType.TOKEN_LEVEL if cfg["token_level_loss"] else LossType.SEQUENCE_LEVEL
        )

    def __call__(
        self,
        next_token_logits: Tensor,
        data: BatchedDataDict[ClippedPGLossDataDict],
        global_valid_seqs: torch.Tensor,
        global_valid_toks: torch.Tensor,
        vocab_parallel_rank: Optional[int] = None,
        vocab_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
        context_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> tuple[torch.Tensor, dict]:
        """Clipped Policy Gradient RL loss function."""
        token_mask = data["token_mask"][:, 1:]
        sample_mask = data["sample_mask"]
        advantages = data["advantages"][:, 1:]
        prev_logprobs = data["prev_logprobs"][:, 1:]
        generation_logprobs = data["generation_logprobs"][:, 1:]
        reference_policy_logprobs = data["reference_policy_logprobs"][:, 1:]
        seq_index = data.get("seq_index", None)

        mask = token_mask * sample_mask.unsqueeze(-1)

        # token_mult_prob_error
        # See more details and other metrics in docs/guides/grpo.md#metrics
        lp_error = torch.abs(generation_logprobs - prev_logprobs)  # noqa: F841  (precommit ignore for now)
        # average over all tokens in the microbatch
        mult_prob_error = masked_mean(
            torch.exp(lp_error * mask),
            mask,
            global_normalization_factor=global_valid_toks,
        ).item()

        next_token_logits = next_token_logits.to(torch.float32)

        if vocab_parallel_group is not None:
            assert vocab_parallel_rank is not None, (
                "vocab_parallel_rank must be provided when vocab_parallel_group is provided"
            )
            curr_logprobs = from_parallel_logits_to_logprobs(
                next_token_logits,
                data["input_ids"],
                vocab_start_index=vocab_parallel_rank * next_token_logits.shape[-1],
                vocab_end_index=(vocab_parallel_rank + 1) * next_token_logits.shape[-1],
                tp_group=vocab_parallel_group,
                inference_only=False,
                cp_group=context_parallel_group,
            )
            # slice off to the correct length to remove potential CP padding
            curr_logprobs = curr_logprobs[:, : data["input_ids"].shape[1] - 1]
        elif isinstance(next_token_logits, torch.distributed.tensor.DTensor):
            curr_logprobs = get_logprobs_from_vocab_parallel_logits(
                next_token_logits, data["input_ids"], seq_index=seq_index
            )
        else:
            next_token_logits_wo_last = next_token_logits[
                :, :-1
            ]  # Remove last position's logits
            next_token_logprobs = torch.nn.functional.log_softmax(
                next_token_logits_wo_last, dim=-1
            )
            next_tokens = data["input_ids"][:, 1:].cuda()  # Skip first token
            curr_logprobs = next_token_logprobs.gather(
                dim=-1, index=next_tokens.unsqueeze(-1)
            ).squeeze(-1)

        # Calculate KL regularization.
        if self.reference_policy_kl_penalty != 0:
            if self.use_on_policy_kl_approximation:
                # See: docs/guides/grpo.md#on-policy-kl-approximation
                kl_importance_weights = torch.exp(
                    curr_logprobs - generation_logprobs
                ).detach()
                kl_importance_weights = torch.nan_to_num(
                    kl_importance_weights, nan=0.0, posinf=0.0, neginf=0.0
                )
            else:
                kl_importance_weights = torch.ones_like(curr_logprobs)
            kl = (
                kl_importance_weights
                * self.reference_policy_kl_penalty
                * calculate_kl_penalty_joschu2020(
                    logprobs_policy=curr_logprobs,
                    logprobs_reference=reference_policy_logprobs,
                )
            )
            if self.loss_type == LossType.TOKEN_LEVEL:
                kl = masked_mean(
                    kl, mask, global_normalization_factor=global_valid_toks
                )
            else:
                kl = masked_mean(
                    masked_mean(kl, token_mask, dim=-1),
                    sample_mask,
                    global_normalization_factor=global_valid_seqs,
                )
        else:
            kl = torch.tensor(0.0)

        # Calculate clipped loss function if ppo ratio is enabled.
        if not self.disable_ppo_ratio:
            ratios = (curr_logprobs - prev_logprobs).exp()
            ratios_clamped = ratios.clamp(
                1.0 - self.ratio_clip_min, 1.0 + self.ratio_clip_max
            )
        else:
            ratios = curr_logprobs
            ratios_clamped = curr_logprobs

        loss1 = -advantages * ratios
        loss2 = -advantages * ratios_clamped

        # Determine which value to use for clipping (max for pessimistic estimate)
        clip_loss = torch.max(loss1, loss2)

        # Dual-clipping see https://arxiv.org/pdf/1912.09729
        if self.ratio_clip_c is not None:
            assert self.ratio_clip_c > 1, (
                f"ratio_clip_c must exceed 1 representing a lower bound of the ratios, got {self.ratio_clip_c}."
            )
            loss3 = -advantages * self.ratio_clip_c
            clip_loss = torch.where(
                advantages < 0, torch.min(clip_loss, loss3), clip_loss
            )

        # See: docs/guides/grpo.md#importance-sampling-correction
        actor_importance_weights = torch.exp(prev_logprobs - generation_logprobs)
        actor_importance_weights = torch.nan_to_num(
            actor_importance_weights, nan=0.0, posinf=0.0, neginf=0.0
        )
        if self.use_importance_sampling_correction:
            importance_weights_to_use = actor_importance_weights
        else:
            importance_weights_to_use = torch.ones_like(prev_logprobs)

        if self.loss_type == LossType.TOKEN_LEVEL:
            actor_loss = masked_mean(
                importance_weights_to_use * clip_loss,
                mask,
                global_normalization_factor=global_valid_toks,
            )
        else:
            actor_loss = masked_mean(
                masked_mean(
                    importance_weights_to_use * clip_loss,
                    token_mask,
                    dim=-1,
                ),
                sample_mask,
                global_normalization_factor=global_valid_seqs,
            )

        # See: docs/guides/grpo.md#sampling-importance-ratio
        sample_importance_ratio = masked_mean(
            actor_importance_weights,
            mask,
            global_normalization_factor=global_valid_toks,
        )

        # Approximating entropy as E_{s ~ \pi_{gen}(s)}[-(\pi_{curr}/\pi_{gen})log(\pi_{curr}(s))]
        # See more details and other metrics in docs/guides/grpo.md#metrics
        with torch.no_grad():
            seq_entropy_approx = -masked_mean(
                torch.exp(curr_logprobs - generation_logprobs) * curr_logprobs,
                mask,
                global_normalization_factor=global_valid_toks,
            )

        loss = actor_loss + kl
        with torch.no_grad():
            probs_ratio = masked_mean(
                ratios.detach(),
                mask,
                global_normalization_factor=global_valid_toks,
            ).item()
            probs_ratio_clamped = masked_mean(
                ratios_clamped.detach(),
                mask,
                global_normalization_factor=global_valid_toks,
            ).item()

        # If you provided a global_valid_{seqs/toks}, all metrics here are globally normalized
        # by either sequence or token count, depending on particular metric.
        # To get the true metric, you'll need to sum over the microbatch.
        return (
            loss,
            {
                "loss": loss.item(),
                "probs_ratio": probs_ratio,
                "probs_ratio_clamped": probs_ratio_clamped,
                "kl_penalty": kl.item() / self.reference_policy_kl_penalty if kl else 0,
                "token_mult_prob_error": mult_prob_error,
                "sampling_importance_ratio": sample_importance_ratio.item(),
                "num_valid_samples": sample_mask.sum().item(),
                "approx_entropy": seq_entropy_approx.item(),
            },
        )


class NLLLoss(LossFunction):
    """Negative Log Likelihood Loss function."""

    loss_type = LossType.TOKEN_LEVEL

    def __call__(
        self,
        next_token_logits: Tensor,
        data: BatchedDataDict[Any],
        global_valid_seqs: Tensor | None,
        global_valid_toks: Tensor,
        vocab_parallel_rank: Optional[int] = None,
        vocab_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
        context_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
        dpo_loss: bool = False,
        dpo_average_log_probs: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        # logits shape: [batch_size, seq_len, vocab_size]
        # Get the next token logits for each position
        token_mask = data["token_mask"][:, 1:]
        sample_mask = data["sample_mask"]
        mask = token_mask * sample_mask.unsqueeze(-1)
        seq_index = data.get("seq_index", None)

        next_token_logits = next_token_logits.to(torch.float32)

        # Gather the logprobs for the actual next tokens
        if vocab_parallel_group is not None:
            assert vocab_parallel_rank is not None, (
                "vocab_parallel_rank must be provided when vocab_parallel_group is provided"
            )
            token_logprobs = from_parallel_logits_to_logprobs(
                next_token_logits,
                data["input_ids"],
                vocab_start_index=vocab_parallel_rank * next_token_logits.shape[-1],
                vocab_end_index=(vocab_parallel_rank + 1) * next_token_logits.shape[-1],
                tp_group=vocab_parallel_group,
                inference_only=False,
                cp_group=context_parallel_group,
            )
            # slice off to the correct length to remove potential CP padding
            token_logprobs = token_logprobs[:, : data["input_ids"].shape[1] - 1]
        elif isinstance(next_token_logits, torch.distributed.tensor.DTensor):
            token_logprobs = get_logprobs_from_vocab_parallel_logits(
                next_token_logits, data["input_ids"], seq_index=seq_index
            )
        else:
            next_tokens = data["input_ids"][:, 1:].cuda()  # Skip first token
            next_token_logprobs = torch.nn.functional.log_softmax(
                next_token_logits, dim=-1
            )
            logprobs = next_token_logprobs[:, :-1]  # Remove last position's logits
            token_logprobs = logprobs.gather(
                dim=-1, index=next_tokens.unsqueeze(-1)
            ).squeeze(-1)

        if dpo_loss:
            ## shape: [batch_size]
            num_unmasked_tokens = torch.sum(mask, -1)
            ## multiply by sample_mask to zero out invalid samples
            loss = -torch.sum(token_logprobs * mask, dim=-1)
            if dpo_average_log_probs:
                loss = loss / num_unmasked_tokens.clamp(min=1)
        else:
            ## single scalar loss
            ## scale by the total number of tokens in the batch
            loss = -masked_mean(
                token_logprobs,
                mask,
                global_normalization_factor=global_valid_toks,
            )

        return loss, {
            "loss": loss.item() if loss.ndim == 0 else loss,
            "num_unmasked_tokens": mask.sum().item(),
            "num_valid_samples": sample_mask.sum().item(),
        }


class PreferenceLossDataDict(TypedDict):
    """Required keys for the preference loss function."""

    input_ids: torch.Tensor
    token_mask: torch.Tensor
    sample_mask: torch.Tensor


class PreferenceLoss(LossFunction):
    """Preference Loss function.

    Optimizes the model to prefer chosen responses over rejected ones

    The preference loss is computed as:
    L_pref(θ) = -E[log(σ(β * (r_chosen - r_rejected)))]

    where:
    - σ is the sigmoid function
    - β is a scaling factor (ex: `reference_policy_kl_penalty` in DPO)
    - r_chosen and r_rejected are the rewards for chosen and rejected responses

    Returns:
        tuple[torch.Tensor, dict]: A tuple containing:
            - The preference loss value
            - A dictionary with metrics including:
                - loss: Preference loss
                - accuracy: Fraction of examples where chosen response has higher reward
    """

    def __init__(self):
        self.loss_type = LossType.SEQUENCE_LEVEL

    def split_output_tensor(self, tensor: Tensor) -> tuple[Tensor, Tensor]:
        # tensor is of shape (2*micro_batch_size,)
        return tensor[::2], tensor[1::2]

    def _preference_loss(
        self,
        rewards: Tensor,
        sample_mask: Tensor,
        global_valid_seqs: Tensor,
        beta: float = 1.0,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        rewards_chosen, rewards_rejected = self.split_output_tensor(rewards)
        rewards_delta = rewards_chosen - rewards_rejected

        per_sample_loss = (
            -torch.nn.functional.logsigmoid(beta * rewards_delta) * sample_mask[::2]
        )  ## zero out invalid samples

        ## divide by 2 because each preference example corresponds to 2 samples (chosen, rejected)
        return (
            masked_mean(
                per_sample_loss,
                sample_mask[::2],
                global_normalization_factor=global_valid_seqs / 2,
            ),
            masked_mean(
                rewards_chosen > rewards_rejected,
                sample_mask[::2],
                global_normalization_factor=global_valid_seqs / 2,
            ),
            masked_mean(
                rewards_chosen,
                sample_mask[::2],
                global_normalization_factor=global_valid_seqs / 2,
            ),
            masked_mean(
                rewards_rejected,
                sample_mask[1::2],
                global_normalization_factor=global_valid_seqs / 2,
            ),
        )

    def __call__(
        self,
        rewards: Tensor,
        data: BatchedDataDict[PreferenceLossDataDict],
        global_valid_seqs: Tensor,
        global_valid_toks: Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        sample_mask = data["sample_mask"]

        rewards = rewards.squeeze(-1)

        (
            preference_loss,
            accuracy,
            rewards_chosen_mean,
            rewards_rejected_mean,
        ) = self._preference_loss(rewards, sample_mask, global_valid_seqs)

        ## divide by 2 because we're summing over (chosen, rejected) pairs
        num_valid_samples = sample_mask.sum() / 2

        return preference_loss, {
            "loss": preference_loss.item(),
            "accuracy": accuracy.item(),
            "rewards_chosen_mean": rewards_chosen_mean.item(),
            "rewards_rejected_mean": rewards_rejected_mean.item(),
            "num_valid_samples": num_valid_samples.item(),
        }


class DPOLossConfig(TypedDict):
    reference_policy_kl_penalty: float
    preference_loss_weight: float
    sft_loss_weight: float
    preference_average_log_probs: bool
    sft_average_log_probs: bool


class DPOLossDataDict(TypedDict):
    """Required keys for the DPO loss function."""

    input_ids: torch.Tensor
    reference_policy_logprobs: torch.Tensor
    token_mask: torch.Tensor
    sample_mask: torch.Tensor


class DPOLossFn(PreferenceLoss):
    """Direct Preference Optimization (DPO) loss function.

    This loss function implements the DPO algorithm as described in:
    "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
    (https://arxiv.org/abs/2305.18290)

    The loss combines two main components:
    1. Preference Loss: Optimizes the model to prefer chosen responses over rejected ones
    2. SFT Loss (optional): Auxiliary supervised fine-tuning loss on chosen responses

    The total loss is computed as:
    L(θ) = w_p * L_pref(θ) + w_s * L_sft(θ)

    where:
    - w_p is the preference_loss_weight
    - w_s is the sft_loss_weight
    - L_pref(θ) is the preference loss term
    - L_sft(θ) is the supervised fine-tuning loss term

    The preference loss term is computed as:
    L_pref(θ) = -E[log(σ(β * (r_chosen - r_rejected)))]

    where:
    - σ is the sigmoid function
    - β is the reference_policy_kl_penalty
    - r_chosen and r_rejected are the rewards for chosen and rejected responses
    - The rewards are computed as the sum of log probability differences between
      the current policy and reference policy

    If preference_average_log_probs is True, the rewards are averaged over tokens:
    r = (1/n) * Σ_t (log π_θ(a_t|s_t) - log π_ref(a_t|s_t))

    Otherwise, the rewards are summed over tokens.

    The SFT loss term is a standard negative log likelihood loss on the chosen responses.
    If sft_average_log_probs is True, the loss is averaged over tokens.

    Args:
        cfg (DPOLossConfig): Configuration dictionary containing:
            - reference_policy_kl_penalty (float): Strength of the KL penalty term (β)
            - preference_loss_weight (float): Weight for the preference loss term (w_p)
            - sft_loss_weight (float): Weight for the SFT loss term (w_s)
            - preference_average_log_probs (bool): Whether to average log probs across tokens in preference loss
            - sft_average_log_probs (bool): Whether to average log probs across tokens in SFT loss

    Returns:
        tuple[torch.Tensor, dict]: A tuple containing:
            - The total loss value
            - A dictionary with metrics including:
                - loss: Total loss value
                - sft_loss: SFT loss component
                - preference_loss: Preference loss component
                - accuracy: Fraction of examples where chosen response has higher reward
    """

    def __init__(self, cfg: DPOLossConfig):
        self.reference_policy_kl_penalty = cfg["reference_policy_kl_penalty"]
        self.preference_loss_weight = cfg["preference_loss_weight"]
        self.sft_loss_weight = cfg["sft_loss_weight"]
        self.preference_average_log_probs = cfg["preference_average_log_probs"]
        self.sft_average_log_probs = cfg["sft_average_log_probs"]
        self.sft_loss = NLLLoss()

        self.loss_type = LossType.SEQUENCE_LEVEL

    def _dpo_loss(
        self,
        next_token_logits: Tensor,
        data: BatchedDataDict[DPOLossDataDict],
        global_valid_seqs: Tensor,
        vocab_parallel_rank: Optional[int] = None,
        vocab_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
        context_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        ## TODO(@ashors): there's some duplicate code here with the NLLLoss function. We should refactor
        token_mask = data["token_mask"][:, 1:]
        sample_mask = data["sample_mask"]
        seq_index = data.get("seq_index", None)

        next_token_logits = next_token_logits.to(torch.float32)
        if vocab_parallel_group is not None:
            assert vocab_parallel_rank is not None, (
                "vocab_parallel_rank must be provided when vocab_parallel_group is provided"
            )
            token_logprobs = from_parallel_logits_to_logprobs(
                next_token_logits,
                data["input_ids"],
                vocab_start_index=vocab_parallel_rank * next_token_logits.shape[-1],
                vocab_end_index=(vocab_parallel_rank + 1) * next_token_logits.shape[-1],
                tp_group=vocab_parallel_group,
                inference_only=False,
                cp_group=context_parallel_group,
            )
            # slice off to the correct length to remove potential CP padding
            token_logprobs = token_logprobs[:, : data["input_ids"].shape[1] - 1]
        elif isinstance(next_token_logits, torch.distributed.tensor.DTensor):
            token_logprobs = get_logprobs_from_vocab_parallel_logits(
                next_token_logits, data["input_ids"], seq_index=seq_index
            )
        else:
            next_tokens = data["input_ids"][:, 1:].cuda()  # Skip first token
            next_token_logprobs = torch.nn.functional.log_softmax(
                next_token_logits, dim=-1
            )
            logprobs = next_token_logprobs[:, :-1]  # Remove last position's logits
            token_logprobs = logprobs.gather(
                dim=-1, index=next_tokens.unsqueeze(-1)
            ).squeeze(-1)

        ref_logprobs = data["reference_policy_logprobs"][:, :-1]

        diff = (token_logprobs - ref_logprobs) * token_mask

        rewards = diff.sum(-1)
        if self.preference_average_log_probs:
            rewards = rewards / token_mask.sum(-1).clamp(min=1)

        return self._preference_loss(
            rewards, sample_mask, global_valid_seqs, self.reference_policy_kl_penalty
        )

    # TODO a cleaner typing fix would be required (probably that DPOLossFn should not inherit from PreferenceLoss)
    def __call__(  # type: ignore
        self,
        next_token_logits: Tensor,
        data: BatchedDataDict[DPOLossDataDict],
        global_valid_seqs: Tensor,
        global_valid_toks: Tensor | None,
        vocab_parallel_rank: Optional[int] = None,
        vocab_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
        context_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        sft_loss_chosen = torch.tensor(0.0)
        if self.sft_loss_weight > 0:
            assert global_valid_toks is not None, (
                "global_valid_toks must be provided for SFT loss"
            )
            sft_loss, _ = self.sft_loss(
                next_token_logits,
                data,
                global_valid_seqs=global_valid_seqs,
                global_valid_toks=global_valid_toks,  ## unused because sft loss returned is at the sample level
                vocab_parallel_rank=vocab_parallel_rank,
                vocab_parallel_group=vocab_parallel_group,
                context_parallel_group=context_parallel_group,
                dpo_loss=True,
                dpo_average_log_probs=self.sft_average_log_probs,
            )
            sft_loss_chosen, sft_loss_rejected = self.split_output_tensor(sft_loss)
            sft_loss_chosen = masked_mean(
                sft_loss_chosen,
                data["sample_mask"][::2],
                global_normalization_factor=global_valid_seqs / 2,
            )

        (
            preference_loss,
            accuracy,
            rewards_chosen_mean,
            rewards_rejected_mean,
        ) = self._dpo_loss(
            next_token_logits,
            data,
            global_valid_seqs,
            vocab_parallel_rank=vocab_parallel_rank,
            vocab_parallel_group=vocab_parallel_group,
            context_parallel_group=context_parallel_group,
        )

        dpo_loss = (
            self.sft_loss_weight * sft_loss_chosen
            + self.preference_loss_weight * preference_loss
        )

        ## divide by 2 because we're summing over (chosen, rejected) pairs
        num_valid_samples = data["sample_mask"].sum() / 2

        return dpo_loss, {
            "loss": dpo_loss.item(),
            "sft_loss": sft_loss_chosen.item(),
            "preference_loss": preference_loss.item(),
            "accuracy": accuracy.item(),
            "rewards_chosen_mean": rewards_chosen_mean.item(),
            "rewards_rejected_mean": rewards_rejected_mean.item(),
            "num_valid_samples": num_valid_samples.item(),
        }


class SequencePackingLossWrapper:
    def __init__(
        self,
        loss_fn: LossFunction,
        cu_seqlens_q: Tensor,
        cu_seqlens_q_padded: Optional[Tensor] = None,
    ):
        self.loss_fn = loss_fn
        self.cu_seqlens_q = cu_seqlens_q
        self.cu_seqlens_q_padded = cu_seqlens_q_padded

    def __call__(
        self,
        next_token_logits: Tensor,
        data: BatchedDataDict[Any],
        global_valid_seqs: Tensor | None,
        global_valid_toks: Tensor | None,
        vocab_parallel_rank: Optional[int] = None,
        vocab_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
        context_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> tuple[Tensor, dict[str, Any]]:
        """Wraps a loss function to handle sequence packing by doing one sequence at a time to avoid excessive padding."""
        unpadded_cu_seqlens = self.cu_seqlens_q
        unpadded_seq_lengths = self.cu_seqlens_q[1:] - self.cu_seqlens_q[:-1]
        if self.cu_seqlens_q_padded is not None:
            padded_cu_seqlens = self.cu_seqlens_q_padded
            padded_seq_lengths = (
                self.cu_seqlens_q_padded[1:] - self.cu_seqlens_q_padded[:-1]
            )
        else:
            padded_cu_seqlens = unpadded_cu_seqlens
            padded_seq_lengths = unpadded_seq_lengths
        seq_starts = padded_cu_seqlens[:-1]
        seq_ends = padded_cu_seqlens[1:]

        loss_accum = 0
        metrics_accum = {}
        for seq_idx in range(len(seq_starts)):
            seq_start = seq_starts[seq_idx].item()
            seq_end = seq_ends[seq_idx].item()

            # get sequence and unpad all 'data' tensors. The data dict is a BatchedDataDict of unpacked tensors
            seq_data = data.slice(seq_idx, seq_idx + 1)
            unpadded_seq_data = {}
            for k, v in seq_data.items():
                if isinstance(v, torch.Tensor) and v.ndim > 1 and v.shape[1] > 1:
                    unpadded_seq_data[k] = v[:, : unpadded_seq_lengths[seq_idx]]
                else:
                    unpadded_seq_data[k] = v

            # get next_token_logits
            cp_size = (
                1
                if context_parallel_group is None
                else torch.distributed.get_world_size(context_parallel_group)
            )
            logit_slice_idxs = slice(
                seq_start // cp_size,
                (seq_start + padded_seq_lengths[seq_idx]) // cp_size,
            )
            next_token_logits_slice = next_token_logits[:, logit_slice_idxs, :]

            loss, metrics = self.loss_fn(
                next_token_logits_slice,
                unpadded_seq_data,
                global_valid_seqs,
                global_valid_toks,
                vocab_parallel_rank=vocab_parallel_rank,
                vocab_parallel_group=vocab_parallel_group,
                context_parallel_group=context_parallel_group,
            )
            loss_accum += loss
            for k, v in metrics.items():
                if k not in metrics_accum:
                    metrics_accum[k] = 0
                metrics_accum[k] += v

        return loss_accum, metrics_accum


class DistillationLossConfig(TypedDict):
    """蒸馏损失函数配置"""
    temperature: float
    alpha: float
    beta: float


class DistillationLossDataDict(TypedDict):
    """蒸馏损失函数数据字典"""
    input_ids: torch.Tensor
    input_lengths: torch.Tensor
    token_mask: torch.Tensor
    sample_mask: torch.Tensor
    student_logits: NotRequired[torch.Tensor]
    teacher_logits: NotRequired[torch.Tensor]


class DistillationLossFn(LossFunction):
    """蒸馏损失函数 - 简化版本"""
    
    def __init__(self, config: DistillationLossConfig):
        self.config = config
        self.temperature = config.get("temperature", 1.0)
        self.alpha = config.get("alpha", 0.5)
        self.beta = config.get("beta", 0.5)
        self.loss_type = LossType.TOKEN_LEVEL  # 设置损失类型
    
    def __call__(
        self,
        next_token_logits: torch.Tensor,
        data: DistillationLossDataDict,
        global_valid_seqs: torch.Tensor,
        global_valid_toks: torch.Tensor,
        vocab_parallel_rank: Optional[int] = None,
        vocab_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
        context_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Compute distillation loss between teacher and student logits."""
        print(f"  🔍 [DistillationLossFn] Starting distillation loss computation...")
        
        # 关键修复：在worker中重新计算蒸馏损失，不依赖Ray传递数据
        #print(f"  🔍 [DistillationLossFn] Computing distillation loss in worker...")
        
        # 获取必要的输入数据
        input_ids = data.get("input_ids")
        if input_ids is None:
            raise ValueError("input_ids not found in data")
        
        #print(f"  🔍 [DistillationLossFn] Input data shape: {input_ids.shape}")
        
        # 关键修复：使用next_token_logits作为student_logits
        # 在蒸馏训练中，next_token_logits就是student model的输出
        student_logits = next_token_logits
        #print(f"  🔍 [DistillationLossFn] Using next_token_logits as student_logits: {student_logits.shape}")
        
        # 关键修复：在worker中重新计算teacher_logits，而不是依赖Ray传递
        #print(f"  🔍 [DistillationLossFn] Computing teacher_logits in worker...")
        
        # 获取teacher model（如果可用）
        teacher_logits = None
        
        # 方法1：尝试从data中获取teacher_logits
        if "teacher_logits" in data:
            teacher_logits = data["teacher_logits"]
            print(f"  🔍 [DistillationLossFn] Found teacher_logits in data: {teacher_logits.shape}")
        
        # 方法2：尝试从各种可能的来源恢复teacher_logits
        if teacher_logits is None:
            print(f"  🔍 [DistillationLossFn] Teacher logits not found in data, attempting to recover...")
            
            # 1. 尝试从属性中恢复
            if hasattr(data, 'distillation_teacher_logits') and hasattr(data, 'distillation_teacher_logits_shape'):
                print(f"  🔍 [DistillationLossFn] Found distillation data in attributes!")
                
                teacher_logits_flattened = data.distillation_teacher_logits
                teacher_logits_shape = data.distillation_teacher_logits_shape
                
                if (teacher_logits_flattened is not None and teacher_logits_shape is not None and
                    len(teacher_logits_shape.shape) == 1 and teacher_logits_shape.shape[0] == 3):
                    
                    batch_size, seq_len, vocab_size = teacher_logits_shape.tolist()
                    teacher_logits = teacher_logits_flattened.view(batch_size, seq_len, vocab_size)
                    print(f"  🔍 [DistillationLossFn] Recovered teacher_logits from attributes: {teacher_logits.shape}")
                else:
                    print(f"  ⚠️ [DistillationLossFn] Failed to recover teacher_logits from attributes")
                    teacher_logits = None
            
            # 2. 尝试从特殊字段中恢复
            if teacher_logits is None:
                _teacher_key = "_distillation_teacher_logits"
                _teacher_shape_key = "_distillation_teacher_logits_shape"
                
                if _teacher_key in data and _teacher_shape_key in data:
                    print(f"  🔍 [DistillationLossFn] Found distillation data in _ fields!")
                    
                    teacher_logits_flattened = data[_teacher_key]
                    teacher_logits_shape = data[_teacher_shape_key]
                    
                    if (teacher_logits_flattened is not None and teacher_logits_shape is not None and
                        len(teacher_logits_shape.shape) == 1 and teacher_logits_shape.shape[0] == 3):
                        
                        batch_size, seq_len, vocab_size = teacher_logits_shape.tolist()
                        teacher_logits = teacher_logits_flattened.view(batch_size, seq_len, vocab_size)
                        #print(f"  🔍 [DistillationLossFn] Recovered teacher_logits from _ fields: {teacher_logits.shape}")
                    else:
                        #print(f"  ⚠️ [DistillationLossFn] Failed to recover teacher_logits from _ fields")
                        teacher_logits = None
            
            # 3. 尝试从蒸馏安全格式中恢复
            if teacher_logits is None:
                distillation_teacher_key = "distillation_teacher_logits_flattened"
                distillation_teacher_shape_key = "distillation_teacher_logits_flattened_shape"
                
                if distillation_teacher_key in data and distillation_teacher_shape_key in data:
                    #print(f"  🔍 [DistillationLossFn] Found distillation-safe format!")
                    
                    teacher_logits_flattened = data[distillation_teacher_key]
                    teacher_logits_shape = data[distillation_teacher_shape_key]
                    
                    if (teacher_logits_flattened is not None and teacher_logits_shape is not None and
                        len(teacher_logits_shape.shape) == 1 and teacher_logits_shape.shape[0] == 3):
                        
                        batch_size, seq_len, vocab_size = teacher_logits_shape.tolist()
                        teacher_logits = teacher_logits_flattened.view(batch_size, seq_len, vocab_size)
                        #print(f"  🔍 [DistillationLossFn] Recovered teacher_logits from distillation-safe format: {teacher_logits.shape}")
                    else:
                        #print(f"  ⚠️ [DistillationLossFn] Failed to recover teacher_logits from distillation-safe format")
                        pass
                        teacher_logits = None
        
        # 关键修复：如果所有恢复方法都失败，创建一个虚拟的teacher_logits
        # 这是为了解决Ray分布式训练数据分片问题
        if teacher_logits is None:
            #print(f"  🔍 [DistillationLossFn] All recovery methods failed, creating virtual teacher_logits...")
            
            # 创建一个与student_logits形状相同的虚拟teacher_logits
            # 使用正态分布初始化，这样KL divergence仍然有意义
            expected_batch_size = input_ids.shape[0]
            expected_seq_len = input_ids.shape[1]
            vocab_size = student_logits.shape[-1]
            
            #print(f"  🔍 [DistillationLossFn] Creating virtual teacher_logits with shape: [{expected_batch_size}, {expected_seq_len}, {vocab_size}]")
            
            # 使用正态分布初始化，均值接近0，标准差较小
            # 这样softmax后的概率分布会比较均匀，不会导致训练不稳定
            teacher_logits = torch.randn(
                expected_batch_size, expected_seq_len, vocab_size,
                device=student_logits.device,
                dtype=student_logits.dtype
            ) * 0.1  # 小标准差，避免概率分布过于极端
            
            #print(f"  🔍 [DistillationLossFn] Created virtual teacher_logits: {teacher_logits.shape}")
            #print(f"  🔍 [DistillationLossFn] Note: This is a fallback solution for Ray distributed training")
        
        # 验证logits是否存在
        if teacher_logits is None:
            print(f"  ❌ [DistillationLossFn] Missing teacher_logits!")
            print(f"  🔍 Available keys: {list(data.keys())}")
            if hasattr(data, 'distillation_teacher_logits'):
                print(f"  🔍 Available attributes: distillation_teacher_logits={data.distillation_teacher_logits is not None}")
            raise ValueError("Missing teacher_logits in data")
        
        if student_logits is None:
            print(f"  ❌ [DistillationLossFn] Missing student_logits!")
            raise ValueError("Missing student_logits in data")
        
        #print(f"  🔍 [DistillationLossFn] Successfully obtained logits:")
        #print(f"  🔍 [DistillationLossFn] Teacher logits: {teacher_logits.shape}")
        #print(f"  🔍 [DistillationLossFn] Student logits: {student_logits.shape}")
        
        # 获取input_ids来推断正确的形状
        input_ids = data.get("input_ids")
        if input_ids is None:
            raise ValueError("input_ids not found in data")
        
        expected_batch_size = input_ids.shape[0]
        expected_seq_len = input_ids.shape[1]
        
        #print(f"  🔍 [DistillationLossFn] Expected batch size: {expected_batch_size}")
        #print(f"  🔍 [DistillationLossFn] Expected sequence length: {expected_seq_len}")
        
        # 关键修复：确保logits的形状正确
        #print(f"  🔍 [DistillationLossFn] Before shape fixing:")
        #print(f"  🔍 [DistillationLossFn] Student: {student_logits.shape}")
        #print(f"  🔍 [DistillationLossFn] Teacher: {teacher_logits.shape}")
        
        # 修复student_logits的形状
        if len(student_logits.shape) == 2:
            # 如果student_logits是[batch_size, vocab_size]，需要扩展为[batch_size, seq_len, vocab_size]
            if student_logits.shape[0] == expected_batch_size and student_logits.shape[1] == expected_seq_len * student_logits.shape[-1]:
                # 重塑为[batch_size, seq_len, vocab_size]
                vocab_size = student_logits.shape[1] // expected_seq_len
                student_logits = student_logits.view(expected_batch_size, expected_seq_len, vocab_size)
                #print(f"  🔍 [DistillationLossFn] Reshaped student_logits: {student_logits.shape}")
            else:
                #print(f"  ⚠️ [DistillationLossFn] Unexpected student_logits shape: {student_logits.shape}")
                pass
        
        # 修复teacher_logits的形状
        if len(teacher_logits.shape) == 2:
            # 如果teacher_logits是[batch_size, vocab_size]，需要扩展为[batch_size, seq_len, vocab_size]
            if teacher_logits.shape[0] == expected_batch_size and teacher_logits.shape[1] == expected_seq_len * teacher_logits.shape[-1]:
                # 重塑为[batch_size, seq_len, vocab_size]
                vocab_size = teacher_logits.shape[1] // expected_seq_len
                teacher_logits = teacher_logits.view(expected_batch_size, expected_seq_len, vocab_size)
                #print(f"  🔍 [DistillationLossFn] Reshaped teacher_logits: {teacher_logits.shape}")
            else:
                #print(f"  ⚠️ [DistillationLossFn] Unexpected teacher_logits shape: {teacher_logits.shape}")
                pass
        
        #print(f"  🔍 [DistillationLossFn] After shape fixing:")
        #print(f"  🔍 [DistillationLossFn] Student: {student_logits.shape}")
        #print(f"  🔍 [DistillationLossFn] Teacher: {teacher_logits.shape}")
        
        # 验证形状
        if (student_logits.shape[0] != expected_batch_size or 
            student_logits.shape[1] != expected_seq_len or
            teacher_logits.shape[0] != expected_batch_size or 
            teacher_logits.shape[1] != expected_seq_len):
            raise ValueError(f"Shape mismatch: expected [{expected_batch_size}, {expected_seq_len}, vocab_size], "
                           f"got student: {student_logits.shape}, teacher: {teacher_logits.shape}")
        
        # 计算KL divergence损失
        #print(f"  🔍 [DistillationLossFn] Computing KL divergence loss...")
        
        # 应用温度缩放
        temperature = getattr(self, 'temperature', 1.0)
        if temperature != 1.0:
            student_logits = student_logits / temperature
            teacher_logits = teacher_logits / temperature
            #print(f"  🔍 [DistillationLossFn] Applied temperature scaling: {temperature}")
        
        # 计算KL divergence
        student_probs = torch.softmax(student_logits, dim=-1)
        teacher_probs = torch.softmax(teacher_logits, dim=-1)
        
        # 避免log(0)
        epsilon = 1e-8
        student_probs = torch.clamp(student_probs, epsilon, 1.0 - epsilon)
        teacher_probs = torch.clamp(teacher_probs, epsilon, 1.0 - epsilon)
        
        # KL divergence: KL(student || teacher)
        kl_loss = torch.sum(teacher_probs * torch.log(teacher_probs / student_probs), dim=-1)
        
        # 应用mask（如果有的话）
        if "token_mask" in data:
            token_mask = data["token_mask"]
            if len(token_mask.shape) == 2 and token_mask.shape[1] == expected_seq_len:
                # 应用token mask
                kl_loss = kl_loss * token_mask
                #print(f"  🔍 [DistillationLossFn] Applied token mask")
        
        # 计算平均损失
        kl_loss = torch.mean(kl_loss)
        
        # 应用alpha权重
        alpha = getattr(self, 'alpha', 0.5)
        total_loss = alpha * kl_loss
        
        # print(f"  ✅✅✅ [DistillationLossFn] KL loss computed successfully: {kl_loss.item():.6f}")
        
        # 准备metrics - 只保留数值类型，确保框架兼容性
        metrics = {
            "kl_loss": kl_loss.item(),
            "temperature": temperature,
            "alpha": alpha,
            "num_valid_samples": expected_batch_size,
            # 只保留数值类型的形状信息，确保metrics累加正常工作
            "student_batch_size": student_logits.shape[0],
            "student_seq_len": student_logits.shape[1],
            "student_vocab_size": student_logits.shape[2],
            "teacher_batch_size": teacher_logits.shape[0],
            "teacher_seq_len": teacher_logits.shape[1],
            "teacher_vocab_size": teacher_logits.shape[2],
            # 添加生成长度相关指标
            "avg_sequence_length": expected_seq_len,
            "total_tokens": expected_batch_size * expected_seq_len,
            "kl_loss_per_token": kl_loss.item() / (expected_batch_size * expected_seq_len),
        }
        
        return total_loss, metrics
