"""This file contains code for text-to-image HART Transformer.

This file is adopted and modified from https://github.com/FoundationVision/VAR/blob/main/models/var.py
"""

import math
import os
from functools import partial
from typing import Optional, Tuple, Union
from copy import deepcopy
import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import time
from transformers import AutoConfig, AutoModel, PreTrainedModel
from copy import deepcopy
from hart.modules.diffusion.diffloss import DiffLoss
from hart.modules.models.autoencoder import (
    HARTAutoEncoder,
    HARTAutoEncoderWithDisc,
    HARTHybridQuantizer,
)
import concurrent.futures
from hart.modules.models.transformer.configuration import HARTForT2IConfig
from hart.modules.networks.basic_hart import (
    AdaLNBeforeHead,
    AdaLNSelfAttn,
    LlamaRMSNormFused,
    TimestepEmbedder,
)
from hart.modules.networks.utils import (
    gumbel_softmax_with_rng,
    sample_with_top_k_top_p_,
)
from hart.utils import get_device
from typing import Callable, Tuple


import torch.nn.functional as F

def select_and_set_zero(tokens,hr_idx):
    tokens_=torch.zeros_like(tokens)
    hr_idx_expanded = hr_idx.unsqueeze(-1).repeat(tokens.shape[0]//hr_idx.shape[0], 1, tokens.shape[-1])
    gathered_values = torch.gather(tokens, 1, index=hr_idx_expanded)
    tokens_.scatter_(1, hr_idx_expanded, gathered_values)
    return tokens_

def calculate_mse(input_tensor, output_tensor):
    if input_tensor.shape != output_tensor.shape:
        raise ValueError("Input and output tensors must have the same shape.")
    
    mse_per_position = ((input_tensor - output_tensor) ** 2).mean(dim=-1)  # Shape: B x N
    return mse_per_position

def generate_anchors_from_flat(N, n):
    # Generate coordinates every n steps
    indices = torch.arange(0, N, n)

    # Ensure the last position (N-1) is included if not divisible
    if n % N != 0:
        indices = torch.cat([indices, torch.tensor([N - 1]).to(indices)])

    # Create row/col meshgrid
    grid_x, grid_y = torch.meshgrid(indices, indices)

    # Flatten into (num_anchors, 2)
    anchors = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)

    return anchors

def mask_by_order(mask_len, order, bsz, seq_len):
    masking = torch.zeros(bsz, seq_len).cuda()
    masking = torch.scatter(
        masking,
        dim=-1,
        index=order[:, : mask_len.long()],
        src=torch.ones(bsz, seq_len).cuda(),
    ).bool()
    return masking


class CopyableGenerator(torch.Generator):
    def __deepcopy__(self, memo):
        new_generator = CopyableGenerator(device=self.device)
        new_generator.set_state(self.get_state())
        return new_generator


class SharedAdaLin(nn.Linear):
    def forward(self, cond_BD):
        C = self.weight.shape[0] // 6
        return super().forward(cond_BD).view(-1, 1, 6, C)  # B16C

def get_reverse_indices(index, L):
    B = index.size(0)
    all_indices = torch.arange(L).unsqueeze(0).expand(B, -1).to(index.device)

    # Mask out the given indices
    mask = torch.ones(B, L, dtype=torch.bool, device=index.device)
    mask.scatter_(1, index, False)

    # Select remaining indices
    remaining_indices = all_indices[mask].view(B, -1)

    return remaining_indices

def map_to_best_corner(N, n, pos, matrix, anchor_indices):
    # Convert flattened positions to row/col indices
    i = pos // N
    j = pos % N
    batch_idx = torch.arange(matrix.shape[0]).unsqueeze(1).expand(-1, pos.shape[1]).to(matrix.device)

    # Gather feature vectors at query positions
    features_at_pos = matrix[batch_idx, i, j, :]  # (B, T, C)

    # Compute squared distance from each pos to candidate anchors
    dis = (i[:, None, :] - anchor_indices[..., None, 0].to(matrix.device)) ** 2 + \
          (j[:, None, :] - anchor_indices[..., None, 1].to(matrix.device)) ** 2

    # Pick 4 nearest anchors for each position
    closest_pos = dis.sort(1)[1][:, :4]
    anchor_positions = anchor_indices.unsqueeze(1).expand(-1, N ** 2, -1, -1) \
        .gather(index=closest_pos.permute(0, 2, 1).unsqueeze(-1).expand(-1, -1, -1, 2), dim=1)

    # Gather anchor features
    B_idx = torch.arange(anchor_positions.shape[0]).unsqueeze(1).unsqueeze(1).expand(
        -1, anchor_positions.shape[1], 4
    ).to(anchor_positions.device)
    corner_features = matrix[B_idx, anchor_positions[..., 0], anchor_positions[..., 1], :]  # (B, T, 4, C)

    # Cosine similarity between pos features and corner features
    similarities = F.cosine_similarity(features_at_pos.unsqueeze(2), corner_features, dim=-1)  # (B, T, 4)

    # Pick best anchor per position
    best_indices = torch.argmax(similarities, dim=-1)  # (B, T)
    best_pos = anchor_positions.gather(
        dim=2,
        index=best_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 2)
    )[:, :, 0]

    # Convert (row, col) back to flat index
    best_pos = best_pos[..., 0] * N + best_pos[..., 1]

    return best_pos, similarities


class HARTForT2I(PreTrainedModel):
    config_class = HARTForT2IConfig
    _no_split_modules = ["AdaLNSelfAttn", "HARTAutoEncoderWithDisc", "AdaLNBeforeHead"]

    def __init__(self, config: HARTForT2IConfig):
        super().__init__(config)
        self.supports_gradient_checkpointing = True
        # 0. hyperparameters

        embed_dim = config.embed_dim
        num_heads = config.num_heads
        vae_path = config.vae_path
        if vae_path is None:
            vae_path = os.path.join(
                os.path.dirname(config._name_or_path.rstrip("/")), "tokenizer"
            )
        depth = config.depth
        drop_rate = config.drop_rate
        cond_drop_rate = config.cond_drop_rate
        drop_path_rate = config.drop_path_rate
        attn_drop_rate = config.attn_drop_rate
        mlp_ratio = config.mlp_ratio
        norm_eps = config.norm_eps
        shared_aln = config.shared_aln
        attn_l2_norm = config.attn_l2_norm
        context_token = config.context_token
        context_dim = config.context_dim
        patch_nums = config.patch_nums
        flash_if_available, fused_if_available = (
            config.flash_if_available,
            config.fused_if_available,
        )
        self.mlp_type = mlp_type = config.mlp_type
        self.attn_type = attn_type = config.attn_type
        if self.attn_type == "gpt2":
            norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        elif self.attn_type == "llama":
            norm_layer = partial(LlamaRMSNormFused, eps=norm_eps)
        else:
            raise NotImplementedError
        self.disable_aln = config.disable_aln
        self.use_timestep_embed = use_timestep_embed = config.use_timestep_embed
        self.sep_aln_pooling_mode = sep_aln_pooling_mode = config.sep_aln_pooling_mode
        self.use_cross_attn = use_cross_attn = config.use_cross_attn

        self.diffusion_head_repeats = diffusion_head_repeats = (
            config.diffusion_head_repeats
        )
        # MAR variant masking ratio, a left-half truncated Gaussian centered at 100% masking ratio with std 0.25
        mask_ratio_min = 0.5
        self.mask_ratio_generator = stats.truncnorm(
            (mask_ratio_min - 1.0) / 0.25, 0, loc=1.0, scale=0.25
        )

        vae_local = HARTAutoEncoderWithDisc.from_pretrained(vae_path,device_map="cpu").vae
        vae_local = vae_local.cuda()
        vae_local.requires_grad_(False)

        assert embed_dim % num_heads == 0
        self.Cvae, self.V = vae_local.Cvae, vae_local.vocab_size
        self.depth, self.C, self.D, self.num_heads = (
            depth,
            embed_dim,
            embed_dim,
            num_heads,
        )

        self.cond_drop_rate = cond_drop_rate
        self.prog_si = -1  # progressive training

        self.patch_nums: Tuple[int] = tuple(patch_nums)
        self.L = sum(pn**2 for pn in self.patch_nums)
        # self.first_l = self.patch_nums[0] ** 2
        self.first_l = context_token
        self.begin_ends = []
        self.begin_ends.append((0, context_token))
        cur = context_token
        for i, pn in enumerate(self.patch_nums[1:]):
            self.begin_ends.append((cur, cur + pn**2))
            cur += pn**2

        self.num_stages_minus_1 = len(self.patch_nums) - 1
        self.rng = CopyableGenerator(device=get_device())

        # 1. input (word) embedding
        quant: HARTHybridQuantizer = vae_local.quantize
        self.vae_proxy = nn.Sequential(vae_local)
        additional_codebooks=[]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        additional_codebooks = []

        for i in range(len(self.patch_nums)):
            # 创建嵌入层，初始时不使用 .cuda()
            embed = nn.Embedding(num_embeddings=4096, embedding_dim=32)

            # 初始化嵌入层的权重
            torch.nn.init.zeros_(embed.weight.data)

            # 将嵌入层添加到列表中
            additional_codebooks.append(embed)
        self.additional_codebooks=nn.ModuleList(additional_codebooks)
        self.vae_quant_proxy= nn.Sequential(quant)
        self.word_embed = nn.Linear(self.Cvae, self.C)

        # 2. class embedding
        init_std = math.sqrt(1 / self.C / 3)

        self.context_token = context_token
        self.context_dim = context_dim
        self.context_shape = (context_token, context_dim)
        self.context_embed = nn.Linear(context_dim, self.D)
        if config.use_context_norm:
            self.context_norm = norm_layer(context_dim, scale=config.context_norm_scale)
        else:
            self.context_norm = nn.Identity()

        nn.init.trunc_normal_(self.context_embed.weight.data, mean=0, std=init_std)
        if attn_type == "gpt2" or self.context_token == 0:
            # gpt2 uses absolute pos emb for context tokens
            # c2i also adds this absolute pos emb
            self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C))
            nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)
        else:
            self.pos_start = None

        # 3. absolute position embedding
        self.last_level_pns = self.patch_nums[-1] ** 2
        pos_1LC = []
        for i, pn in enumerate(self.patch_nums):
            if i > 0:
                pe = torch.empty(1, pn * pn, self.C)
            else:
                pe = torch.empty(1, context_token, self.C)
            nn.init.trunc_normal_(pe, mean=0, std=init_std)
            pos_1LC.append(pe)
        pos_1LC = torch.cat(pos_1LC, dim=1)  # 1, L, C
        assert tuple(pos_1LC.shape) == (1, self.L + context_token - 1, self.C)
        if self.attn_type == "gpt2":
            self.pos_1LC = nn.Parameter(pos_1LC)
        elif self.attn_type == "llama":
            self.pos_1LC = None
        else:
            raise NotImplementedError

        if not self.use_timestep_embed:
            # level embedding (similar to GPT's segment embedding, used to distinguish different levels of token pyramid)
            self.lvl_embed = nn.Embedding(len(self.patch_nums), self.C)
            nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)
        else:
            self.lvl_embed = TimestepEmbedder(embed_dim)

        # 4. backbone blocks
        self.shared_ada_lin = nn.Identity()
        self.fix_compress_ratio=None

        self.drop_path_rate = drop_path_rate
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule (linearly increasing)
        self.blocks = nn.ModuleList(
            [
                AdaLNSelfAttn(
                    cond_dim=self.D,
                    shared_aln=shared_aln,
                    block_idx=block_idx,
                    embed_dim=self.C,
                    norm_layer=norm_layer,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[block_idx],
                    last_drop_p=0 if block_idx == 0 else dpr[block_idx - 1],
                    attn_l2_norm=attn_l2_norm,
                    flash_if_available=flash_if_available,
                    fused_if_available=fused_if_available,
                    mlp_type=mlp_type,
                    attn_type=attn_type,
                    max_position_embeddings=2
                    ** int(math.ceil(math.log2(self.L + context_token - 1))),
                    patch_nums=self.patch_nums,
                    context_token=self.context_token,
                    disable_aln=self.disable_aln,
                    sep_aln_pooling_mode=self.sep_aln_pooling_mode,
                    use_cross_attn=self.use_cross_attn,
                )
                for block_idx in range(depth)
            ]
        )

        fused_add_norm_fns = [b.fused_add_norm_fn is not None for b in self.blocks]
        self.using_fused_add_norm_fn = any(fused_add_norm_fns)

        # 5. attention mask used in training (for masking out the future)
        #    it won't be used in inference, since kv cache is enabled
        d: torch.Tensor = torch.cat(
            [torch.full((context_token,), 0)]
            + [
                torch.full((pn * pn,), i + 1)
                for i, pn in enumerate(self.patch_nums[1:])
            ]
        ).view(1, self.L + context_token - 1, 1)
        dT = d.transpose(1, 2)  # dT: 11L
        lvl_1L = dT[:, 0].contiguous()
        self.register_buffer("lvl_1L", lvl_1L)
        attn_bias_for_masking = torch.where(d >= dT, 0.0, -torch.inf).reshape(
            1, 1, self.L + context_token - 1, self.L + context_token - 1
        )
        self.register_parameter(
            "attn_bias_for_masking", nn.Parameter(attn_bias_for_masking.contiguous(), requires_grad=False)
        )
        print(attn_bias_for_masking.shape)

        # 6. classifier head
        self.head_nm = AdaLNBeforeHead(self.C, self.D, norm_layer=norm_layer)
        self.head = nn.Linear(self.C, self.V)

        self.decoder_norm = norm_layer(self.C)

        self.diffloss = DiffLoss(
            target_channels=self.Cvae,
            z_channels=self.C,
            width=config.diff_width,
            depth=config.diff_depth,
            num_sampling_steps=config.num_sampling_steps,
            sampler=config.sampler,
        )
        self.diffloss_low = DiffLoss(
                target_channels=self.Cvae,
                z_channels=self.C,
                width=self.config.diff_width,
                depth=self.config.diff_depth,
                num_sampling_steps=self.config.num_sampling_steps,
                sampler=self.config.sampler,
            )
        self.diffusion_batch_mul = config.diffusion_batch_mul

    def init_diffloss_low(self):
        self.diffloss_low.load_state_dict(self.diffloss.state_dict())

    def get_logits(
        self,
        h_or_h_and_residual: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        cond_BD: Optional[torch.Tensor],
    ):
        if not isinstance(h_or_h_and_residual, torch.Tensor):
            h, resi = h_or_h_and_residual  # fused_add_norm must be used
            h = resi + self.blocks[-1].drop_path(h)
        else:  # fused_add_norm is not used
            h = h_or_h_and_residual
        return self.head(self.head_nm(h.half(), cond_BD).half()).half()

    def forward_diff_loss(self, z, target, mask=None, low_frequency=False):
        bs, seq_len, _ = target.shape
        target = target.reshape(bs * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        z = z.reshape(bs * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        if low_frequency:
            loss = self.diffloss_low(z=z, target=target, mask=mask)
        else:
            loss = self.diffloss(z=z, target=target, mask=mask)
        return loss

    def lf_anchor(
        self,
        logits_BlV,
        si,
        mse_difference,
        compress_ratio_,
        local_window_size,
    ):
        B = logits_BlV.shape[0]
        patch_size = self.patch_nums[si]

        # Upsample logits to current resolution
        upsampled_logits = torch.nn.functional.interpolate(
            logits_BlV.permute(0, 2, 1).reshape(B, 4096, self.patch_nums[si - 1], self.patch_nums[si - 1]),
            size=(patch_size, patch_size),
            mode='bilinear'
        )

        # Upsample ΔF (MSE difference) to current resolution
        upsampled_mse = torch.nn.functional.interpolate(
            mse_difference.reshape(B, 1, self.patch_nums[si - 1], self.patch_nums[si - 1]),
            size=(patch_size, patch_size),
            mode='bilinear'
        )

        # Step 1: Select high/low-frequency tokens
        metric = upsampled_mse
        high_freq_idx = metric.reshape(B, -1).sort(1, descending=True)[1][:, :int(patch_size * patch_size * compress_ratio_)]
        low_freq_idx = get_reverse_indices(high_freq_idx, patch_size ** 2)

        # Step 2: Generate anchors from local windows
        anchor_candidates = generate_anchors_from_flat(patch_size, local_window_size).to(logits_BlV.device)
        anchor_candidates = anchor_candidates.unsqueeze(0).repeat(B, 1, 1)

        # Step 3: Match tokens to best anchors
        best_anchor_map, anchor_similarities = map_to_best_corner(
            patch_size,
            local_window_size,
            torch.arange(0, patch_size ** 2)[None, :].repeat(B, 1).to(upsampled_logits.device),
            upsampled_logits.permute(0, 2, 3, 1),
            anchor_candidates
        )
        anchor_candidates = anchor_candidates[..., 0] * patch_size + anchor_candidates[..., 1]

        # Step 4: Merge anchors with high-frequency tokens (batch alignment)
        if B > 1:
            combined_idx = torch.cat([anchor_candidates, high_freq_idx], dim=1)
            min_len = min([i.unique().shape[0] for i in combined_idx])
            high_freq_idx = torch.stack([i.unique(sorted=False)[:min_len] for i in combined_idx])
        else:
            high_freq_idx=torch.cat([high_freq_idx,anchor_candidates],dim=1)

        # Step 5: Update low-frequency set
        low_freq_idx = get_reverse_indices(high_freq_idx, patch_size ** 2)

        # Step 6: Decide recoverability of low-frequency tokens
        max_sim = anchor_similarities.max(-1)[0].gather(index=low_freq_idx, dim=1)
        recoverable_mask = max_sim > 0.9  # fixed threshold for HART
        low_to_anchor_idx = best_anchor_map.gather(dim=1, index=low_freq_idx)

        return high_freq_idx, low_freq_idx, recoverable_mask, low_to_anchor_idx


    @torch.no_grad()
    def autoregressive_infer_cfg(
        self,
        B: int,
        label_B: Optional[Union[int, torch.LongTensor]],
        g_seed: Optional[int] = None,
        cfg=1.5,
        top_k=0,
        top_p=0.0,
        more_smooth=False,
        context_position_ids: torch.Tensor = None,
        context_mask: torch.Tensor = None,
        final_stage=0,
        num_maskgit_iters=1,
        start_prune_stage=9,
        compress_method="none",
        compress_ratio=0.9,
        local_window_size=4,
        specific_mse_layer=16,
    ) -> torch.Tensor:  # returns reconstructed image (B, 3, H, W) in [0, 1]
        """
        only used for inference, on autoregressive mode
        :param B: batch size
        :param label_B: imagenet label; if None, randomly sampled
        :param g_seed: random seed
        :param cfg: classifier-free guidance ratio
        :param top_k: top-k sampling
        :param top_p: top-p sampling
        :param more_smooth: smoothing the pred using gumbel softmax; only used in visualization, not used in FID/IS benchmarking
        :return: if returns_vemb: list of embedding h_BChw := vae_embed(idx_Bl), else: list of idx_Bl
        """
        # num_maskgit_iters = 1
        # final_stage = 2
        if g_seed is None:
            rng = None
        else:
            self.rng.manual_seed(g_seed)
            rng = self.rng
        assert label_B is not None
        assert label_B.shape[1] == self.context_token

        sos = cond_BD = self.context_embed(
            self.context_norm(
                torch.cat((label_B, torch.full_like(label_B, fill_value=0.0)), dim=0)
            )
        )
        # Haotian: need to handle CFG here so we replicate context position ids
        context_position_ids = torch.cat(
            (context_position_ids, torch.full_like(context_position_ids, fill_value=0)),
            dim=0,
        )

        b = context_mask.shape[0]
        context_mask = torch.cat(
            (context_mask, torch.full_like(context_mask, fill_value=0))
        )
        context_mask[b:, 0] = 1

        if self.pos_1LC is not None:
            lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC
        else:
            lvl_pos = self.lvl_embed(self.lvl_1L)

        if self.pos_start is not None:
            next_token_map = (
                sos.expand(2 * B, self.first_l, -1)
                + self.pos_start.expand(2 * B, self.first_l, -1)
                + lvl_pos[:, : self.first_l]
            )
        else:
            next_token_map = (
                sos.expand(2 * B, self.first_l, -1) + lvl_pos[:, : self.first_l]
            )

        cur_L = 0
        f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])
        
        hr_idx=None
        mse_difference=None

        for b in self.blocks:
            b.attn.kv_caching(True)

        for si, pn in enumerate(self.patch_nums[:-1]):  # si: i-th segment
            ratio = si / self.num_stages_minus_1
            # last_L = cur_L
            if si > 0:
                cur_L += pn * pn
            else:
                cur_L += self.context_token
            # assert self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].sum() == 0, f'AR with {(self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L] != 0).sum()} / {self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].numel()} mask item'
            cond_BD_or_gss = self.shared_ada_lin(cond_BD)
            x = next_token_map
            AdaLNSelfAttn.forward
            if si > start_prune_stage and compress_method == "sparsevar":
                # Step 1: Select high- and low-frequency tokens using anchor assignment
                hr_idx, lr_idx, recover_lr, anchor_lr = self.lf_anchor(
                    logits_BlV, si, mse_difference, compress_ratio_, local_window_size
                )
                hr_idx = hr_idx.reshape(B, -1)
                hr_idx_repeat = hr_idx.repeat(2, 1)

                # Step 2: Build mask for selected tokens (for maskgit conditioning)
                mask_to_pred = torch.ones(B, pn * pn).cuda()
                cur_mask_ = torch.cat([
                    torch.gather(mask_to_pred, dim=1, index=hr_idx),
                    torch.gather(mask_to_pred, dim=1, index=hr_idx)
                ], dim=0)
                cur_mask_ = cur_mask_.nonzero(as_tuple=True)
                cur_mask = (cur_mask_[0], hr_idx_repeat.reshape(-1))

                # Step 3: Gather only high-frequency tokens as input
                x_origin = x
                hr_idx_expand = hr_idx_repeat[..., None].expand(-1, -1, x.shape[-1])
                x = torch.gather(x, dim=1, index=hr_idx_expand)
            else:
                cur_mask = None

            for block_idx, b in enumerate(self.blocks):
                origin_x = x

                # Forward pass through block, with optional masked conditioning
                x, _ = b(
                    x=x,
                    cond_BD=cond_BD_or_gss,
                    attn_bias=None,
                    si=si,
                    context_position_ids=context_position_ids,
                    context_mask=context_mask,
                    m_maskgit=cur_mask
                )

                # Step 4: Compute ΔF at a specific layer to guide pruning
                if si >= start_prune_stage:
                    if block_idx == specific_mse_layer and compress_method == "sparsevar":
                        if hr_idx is not None:
                            x_after = x_origin
                            x_after = x_after.scatter(1, hr_idx.unsqueeze(-1).repeat(2, 1, x.shape[-1]), x)

                            x_before = x_origin
                            x_before = x_before.scatter(1, hr_idx.unsqueeze(-1).repeat(2, 1, x.shape[-1]), origin_x)
                        else:
                            x_after = x
                            x_before = origin_x

                        mse_difference = calculate_mse(x_after, x_before)
                        mse_difference = torch.stack([mse_difference[:B], mse_difference[B:]]).mean(0)

                        compress_ratio_ = ((mse_difference / mse_difference.max()) > compress_ratio).float().mean(1).max()

            # Step 5: Recover low-frequency tokens via anchors and merge back into sequence
            if si > start_prune_stage and compress_method == "sparsevar":
                x_origin.scatter_(1, hr_idx_expand, x)

                source_flag = torch.gather(x_origin, dim=1, index=anchor_lr.unsqueeze(-1).repeat(2, 1, x.shape[-1]))
                dst = torch.gather(x_origin, dim=1, index=lr_idx.unsqueeze(-1).repeat(2, 1, x.shape[-1]))

                # Mix recovered tokens with anchors based on recoverability mask
                source_flag = dst * (1 - recover_lr.to(dst).repeat(2, 1).unsqueeze(-1)) + \
                              (recover_lr.to(dst).repeat(2, 1).unsqueeze(-1) * source_flag)

                # Scatter recovered tokens back into low-frequency positions
                x_origin.scatter_(1, lr_idx.unsqueeze(-1).repeat(2, 1, x.shape[-1]), source_flag)
                x = x_origin


            logits_BlV = self.get_logits(x, cond_BD)

            if si == self.num_stages_minus_1:
                last_layer_cond = x

            t = cfg * ratio
            logits_BlV = (1 + t) * logits_BlV[:B] - t * logits_BlV[B:]

            # Haotian: Added for text-conditioned generation
            if si == 0:
                logits_BlV = logits_BlV[:, [-1], :]
                
            idx_Bl = sample_with_top_k_top_p_(
                logits_BlV.detach().clone(),
                rng=rng,
                top_k=(600 if si < 7 else 300),
                top_p=top_p,
                num_samples=1,
            )[:, :, 0]
            if not more_smooth:  # this is the default case
                h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)  # B, l, Cvae
            else:  # not used when evaluating FID/IS/Precision/Recall
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)  # refer to mask-git
                h_BChw = gumbel_softmax_with_rng(
                    logits_BlV.detach().clone().mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng
                ) @ self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)

            if si > start_prune_stage and compress_method == "sparsevar":
                # Build a temporary mask for low-frequency tokens
                index_temp = torch.zeros(B, pn**2, device=lr_idx.device)

                # Mark low-frequency tokens that are NOT recoverable (recover_lr == False)
                index_temp.scatter_(1, lr_idx, (~recover_lr).to(index_temp))

                # Zero out these tokens in the latent feature map
                h_BChw[index_temp.bool()] = 0


            h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)

            f_hat, next_token_map,h = self.vae_quant_proxy[
                0
            ].get_next_autoregressive_input(
                si, len(self.patch_nums), f_hat, h_BChw, patch_nums=self.patch_nums
            )

            next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)

            next_token_map = (
                self.word_embed(next_token_map)
                + lvl_pos[:, cur_L : cur_L + self.patch_nums[si + 1] ** 2]
            )
            next_token_map = next_token_map.repeat(
                2, 1, 1
            )  # double the batch sizes due to CFG
            
        if compress_method == "sparsevar":
            # At the final stage, set stage index explicitly
            si = len(self.patch_nums) - 1

            # Select high-/low-frequency tokens and anchors for this stage
            hr_idx, lr_idx, recover_lr, anchor_lr = self.lf_anchor(
                logits_BlV, si, mse_difference, compress_ratio_, local_window_size
            )

            # Reshape high-frequency indices into (B, N_keep)
            hr_idx = hr_idx.reshape(B, -1)


        ################ last stage maskgit ################
        si = len(self.patch_nums) - 1
        mask = torch.ones(B, self.last_level_pns).cuda()
        tokens = torch.zeros(B, self.last_level_pns, self.Cvae).cuda()
        orders = self.sample_orders(B)

        num_iter = num_maskgit_iters
        indices = list(range(num_iter))
                

        for step in indices:

            # mask_ratio = 1 - (step + 1) / num_iter
            mask_ratio = np.cos(math.pi / 2.0 * (step + 1) / num_iter)
            mask_len = torch.Tensor([np.floor(self.last_level_pns * mask_ratio)]).cuda()
            # masks out at least one for the next iteration
            mask_len = torch.maximum(
                torch.Tensor([1]).cuda(),
                torch.minimum(torch.sum(mask, dim=-1, keepdims=True) - 1, mask_len),
            )
            # get masking for next iteration and locations to be predicted in this iteration
            mask_next = mask_by_order(mask_len[0], orders, B, self.last_level_pns)
            if step >= num_iter - 1:
                mask_to_pred = mask[:B].bool()
            else:
                mask_to_pred = torch.logical_xor(mask[:B].bool(), mask_next.bool())
            mask = mask_next
            cur_mask = torch.cat([mask_to_pred, mask_to_pred], dim=0)
            cur_mask = cur_mask.nonzero(as_tuple=True)
            x = next_token_map[cur_mask].reshape(2 * B, -1, self.C)
            
            if compress_method == "sparsevar":
                # Repeat high-frequency indices for both conditional/unconditional branches
                hr_idx_repeat = hr_idx.repeat(2, 1)

                # Build mask for selected tokens (used for maskgit conditioning)
                cur_mask_ = torch.cat([
                    torch.gather(mask_to_pred, dim=1, index=hr_idx),
                    torch.gather(mask_to_pred, dim=1, index=hr_idx)
                ], dim=0)
                cur_mask_ = cur_mask_.nonzero(as_tuple=True)
                cur_mask = (cur_mask_[0], hr_idx_repeat.reshape(-1))

                # Gather only high-frequency tokens as the current input
                x_origin = x
                hr_idx_expand = hr_idx_repeat[..., None].expand(-1, -1, x.shape[-1])
                x = torch.gather(x, dim=1, index=hr_idx_expand)


            for block_idx,b in enumerate(self.blocks):
                # Haotian: si used for position embed
                # note: m_maskgit makes sure that PEs are correct.
                origin_x=x
                x,_ = b(
                    x=x,
                    cond_BD=cond_BD_or_gss,
                    attn_bias=None,
                    si=len(self.patch_nums) - 1,
                    m_maskgit=cur_mask,
                    context_position_ids=context_position_ids,
                    context_mask=context_mask
                )

            if compress_method=="sparsevar":
                x_origin.scatter_(1,hr_idx_expand,x)
                source_flag=torch.gather(x_origin,dim=1,index=anchor_lr.unsqueeze(-1).repeat(2,1,x.shape[-1]))
                x_origin.scatter_(1,lr_idx.unsqueeze(-1).repeat(2,1,x.shape[-1]),source_flag)
                x=x_origin

            logits_BlV = self.get_logits(x, cond_BD)
            last_layer_cond = x

            t = cfg * ratio
            logits_BlV = (1 + t) * logits_BlV[:B] - t * logits_BlV[B:]
 
            si = len(self.patch_nums) - 1
            idx_Bl = sample_with_top_k_top_p_(
                logits_BlV.detach().clone(),
                rng=rng,
                top_k=(600 if si < 7 else 300),
                top_p=top_p,
                num_samples=1,
            )[:, :, 0]
            if not more_smooth:  # this is the default case
                h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)  # B, l, Cvae
            else:  # not used when evaluating FID/IS/Precision/Recall
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)  # refer to mask-git
                h_BChw = gumbel_softmax_with_rng(
                    logits_BlV.detach().clone().mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng
                ) @ self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)

            if final_stage == 0:
                # sample with diffusion model
                
                last_stage_discrete_cond = self.vae_quant_proxy[0].embedding(idx_Bl)
                last_stage_discrete_cond = self.word_embed(last_stage_discrete_cond)
                last_stage_discrete_cond = torch.cat(
                    [last_stage_discrete_cond, last_stage_discrete_cond], dim=0
                )
                add_con=last_layer_cond + last_stage_discrete_cond

                last_stage_cond = self.decoder_norm(
                    add_con
                )

                bs, cur_seq_len, _ = last_stage_cond.shape
                ##### begin baseline sampling #####
                last_stage_cond = last_stage_cond.reshape(-1, last_stage_cond.shape[-1])
                h_BChw_diff = self.diffloss.sample(
                    z=last_stage_cond, temperature=1.0, cfg=t
                )
                h_BChw_diff = h_BChw_diff.reshape(bs, -1, h_BChw_diff.shape[-1])
                h_BChw_diff, _ = h_BChw_diff.chunk(2, dim=0)
                ##### end baseline sampling #####
                #h_BChw_diff[h_BChw_diff.isnan()]=0
                # [B, L, Cvae]
                tokens[mask_to_pred] = (h_BChw+h_BChw_diff).reshape(-1, self.Cvae)
            else:
                tokens[mask_to_pred] = h_BChw.reshape(-1, self.Cvae)

        if compress_method == "sparsevar":
            # Build a mask for low-frequency tokens that are NOT recoverable by anchors
            index_temp = torch.zeros(B, self.patch_nums[-1]**2, device=lr_idx.device)
            # Mark lr positions where recover_lr == False
            index_temp.scatter_(1, lr_idx, (~recover_lr).to(index_temp))  # shape: (B, H*W)

            # Zero out those tokens before final output
            tokens[index_temp.bool()] = 0

        h_BChw_final = tokens.transpose(1, 2).reshape(
            B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1]
        )

        f_hat = f_hat+h_BChw_final

        ################ last stage maskgit ################

        for b in self.blocks:
            b.attn.kv_caching(False)

        with torch.no_grad():
            res=self.vae_proxy[0].fhat_to_img(f_hat).add(1).mul(0.5) # de-normalize, from [-1, 1] to [0, 1]

        return (
            res
        )  

    def sample_orders(self, bsz,hr_idx=None):
        # generate a batch of random generation orders
        orders = []
        for _ in range(bsz):
            if hr_idx is None:
                order = np.array(list(range(self.last_level_pns)))
            else:
                order = np.array(list(range(hr_idx.shape[1])))
            np.random.shuffle(order)
            orders.append(order)
        orders = torch.Tensor(np.array(orders)).cuda().long()
        return orders

    def random_masking(self, x, orders, select_L=None, hr_idx=None):
        # generate token mask
        bsz, seq_len, embed_dim = x.shape
        #mask_rate = self.mask_ratio_generator.rvs(1)[0]
        mask_rate = 0.05
        # we cannot mask out all the tokens
        num_masked_tokens = min(int(np.ceil(hr_idx.shape[1] * mask_rate)), seq_len - 32)
        mask = torch.zeros(bsz, seq_len, device=x.device)
        # all first few stages are kept
        if select_L is None:
            mask_keep = torch.zeros(
                bsz, self.L - seq_len + self.context_token - 1, device=x.device
            )
        else:
            mask_keep = torch.zeros(
                bsz, select_L - seq_len, device=x.device
            )
        mask = torch.scatter(
            mask,
            dim=-1,
            index=orders[:, :num_masked_tokens],
            src=torch.ones(bsz, seq_len, device=x.device),
        )
        mask_full = torch.cat([mask_keep, mask], dim=1).contiguous()
        return mask_full, mask

    def forward(
        self,
        context: torch.Tensor,
        x_BLCv_wo_first_l: torch.Tensor,
        context_position_ids: torch.Tensor,
        context_mask: torch.Tensor,
        last_layer_gt: torch.Tensor = None,
        last_layer_gt_discrete: torch.Tensor = None,
        hr_idx: torch.Tensor = None,
        last_stage_cond_cache: torch.Tensor = None,
        hr_idxs: torch.Tensor = None,
        idx_Bl: torch.Tensor = None,
        split_frequency: bool = False
    ) -> torch.Tensor:  # returns logits_BLV
        """
        :param label_B: label_B
        :param x_BLCv_wo_first_l: teacher forcing input (B, self.L-self.first_l, self.Cvae)
        :return: logits BLV, V is vocab_sizee
        """
        bg, ed = (
            self.begin_ends[self.prog_si]
            if self.prog_si >= 0
            else (0, self.L + self.context_token - 1)
        )
        if hr_idx is not None:
            bg, ed = (
                self.begin_ends[self.prog_si]
                if self.prog_si >= 0
                else (0, x_BLCv_wo_first_l.shape[1] + self.context_token - 1)
            )
        B = x_BLCv_wo_first_l.shape[0]
        orders = self.sample_orders(bsz=B,hr_idx=hr_idx)
        #if hr_idx is None:
        #    mask, mask_wo_prev_stages = self.random_masking(
        #        x_BLCv_wo_first_l[:, -self.last_level_pns :, :], orders
        #    )
        #else:
        #    mask, mask_wo_prev_stages = self.random_masking(
        #        x_BLCv_wo_first_l[:, -4096 :, :], orders, select_L=x_BLCv_wo_first_l.shape[1], hr_idx=hr_idx
        #    )
        mask=torch.zeros(B,x_BLCv_wo_first_l.shape[1]).to(x_BLCv_wo_first_l.device)
        mask_wo_prev_stages=torch.zeros(B,self.last_level_pns).to(x_BLCv_wo_first_l.device)
        mask_for_attn = (1 - mask)[:, self.context_token :].nonzero(as_tuple=True) # mask for remain tokens(start from context_len(0))
        mask = (1 - mask).nonzero(as_tuple=True) # mask for remain tokens(start from context_len)
        mask_wo_prev_stages = (1 - mask_wo_prev_stages).nonzero(as_tuple=True) # mask for remain last stage tokens(start from context_len)
        last_layer_gt = last_layer_gt[mask_wo_prev_stages].reshape(
            B, -1, last_layer_gt.shape[-1]
        )
        last_layer_gt_discrete = last_layer_gt_discrete[mask_wo_prev_stages].reshape(
            B, -1
        )
        if hr_idx is not None:
            ed = (
                last_layer_gt.shape[1]
                + x_BLCv_wo_first_l.shape[1]
                - 4096
            )
        else:
            ed = (
                last_layer_gt.shape[1]
                + self.L
                + self.context_token
                - 1
                - self.last_level_pns
            )

        with torch.inference_mode():
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                drop_pos = torch.where(
                    torch.randn(B, device=context.device) < self.cond_drop_rate
                )[0]
                if drop_pos.shape==0:
                    last_stage_cond_cache=last_stage_cond_cache[:B]
                else:
                    last_stage_cond_cache=last_stage_cond_cache[B:]
                context[drop_pos] *= 0

                sos = cond_BD = self.context_embed(self.context_norm(context.half()))
                if self.pos_start is not None:
                    sos = sos.expand(B, self.first_l, -1) + self.pos_start.expand(
                        B, self.first_l, -1
                    )
                else:
                    sos = sos.expand(B, self.first_l, -1)

                if self.prog_si == 0:
                    x_BLC = sos
                else:
                    x_BLC = torch.cat(
                        (sos, self.word_embed(x_BLCv_wo_first_l.float())), dim=1
                    )

                # apply maskgit
                x_BLC = x_BLC[mask].reshape(B, -1, x_BLC.shape[-1])

                select_indexs=[torch.arange(0,300).to(x_BLC.device).unsqueeze(0).repeat(B,1)]

                cur_L=300
                for ni in range(1,len(hr_idxs)-1):
                    now_index=torch.arange(0,self.patch_nums[ni]**2).to(x_BLC.device).unsqueeze(0).repeat(B,1)+cur_L
                    if hr_idxs[ni] is not None:
                        now_index = torch.gather(now_index,dim=1,index=hr_idxs[ni])
                    cur_L+=self.patch_nums[ni]**2
                    select_indexs.append(now_index)
                select_indexs.append(torch.arange(0,mask_wo_prev_stages[0].shape[0]).to(x_BLC.device).unsqueeze(0).repeat(B,1)+cur_L)
                select_indexs=torch.cat(select_indexs,dim=1)

                if self.pos_1LC is not None:
                    x_BLC += (
                        self.lvl_embed(self.lvl_1L[:, :ed].expand(B, -1))
                        + self.pos_1LC[:, :ed]
                    )  # lvl: BLC;  pos: 1LC
                else:
                    if hr_idx is None:
                        x_BLC += self.lvl_embed(self.lvl_1L[:, :ed].expand(B, -1))
                    else:
                        x_BLC += self.lvl_embed(torch.gather(self.lvl_1L,dim=1,index=select_indexs).expand(B, -1))

        attn_bias = self.attn_bias_for_masking[:, :, :ed, :ed]
        cond_BD_or_gss = self.shared_ada_lin(cond_BD)

        # hack: get the dtype if mixed precision is used
        temp = x_BLC.new_ones(8, 8)
        main_type = torch.matmul(temp, temp).dtype

        x_BLC = x_BLC.to(dtype=main_type)
        cond_BD_or_gss = cond_BD_or_gss.to(dtype=main_type)
        attn_bias = attn_bias.to(dtype=main_type)

        AdaLNSelfAttn.forward
        if hr_idx is not None:
            prev_token_len=x_BLC.shape[1]-mask_wo_prev_stages[1].shape[0] # 前面stage的token总和
            all_after_mask=torch.cat([torch.arange(prev_token_len).unsqueeze(0).repeat(B,1).to(x_BLC.device),\
                                              hr_idx+prev_token_len],dim=1).unsqueeze(-1).expand(-1,-1,x_BLC.shape[-1]).long()
            hr_idx_after_mask=hr_idx.unsqueeze(-1).expand(-1,-1,x_BLC.shape[-1])
            x_BLC_origin=x_BLC
            x_BLC=torch.gather(x_BLC,dim=1,index=all_after_mask)
            attn_bias=attn_bias[:,:,:,all_after_mask[0][...,0]][:,:,all_after_mask[0][...,0]]
            mask_for_attn=(torch.cat([mask_for_attn[0][:prev_token_len-self.context_token],\
                                      mask_for_attn[0][prev_token_len-self.context_token:][hr_idx_after_mask[0,:,0]]]),\
                                        torch.cat([mask_for_attn[1][:prev_token_len-self.context_token],\
                                      mask_for_attn[1][prev_token_len-self.context_token:][hr_idx_after_mask[0,:,0]]]))
        for i, b in enumerate(self.blocks):
            x_BLC,_ = b(
                x=x_BLC,
                cond_BD=cond_BD_or_gss,
                attn_bias=attn_bias,
                m_maskgit=mask_for_attn,
                context_position_ids=context_position_ids,
                context_mask=context_mask,
            )
        if hr_idx is not None:
            lr_idx_in_mask=get_reverse_indices(hr_idx,4096).to(x_BLC.device)+prev_token_len #在最后序列中丢掉的token在mask后序列中的index
            lr_idx_in_cache=mask_wo_prev_stages[1][lr_idx_in_mask[0].int()-prev_token_len].unsqueeze(0)
            x_BLC=x_BLC_origin.scatter(1,all_after_mask,x_BLC)
            start_this_stage=0
            end_this_stage=299
            last_stage_cache=torch.zeros(B,self.patch_nums[-1]*self.patch_nums[-1],x_BLC.shape[-1]).to(x_BLC)
            for ni in range(1,len(hr_idxs)-1):
                start_this_stage=end_this_stage+1
                pn=self.patch_nums[ni]
                if hr_idxs[ni] is not None:
                    end_this_stage=start_this_stage+hr_idxs[ni].shape[1]-1
                    this_stage_x_BLCs=x_BLC[:,start_this_stage:end_this_stage+1]
                    flags=torch.zeros(B,pn*pn,x_BLC.shape[-1]).to(x_BLC)
                    flags.scatter_(1,hr_idxs[ni].unsqueeze(-1).expand(-1,-1,x_BLC.shape[-1]),this_stage_x_BLCs)
                    this_stage_x_BLCs=flags
                else:
                    end_this_stage=start_this_stage+pn*pn-1
                    this_stage_x_BLCs=x_BLC[:,start_this_stage:end_this_stage+1]
                this_stage_cache=torch.nn.functional.interpolate(this_stage_x_BLCs.reshape(B,pn,pn,-1).permute(0,3,1,2),\
                                                    size=(self.patch_nums[-1],self.patch_nums[-1]),mode='nearest').permute(0,2,3,1).reshape(B,-1,x_BLC.shape[-1])
                
                last_stage_cache[this_stage_cache!=0]=this_stage_cache[this_stage_cache!=0]
            x_BLC.scatter_(1,lr_idx_in_mask.unsqueeze(-1).expand(-1,-1,x_BLC.shape[-1]),
                                torch.gather(last_stage_cond_cache,dim=1,index=lr_idx_in_cache.unsqueeze(-1).expand(-1,-1,last_stage_cache.shape[-1])).expand(-1,-1,x_BLC.shape[-1]))
            #x_BLC=last_stage_cache
        # parallel generation of discrete and continuous tokens
        if hr_idx is None:
            x_BLC_logits, last_layer_cond = (
                x_BLC,
                x_BLC[:, self.L + self.context_token - 1 - self.last_level_pns :, :],
            )
        else:
            x_BLC_logits, last_layer_cond = (
                x_BLC,
                x_BLC[:, x_BLCv_wo_first_l.shape[1] - 4096 :, :],
            )

        x_BLC_logits = self.get_logits(x_BLC_logits.float(), cond_BD)
        with torch.no_grad():
            # important to clone the last stage logits
            # Haotian: autoregressive LLM sometimes have this error:
            # RuntimeError: probability tensor contains either `inf`, `nan` or element < 0
            try:
                temp=self.L + self.context_token - 1 - self.last_level_pns
                if hr_idx is not None:
                    temp = x_BLCv_wo_first_l.shape[1] - 4096
                idx_BL_sampled = sample_with_top_k_top_p_(
                    x_BLC_logits[
                        :, temp :
                    ]
                    .clone()
                    .detach(),
                    rng=self.rng,
                    top_k=600,
                    top_p=0.96,
                    num_samples=1,
                )[:, :, 0]
            except:
                idx_BL_sampled = last_layer_gt_discrete
        #idx_BL_sampled=idx_Bl
        last_stage_discrete_embed = self.vae_quant_proxy[0].embedding(idx_BL_sampled)
        last_stage_discrete_cond = self.word_embed(last_stage_discrete_embed)
        if hr_idx is not None:
            last_stage_discrete_cond=select_and_set_zero(last_stage_discrete_cond,hr_idx_after_mask[...,0])
            last_stage_discrete_embed=select_and_set_zero(last_stage_discrete_embed,hr_idx_after_mask[...,0])
        last_layer_cond = self.decoder_norm(last_layer_cond + last_stage_discrete_cond)
            

        last_layer_gt_continuous = last_layer_gt - last_stage_discrete_embed

        if hr_idx is not None and split_frequency:
            hr_idx_expand=hr_idx.unsqueeze(-1).expand(-1,-1,last_layer_cond.shape[-1])
            reverse_select=get_reverse_indices(hr_idx,self.patch_nums[-1]**2)
            reverse_select_expand=reverse_select.unsqueeze(-1).expand(-1,-1,last_layer_cond.shape[-1])
            last_layer_cond_high=last_layer_cond.gather(1,hr_idx_expand)
            last_layer_cond_low=last_layer_cond.gather(1,reverse_select_expand)
            last_layer_gt_continuous_high=last_layer_gt_continuous.gather(1,hr_idx.unsqueeze(-1).expand(-1,-1,self.Cvae))
            last_layer_gt_continuous_low=last_layer_gt_continuous.gather(1,reverse_select.unsqueeze(-1).expand(-1,-1,self.Cvae))
            diff_loss_high = self.forward_diff_loss(
                z=last_layer_cond_high, target=last_layer_gt_continuous_high
            )
            diff_loss_low = self.forward_diff_loss(
                z=last_layer_cond_low, target=last_layer_gt_continuous_low, low_frequency=True
            )
            diff_loss = diff_loss_high + diff_loss_low
        else:
            diff_loss = self.forward_diff_loss(
                z=last_layer_cond[:,lr_idx_in_mask[0]-prev_token_len], target=last_layer_gt_continuous[:,lr_idx_in_mask[0]-prev_token_len]
            )
        # Haotian: important, we should start from self.context_token - 1.
        return (
            x_BLC_logits[:, self.context_token - 1 :, :],
            diff_loss,
            mask_wo_prev_stages,
        )  # logits BLV, V is vocab_size
    
    def cal_loss_wlogits(self,x_BLC_logits,last_layer_cond,last_layer_gt,hr_idx,last_stage_cond_cache):
        idx_BL_sampled = sample_with_top_k_top_p_(
                    x_BLC_logits
                    .clone()
                    .detach(),
                    rng=self.rng,
                    top_k=600,
                    top_p=0.96,
                    num_samples=1,
                )[:, :, 0]
        last_stage_discrete_embed = self.vae_quant_proxy[0].embedding(idx_BL_sampled)
        last_stage_discrete_cond = self.word_embed(last_stage_discrete_embed)
        last_stage_discrete_cond=select_and_set_zero(last_stage_discrete_cond,hr_idx[-1])
        last_stage_discrete_embed=select_and_set_zero(last_stage_discrete_embed,hr_idx[-1])
        lr_index_expand=torch.tensor(get_reverse_indices(hr_idx[-1],self.patch_nums[-1]**2).unsqueeze(-1).repeat(1,1,last_stage_cond_cache.shape[-1]))
        last_layer_cond=last_layer_cond.scatter(1,lr_index_expand,torch.gather(last_stage_cond_cache,dim=1,index=lr_index_expand))
        last_layer_cond = self.decoder_norm(last_layer_cond + last_stage_discrete_cond)
        last_layer_gt_continuous = last_layer_gt - last_stage_discrete_embed
        diff_loss = self.forward_diff_loss(
                z=last_layer_cond, target=last_layer_gt_continuous
            )
        return diff_loss
    
    def init_weights(
        self,
        init_adaln=0.5,
        init_adaln_gamma=1e-5,
        init_head=0.02,
        init_std=0.02,
        conv_std_or_gain=0.02,
    ):
        if init_std < 0:
            init_std = (1 / self.C / 3) ** 0.5  # init_std < 0: automated

        print(f"[init_weights] {type(self).__name__} with {init_std=:g}")
        for m in self.modules():
            with_weight = hasattr(m, "weight") and m.weight is not None
            with_bias = hasattr(m, "bias") and m.bias is not None
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if with_bias:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()
            elif isinstance(
                m,
                (
                    nn.LayerNorm,
                    nn.BatchNorm1d,
                    nn.BatchNorm2d,
                    nn.BatchNorm3d,
                    nn.SyncBatchNorm,
                    nn.GroupNorm,
                    nn.InstanceNorm1d,
                    nn.InstanceNorm2d,
                    nn.InstanceNorm3d,
                ),
            ):
                if with_weight:
                    m.weight.data.fill_(1.0)
                if with_bias:
                    m.bias.data.zero_()
            # conv: VAR has no conv, only VQVAE has conv
            elif isinstance(
                m,
                (
                    nn.Conv1d,
                    nn.Conv2d,
                    nn.Conv3d,
                    nn.ConvTranspose1d,
                    nn.ConvTranspose2d,
                    nn.ConvTranspose3d,
                ),
            ):
                if conv_std_or_gain > 0:
                    nn.init.trunc_normal_(m.weight.data, std=conv_std_or_gain)
                else:
                    nn.init.xavier_normal_(m.weight.data, gain=-conv_std_or_gain)
                if with_bias:
                    m.bias.data.zero_()

        if init_head >= 0:
            if isinstance(self.head, nn.Linear):
                self.head.weight.data.mul_(init_head)
                self.head.bias.data.zero_()
            elif isinstance(self.head, nn.Sequential):
                self.head[-1].weight.data.mul_(init_head)
                self.head[-1].bias.data.zero_()

        if isinstance(self.head_nm, AdaLNBeforeHead):
            self.head_nm.ada_lin[-1].weight.data.mul_(init_adaln)
            if (
                hasattr(self.head_nm.ada_lin[-1], "bias")
                and self.head_nm.ada_lin[-1].bias is not None
            ):
                self.head_nm.ada_lin[-1].bias.data.zero_()

        depth = len(self.blocks)
        for block_idx, sab in enumerate(self.blocks):
            sab: AdaLNSelfAttn
            sab.attn.proj.weight.data.div_(math.sqrt(2 * depth))
            if hasattr(sab.ffn, "fc2"):
                sab.ffn.fc2.weight.data.div_(math.sqrt(2 * depth))
            if hasattr(sab.ffn, "fcg") and sab.ffn.fcg is not None:
                nn.init.ones_(sab.ffn.fcg.bias)
                nn.init.trunc_normal_(sab.ffn.fcg.weight, std=1e-5)
            if hasattr(sab, "ada_lin"):
                sab.ada_lin[-1].weight.data[2 * self.C :].mul_(init_adaln)
                sab.ada_lin[-1].weight.data[: 2 * self.C].mul_(init_adaln_gamma)
                if (
                    hasattr(sab.ada_lin[-1], "bias")
                    and sab.ada_lin[-1].bias is not None
                ):
                    sab.ada_lin[-1].bias.data.zero_()
            elif hasattr(sab, "ada_gss"):
                sab.ada_gss.data[:, :, 2:].mul_(init_adaln)
                sab.ada_gss.data[:, :, :2].mul_(init_adaln_gamma)

        self.diffloss.initialize_weights()

    def extra_repr(self):
        return f"drop_path_rate={self.drop_path_rate:g}"


AutoConfig.register("hart_transformer_t2i", HARTForT2IConfig)
AutoModel.register(HARTForT2IConfig, HARTForT2I)

def get_new_anomalous_indices(hr_idx, select_index):
    """
    根据异常值索引 (hr_idx) 和选择索引 (select_index)，
    返回新 tensor 中异常值的索引。

    参数:
    - hr_idx (torch.Tensor): 异常值索引，形状 BxC。
    - select_index (torch.Tensor): 新选择的索引，形状 BxN。

    返回:
    - new_anomalous_index (list): 每个 batch 新 tensor 中异常值的索引 (B x C')。
    """
    B = hr_idx.shape[0]  # Batch size
    new_anomalous_index = []

    for b in range(B):
        # 获取当前 batch 中异常值在新索引 (select_index) 中的位置
        indices_in_select = torch.nonzero(
            torch.isin(select_index[b], hr_idx[b]), as_tuple=True
        )[0]
        new_anomalous_index.append(indices_in_select.cpu().tolist())

    return new_anomalous_index