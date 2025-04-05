# modified from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py
# and https://github.com/tencent-ailab/IP-Adapter/blob/main/ip_adapter/attention_processor.py
import torch
import torch.nn as nn
import torch.nn.functional as F


def minmax_normalize(batch_maps):
    min_val = batch_maps.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
    max_val = batch_maps.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]

    return (batch_maps - min_val) / (max_val - min_val + 1e-5)


class AttnProcessor2_0(torch.nn.Module):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(
        self,
        hidden_size=None,
        cross_attention_dim=None,
    ):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        boxes=None,
        phrase_idxes=None,
        eot_idxes=None,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class MaskedIPAttnProcessor2_0(nn.Module):

    def __init__(self, hidden_size, cross_attention_dim=None, scale=1.0, num_tokens=4, text_tokens=77,
                 need_text_attention_map=False, need_image_attention_map=True, num_dummy_tokens=4, mask_threshold=0.5,
                 use_psuedo_attention_mask=False, subject_scales=None, start_step=5):
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale
        self.num_tokens = num_tokens
        self.text_tokens = text_tokens
        self.num_dummy_tokens = num_dummy_tokens
        self.mask_threshold = mask_threshold
        self.subject_scales = subject_scales
        self.start_step = start_step

        self.to_k_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)

        self.need_text_attention_map = need_text_attention_map
        self.need_image_attention_map = need_image_attention_map

        self.use_psuedo_attention_mask = use_psuedo_attention_mask
        self.attention_maps = []

    def prepare_attention_mask_qk(self, boxes, phrase_idxes, sequence_length_q, sequence_length_k,
                                  batch_size, head_size, dtype, device, use_masked_text_attention=False):
        if boxes is None:
            return None, None

        # TODO: only support square image now
        num_patches_per_row = int(sequence_length_q ** 0.5)
        box_idxes_start = torch.floor(boxes[:, :, 0:2] * num_patches_per_row)
        box_idxes_end = torch.ceil(boxes[:, :, 2:4] * num_patches_per_row)
        box_idxes = torch.cat([box_idxes_start, box_idxes_end], dim=-1)
        box_masks = []
        dummy_attention_mask = torch.ones((batch_size, sequence_length_q), dtype=dtype, device=device)
        for box_idx in box_idxes.unbind(dim=1):
            x_start_patch_idx, y_start_patch_idx, x_end_patch_idx, y_end_patch_idx = box_idx.unbind(dim=1)
            x_indices = torch.arange(num_patches_per_row).unsqueeze(0).expand(batch_size, -1).to(device)
            y_indices = torch.arange(num_patches_per_row).unsqueeze(0).expand(batch_size, -1).to(device)
            x_mask = ((x_indices >= x_start_patch_idx.unsqueeze(1)) & (x_indices < x_end_patch_idx.unsqueeze(1))).to(dtype)
            y_mask = ((y_indices >= y_start_patch_idx.unsqueeze(1)) & (y_indices < y_end_patch_idx.unsqueeze(1))).to(dtype)
            box_mask = torch.bmm(y_mask.unsqueeze(2), x_mask.unsqueeze(1)).reshape(batch_size, -1)
            box_masks.append(box_mask)
            dummy_attention_mask = torch.clamp(dummy_attention_mask - box_mask, min=0)

        # post mask
        post_dummy_attention_mask = dummy_attention_mask.to(torch.bool)
        post_dummy_attention_mask = post_dummy_attention_mask.repeat_interleave(head_size, dim=0)
        
        attention_mask_qk_image = torch.stack(box_masks, dim=-1)
        attention_mask_qk_image = attention_mask_qk_image.repeat_interleave(self.num_tokens, dim=-1)
        attention_mask_qk_image = (1 - attention_mask_qk_image.to(dtype)) * -10000.0  # mask to bias
        # use dummy image tokens to process the background
        dummy_attention_mask = dummy_attention_mask.unsqueeze(-1).repeat_interleave(self.num_dummy_tokens, dim=-1)
        dummy_attention_mask = (1 - dummy_attention_mask) * -10000.0
        attention_mask_qk_image = torch.cat([dummy_attention_mask, attention_mask_qk_image], dim=-1)
        if attention_mask_qk_image.shape[0] < batch_size*head_size:
            attention_mask_qk_image = attention_mask_qk_image.repeat_interleave(head_size, dim=0)
        
        if use_masked_text_attention:
            attention_mask_qk_text = torch.ones((batch_size, sequence_length_q, sequence_length_k), dtype=dtype, device=device)
            for i in range(batch_size):
                for j in range(len(box_masks)):
                    start_idx, end_idx = int(phrase_idxes[i, j, 0].item()), int(phrase_idxes[i, j, 1].item())
                    if start_idx == 0 and end_idx == 0:
                        continue
                    attention_mask_qk_text[i, :, start_idx:end_idx] = box_masks[j][i, ...].unsqueeze(-1)
            attention_mask_qk_text = (1 - attention_mask_qk_text) * -10000.0
            if attention_mask_qk_text.shape[0] < batch_size*head_size:
                attention_mask_qk_text = attention_mask_qk_text.repeat_interleave(head_size, dim=0)
        else:
            attention_mask_qk_text = None

        return attention_mask_qk_image, attention_mask_qk_text, post_dummy_attention_mask

    def get_text_attention_maps(self, attention_probs, boxes, phrase_idxes, head_size):
        bsz = boxes.shape[0]
        _, num_tokens_q, num_tokens_k = attention_probs.shape
        attention_probs = attention_probs.view(bsz, head_size, num_tokens_q, num_tokens_k)
        num_ref = boxes.shape[1]
        h = w = int(num_tokens_q ** 0.5)
        batch_attention_maps = []
        for i in range(bsz):
            sample_attention_maps = []
            for j in range(num_ref):
                start_idx, end_idx = int(phrase_idxes[i, j, 0].item()), int(phrase_idxes[i, j, 1].item())
                if start_idx == 0 and end_idx == 0:
                    sample_attention_maps.append(
                        torch.zeros(num_tokens_q, dtype=attention_probs.dtype, device=attention_probs.device))
                else:
                    attention_map = attention_probs[i, :, :,
                                    start_idx:end_idx]  # [num_heads, num_tokens_q, num_tokens_phrase]
                    attention_map = torch.mean(torch.mean(attention_map, dim=-1), dim=0)  # [num_tokens_q]
                    sample_attention_maps.append(attention_map)
            batch_attention_maps.append(torch.stack(sample_attention_maps))

        self.attention_maps.append(torch.stack(batch_attention_maps).reshape(bsz, num_ref, h, w))

    def get_psuedo_attention_mask(self, head_size):
        # text_attention_maps = self.attention_maps[-1]  # [bsz, num_ref, h, w]
        if not self.use_psuedo_attention_mask or len(self.attention_maps) < self.start_step:
            return None, None
        text_attention_maps = torch.stack(self.attention_maps).mean(dim=0)  # [bsz, num_ref, h, w]
        text_attention_maps = minmax_normalize(text_attention_maps)
        dtype, device = text_attention_maps.dtype, text_attention_maps.device
        bsz, num_ref, h, w = text_attention_maps.shape
        seq_len_q = h * w
        text_attention_maps = text_attention_maps.view(bsz, num_ref, -1)
        text_attention_maps = text_attention_maps.transpose(1, 2)  # [bsz, h*w, num_ref]

        # use threshold to get the mask
        psuedo_attention_mask = (text_attention_maps > self.mask_threshold).to(dtype)
        psuedo_dummy_attention_mask = torch.ones((bsz, seq_len_q), dtype=dtype, device=device)
        for i in range(num_ref):
            psuedo_box_mask = psuedo_attention_mask[..., i]
            psuedo_dummy_attention_mask = torch.clamp(psuedo_dummy_attention_mask - psuedo_box_mask, min=0)

        # post mask
        post_psuedo_dummy_attention_mask = psuedo_dummy_attention_mask.to(torch.bool)
        post_psuedo_dummy_attention_mask = post_psuedo_dummy_attention_mask.repeat_interleave(head_size, dim=0)

        psuedo_attention_mask = psuedo_attention_mask.repeat_interleave(self.num_tokens, dim=-1)
        psuedo_attention_mask = (1 - psuedo_attention_mask) * -10000.0  # mask to bias
        psuedo_dummy_attention_mask = psuedo_dummy_attention_mask.unsqueeze(-1).repeat_interleave(self.num_dummy_tokens, dim=-1)
        psuedo_dummy_attention_mask = (1 - psuedo_dummy_attention_mask) * -10000.0
        psuedo_attention_mask = torch.cat([psuedo_dummy_attention_mask, psuedo_attention_mask], dim=-1)
        if psuedo_attention_mask.shape[0] < bsz * head_size:
            psuedo_attention_mask = psuedo_attention_mask.repeat_interleave(head_size, dim=0)

        return psuedo_attention_mask, post_psuedo_dummy_attention_mask

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        boxes=None,
        phrase_idxes=None,
        eot_idxes=None,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])
        rf_attention_mask = None
        #print(boxes, phrase_idxes, hidden_states.shape[1],self.text_tokens, batch_size, attn.heads,
                                                                #hidden_states.dtype, hidden_states.device) #one None 2304 77 2 10 torch.float16 cuda:0
        custom_attention_masks = self.prepare_attention_mask_qk(boxes, phrase_idxes, hidden_states.shape[1],
                                                                self.text_tokens, batch_size, attn.heads,
                                                                hidden_states.dtype, hidden_states.device,
                                                                use_masked_text_attention=False)
        #print(custom_attention_masks)
        attention_mask_qk_image, attention_mask_qk_text, dummy_attention_mask = custom_attention_masks
        if attention_mask_qk_image is not None:
            attention_mask_qk_image = attention_mask_qk_image.view(batch_size, attn.heads, -1, attention_mask_qk_image.shape[-1])
        if attention_mask_qk_text is not None:
            attention_mask_qk_text = attention_mask_qk_text.view(batch_size, attn.heads, -1, attention_mask_qk_text.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            # get encoder_hidden_states, ip_hidden_states
            # end_pos = encoder_hidden_states.shape[1] - self.num_tokens
            end_pos = self.text_tokens
            encoder_hidden_states, ip_hidden_states = (
                encoder_hidden_states[:, :end_pos, :],
                encoder_hidden_states[:, end_pos:, :],
            )
            attention_mask, rf_attention_mask = (
                attention_mask[:, :, :, :end_pos],
                attention_mask[:, :, :, end_pos:],
            ) if attention_mask is not None else (None, None)
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        attention_mask = attention_mask_qk_text if attention_mask_qk_text is not None else attention_mask
        if not self.need_text_attention_map:
            # original attention 2.0
            new_query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            # the output of sdp = (batch, num_heads, seq_len, head_dim)
            hidden_states = F.scaled_dot_product_attention(
                new_query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )

            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            hidden_states = hidden_states.to(query.dtype)
        else:
            # we need get the attention map, so use the previous attention
            new_query = attn.head_to_batch_dim(query)
            key = attn.head_to_batch_dim(key)
            value = attn.head_to_batch_dim(value)

            if attention_mask is not None:
                attention_mask = attention_mask.view(batch_size*attn.heads, -1, attention_mask.shape[-1])
            attention_probs = attn.get_attention_scores(new_query, key, attention_mask)
            self.get_text_attention_maps(attention_probs, boxes, phrase_idxes, attn.heads)
            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = attn.batch_to_head_dim(hidden_states)

        # get psuedo attention mask for image: better start after some timesteps
        psuedo_attention_mask, psuedo_dummy_attention_mask = self.get_psuedo_attention_mask(attn.heads)
        if psuedo_attention_mask is not None:
            psuedo_attention_mask = psuedo_attention_mask.view(batch_size, attn.heads, -1,
                                                               psuedo_attention_mask.shape[-1])

        ip_key = self.to_k_ip(ip_hidden_states)
        ip_value = self.to_v_ip(ip_hidden_states)
        rf_attention_mask = attention_mask_qk_image if attention_mask_qk_image is not None else rf_attention_mask
        rf_attention_mask = psuedo_attention_mask if psuedo_attention_mask is not None else rf_attention_mask
        dummy_attention_mask = psuedo_dummy_attention_mask if psuedo_dummy_attention_mask is not None else dummy_attention_mask
        if not self.need_image_attention_map:
            new_query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            # the output of sdp = (batch, num_heads, seq_len, head_dim)
            ip_hidden_states = F.scaled_dot_product_attention(
                new_query, ip_key, ip_value, attn_mask=rf_attention_mask, dropout_p=0.0, is_causal=False
            )

            ip_hidden_states = ip_hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            ip_hidden_states = ip_hidden_states.to(query.dtype)
        else:
            new_query = attn.head_to_batch_dim(query)
            ip_key = attn.head_to_batch_dim(ip_key)
            ip_value = attn.head_to_batch_dim(ip_value)

            if rf_attention_mask is not None:
                rf_attention_mask = rf_attention_mask.view(batch_size*attn.heads, -1, rf_attention_mask.shape[-1])
            ip_attention_probs = attn.get_attention_scores(new_query, ip_key, rf_attention_mask)
            # mask attention_probs in background
            ip_attention_probs = torch.where(dummy_attention_mask.unsqueeze(-1), torch.zeros_like(ip_attention_probs), ip_attention_probs)
            if self.subject_scales is not None:
                # apply different scales to different subjects
                subject_scales = torch.tensor(self.subject_scales, dtype=ip_attention_probs.dtype, device=ip_attention_probs.device)
                subject_scales = subject_scales.unsqueeze(0).unsqueeze(0).repeat_interleave(self.num_tokens, dim=-1)
                dummy_subject_scales = torch.ones((1, 1, 1), dtype=ip_attention_probs.dtype, device=ip_attention_probs.device).repeat_interleave(self.num_dummy_tokens, dim=-1)
                subject_scales = torch.cat([dummy_subject_scales, subject_scales], dim=-1)
                ip_attention_probs = ip_attention_probs * subject_scales
            ip_hidden_states = torch.bmm(ip_attention_probs, ip_value)
            ip_hidden_states = attn.batch_to_head_dim(ip_hidden_states)

        if self.subject_scales is None:
            hidden_states = hidden_states + self.scale * ip_hidden_states
        else:
            hidden_states = hidden_states + ip_hidden_states

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class CNAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self, num_tokens=4, text_tokens=77):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        self.num_tokens = num_tokens
        self.text_tokens = text_tokens

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])
        rf_attention_mask = None

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            # end_pos = encoder_hidden_states.shape[1] - self.num_tokens
            end_pos = self.text_tokens
            encoder_hidden_states = encoder_hidden_states[:, :end_pos]  # only use text
            attention_mask = attention_mask[:, :, :end_pos]
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
