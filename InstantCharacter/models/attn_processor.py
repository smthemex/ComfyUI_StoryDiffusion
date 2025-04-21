from typing import Optional

import torch.nn as nn
import torch
import torch.nn.functional as F
from diffusers.models.embeddings import apply_rotary_emb
from einops import rearrange

from .norm_layer import RMSNorm


class FluxIPAttnProcessor(nn.Module):
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(
        self,
        hidden_size=None,
        ip_hidden_states_dim=None,
    ):
        super().__init__()
        self.norm_ip_q = RMSNorm(128, eps=1e-6)
        self.to_k_ip = nn.Linear(ip_hidden_states_dim, hidden_size)
        self.norm_ip_k = RMSNorm(128, eps=1e-6)
        self.to_v_ip = nn.Linear(ip_hidden_states_dim, hidden_size)


    def __call__(
        self,
        attn,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        emb_dict={},
        subject_emb_dict={},
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        # IPadapter
        ip_hidden_states = self._get_ip_hidden_states(
            attn, 
            query if encoder_hidden_states is not None else query[:, emb_dict['length_encoder_hidden_states']:],
            subject_emb_dict.get('ip_hidden_states', None)
        )

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)


        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )
                
            if ip_hidden_states is not None:
                hidden_states = hidden_states + ip_hidden_states * subject_emb_dict.get('scale', 1.0)

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:

            if ip_hidden_states is not None:
                hidden_states[:, emb_dict['length_encoder_hidden_states']:] = \
                    hidden_states[:, emb_dict['length_encoder_hidden_states']:] + \
                    ip_hidden_states * subject_emb_dict.get('scale', 1.0)

            return hidden_states


    def _scaled_dot_product_attention(self, query, key, value, attention_mask=None, heads=None):
        query = rearrange(query, '(b h) l c -> b h l c', h=heads)
        key = rearrange(key, '(b h) l c -> b h l c', h=heads)
        value = rearrange(value, '(b h) l c -> b h l c', h=heads)
        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False, attn_mask=None)
        hidden_states = rearrange(hidden_states, 'b h l c -> (b h) l c', h=heads)
        hidden_states = hidden_states.to(query)
        return hidden_states


    def _get_ip_hidden_states(
            self, 
            attn, 
            img_query, 
            ip_hidden_states, 
        ):
        if ip_hidden_states is None:
            return None
        
        if not hasattr(self, 'to_k_ip') or not hasattr(self, 'to_v_ip'):
            return None

        ip_query = self.norm_ip_q(rearrange(img_query, 'b l (h d) -> b h l d', h=attn.heads))
        ip_query = rearrange(ip_query, 'b h l d -> (b h) l d')
        ip_key = self.to_k_ip(ip_hidden_states)
        ip_key = self.norm_ip_k(rearrange(ip_key, 'b l (h d) -> b h l d', h=attn.heads))
        ip_key = rearrange(ip_key, 'b h l d -> (b h) l d')
        ip_value = self.to_v_ip(ip_hidden_states)
        ip_value = attn.head_to_batch_dim(ip_value)
        ip_hidden_states = self._scaled_dot_product_attention(
            ip_query.to(ip_value.dtype), ip_key.to(ip_value.dtype), ip_value, None, attn.heads)
        ip_hidden_states = ip_hidden_states.to(img_query.dtype)
        ip_hidden_states = attn.batch_to_head_dim(ip_hidden_states)
        return ip_hidden_states

