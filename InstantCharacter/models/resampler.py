import torch.nn as nn
import torch
import math

from diffusers.models.transformers.transformer_2d import BasicTransformerBlock
from diffusers.models.embeddings import Timesteps, TimestepEmbedding
from timm.models.vision_transformer import Mlp

from .norm_layer import RMSNorm


# FFN
def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )

    
def reshape_tensor(x, heads):
    bs, length, width = x.shape
    #(bs, length, width) --> (bs, length, n_heads, dim_per_head)
    x = x.view(bs, length, heads, -1)
    # (bs, length, n_heads, dim_per_head) --> (bs, n_heads, length, dim_per_head)
    x = x.transpose(1, 2)
    # (bs, n_heads, length, dim_per_head) --> (bs*n_heads, length, dim_per_head)
    x = x.reshape(bs, heads, length, -1)
    return x


class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)


    def forward(self, x, latents, shift=None, scale=None):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, n1, D)
            latent (torch.Tensor): latent features
                shape (b, n2, D)
        """
        x = self.norm1(x)
        latents = self.norm2(latents)

        if shift is not None and scale is not None:
            latents = latents * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        
        b, l, _ = latents.shape

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        
        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        # attention
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1) # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v
        
        out = out.permute(0, 2, 1, 3).reshape(b, l, -1)

        return self.to_out(out)


class ReshapeExpandToken(nn.Module):
    def __init__(self, expand_token, token_dim):
        super().__init__()
        self.expand_token = expand_token
        self.token_dim = token_dim

    def forward(self, x):
        x = x.reshape(-1, self.expand_token, self.token_dim)
        return x


class TimeResampler(nn.Module):
    def __init__(
        self,
        dim=1024,
        depth=8,
        dim_head=64,
        heads=16,
        num_queries=8,
        embedding_dim=768,
        output_dim=1024,
        ff_mult=4,
        timestep_in_dim=320,
        timestep_flip_sin_to_cos=True,
        timestep_freq_shift=0,
        expand_token=None,
        extra_dim=None,
    ):
        super().__init__()
        
        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / dim**0.5)

        self.expand_token = expand_token is not None
        if expand_token:
            self.expand_proj = torch.nn.Sequential(
                torch.nn.Linear(embedding_dim, embedding_dim * 2),
                torch.nn.GELU(),
                torch.nn.Linear(embedding_dim * 2, embedding_dim * expand_token),
                ReshapeExpandToken(expand_token, embedding_dim),
                RMSNorm(embedding_dim, eps=1e-8),
            )

        self.proj_in = nn.Linear(embedding_dim, dim)
        
        self.extra_feature = extra_dim is not None
        if self.extra_feature:
            self.proj_in_norm = RMSNorm(dim, eps=1e-8)
            self.extra_proj_in = torch.nn.Sequential(
                nn.Linear(extra_dim, dim),
                RMSNorm(dim, eps=1e-8),
            )

        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)
        
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        # msa
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        # ff
                        FeedForward(dim=dim, mult=ff_mult),
                        # adaLN
                        nn.Sequential(nn.SiLU(), nn.Linear(dim, 4 * dim, bias=True))
                    ]
                )
            )

        # time
        self.time_proj = Timesteps(timestep_in_dim, timestep_flip_sin_to_cos, timestep_freq_shift)
        self.time_embedding = TimestepEmbedding(timestep_in_dim, dim, act_fn="silu")


    def forward(self, x, timestep, need_temb=False, extra_feature=None):
        timestep_emb = self.embedding_time(x, timestep)  # bs, dim

        latents = self.latents.repeat(x.size(0), 1, 1)
        
        if self.expand_token:
            x = self.expand_proj(x)

        x = self.proj_in(x)

        if self.extra_feature:
            extra_feature = self.extra_proj_in(extra_feature)
            x = self.proj_in_norm(x)
            x = torch.cat([x, extra_feature], dim=1)
            
        x = x + timestep_emb[:, None]

        for attn, ff, adaLN_modulation in self.layers:
            shift_msa, scale_msa, shift_mlp, scale_mlp = adaLN_modulation(timestep_emb).chunk(4, dim=1)
            latents = attn(x, latents, shift_msa, scale_msa) + latents

            res = latents
            for idx_ff in range(len(ff)):
                layer_ff = ff[idx_ff]
                latents = layer_ff(latents)
                if idx_ff == 0 and isinstance(layer_ff, nn.LayerNorm):  # adaLN
                    latents = latents * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
            latents = latents + res

            # latents = ff(latents) + latents
            
        latents = self.proj_out(latents)
        latents = self.norm_out(latents)

        if need_temb:
            return latents, timestep_emb
        else:
            return latents


    def embedding_time(self, sample, timestep):

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=sample.dtype)

        emb = self.time_embedding(t_emb, None)
        return emb


class CrossLayerCrossScaleProjector(nn.Module):
    def __init__(
        self,
        inner_dim=2688,
        num_attention_heads=42,
        attention_head_dim=64,
        cross_attention_dim=2688,
        num_layers=4,

        # resampler
        dim=1280,
        depth=4,
        dim_head=64,
        heads=20,
        num_queries=1024,
        embedding_dim=1152 + 1536,
        output_dim=4096,
        ff_mult=4,
        timestep_in_dim=320,
        timestep_flip_sin_to_cos=True,
        timestep_freq_shift=0,
    ):
        super().__init__()

        self.cross_layer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=0,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn="geglu",
                    num_embeds_ada_norm=None,
                    attention_bias=False,
                    only_cross_attention=False,
                    double_self_attention=False,
                    upcast_attention=False,
                    norm_type='layer_norm',
                    norm_elementwise_affine=True,
                    norm_eps=1e-6,
                    attention_type="default",
                )
                for _ in range(num_layers)
            ]
        )

        self.cross_scale_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=0,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn="geglu",
                    num_embeds_ada_norm=None,
                    attention_bias=False,
                    only_cross_attention=False,
                    double_self_attention=False,
                    upcast_attention=False,
                    norm_type='layer_norm',
                    norm_elementwise_affine=True,
                    norm_eps=1e-6,
                    attention_type="default",
                )
                for _ in range(num_layers)
            ]
        )

        self.proj = Mlp(
            in_features=inner_dim, 
            hidden_features=int(inner_dim*2), 
            act_layer=lambda: nn.GELU(approximate="tanh"), 
            drop=0
        )

        self.proj_cross_layer = Mlp(
            in_features=inner_dim, 
            hidden_features=int(inner_dim*2), 
            act_layer=lambda: nn.GELU(approximate="tanh"), 
            drop=0
        )

        self.proj_cross_scale = Mlp(
            in_features=inner_dim, 
            hidden_features=int(inner_dim*2), 
            act_layer=lambda: nn.GELU(approximate="tanh"), 
            drop=0
        )

        self.resampler = TimeResampler(
            dim=dim,
            depth=depth,
            dim_head=dim_head,
            heads=heads,
            num_queries=num_queries,
            embedding_dim=embedding_dim,
            output_dim=output_dim,
            ff_mult=ff_mult,
            timestep_in_dim=timestep_in_dim,
            timestep_flip_sin_to_cos=timestep_flip_sin_to_cos,
            timestep_freq_shift=timestep_freq_shift,
        )

    def forward(self, low_res_shallow, low_res_deep, high_res_deep, timesteps, cross_attention_kwargs=None, need_temb=True):
        '''
            low_res_shallow [bs, 729*l, c]
            low_res_deep    [bs, 729, c]
            high_res_deep   [bs, 729*4, c]
        '''

        cross_layer_hidden_states = low_res_deep
        for block in self.cross_layer_blocks:
            cross_layer_hidden_states = block(
                cross_layer_hidden_states,
                encoder_hidden_states=low_res_shallow,
                cross_attention_kwargs=cross_attention_kwargs,
            )
        cross_layer_hidden_states = self.proj_cross_layer(cross_layer_hidden_states)

        cross_scale_hidden_states = low_res_deep
        for block in self.cross_scale_blocks:
            cross_scale_hidden_states = block(
                cross_scale_hidden_states,
                encoder_hidden_states=high_res_deep,
                cross_attention_kwargs=cross_attention_kwargs,
            )
        cross_scale_hidden_states = self.proj_cross_scale(cross_scale_hidden_states)
        
        hidden_states = self.proj(low_res_deep) + cross_scale_hidden_states
        hidden_states = torch.cat([hidden_states, cross_layer_hidden_states], dim=1)

        hidden_states, timestep_emb = self.resampler(hidden_states, timesteps, need_temb=True)
        return hidden_states, timestep_emb

