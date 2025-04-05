# modified from https://github.com/mlfoundations/open_flamingo/blob/main/open_flamingo/src/helpers.py
# and https://github.com/lucidrains/imagen-pytorch/blob/main/imagen_pytorch/imagen_pytorch.py
# and https://github.com/tencent-ailab/IP-Adapter/blob/main/ip_adapter/ip_adapter.py
# and https://github.com/tencent-ailab/IP-Adapter/blob/main/ip_adapter/resampler.py

import math

import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange


class FourierEmbedder(nn.Module):
    def __init__(self, num_freqs=64, temperature=100):
        super().__init__()

        self.num_freqs = num_freqs
        self.temperature = temperature

        freq_bands = temperature ** (torch.arange(num_freqs) / num_freqs)
        freq_bands = freq_bands[None, None]
        self.register_buffer("freq_bands", freq_bands, persistent=False)

    def __call__(self, x):
        x = self.freq_bands * x.unsqueeze(-1)
        return torch.stack((x.sin(), x.cos()), dim=-1).permute(0, 2, 3, 1).reshape(x.shape[0], -1)


class ImageProjModel(torch.nn.Module):
    """Projection Model"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens


# FFN
def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
        # nn.LayerNorm(dim),
    )


def reshape_tensor(x, heads):
    bs, length, width = x.shape
    # (bs, length, width) --> (bs, length, n_heads, dim_per_head)
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

    def forward(self, x, latents):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, n1, D)
            latent (torch.Tensor): latent features
                shape (b, n2, D)
        """
        x = self.norm1(x)
        latents = self.norm2(latents)

        b, l, _ = latents.shape

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        # attention
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1)  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v

        out = out.permute(0, 2, 1, 3).reshape(b, l, -1)

        return self.to_out(out)


class Resampler(nn.Module):
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
        max_seq_len: int = 257,  # CLIP tokens + CLS token
        apply_pos_emb: bool = False,
        num_latents_mean_pooled: int = 0,  # number of latents derived from mean pooled representation of the sequence
        latent_init_mode: str = "random",
        phrase_embeddings_dim: int = 1024,
        fourier_freqs: int = 8,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.grounding_token_num = self.num_queries
        self.dim = dim
        self.pos_emb = nn.Embedding(max_seq_len, embedding_dim) if apply_pos_emb else None

        self.latent_init_mode = latent_init_mode
        if latent_init_mode == "random":
            self.latents = nn.Parameter(torch.randn(1, self.latents_token_num, dim) / dim**0.5)
            self.fourier_embedder = None
            self.latent_proj = None
            self.latent_norm = None
        elif latent_init_mode == "grounding":
            self.latents = None
            self.grounding_latents = nn.Parameter(torch.randn(1, self.grounding_token_num, dim) / dim ** 0.5)
            self.fourier_embedder = FourierEmbedder(num_freqs=fourier_freqs)
            grounding_embedding_dim = phrase_embeddings_dim + fourier_freqs * 2 * 4  # 2: sin/cos, 4: xyxy
            self.latent_proj = torch.nn.Sequential(
                torch.nn.Linear(grounding_embedding_dim, grounding_embedding_dim * 2),
                torch.nn.GELU(),
                torch.nn.Linear(grounding_embedding_dim * 2, dim * self.grounding_token_num),
            )
            self.latent_norm = nn.LayerNorm(dim)
        else:
            raise ValueError(f"Invalid latent_init_mode: {latent_init_mode}")

        self.proj_in = nn.Linear(embedding_dim, dim)
        self.attention_norm = nn.LayerNorm(dim)

        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)

        self.to_latents_from_mean_pooled_seq = (
            nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim * num_latents_mean_pooled),
                Rearrange("b (n d) -> b n d", n=num_latents_mean_pooled),
            )
            if num_latents_mean_pooled > 0
            else None
        )

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

    def forward(self, x, grounding_kwargs=None, shortcut=False, scale=1.0):
        if self.pos_emb is not None:
            n, device = x.shape[1], x.device
            pos_emb = self.pos_emb(torch.arange(n, device=device))
            x = x + pos_emb

        if self.latent_init_mode == "random":
            latents = self.latents.repeat(x.size(0), 1, 1)
        elif self.latent_init_mode == "grounding":
            boxes = grounding_kwargs["boxes"]
            phrase_embeds = grounding_kwargs["phrase_embeds"]
            fourier_embeds = self.fourier_embedder(boxes)
            #print(phrase_embeds.shape,fourier_embeds.shape)
            grounding_embeds = torch.cat((phrase_embeds, fourier_embeds), dim=-1)
            
            drop_grounding_tokens = grounding_kwargs["drop_grounding_tokens"]
            num_ref = x.shape[0] // len(drop_grounding_tokens)
            drop_grounding_tokens = [item for item in drop_grounding_tokens for _ in range(num_ref)]

            latents = self.latent_proj(grounding_embeds)
            latents = latents.view(-1, self.grounding_token_num, self.dim)
            latents = self.latent_norm(latents)

            # drop grounding tokens to learnable latents
            drop_num = len([item for item in drop_grounding_tokens if item == 1])
            if drop_num > 0:
                latents_ = []
                learnable_latents = self.grounding_latents.repeat(drop_num, 1, 1)
                cur_idx = 0
                for latent, drop_grounding_token in zip(latents, drop_grounding_tokens):
                    if drop_grounding_token == 1:
                        latent = learnable_latents[cur_idx]
                        cur_idx += 1
                    latents_.append(latent)
                latents = torch.stack(latents_)
        else:
            raise ValueError(f"Invalid latent_init_mode: {self.latent_init_mode}")

        x = self.proj_in(x)

        if self.to_latents_from_mean_pooled_seq:
            meanpooled_seq = masked_mean(x, dim=1, mask=torch.ones(x.shape[:2], device=x.device, dtype=torch.bool))
            meanpooled_latents = self.to_latents_from_mean_pooled_seq(meanpooled_seq)
            latents = torch.cat((meanpooled_latents, latents), dim=-2)

        init_latents = latents

        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents

        latents = self.attention_norm(latents)
        latents = self.proj_out(latents)
        if shortcut:
            latents = init_latents + latents * scale

        return self.norm_out(latents)


def masked_mean(t, *, dim, mask=None):
    if mask is None:
        return t.mean(dim=dim)

    denom = mask.sum(dim=dim, keepdim=True)
    mask = rearrange(mask, "b n -> b n 1")
    masked_t = t.masked_fill(~mask, 0.0)

    return masked_t.sum(dim=dim) / denom.clamp(min=1e-5)
