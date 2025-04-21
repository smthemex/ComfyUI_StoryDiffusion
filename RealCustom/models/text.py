# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
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

import torch
from dataclasses import dataclass
from torch import nn
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import convert_open_clip_checkpoint,convert_ldm_clip_checkpoint
from transformers import AutoTokenizer, CLIPTokenizerFast, CLIPTextModel, T5EncoderModel,CLIPTextModelWithProjection,CLIPTextConfig
from typing import List



@dataclass
class TextModelOutput:
    embeddings: torch.Tensor
    masks: torch.Tensor
    pooled: List


class TextModel(nn.Module):
    available_modes = [
        "last",                 # If present, use last layer.
        "penultimate",          # If present, use penultimate layer.
        "penultimate_nonorm",   # If present, use penultimate layer without final norm.
        "token_cat",            # If present, concat in token dimension, default concat in channel dimension.
        "pad0",                 # If present, use 0 padding, default use EOT padding.
        "masked",               # If present, pass attention mask to encoder.
    ]

    def __init__(self, variant: List[str],token_paths, mode: List[str]):
        super().__init__()
        self.mode = set(mode)
        self.tokenizers = []
        self.models = nn.ModuleList([])
        
        for index,(clip,token) in enumerate(zip(variant,token_paths)):
            self.tokenizers.append(CLIPTokenizerFast.from_pretrained(token, model_max_length=77))
            self.models.append(clip)
        # for v in variant:
        #     if "clip" in v.lower():
        #         self.tokenizers.append(CLIPTokenizerFast.from_pretrained(v, model_max_length=77))
        #         self.models.append(CLIPTextModel.from_pretrained(v))
        #     elif "t5" in v.lower() or "ul2" in v.lower():
        #         self.tokenizers.append(AutoTokenizer.from_pretrained(v, model_max_length=77))
        #         self.models.append(T5EncoderModel.from_pretrained(v, torch_dtype=torch.bfloat16))
        #     else:
        #         raise NotImplementedError
    
    def get_vaild_token_length(self, text): # Return the length of the BPE encoding of the text, excluding `<sos>` and `<eos>`.
        lengths = []
        for tokenizer, model in zip(self.tokenizers, self.models):

            tokens = tokenizer(
                text=text,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            ).to(model.device)
            token_length = tokens["attention_mask"].sum() - 2 # In the attention mask, both the SOS and EOS (first PAD) have a value of 1.
            # if token_length.is_meta:
            #     token_length = token_length.to("cuda")  # 将
            lengths.append(token_length.item())
        length = int(sum(lengths) / len(lengths))
        return length

    def forward(self, text: List[str]) -> TextModelOutput:
        embeddings = []
        masks = []
        pooled = []

        for tokenizer, model in zip(self.tokenizers, self.models):

            tokens = tokenizer(
                text=text,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            ).to(model.device)

            if "pad0" in self.mode:
                tokens.input_ids *= tokens.attention_mask

            output = model(
                input_ids=tokens.input_ids,
                attention_mask=tokens.attention_mask if "masked" in self.mode else None,
                output_hidden_states=True
            )

            if "last" in self.mode:
                embeddings.append(output.last_hidden_state)
            if "penultimate" in self.mode:
                embeddings.append(model.text_model.final_layer_norm(output.hidden_states[-2]))
            if "penultimate_nonorm" in self.mode:
                embeddings.append(output.hidden_states[-2])
            masks.append(tokens.attention_mask)
            if hasattr(output, "pooler_output"):
                pooled.append(output.pooler_output)

        if "token_cat" in self.mode:
            return TextModelOutput(
                embeddings=torch.cat(embeddings, dim=1),
                masks=torch.cat(masks, dim=1),
                pooled=pooled
            )
        else:
            return TextModelOutput(
                embeddings=torch.cat(embeddings, dim=2),
                masks=torch.stack(masks, dim=2).sum(2).clamp_max(1),
                pooled=pooled
            )