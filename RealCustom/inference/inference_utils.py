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

def find_phrase_positions_in_text(text, phrase):
    """
    Return the position of the first character of the phrase in the text.
    """
    
    position = -1
    positions = []
    while True:
        position = text.find(phrase, position + 1)
        if position == -1:
            break
        positions.append(position)
    return positions

def classifier_free_guidance_image_prompt_cascade(
    pred_t_cond, pred_ti_cond, pred_uncond, guidance_weight_t=7.5, guidance_weight_i=7.5, 
    guidance_stdev_rescale_factor=0.7, cfg_rescale_mode="none", super_cross_mask=None
    ):

    if cfg_rescale_mode == "none":
        pred = pred_uncond + guidance_weight_t * (pred_t_cond - pred_uncond) + guidance_weight_i * (pred_ti_cond - pred_t_cond)
    elif cfg_rescale_mode == "none_direct":
        pred = pred_uncond + guidance_weight_i * (pred_ti_cond - pred_uncond)
    elif cfg_rescale_mode == "naive":
        assert super_cross_mask is not None
        pred_std_t_before = pred_t_cond.std([1, 2, 3], keepdim=True)
        pred_std_ti_before = pred_ti_cond.std([1, 2, 3], keepdim=True)

        pred = pred_uncond + guidance_weight_t * (pred_t_cond - pred_uncond) + guidance_weight_i * (pred_ti_cond - pred_t_cond)

        pred_std_after = pred.std([1, 2, 3], keepdim=True)

        pred_rescale_t_factor = guidance_stdev_rescale_factor * (pred_std_t_before / pred_std_after) + (1 - guidance_stdev_rescale_factor)
        pred_rescale_ti_factor = guidance_stdev_rescale_factor * (pred_std_ti_before / pred_std_after) + (1 - guidance_stdev_rescale_factor)

        pred_ti = pred * super_cross_mask
        pred_t = pred * (1 - super_cross_mask)
        pred = pred_ti * pred_rescale_ti_factor + pred_t * pred_rescale_t_factor
    elif cfg_rescale_mode == "naive_global":
        pred_std_ti_before = pred_ti_cond.std([1, 2, 3], keepdim=True)

        pred = pred_uncond + guidance_weight_t * (pred_t_cond - pred_uncond) + guidance_weight_i * (pred_ti_cond - pred_t_cond)

        pred_std_after = pred.std([1, 2, 3], keepdim=True)

        pred_rescale_ti_factor = guidance_stdev_rescale_factor * (pred_std_ti_before / pred_std_after) + (1 - guidance_stdev_rescale_factor)

        pred = pred * pred_rescale_ti_factor
    elif cfg_rescale_mode == "naive_global_direct":
        pred_std_ti_before = pred_ti_cond.std([1, 2, 3], keepdim=True)

        pred = pred_uncond + guidance_weight_i * (pred_ti_cond - pred_uncond)

        pred_std_after = pred.std([1, 2, 3], keepdim=True)

        pred_rescale_ti_factor = guidance_stdev_rescale_factor * (pred_std_ti_before / pred_std_after) + (1 - guidance_stdev_rescale_factor)

        pred = pred * pred_rescale_ti_factor
    else:
        raise NotImplementedError()

    return pred