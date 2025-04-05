# Copyright (C) 2024 NVIDIA Corporation.  All rights reserved.
#
# This work is licensed under the LICENSE file
# located at the root directory.
import os.path

import torch
from diffusers import DDIMScheduler
from .consistory_unet_sdxl import ConsistorySDXLUNet2DConditionModel
from .consistory_pipeline import ConsistoryExtendAttnSDXLPipeline
from .consistory_utils import FeatureInjector, AnchorCache
from .utils.general_utils import *
import gc
import folder_paths

LATENT_RESOLUTIONS = [32, 64]

def clear_memory():
    torch.cuda.empty_cache()
    gc.collect()




def load_pipeline(repo_id,unet_path,gpu_id=0):
    float_type = torch.float16
    #sd_id = "stabilityai/stable-diffusion-xl-base-1.0"
    device = torch.device(f'cuda:{gpu_id}') if torch.cuda.is_available() else torch.device('cpu')
    if repo_id:
        unet = ConsistorySDXLUNet2DConditionModel.from_pretrained(repo_id, subfolder="unet", torch_dtype=float_type)
        scheduler = DDIMScheduler.from_pretrained(repo_id, subfolder="scheduler")
        story_pipeline = ConsistoryExtendAttnSDXLPipeline.from_pretrained(
            repo_id, unet=unet, torch_dtype=float_type, variant="fp16", use_safetensors=True, scheduler=scheduler
        ).to(device)
    elif not repo_id and unet_path:
        config_file = os.path.join(folder_paths.base_path, "custom_nodes/ComfyUI_StoryDiffusion/local_repo/unet/config.json")
        sdxl_repo = os.path.join(folder_paths.base_path, "custom_nodes/ComfyUI_StoryDiffusion/local_repo")
        original_config_file = os.path.join(folder_paths.base_path, "custom_nodes/ComfyUI_StoryDiffusion/config/sd_xl_base.yaml")
        from safetensors.torch import load_file
        from diffusers.pipelines.stable_diffusion.convert_from_ckpt import convert_ldm_unet_checkpoint
        state_dict = load_file(unet_path)
        unet_config = ConsistorySDXLUNet2DConditionModel.load_config(config_file)
        Unet = ConsistorySDXLUNet2DConditionModel.from_config(unet_config).to(device,torch.float16)
        state_dict = convert_ldm_unet_checkpoint(state_dict, Unet.config)
        Unet.load_state_dict(state_dict, strict=False)
        del state_dict
        clear_memory()
        scheduler = DDIMScheduler.from_pretrained(sdxl_repo, subfolder="scheduler")
        try:
            story_pipeline = ConsistoryExtendAttnSDXLPipeline.from_single_file(
                unet_path,unet=Unet, config=sdxl_repo, original_config=original_config_file, torch_dtype=float_type, variant="fp16", use_safetensors=True, scheduler=scheduler
            ).to(device)
        except:
            story_pipeline = ConsistoryExtendAttnSDXLPipeline.from_single_file(
                unet_path, unet=Unet,config=sdxl_repo, original_config_file=original_config_file, torch_dtype=float_type, variant="fp16", use_safetensors=True, scheduler=scheduler
            ).to(device)
       
    else:
        raise "need a repo or chocie a sdxl checkpoints"
    
    
    story_pipeline.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)
    return story_pipeline

def create_anchor_mapping(bsz, anchor_indices=[0]):
    anchor_mapping = torch.eye(bsz, dtype=torch.bool)
    for anchor_idx in anchor_indices:
        anchor_mapping[:, anchor_idx] = True

    return anchor_mapping

def create_token_indices(prompts, batch_size, concept_token, tokenizer):
    if isinstance(concept_token, str):
        concept_token = [concept_token]

    concept_token_id = [tokenizer.encode(x, add_special_tokens=False)[0] for x in concept_token]
    tokens = tokenizer.batch_encode_plus(prompts, padding=True, return_tensors='pt')['input_ids']

    token_indices = torch.full((len(concept_token), batch_size), -1, dtype=torch.int64)
    for i, token_id in enumerate(concept_token_id):
        batch_loc, token_loc = torch.where(tokens == token_id)
        token_indices[i, batch_loc] = token_loc

    return token_indices

def create_latents(story_pipeline, seed, batch_size, same_latent, device, float_type):
    # if seed is int
    if isinstance(seed, int):
        #g = torch.Generator('cuda').manual_seed(seed)
        g = torch.Generator(device).manual_seed(seed)
        shape = (batch_size, story_pipeline.unet.config.in_channels, 128, 128)
        latents = randn_tensor(shape, generator=g, device=device, dtype=float_type)
    elif isinstance(seed, list):
        shape = (batch_size, story_pipeline.unet.config.in_channels, 128, 128)
        latents = torch.empty(shape, device=device, dtype=float_type)
        for i, seed_i in enumerate(seed):
            #g = torch.Generator('cuda').manual_seed(seed_i)
            g = torch.Generator(device).manual_seed(seed_i)
            curr_latent = randn_tensor(shape, generator=g, device=device, dtype=float_type)
            latents[i] = curr_latent[i]
    else:
        raise "seed cause error"
    if same_latent:
        latents = latents[:1].repeat(batch_size, 1, 1, 1)

    return latents, g

# Batch inference
def run_batch_generation(story_pipeline, prompts, concept_token,negative_prompt,
                        seed=40, n_steps=50, mask_dropout=0.5,
                        same_latent=False, share_queries=True,
                        perform_sdsa=True, perform_injection=True,
                        downscale_rate=4, n_achors=2,cf_clip=None):
    device = story_pipeline.device
    tokenizer = story_pipeline.tokenizer
    float_type = story_pipeline.dtype
    unet = story_pipeline.unet

    batch_size = len(prompts)

    token_indices = create_token_indices(prompts, batch_size, concept_token, tokenizer)
    anchor_mappings = create_anchor_mapping(batch_size, anchor_indices=list(range(n_achors)))

    default_attention_store_kwargs = {
        'token_indices': token_indices,
        'mask_dropout': mask_dropout,
        'extended_mapping': anchor_mappings
    }

    default_extended_attn_kwargs = {'extend_kv_unet_parts': ['up']}
    query_store_kwargs= {'t_range': [0,n_steps//10], 'strength_start': 0.9, 'strength_end': 0.81836735}

    latents, g = create_latents(story_pipeline, seed, batch_size, same_latent, device, float_type)

    # ------------------ #
    # Extended attention First Run #

    if perform_sdsa:
        extended_attn_kwargs = {**default_extended_attn_kwargs, 't_range': [(1, n_steps)]}
    else:
        extended_attn_kwargs = {**default_extended_attn_kwargs, 't_range': []}

    print(extended_attn_kwargs['t_range'])
    
    out = story_pipeline(prompt=prompts, negative_prompt=negative_prompt, generator=g, latents=latents,
                         attention_store_kwargs=default_attention_store_kwargs,
                         extended_attn_kwargs=extended_attn_kwargs,
                         share_queries=share_queries,
                         query_store_kwargs=query_store_kwargs,
                         num_inference_steps=n_steps,cf_clip=cf_clip)
    
    
    # Extended attention with nn_map #
    
    if  perform_injection:
        last_masks = story_pipeline.attention_store.last_mask
        dift_features = unet.latent_store.dift_features['251_0'][batch_size:]
        unet.latent_store.reset() # turn to {}
        del out
        clear_memory()
        dift_features = torch.stack([gaussian_smooth(x, kernel_size=3, sigma=1) for x in dift_features], dim=0)
        nn_map, nn_distances = cyclic_nn_map(dift_features, last_masks, LATENT_RESOLUTIONS, device)
        
        clear_memory()
        feature_injector = FeatureInjector(nn_map, nn_distances, last_masks,
                                           inject_range_alpha=[(n_steps // 10, n_steps // 3, 0.8)],
                                           swap_strategy='min', inject_unet_parts=['up', 'down'],
                                           dist_thr='dynamic')
       
        out = story_pipeline(prompt=prompts, generator=g, latents=latents,
                             attention_store_kwargs=default_attention_store_kwargs,
                             extended_attn_kwargs=extended_attn_kwargs,
                             share_queries=share_queries,
                             query_store_kwargs=query_store_kwargs,
                             feature_injector=feature_injector,
                             num_inference_steps=n_steps,cf_clip=cf_clip)
        
        #img_all = view_images([np.array(x) for x in out.images], display_image=False, downscale_rate=downscale_rate)
        # display_attn_maps(story_pipeline.attention_store.last_mask, out.images)
        clear_memory()

    return out.images

# Anchors
def run_anchor_generation(story_pipeline, prompts, concept_token,negative_prompt,
                        seed=40, n_steps=50, mask_dropout=0.5,
                        same_latent=False, share_queries=True,
                        perform_sdsa=True, perform_injection=True,
                        downscale_rate=4, cache_cpu_offloading=False,cf_clip=None):
    device = story_pipeline.device
    tokenizer = story_pipeline.tokenizer
    float_type = story_pipeline.dtype
    unet = story_pipeline.unet

    batch_size = len(prompts)

    token_indices = create_token_indices(prompts, batch_size, concept_token, tokenizer)

    default_attention_store_kwargs = {
        'token_indices': token_indices,
        'mask_dropout': mask_dropout
    }

    default_extended_attn_kwargs = {'extend_kv_unet_parts': ['up']}
    query_store_kwargs={'t_range': [0,n_steps//10], 'strength_start': 0.9, 'strength_end': 0.81836735}

    latents, g = create_latents(story_pipeline, seed, batch_size, same_latent, device, float_type)

    anchor_cache_first_stage = AnchorCache()
    anchor_cache_second_stage = AnchorCache()

    # ------------------ #
    # Extended attention First Run #

    if perform_sdsa:
        extended_attn_kwargs = {**default_extended_attn_kwargs, 't_range': [(1, n_steps)]}
    else:
        extended_attn_kwargs = {**default_extended_attn_kwargs, 't_range': []}

    print(extended_attn_kwargs['t_range'])
    
    
    out = story_pipeline(prompt=prompts, negative_prompt=negative_prompt, generator=g, latents=latents,
                         attention_store_kwargs=default_attention_store_kwargs,
                         extended_attn_kwargs=extended_attn_kwargs,
                         share_queries=share_queries,
                         query_store_kwargs=query_store_kwargs,
                         anchors_cache=anchor_cache_first_stage,
                         num_inference_steps=n_steps,cf_clip=cf_clip)
   
    last_masks = story_pipeline.attention_store.last_mask
    
    dift_features = unet.latent_store.dift_features['251_0'][batch_size:]
    unet.latent_store.reset()  # turn to {}
    clear_memory()
    dift_features = torch.stack([gaussian_smooth(x, kernel_size=3, sigma=1) for x in dift_features], dim=0)
    
    anchor_cache_first_stage.dift_cache = dift_features
    anchor_cache_first_stage.anchors_last_mask = last_masks
    
    if cache_cpu_offloading:
        anchor_cache_first_stage.to_device(torch.device('cpu'))
    
    nn_map, nn_distances = cyclic_nn_map(dift_features, last_masks, LATENT_RESOLUTIONS, device)
    clear_memory()
    # ------------------ #
    # Extended attention with nn_map #
    
    if  perform_injection:
        
        feature_injector = FeatureInjector(nn_map, nn_distances, last_masks, inject_range_alpha=[(n_steps//10, n_steps//3,0.8)], 
                                        swap_strategy='min', inject_unet_parts=['up', 'down'], dist_thr='dynamic')

        out = story_pipeline(prompt=prompts,negative_prompt=negative_prompt, generator=g, latents=latents,
                            attention_store_kwargs=default_attention_store_kwargs,
                            extended_attn_kwargs=extended_attn_kwargs,
                            share_queries=share_queries,
                            query_store_kwargs=query_store_kwargs,
                            feature_injector=feature_injector,
                            anchors_cache=anchor_cache_second_stage,
                            num_inference_steps=n_steps,cf_clip=cf_clip)
        #img_all = view_images([np.array(x) for x in out.images], display_image=False, downscale_rate=downscale_rate)
        # display_attn_maps(story_pipeline.attention_store.last_mask, out.images)

        anchor_cache_second_stage.dift_cache = dift_features
        anchor_cache_second_stage.anchors_last_mask = last_masks

        if cache_cpu_offloading:
            anchor_cache_second_stage.to_device(torch.device('cpu'))
        
        clear_memory()
    # else:
    #     img_all = view_images([np.array(x) for x in out.images], display_image=False, downscale_rate=downscale_rate)
    
    return out.images, anchor_cache_first_stage, anchor_cache_second_stage

def run_extra_generation(story_pipeline, prompts, concept_token, negative_prompt,
                         anchor_cache_first_stage, anchor_cache_second_stage,
                         seed=40, n_steps=50, mask_dropout=0.5,
                         same_latent=False, share_queries=True,
                         perform_sdsa=True, perform_injection=True,
                         downscale_rate=4, cache_cpu_offloading=False,cf_clip=None):
    device = story_pipeline.device
    tokenizer = story_pipeline.tokenizer
    float_type = story_pipeline.dtype
    unet = story_pipeline.unet

    batch_size = len(prompts)

    token_indices = create_token_indices(prompts, batch_size, concept_token, tokenizer)

    default_attention_store_kwargs = {
        'token_indices': token_indices,
        'mask_dropout': mask_dropout
    }

    default_extended_attn_kwargs = {'extend_kv_unet_parts': ['up']}
    query_store_kwargs={'t_range': [0,n_steps//10], 'strength_start': 0.9, 'strength_end': 0.81836735}

    extra_batch_size = batch_size + 2
    if isinstance(seed, list):
        seed = [seed[0], seed[0], *seed]

    latents, g = create_latents(story_pipeline, seed, extra_batch_size, same_latent, device, float_type)
    latents = latents[2:]

    anchor_cache_first_stage.set_mode_inject()
    anchor_cache_second_stage.set_mode_inject()

    # ------------------ #
    # Extended attention First Run #

    if cache_cpu_offloading:
        anchor_cache_first_stage.to_device(device)

    if perform_sdsa:
        extended_attn_kwargs = {**default_extended_attn_kwargs, 't_range': [(1, n_steps)]}
    else:
        extended_attn_kwargs = {**default_extended_attn_kwargs, 't_range': []}

    print(extended_attn_kwargs['t_range'])
    out = story_pipeline(prompt=prompts,negative_prompt=negative_prompt, generator=g, latents=latents,
                        attention_store_kwargs=default_attention_store_kwargs,
                        extended_attn_kwargs=extended_attn_kwargs,
                        share_queries=share_queries,
                        query_store_kwargs=query_store_kwargs,
                        anchors_cache=anchor_cache_first_stage,
                        num_inference_steps=n_steps,cf_clip=cf_clip)
   

    # ------------------ #
    # Extended attention with nn_map #
    last_masks = story_pipeline.attention_store.last_mask
    dift_features = unet.latent_store.dift_features['251_0'][batch_size:] # 261_0 cause error
    unet.latent_store.reset()  # turn to {}
    clear_memory()
    dift_features = torch.stack([gaussian_smooth(x, kernel_size=3, sigma=1) for x in dift_features], dim=0)
    
    anchor_dift_features = anchor_cache_first_stage.dift_cache
    anchor_last_masks = anchor_cache_first_stage.anchors_last_mask
    
    nn_map, nn_distances = anchor_nn_map(dift_features, anchor_dift_features, last_masks, anchor_last_masks,
                                         LATENT_RESOLUTIONS, device)
    
    if cache_cpu_offloading:
        anchor_cache_first_stage.to_device(torch.device('cpu'))
    
    clear_memory()
    
    if  perform_injection:

        if cache_cpu_offloading:
            anchor_cache_second_stage.to_device(device)

        feature_injector = FeatureInjector(nn_map, nn_distances, last_masks, inject_range_alpha=[(n_steps//10, n_steps//3,0.8)], 
                                        swap_strategy='min', inject_unet_parts=['up', 'down'], dist_thr='dynamic')

        out = story_pipeline(prompt=prompts,negative_prompt=negative_prompt, generator=g, latents=latents,
                            attention_store_kwargs=default_attention_store_kwargs,
                            extended_attn_kwargs=extended_attn_kwargs,
                            share_queries=share_queries,
                            query_store_kwargs=query_store_kwargs,
                            feature_injector=feature_injector,
                            anchors_cache=anchor_cache_second_stage,
                            num_inference_steps=n_steps,cf_clip=cf_clip)
        
        
        #img_all = view_images([np.array(x) for x in out.images], display_image=False, downscale_rate=downscale_rate)
        # display_attn_maps(story_pipeline.attention_store.last_mask, out.images)

        if cache_cpu_offloading:
            anchor_cache_second_stage.to_device(torch.device('cpu'))
        clear_memory()
    # else:
    #     img_all = view_images([np.array(x) for x in out.images], display_image=False, downscale_rate=downscale_rate)
    
    return out.images
