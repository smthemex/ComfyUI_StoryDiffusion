# Copyright (C) 2024 NVIDIA Corporation.  All rights reserved.
#
# This work is licensed under the LICENSE file
# located at the root directory.

import os
import argparse
from .consistory_run import load_pipeline, run_batch_generation, run_anchor_generation, run_extra_generation

def run_batch(gpu, seed=40, mask_dropout=0.5, same_latent=False,
              style="A photo of ", subject="a cute dog", concept_token=['dog'],
              settings=["sitting in the beach", "standing in the snow"],
              out_dir = None):
    
    story_pipeline = load_pipeline(gpu)
    prompts = [f'{style}{subject} {setting}' for setting in settings]

    images, image_all = run_batch_generation(story_pipeline, prompts, concept_token, seed, mask_dropout=mask_dropout, same_latent=same_latent)

    if out_dir is not None:
        for i, image in enumerate(images):
            image.save(f'{out_dir}/image_{i}.png')

    return images, image_all

def run_cached_anchors(gpu, seed=40, mask_dropout=0.5, same_latent=False,
                style="A photo of ", subject="a cute dog", concept_token=['dog'],
                settings=["sitting in the beach", "standing in the snow"],
                cache_cpu_offloading=False, out_dir = None):
    
    story_pipeline = load_pipeline(gpu)
    prompts = [f'{style}{subject} {setting}' for setting in settings]
    anchor_prompts = prompts[:2]
    extra_prompts = prompts[2:]

    anchor_out_images, anchor_image_all, anchor_cache_first_stage, anchor_cache_second_stage = run_anchor_generation(story_pipeline, anchor_prompts, concept_token, 
                                                                                                        seed=seed, mask_dropout=mask_dropout, same_latent=same_latent,
                                                                                                        cache_cpu_offloading=cache_cpu_offloading)
    
    if out_dir is not None:
        for i, image in enumerate(anchor_out_images):
            image.save(f'{out_dir}/anchor_image_{i}.png')

    for i, extra_prompt in enumerate(extra_prompts):
        extra_out_images, extra_image_all = run_extra_generation(story_pipeline, [extra_prompt], concept_token, anchor_cache_first_stage, anchor_cache_second_stage, 
                                                    seed=seed, mask_dropout=mask_dropout, same_latent=same_latent, cache_cpu_offloading=cache_cpu_offloading)
        
        if out_dir is not None:
            extra_out_images[0].save(f'{out_dir}/extra_image_{i}.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_type', default="batch", type=str, required=False) # batch, cached

    parser.add_argument('--gpu', default=0, type=int, required=False)
    parser.add_argument('--seed', default=40, type=int, required=False)
    parser.add_argument('--mask_dropout', default=0.5, type=float, required=False)
    parser.add_argument('--same_latent', default=False, type=bool, required=False)

    parser.add_argument('--style', default="A photo of ", type=str, required=False)
    parser.add_argument('--subject', default="a cute dog", type=str, required=False)
    parser.add_argument('--concept_token', default=["dog"], 
                        type=str, nargs='*', required=False)
    parser.add_argument('--settings', default=["sitting in the beach", "standing in the snow"], 
                        type=str, nargs='*', required=False)
    parser.add_argument('--cache_cpu_offloading', default=False, type=bool, required=False)
    
    parser.add_argument('--out_dir', default=None, type=str, required=False)

    args = parser.parse_args()

    if args.out_dir is not None:
        os.makedirs(args.out_dir, exist_ok=True)

    if args.run_type == "batch":
        run_batch(args.gpu, args.seed, args.mask_dropout, args.same_latent, args.style, 
                  args.subject, args.concept_token, args.settings, args.out_dir)
    elif args.run_type == "cached":
        run_cached_anchors(args.gpu, args.seed, args.mask_dropout, args.same_latent, args.style, 
                           args.subject, args.concept_token, args.settings, args.cache_cpu_offloading, args.out_dir)
    else:
        print("Invalid run type")