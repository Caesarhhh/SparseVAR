import os
import os.path as osp
import hashlib
import time
import argparse
import json
import shutil
import glob
import re
import sys
sys.path.insert(1,"./")

import torchvision
import cv2
from tqdm import tqdm
import torch
import numpy as np
from PIL import Image
from pytorch_lightning import seed_everything
from transformers import (
    AutoModel,
    AutoTokenizer,
)
from hart.utils import encode_prompts, llm_system_prompt
from hart.modules.models.transformer import HARTForT2I

# set environment variables
os.environ['XFORMERS_FORCE_DISABLE_TRITON'] = '1'

def save_images(sample_imgs, save_path, store_separately, prompts, dataset_name="", metadata=None):
    sample_folder_dir=os.path.dirname(save_path)
    if dataset_name == "GENEVAL":
        sample_folder_dir=save_path.replace("."+save_path.split(".")[-1],"")
        os.makedirs(sample_folder_dir, exist_ok=True)

        os.makedirs(f"{sample_folder_dir}/samples", exist_ok=True)
        sample_imgs_np = sample_imgs.mul(255).cpu().numpy()
        num_imgs = sample_imgs_np.shape[0]
        os.makedirs(sample_folder_dir, exist_ok=True)
        for img_idx in range(num_imgs):
            cur_img = sample_imgs_np[img_idx]
            cur_img = cur_img.transpose(1, 2, 0).astype(np.uint8)
            cur_img_store = Image.fromarray(cur_img)
            cur_img_store.save(f"{sample_folder_dir}/samples/{str(img_idx).zfill(4)}.png")

        grid = torchvision.utils.make_grid(sample_imgs, nrow=2)
        grid_np = grid.to(torch.float16).permute(1, 2, 0).mul(255).cpu().numpy()
        grid_np = Image.fromarray(grid_np.astype(np.uint8))
        grid_np.save(f"{sample_folder_dir}/grid.png")

        with open(f"{sample_folder_dir}/metadata.jsonl", 'w') as f:
            f.write(json.dumps(metadata) + '\n')

        return
    
    if not store_separately and len(sample_imgs) > 1:
        grid = torchvision.utils.make_grid(sample_imgs, nrow=2)
        grid_np = grid.to(torch.float16).permute(1, 2, 0).mul_(255).cpu().numpy()

        os.makedirs(sample_folder_dir, exist_ok=True)
        grid_np = Image.fromarray(grid_np.astype(np.uint8))
        grid_np.save(save_path)
    else:
        # bs, 3, r, r
        sample_imgs_np = sample_imgs.mul_(255).cpu().numpy()
        num_imgs = sample_imgs_np.shape[0]
        os.makedirs(sample_folder_dir, exist_ok=True)
        for img_idx in range(num_imgs):
            cur_img = sample_imgs_np[img_idx]
            cur_img = cur_img.transpose(1, 2, 0).astype(np.uint8)
            cur_img_store = Image.fromarray(cur_img)
            cur_img_store.save(save_path)
            #print(f"Image {img_idx} saved.")

    with open(os.path.join(sample_folder_dir, "prompt.txt"), "w") as f:
        f.write("\n".join(prompts))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--precision", type=str, default="int4", choices=["int4", "bf16"], help="Which precision to use"
    )
    parser.add_argument(
        "-d", "--datasets", type=str, nargs="*", default=["MJHQ", "DCI"], help="The benchmark datasets to evaluate on."
    )
    parser.add_argument("-t", "--num-inference-steps", type=int, default=4, help="Number of inference steps")
    parser.add_argument("-g", "--guidance-scale", type=float, default=0, help="Guidance scale.")
    parser.add_argument("-o", "--output-root", type=str, default=None, help="Image output path")
    parser.add_argument("--load", type=str, help="additional checkpoints", default=None)
    parser.add_argument(
        "--chunk-step",
        type=int,
        default=1,
        help="You will generate images for the subset specified by [chunk-start::chunk-step].",
    )
    parser.add_argument(
        "--chunk-start",
        type=int,
        default=0,
        help="You will generate images for the subset specified by [chunk-start::chunk-step].",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="The path to HART model.",
        default="pretrained_models/HART-1024",
    )
    parser.add_argument(
        "--more_smooth",
        type=bool,
        help="Turn on for more visually smooth samples.",
        default=True,
    )
    parser.add_argument(
        "--text_model_path",
        type=str,
        help="The path to text model, we employ Qwen2-VL-1.5B-Instruct by default.",
        default="Qwen2-VL-1.5B-Instruct",
    )
    parser.add_argument(
        "--use_metric",
        type=str,
        help="metric to identify low frequency",
        default="MSE",
    )
    parser.add_argument(
        "--shield_model_path",
        type=str,
        help="The path to shield model, we employ ShieldGemma-2B by default.",
        default="pretrained_models/shieldgemma-2b",
    )
    parser.add_argument("--prompt", type=str, help="A single prompt.", default="")
    parser.add_argument("--compress_method", type=str, help="compress method", default="none")
    parser.add_argument("--compress_ratio", type=float, help="compress ratio", default=0.5)
    parser.add_argument("--prompt_list", type=list[str], default=[])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--local_window_size", type=int, default=4)
    parser.add_argument("--use_ema", type=bool, default=True)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--start_prune_stage", type=int, default=10)
    parser.add_argument("--max_token_length", type=int, default=300)
    parser.add_argument("--specific_mse_layer", type=int, default=16)
    parser.add_argument("--use_llm_system_prompt", type=bool, default=True)
    parser.add_argument(
        "--cfg", type=float, help="Classifier-free guidance scale.", default=16
    )
    parser.add_argument('--outdir', type=str, default='')
    parser.add_argument('--n_samples', type=int, default=10)
    parser.add_argument('--metadata_file', type=str, default='evaluation/image_reward/benchmark-prompts.json')
    parser.add_argument('--rewrite_prompt', type=int, default=0, choices=[0,1])
    known_args, _ = parser.parse_known_args()

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()

    # parse cfg
    #args.cfg = list(map(float, args.cfg.split(',')))
    #if len(args.cfg) == 1:
    #    args.cfg = args.cfg[0]
    
    with open(args.metadata_file) as fp:
        metadatas = json.load(fp)

    model = AutoModel.from_pretrained(args.model_path,device_map="cuda",torch_dtype=torch.float16)
    model = model
    model.eval()

    text_tokenizer = AutoTokenizer.from_pretrained(args.text_model_path)
    text_model = AutoModel.from_pretrained(args.text_model_path,device_map="auto",torch_dtype=torch.float16)
    text_model.eval()
    
    if args.use_ema:
        ema_model = model
        ema_model.load_state_dict(
            torch.load(os.path.join(args.model_path, "ema_model.bin")),strict=False
        )
        del model
        torch.cuda.empty_cache()
    
    save_metadatas = []
    all_times=[]
    pbar=tqdm((metadatas),ncols=150)
    for index, metadata in enumerate(pbar):
        seed_everything(args.seed,verbose=False)
        prompt_id = metadata['id']
        prompt = metadata['prompt']
        sample_path = os.path.join(args.outdir, prompt_id)
        os.makedirs(sample_path, exist_ok=True)
        #print(f"Prompt ({index: >3}/{len(metadatas)}): '{prompt}'")
        
        images = []
        for ii in range(args.n_samples):
                with torch.inference_mode():
                    with torch.autocast(
                        "cuda", enabled=True, dtype=torch.float16, cache_enabled=True
                    ):
                        (
                            context_tokens,
                            context_mask,
                            context_position_ids,
                            context_tensor,
                        ) = encode_prompts(
                            [prompt],
                            text_model,
                            text_tokenizer,
                            args.max_token_length,
                            llm_system_prompt,
                            args.use_llm_system_prompt,
                        )

                        infer_func = (
                            ema_model.autoregressive_infer_cfg
                            if args.use_ema
                            else model.autoregressive_infer_cfg
                        )
                        st=time.time()
                        output_imgs = infer_func(
                            B=context_tensor.size(0),
                            label_B=context_tensor,
                            cfg=args.cfg,
                            g_seed=args.seed+ii,
                            more_smooth=args.more_smooth,
                            context_position_ids=context_position_ids,
                            context_mask=context_mask,
                            compress_ratio=args.compress_ratio,
                            compress_method=args.compress_method,
                            start_prune_stage=args.start_prune_stage,
                            local_window_size=args.local_window_size,
                            specific_mse_layer=args.specific_mse_layer
                        )
                        ed=time.time()
                        images.append(output_imgs)
                        all_times.append(ed-st)
        
        os.makedirs(sample_path, exist_ok=True)
        metadata['gen_image_paths'] = []
        for i, image in enumerate(images):
            save_file_path = os.path.join(sample_path, f"{prompt_id}_{i}.jpg")
            save_images(
                    image.clone(), save_file_path, False, prompt, "imagereward", metadata=metadata
                )
            metadata['gen_image_paths'].append(save_file_path)
        #print(save_file_path)
        save_metadatas.append(metadata)
        with open(f"{args.outdir}/time_log.txt","w") as f:
            f.write(f"average time : {sum(all_times)/len(all_times):.2f} s.")

        save_metadata_file_path = os.path.join(args.outdir, "metadata.jsonl")
        with open(save_metadata_file_path, "w") as fp:
            json.dump(save_metadatas, fp)
        pbar.set_description(f'average var time : {sum(all_times)/len(all_times):.2f}s')
