import argparse
import os
import json
import torch
import time
import torchvision
from PIL import Image
from tqdm import tqdm
import traceback
from eval_tool.data import get_dataset
from eval_tool.utils import hash_str_to_int
import numpy as np
from transformers import (
    AutoModel,
    AutoTokenizer,
)
import sys
sys.path.insert(0,".")
from hart.utils import encode_prompts, llm_system_prompt
from hart.modules.models.transformer import HARTForT2I

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
        "-d", "--datasets", type=str, nargs="*", default=["DPG"], help="The benchmark datasets to evaluate on."
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
    known_args, _ = parser.parse_known_args()

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    print(args)
    assert args.chunk_step > 0
    #assert 0 <= args.chunk_start < args.chunk_step

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

    if args.load is not None:
        ema_model.load_state_dict(torch.load(args.load),strict=False)
        

    output_root = args.output_root
    if output_root is None:
        model_name=args.model_path.split("/")[-1]
        output_root = f"results/{model_name}/{args.precision}/"
    var_alltimes=[]
    for dataset_name in args.datasets:
        output_dirname = os.path.join(output_root, dataset_name)
        os.makedirs(output_dirname, exist_ok=True)
        if args.resume:
            exist_filenames=os.listdir(output_dirname)
        if dataset_name == "DPG":
            from cus_datasets.DPG import DPGDataset
            dataset = DPGDataset("./cus_datasets/dpg_bench/prompts")
            args.repeat=4
        elif dataset_name == "GENEVAL":
            from cus_datasets.GENEVAL import GENEVALDataset
            dataset = GENEVALDataset("./cus_datasets/geneval/prompts/evaluation_metadata.jsonl")
            args.repeat=4
        pbar=tqdm(dataset,ncols=150)
        for idx,row in enumerate(pbar):
            try:
                torch.cuda.empty_cache()
                filename = row["filename"]
                if args.resume and f"{filename}.png" in exist_filenames:
                    continue
                prompt = row["prompt"]
                metadata = row.get("metadata",None)
                seed = hash_str_to_int(filename)
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
                        output_imgs=[]
                        for i in range(args.repeat):
                            output_img = infer_func(
                                B=context_tensor.size(0),
                                label_B=context_tensor,
                                cfg=args.cfg,
                                g_seed=seed+i,
                                more_smooth=args.more_smooth,
                                context_position_ids=context_position_ids,
                                context_mask=context_mask,
                                compress_ratio=args.compress_ratio,
                                compress_method=args.compress_method,
                                start_prune_stage=args.start_prune_stage,
                                local_window_size=args.local_window_size,
                                specific_mse_layer=args.specific_mse_layer
                            )
                            output_imgs.append(output_img)
                        output_imgs=torch.cat(output_imgs)
                        ed=time.time()
                        var_alltimes.append((ed-st)/args.repeat)
                        avg_time=sum(var_alltimes)/len(var_alltimes)
                        pbar.set_description(f"average var time : {avg_time:.2f}s")
                    filename=filename.split(".")[0]
                    save_images(
                            output_imgs.clone(), os.path.join(output_dirname, f"{filename}.png"), False, prompt, dataset_name, metadata=metadata
                        )
                    with open(os.path.join(output_dirname, f"../{dataset_name}_time_log.txt"),"w") as f:
                        f.write(f"average var time : {avg_time:.2f}s\n")
            except Exception as e:
                traceback.print_exc()
                print(e)
                exit()


if __name__ == "__main__":
    main()
