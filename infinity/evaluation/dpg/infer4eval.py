import argparse
import os
import json
import torch
import time
import torchvision
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import traceback
import sys
sys.path.insert(1,"./")
from eval_tool.data import get_dataset
from pytorch_lightning import seed_everything
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates


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

def save_2x2_image(image_list, save_path):
    """
    拼接一个列表中的四个numpy数组成2x2图片并保存。

    Args:
        image_list (list): 包含4个numpy数组，每个可以通过cv2.imwrite保存为图片。
        save_path (str): 保存拼接后图片的路径。

    Raises:
        ValueError: 如果输入的list不是长度为4。
    """
    if len(image_list) != 4:
        raise ValueError("image_list必须包含4个numpy数组")

    # 检查每个图片的大小是否相同
    h, w, c = image_list[0].shape
    for img in image_list:
        if img.shape != (h, w, c):
            raise ValueError("所有图片的尺寸和通道数必须一致")

    # 拼接图片
    top_row = np.hstack((image_list[0], image_list[1]))
    bottom_row = np.hstack((image_list[2], image_list[3]))
    combined_image = np.vstack((top_row, bottom_row))

    # 保存图片
    cv2.imwrite(save_path, combined_image)

def main():
    parser = argparse.ArgumentParser()
    from tools.run_infinity import gen_one_img,add_common_arguments,load_tokenizer,load_visual_tokenizer,load_transformer
    add_common_arguments(parser)
    parser.add_argument('--prompt', type=str, default='a dog')
    parser.add_argument('--save_file', type=str, default='./tmp.jpg')
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("-o", "--output-root", type=str, default=None, help="Image output path")
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
        "-d", "--datasets", type=str, nargs="*", default=["DPG"], help="The benchmark datasets to evaluate on."
    )
    args = parser.parse_args()
    print(args)
    assert args.chunk_step > 0
    #assert 0 <= args.chunk_start < args.chunk_step

    # parse cfg
    
    tau = args.tau
    cfg = args.cfg
    
    # load text encoder
    text_tokenizer, text_encoder = load_tokenizer(t5_path =args.text_encoder_ckpt)
    # load vae
    vae = load_visual_tokenizer(args)
    # load infinity
    infinity = load_transformer(vae, args)

    output_root = args.output_root
    if output_root is None:
        model_name=args.model_path.split("/")[-1]
        output_root = f"results/{model_name}/{args.precision}/"
    

    sparse_args={
        "compress_method": args.compress_method,
        "local_window_size": args.local_window_size,
        "compress_ratio": args.compress_ratio,
        "start_prune_stage":args.start_prune_stage,
        "specific_mse_layer":args.specific_mse_layer,
        "beta":args.beta
    }
    var_alltimes=[]
    for dataset_name in args.datasets:
        seed_everything(args.seed)
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
        pbar=tqdm(dataset,ncols=100)
        for idx,row in enumerate(pbar):
            try:
                torch.cuda.empty_cache()
                filename = row["filename"]
                if args.resume and f"{filename}.png" in exist_filenames:
                    continue
                prompt = row["prompt"]
                metadata = row.get("metadata",None)
                images=[]
                with torch.no_grad():
                    h_div_w_template = 1.000
                    scale_schedule = dynamic_resolution_h_w[h_div_w_template][args.pn]['scales']
                    scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]
                    tgt_h, tgt_w = dynamic_resolution_h_w[h_div_w_template][args.pn]['pixel']
                    for i in range(args.repeat):
                        image,infinity_time = gen_one_img(infinity, vae, text_tokenizer, text_encoder, prompt, tau_list=tau, cfg_sc=3, cfg_list=float(cfg), scale_schedule=scale_schedule, cfg_insertion_layer=[args.cfg_insertion_layer], vae_type=args.vae_type,sparse_args=sparse_args)
                        images.append(image)
                var_alltimes.append(infinity_time)
                avg_time=sum(var_alltimes)/len(var_alltimes)
                pbar.set_description(f"average infinity time : {avg_time:.2f}s")
                filename=filename.split(".")[0]
                save_2x2_image([i.cpu().numpy() for i in images],os.path.join(output_dirname, f"{filename}.png"))
                with open(os.path.join(output_dirname, f"../{dataset_name}_time_log.txt"),"w") as f:
                    f.write(f"average var time : {avg_time:.2f}s\n")
            except Exception as e:
                traceback.print_exc()
                print(e)
                exit()


if __name__ == "__main__":
    main()
