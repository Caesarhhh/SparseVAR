import argparse
import os
from copy import deepcopy,copy
import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from hart.utils import default_prompts, encode_prompts, llm_system_prompt, safety_check
from eval_tool.data import get_dataset
from eval_tool.utils import get_pipeline, hash_str_to_int
import numpy as np
import matplotlib
matplotlib.use('agg') 
import matplotlib.pyplot as plt
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)
import cv2
from hart.modules.models.transformer import HARTForT2I
from hart.utils import default_prompts, encode_prompts, llm_system_prompt, safety_check
import torch
import random
from cus_datasets.prompt_dataset import PromptDataset
import numpy as np

def combine_images_with_titles(images, titles, output_path, spacing=20, font_path="tnr.ttf", font_size=80):
    """
    Combine images horizontally with titles below each image.

    Args:
        image_paths (list): List of image file paths.
        titles (list): List of titles corresponding to the images.
        output_path (str): Path to save the combined image.
        spacing (int): Space between images, default is 20 pixels.
        font_path (str): Path to the font file for titles.
        font_size (int): Font size for the titles.

    Returns:
        None
    """
    if len(images) != len(titles):
        raise ValueError("The number of image paths must match the number of titles.")

    # Get image sizes
    widths, heights = zip(*(img.size for img in images))

    # Set up font
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        raise IOError(f"Font file not found: {font_path}")

    # Calculate total size of the output image
    total_width = sum(widths) + spacing * (len(images) - 1)
    max_height = max(heights)

    # Estimate extra height for titles
    combined_image = Image.new("RGB", (1, 1))  # Temporary image to use for drawing
    draw = ImageDraw.Draw(combined_image)
    title_height = max(draw.textbbox((0, 0), title, font=font)[3] - draw.textbbox((0, 0), title, font=font)[1] for title in titles)

    # Create a new blank image with a white background
    combined_image = Image.new("RGB", (total_width, max_height + title_height + spacing), "white")

    # Paste images and add titles
    draw = ImageDraw.Draw(combined_image)
    x_offset = 0
    for i, (img, title) in enumerate(zip(images, titles)):
        combined_image.paste(img, (x_offset, 0))

        # Draw the title centered under the image
        text_bbox = draw.textbbox((0, 0), title, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_x = x_offset + (img.width - text_width) // 2
        text_y = max_height + spacing // 2
        draw.text((text_x, text_y), title, fill="black", font=font)

        # Update x_offset for the next image
        x_offset += img.width + spacing

    # Save the resulting image
    combined_image.save(output_path)

def set_seed(seed: int):
    """
    设置PyTorch、numpy和random的随机种子以确保实验可重复。
    """
    # Python random seed
    random.seed(seed)
    
    # Numpy seed
    np.random.seed(seed)
    
    # PyTorch seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 对所有GPU设置相同的种子
    torch.backends.cudnn.deterministic = True  # 确保每次卷积运算结果一致
    torch.backends.cudnn.benchmark = False  # 禁用benchmark模式，以确保复现性

    # 如果使用了torch.device("cuda")，还需要设置以下内容
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # print(f"Random seed set to {seed}")

def save_images(sample_imgs, save_path, store_separately, prompts):
    sample_folder_dir=os.path.dirname(save_path)
    if not store_separately and len(sample_imgs) > 1:
        grid = torchvision.utils.make_grid(sample_imgs, nrow=12)
        grid_np = grid.to(torch.float16).permute(1, 2, 0).mul_(255).cpu().numpy()

        os.makedirs(sample_folder_dir, exist_ok=True)
        grid_np = Image.fromarray(grid_np.astype(np.uint8))
        grid_np.save(os.path.join(sample_folder_dir, f"sample_images.png"))
        cur_img_store=grid_np
        print(f"Example images are saved to {sample_folder_dir}")
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
        f.write(prompts)

    return cur_img_store

def visualize_attention_map(attention_map: torch.Tensor, save_path: str):
    """
    可视化二维 attention map 并保存为图像。
    
    参数：
    - attention_map: torch.Tensor, 2D Tensor, attention map，例如大小为 (H, W)。
    - save_path: str, 图像保存路径。
    """
    if len(attention_map.shape) != 2:
        raise ValueError("Attention map 应该是一个二维张量 (H, W)")

    # 将 attention map 转换为 numpy 数组以便于 matplotlib 可视化
    attention_map = attention_map.detach().cpu().numpy()

    # 创建热力图并保存图像（不需要显示）
    plt.imsave(save_path, attention_map, cmap='viridis')

    image=Image.open(save_path)

    resized_image = image.resize((1024, 1024))

    return resized_image

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--precision", type=str, default="int4", choices=["int4", "bf16"], help="Which precision to use"
    )
    parser.add_argument(
        "-d", "--dataset", type=str, default="MJHQ", help="The benchmark datasets to evaluate on."
    )
    parser.add_argument("-t", "--num-inference-steps", type=int, default=4, help="Number of inference steps")
    parser.add_argument("-g", "--guidance-scale", type=float, default=0, help="Guidance scale.")
    parser.add_argument("-o", "--output_root", type=str, default=None, help="Image output path")
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
    parser.add_argument("--end_compress_ratio", type=float, help="compress ratio", default=0.8)
    parser.add_argument("--learning_rate", type=float, help="learning_rate", default=0.0001)
    parser.add_argument("--prompt_list", type=list[str], default=[])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--number_samples", type=int, default=1000)
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--use_ema", type=bool, default=True)
    parser.add_argument("--split_frequency", action="store_true", default=False)
    parser.add_argument("--start_prune_stage", type=int, default=9)
    parser.add_argument("--flag_start_prune_stage", type=int, default=9)
    parser.add_argument("--max_token_length", type=int, default=300)
    parser.add_argument("--use_llm_system_prompt", type=bool, default=True)
    parser.add_argument("--load", type=str, help="additional checkpoints", default=None)
    parser.add_argument(
        "--cfg", type=float, help="Classifier-free guidance scale.", default=4.5
    )
    known_args, _ = parser.parse_known_args()

    args = parser.parse_args()
    return args

def save_grad_params(model: torch.nn.Module, save_path: str) -> None:
    """
    将模型中所有 requires_grad=True 的参数保存到指定路径。

    Args:
        model (torch.nn.Module): 目标模型，通常是一个 nn.Module 的子类。
        save_path (str): 保存模型参数的路径。
    """
    # 筛选出 requires_grad=True 的参数
    grad_params = {name: param for name, param in model.named_parameters() if param.requires_grad}
    
    # 保存到指定路径
    torch.save(grad_params, save_path)

def reset_model_parameters(model):
    for layer in model.modules():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

def main():
    args = get_args()
    print(args)
    assert args.chunk_step > 0
    assert 0 <= args.chunk_start < args.chunk_step

    model = AutoModel.from_pretrained(args.model_path,device_map="cuda",torch_dtype=torch.float16)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-7)
    criterion = torch.nn.KLDivLoss(reduction='batchmean') 
    criterion2 = torch.nn.MSELoss("mean")
    model = model

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

    teacher_model=deepcopy(ema_model)
    ema_model.reset_addtional_codebooks()
    ema_model.init_diffloss_low()
    teacher_model.init_diffloss_low()
    teacher_model.eval()
    ema_model.eval()
    #reset_model_parameters(ema_model.diffloss)
    for name,param in teacher_model.named_parameters():
        param.requires_grad = False
    for name,param in ema_model.named_parameters():
        if "diffloss" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    output_root = args.output_root
    if output_root is None:
        model_name=args.model_path.split("/")[-1]
        output_root = f"results/{model_name}/{args.precision}/"
    os.makedirs(output_root,exist_ok=True)
    dataset_name=args.dataset
    output_dirname = os.path.join(output_root, dataset_name)
    os.makedirs(output_dirname, exist_ok=True)
    #dataset = get_dataset(name=dataset_name)
    origin_flag_start_prune_stage=args.flag_start_prune_stage
    while(args.compress_ratio<=args.end_compress_ratio+0.001):
        args.flag_start_prune_stage=origin_flag_start_prune_stage
        while(args.flag_start_prune_stage>=args.start_prune_stage):
            for epoch_i in range(args.epoch):
                idx=0
                if dataset_name=="prompt":
                    dataset = PromptDataset(json_path="cus_datasets/generated_prompts500.json",number_samples=args.number_samples)
                else:
                    dataset = get_dataset(name=dataset_name)
                pbar=tqdm(dataset,ncols=150)
                for row in pbar:
                    idx+=1
                    prompt = row["prompt"]
                    #prompt = "a man eats an apple"
                    #prompt="A colossal jellyfish-shaped hot air balloon floating over a serene sunflower field at dawn, glowing softly with bioluminescent lights, intricate lace-like patterns illuminating its translucent body, the horizon painted in soft pastel hues, whimsical, magical realism, ultra-detailed, cinematic lighting, photographed with a Leica M10-P, 90mm lens, f2.0."
                    seed = 1
                    with torch.inference_mode():
                        with torch.autocast(
                            "cuda", enabled=True, dtype=torch.float16, cache_enabled=False
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
                            output_imgs_student,last_feature_students,h_BChw_final_student,f_hat_discrete_student,last_stage_cond_student,\
                                hr_idx,mse_difference,last_train_args,last_stage_cond_cache,hr_idxs,idx_Bl,origin_logits_BlV,f_hat_wo_last,last_stage_x,f_hats = infer_func(
                                B=context_tensor.size(0),
                                label_B=context_tensor,
                                cfg=args.cfg,
                                g_seed=seed,
                                more_smooth=args.more_smooth,
                                context_position_ids=context_position_ids,
                                context_mask=context_mask,
                                compress_ratio=args.compress_ratio,
                                compress_method=args.compress_method,
                                return_features=True,
                                split_frequency=args.split_frequency,
                                start_prune_stage=args.flag_start_prune_stage
                            )
                            infer_func = (
                                teacher_model.autoregressive_infer_cfg
                                if args.use_ema
                                else model.autoregressive_infer_cfg
                            )
                            output_imgs_teacher,last_feature_teachers,h_BChw_final_teacher,f_hat_discrete_teacher,last_stage_cond_teacher,_,_,_,_,_,_,_,_,_,_ = infer_func(
                                B=context_tensor.size(0),
                                label_B=context_tensor,
                                cfg=args.cfg,
                                g_seed=seed,
                                more_smooth=args.more_smooth,
                                context_position_ids=context_position_ids,
                                context_mask=context_mask,
                                compress_ratio=args.compress_ratio,
                                compress_method="none",
                                return_features=True,
                                start_prune_stage=args.flag_start_prune_stage
                            )
                    B=last_feature_students[0].shape[0]
                    last_layer_gt=(output_imgs_teacher-f_hat_wo_last).permute(0,2,3,1).reshape(1,-1,32)
                    _,diff_loss,_=ema_model(last_train_args["context"][0:1],torch.cat(last_train_args["x_BLCv_wo_first_l"],dim=1),\
                        last_train_args["context_position_ids"][0:1],last_train_args["context_mask"][0:1],last_layer_gt,\
                            f_hat_wo_last.permute(0,2,3,1).reshape(1,-1,32),hr_idx=hr_idx,\
                                last_stage_cond_cache=last_stage_cond_cache,hr_idxs=hr_idxs,idx_Bl=idx_Bl,split_frequency=args.split_frequency)

                    #if random.random()<0.1:
                    #    diff_loss=ema_model.cal_loss_wlogits(origin_logits_BlV[B:],last_stage_x[B:],last_layer_gt,hr_idxs,last_stage_cond_cache[B:])
                    #else:
                    #    diff_loss=ema_model.cal_loss_wlogits(origin_logits_BlV[:B],last_stage_x[:B],last_layer_gt,hr_idxs,last_stage_cond_cache[:B])
                    loss=diff_loss
                    loss.backward()
                    if idx%1==0:
                        torch.cuda.empty_cache()
                        optimizer.step()
                        optimizer.zero_grad()
                        pbar.set_description(f"epoch={epoch_i}/{args.epoch} compress_ratio={args.compress_ratio} flag_start_prune_stage={args.flag_start_prune_stage} loss = {loss.item()}")

                    with torch.no_grad():
                        output_imgs_student=ema_model.vae_proxy[0].fhat_to_img(output_imgs_student.half()).add_(1).mul_(0.5)
                        output_imgs_teacher=teacher_model.vae_proxy[0].fhat_to_img(output_imgs_teacher.half()).add_(1).mul_(0.5)
                    if idx%8==0:
                        with torch.no_grad():
                            with torch.autocast(
                                "cuda", enabled=True, dtype=torch.float16, cache_enabled=False
                            ):
                                infer_func = (
                                    teacher_model.autoregressive_infer_cfg
                                    if args.use_ema
                                    else model.autoregressive_infer_cfg
                                )
                                original_output_imgs_student,last_feature_students,h_BChw_final_student,f_hat_discrete_student,last_stage_cond_student,hr_idx,_,last_train_args,_,_,_,_,_,_,_ = infer_func(
                                    B=context_tensor.size(0),
                                    label_B=context_tensor,
                                    cfg=args.cfg,
                                    g_seed=seed,
                                    more_smooth=args.more_smooth,
                                    context_position_ids=context_position_ids,
                                    context_mask=context_mask,
                                    compress_ratio=args.compress_ratio,
                                    compress_method=args.compress_method,
                                    return_features=True,
                                    split_frequency=args.split_frequency,
                                    start_prune_stage=args.flag_start_prune_stage
                                )

                                original_training_free_output_imgs_student,_,h_BChw_final_student,f_hat_discrete_student,last_stage_cond_student,hr_idx,_,last_train_args,_,_,_,_,_,_,_ = infer_func(
                                    B=context_tensor.size(0),
                                    label_B=context_tensor,
                                    cfg=args.cfg,
                                    g_seed=seed,
                                    more_smooth=args.more_smooth,
                                    context_position_ids=context_position_ids,
                                    context_mask=context_mask,
                                    compress_ratio=args.compress_ratio,
                                    compress_method=args.compress_method,
                                    return_features=True,
                                    start_prune_stage=args.flag_start_prune_stage
                                )
                                original_output_imgs_student=teacher_model.vae_proxy[0].fhat_to_img(original_output_imgs_student.half()).add_(1).mul_(0.5)
                                original_training_free_output_imgs_student=teacher_model.vae_proxy[0].fhat_to_img(original_training_free_output_imgs_student.half()).add_(1).mul_(0.5)
                        train_teacher=save_images(
                                output_imgs_teacher.clone(), os.path.join("./", f"{output_root}/train_teacher.png"), True, prompt
                            )
                        original_student=save_images(
                                original_output_imgs_student.clone(), os.path.join("./", f"{output_root}/original_student.png"), True, prompt
                            )
                        original_training_free_student=save_images(
                                original_training_free_output_imgs_student.clone(), os.path.join("./", f"{output_root}/original_training_free_student.png"), True, prompt
                            )
                        train_student=save_images(
                            output_imgs_student.clone(), os.path.join("./", f"{output_root}/train_student.png"), True, prompt
                        )
                        if mse_difference is not None:
                            vis=visualize_attention_map(mse_difference[0].reshape(64,64),f"{output_root}/vis.png")
                            combine_images_with_titles(
                                [original_training_free_student,train_teacher,train_student,original_student,vis],
                                ["drop prev","teacher","student","keep prev","vis"],
                                f"{output_root}/combine.png"
                            )
                        else:
                            combine_images_with_titles(
                                [original_training_free_student,train_teacher,train_student,original_student],
                                ["drop prev","teacher","student","keep prev"],
                                f"{output_root}/combine.png"
                            )
                    if idx>args.number_samples:
                        break
                save_grad_params(ema_model,f"{output_root}/save_distill_{args.compress_ratio:.2f}_prune{args.flag_start_prune_stage}.pt")
            args.flag_start_prune_stage=args.flag_start_prune_stage-1
        args.compress_ratio=args.compress_ratio+0.05


if __name__ == "__main__":  
    main()
