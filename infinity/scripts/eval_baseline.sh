#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

infer_eval_image_reward() {
    #${pip_ext} install image-reward pytorch_lightning
    #${pip_ext} install -U timm diffusers
    #${pip_ext} install openai==1.34.0 
    #${pip_ext} install httpx==0.20.0 

    # step 1, infer images
    ${python_ext} evaluation/image_reward/infer4eval.py \
    --cfg ${cfg} \
    --tau ${tau} \
    --pn ${pn} \
    --model_path ${infinity_model_path} \
    --vae_type ${vae_type} \
    --vae_path ${vae_path} \
    --add_lvl_embeding_only_first_block ${add_lvl_embeding_only_first_block} \
    --use_bit_label ${use_bit_label} \
    --model_type ${model_type} \
    --rope2d_each_sa_layer ${rope2d_each_sa_layer} \
    --rope2d_normalized_by_hw ${rope2d_normalized_by_hw} \
    --use_scale_schedule_embedding ${use_scale_schedule_embedding} \
    --cfg ${cfg} \
    --tau ${tau} \
    --checkpoint_type ${checkpoint_type} \
    --text_encoder_ckpt ${text_encoder_ckpt} \
    --text_channels ${text_channels} \
    --apply_spatial_patchify ${apply_spatial_patchify} \
    --cfg_insertion_layer ${cfg_insertion_layer} \
    --compress_method ${compress_method} \
    --outdir  ${out_dir}

    # step 2, compute image reward
    #${pip_ext} install diffusers==0.16.0
    #${pip_ext} install git+https://github.com/openai/CLIP.git ftfy
    ${python_ext} evaluation/image_reward/cal_imagereward.py \
    --meta_file ${out_dir}/metadata.jsonl
}

infer_eval_hpsv21() {
    #${pip_ext} install hpsv2
    #${pip_ext}install -U diffusers
    ##sudo apt install python3-tk
    #wget https://dl.fbaipublicfiles.com/mmf/clip/bpe_simple_vocab_16e6.txt.gz
    #mv bpe_simple_vocab_16e6.txt.gz /home/tiger/.local/lib/python3.9/site-packages/hpsv2/src/open_clip

    mkdir -p ${out_dir}
    ${python_ext} evaluation/hpsv2/eval_hpsv2.py \
    --cfg ${cfg} \
    --tau ${tau} \
    --pn ${pn} \
    --model_path ${infinity_model_path} \
    --vae_type ${vae_type} \
    --vae_path ${vae_path} \
    --add_lvl_embeding_only_first_block ${add_lvl_embeding_only_first_block} \
    --use_bit_label ${use_bit_label} \
    --model_type ${model_type} \
    --rope2d_each_sa_layer ${rope2d_each_sa_layer} \
    --rope2d_normalized_by_hw ${rope2d_normalized_by_hw} \
    --use_scale_schedule_embedding ${use_scale_schedule_embedding} \
    --cfg ${cfg} \
    --tau ${tau} \
    --checkpoint_type ${checkpoint_type} \
    --text_encoder_ckpt ${text_encoder_ckpt} \
    --text_channels ${text_channels} \
    --apply_spatial_patchify ${apply_spatial_patchify} \
    --cfg_insertion_layer ${cfg_insertion_layer} \
    --compress_method ${compress_method} \
    --outdir ${out_dir}/images | tee ${out_dir}/log.txt
}

test_gen_eval() {
    #${pip_ext} install -U openmim
    #mim install mmengine mmcv-full==1.7.2
    #${pip_ext} install mmdet==2.28.2 pytorch_lightning clip_benchmark open-clip-torch==2.20.0
    #${pip_ext} install -U diffusers
    ##sudo apt install libgl1
    #${pip_ext} install openai
    #${pip_ext} install httpx==0.20.0

    # run inference
    ${python_ext} evaluation/gen_eval/infer4eval.py \
    --cfg ${cfg} \
    --tau ${tau} \
    --pn ${pn} \
    --model_path ${infinity_model_path} \
    --vae_type ${vae_type} \
    --vae_path ${vae_path} \
    --add_lvl_embeding_only_first_block ${add_lvl_embeding_only_first_block} \
    --use_bit_label ${use_bit_label} \
    --model_type ${model_type} \
    --rope2d_each_sa_layer ${rope2d_each_sa_layer} \
    --rope2d_normalized_by_hw ${rope2d_normalized_by_hw} \
    --use_scale_schedule_embedding ${use_scale_schedule_embedding} \
    --cfg ${cfg} \
    --tau ${tau} \
    --checkpoint_type ${checkpoint_type} \
    --text_encoder_ckpt ${text_encoder_ckpt} \
    --text_channels ${text_channels} \
    --apply_spatial_patchify ${apply_spatial_patchify} \
    --cfg_insertion_layer ${cfg_insertion_layer} \
    --outdir ${out_dir}/images \
    --compress_method ${compress_method} \
    --rewrite_prompt ${rewrite_prompt}

    # detect objects
    ${python_ext} evaluation/gen_eval/evaluate_images.py ${out_dir}/images \
    --outfile ${out_dir}/results/det.jsonl \
    --model-config evaluation/gen_eval/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py \
    --model-path weights/mask2former

    # accumulate results
    ${python_ext} evaluation/gen_eval/summary_scores.py ${out_dir}/results/det.jsonl > ${out_dir}/results/res.txt
    cat ${out_dir}/results/res.txt
}

test_dpg() {
    ${python_ext} evaluation/dpg/infer4eval.py \
        --cfg ${cfg} \
        --tau ${tau} \
        --pn ${pn} \
        --model_path ${infinity_model_path} \
        --vae_type ${vae_type} \
        --vae_path ${vae_path} \
        --add_lvl_embeding_only_first_block ${add_lvl_embeding_only_first_block} \
        --use_bit_label ${use_bit_label} \
        --model_type ${model_type} \
        --rope2d_each_sa_layer ${rope2d_each_sa_layer} \
        --rope2d_normalized_by_hw ${rope2d_normalized_by_hw} \
        --use_scale_schedule_embedding ${use_scale_schedule_embedding} \
        --cfg ${cfg} \
        --tau ${tau} \
        --checkpoint_type ${checkpoint_type} \
        --text_encoder_ckpt ${text_encoder_ckpt} \
        --text_channels ${text_channels} \
        --apply_spatial_patchify ${apply_spatial_patchify} \
        --save_file tmp.jpg \
        --datasets DPG \
        --start_prune_stage 9 \
        --compress_method ${compress_method} --test_speed \
        -o ${out_dir}
    
    cd cus_datasets
    bash dpg_bench/dist_eval.sh ../${out_dir}/DPG 1024
    cd ..
}

python_ext=python3
pip_ext=pip3

# set arguments for inference
pn=1M
model_type=infinity_2b
use_scale_schedule_embedding=0
use_bit_label=1
checkpoint_type='torch'
infinity_model_path=weights/infinity_2b_reg.pth
vae_type=32
vae_path=weights/infinity_vae_d32_reg.pth
cfg=4
tau=1
rope2d_normalized_by_hw=2
add_lvl_embeding_only_first_block=1
rope2d_each_sa_layer=1
text_encoder_ckpt=google/flan-t5-xl
text_channels=2048
apply_spatial_patchify=0
cfg_insertion_layer=0
sub_fix=cfg${cfg}_tau${tau}_cfg_insertion_layer${cfg_insertion_layer}

compress_methods=("none")
for compress_method in "${compress_methods[@]}"; do
out_dir_root=outputs/baseline/infinity_2b_evaluation
# HPS v2.1
echo "========== Running HPS v2.1 Evaluation =========="
out_dir=${out_dir_root}/hpsv21_${sub_fix}
infer_eval_hpsv21

# ImageReward
echo "========== Running ImageReward Evaluation =========="
out_dir=${out_dir_root}/image_reward_${sub_fix}
infer_eval_image_reward

# DPG-Bench
echo "========== Running DPG-Bench Evaluation =========="
out_dir=${out_dir_root}/dpg_${sub_fix}
test_dpg

# GenEval
echo "========== Running GenEval Evaluation =========="
out_dir=${out_dir_root}/geneval_${sub_fix}
rewrite_prompt=1
test_gen_eval

done