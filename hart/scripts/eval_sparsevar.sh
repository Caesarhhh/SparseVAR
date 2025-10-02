#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

compress_ratios=(0.5 0.6 0.7)

# evaluate HPSV2 dataset
evaluate_hpsv2() {
    local compress_ratio=$1
    EXP="hpsv2/sparsevar${compress_ratio}"
    python evaluation/hpsv2/infer4eval.py --model_path weights/hart-0.7b-1024px/llm \
       --text_model_path mit-han-lab/Qwen2-VL-1.5B-Instruct \
       --shield_model_path google/shieldgemma-2b \
       -o ./outputs/${EXP} \
       --chunk-step 1 \
       --chunk-start 0 \
       --compress_ratio ${compress_ratio} \
       --compress_method sparsevar \
       --outdir ./outputs/${EXP}
}

# evaluate GenEval dataset
evaluate_geneval() {
    local compress_ratio=$1
    EXP="geneval/sparsevar${compress_ratio}"
    python evaluation/gen_eval/infer4eval.py --model_path weights/hart-0.7b-1024px/llm \
       --text_model_path mit-han-lab/Qwen2-VL-1.5B-Instruct \
       --shield_model_path google/shieldgemma-2b \
       -d GENEVAL \
       -o ./outputs/${EXP} \
       --chunk-step 1 \
       --chunk-start 0 \
       --compress_ratio ${compress_ratio} \
       --compress_method sparsevar

    cd ./cus_datasets/geneval
    python evaluation/evaluate_images.py \
        ../../outputs/${EXP}/GENEVAL \
        --outfile "../../outputs/${EXP}/results.jsonl" \
        --model-path .
    python evaluation/summary_scores.py "../../outputs/${EXP}/results.jsonl"
    cd ../..
}

# evaluate DPG dataset
evaluate_dpg() {
    local compress_ratio=$1
    EXP="dpg/sparsevar${compress_ratio}"
    python evaluation/gen_eval/infer4eval.py --model_path weights/hart-0.7b-1024px/llm \
       --text_model_path mit-han-lab/Qwen2-VL-1.5B-Instruct \
       --shield_model_path google/shieldgemma-2b \
       -d DPG \
       -o ./outputs/${EXP} \
       --chunk-step 1 \
       --chunk-start 0 \
       --compress_ratio $compress_ratio \
       --compress_method sparsevar

    cd ./cus_datasets
    bash dpg_bench/dist_eval.sh ../outputs/${EXP}/DPG 1024
    cd ..
}

# evaluate ImageReward dataset
evaluate_imagereward() {
    local compress_ratio=$1
    EXP="imagereward/sparsevar${compress_ratio}"
    python evaluation/image_reward/infer4eval.py --model_path weights/hart-0.7b-1024px/llm \
       --text_model_path mit-han-lab/Qwen2-VL-1.5B-Instruct \
       --shield_model_path google/shieldgemma-2b \
       -o ./outputs/${EXP} \
       --chunk-step 1 \
       --chunk-start 0 \
       --compress_ratio ${compress_ratio} \
       --compress_method sparsevar \
       --outdir ./outputs/${EXP}

    python evaluation/image_reward/cal_imagereward.py \
          --meta_file ./outputs/${EXP}/metadata.jsonl
}

# Main execution

for ratio in "${compress_ratios[@]}"; do
    echo "Evaluating with compression ratio: $ratio"
    
    echo "Evaluating HPSV2..."
    evaluate_hpsv2 $ratio
    
    echo "Evaluating GenEval..."
    evaluate_geneval $ratio
    
    echo "Evaluating DPG..."
    evaluate_dpg $ratio
    
    echo "Evaluating ImageReward..."
    evaluate_imagereward $ratio
    
    echo "Completed evaluation for ratio: $ratio"
    echo "----------------------------------------"
done

echo "All evaluations completed!"