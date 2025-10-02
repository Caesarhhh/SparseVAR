#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

# evaluate HPSV2 dataset (baseline)
evaluate_hpsv2_baseline() {
    EXP="hpsv2/baseline"
    python evaluation/hpsv2/infer4eval.py --model_path weights/hart-0.7b-1024px/llm \
       --text_model_path mit-han-lab/Qwen2-VL-1.5B-Instruct \
       --shield_model_path google/shieldgemma-2b \
       -o ./outputs/${EXP} \
       --chunk-step 1 \
       --chunk-start 0 \
       --compress_method none \
       --outdir ./outputs/${EXP}
}

# evaluate GenEval dataset (baseline)
evaluate_geneval_baseline() {
    EXP="geneval/baseline"
    python evaluation/gen_eval/infer4eval.py --model_path weights/hart-0.7b-1024px/llm \
       --text_model_path mit-han-lab/Qwen2-VL-1.5B-Instruct \
       --shield_model_path google/shieldgemma-2b \
       -d GENEVAL \
       -o ./outputs/${EXP} \
       --chunk-step 1 \
       --chunk-start 0 \
       --compress_method none

    cd ./cus_datasets/geneval
    python evaluation/evaluate_images.py \
        ../../outputs/${EXP}/GENEVAL \
        --outfile "../../outputs/${EXP}/results.jsonl" \
        --model-path .
    python evaluation/summary_scores.py "../../outputs/${EXP}/results.jsonl"
    cd ../..
}

# evaluate DPG dataset (baseline)
evaluate_dpg_baseline() {
    EXP="dpg/baseline"
    python evaluation/gen_eval/infer4eval.py --model_path weights/hart-0.7b-1024px/llm \
       --text_model_path mit-han-lab/Qwen2-VL-1.5B-Instruct \
       --shield_model_path google/shieldgemma-2b \
       -d DPG \
       -o ./outputs/${EXP} \
       --chunk-step 1 \
       --chunk-start 0 \
       --compress_method none

    cd ./cus_datasets
    bash dpg_bench/dist_eval.sh ../outputs/${EXP}/DPG 1024
    cd ..
}

# evaluate ImageReward dataset (baseline)
evaluate_imagereward_baseline() {
    EXP="imagereward/baseline"
    python evaluation/image_reward/infer4eval.py --model_path weights/hart-0.7b-1024px/llm \
       --text_model_path mit-han-lab/Qwen2-VL-1.5B-Instruct \
       --shield_model_path google/shieldgemma-2b \
       -o ./outputs/${EXP} \
       --chunk-step 1 \
       --chunk-start 0 \
       --compress_method none \
       --outdir ./outputs/${EXP}

    python evaluation/image_reward/cal_imagereward.py \
          --meta_file ./outputs/${EXP}/metadata.jsonl
}

# Main execution for baseline
echo "Evaluating baseline (no compression)..."

echo "Evaluating HPSV2 baseline..."
evaluate_hpsv2_baseline

echo "Evaluating GenEval baseline..."
evaluate_geneval_baseline

echo "Evaluating DPG baseline..."
evaluate_dpg_baseline

echo "Evaluating ImageReward baseline..."
evaluate_imagereward_baseline

echo "All baseline evaluations completed!"