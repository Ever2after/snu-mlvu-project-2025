export CUDA_VISIBLE_DEVICES=0

export TOKENIZERS_PARALLELISM=false

python3 inference.py \
    --dataDir ./data \
    --outputDir ./output \
    --dataset test \
    --model qwen2.5-vl-3b-sft \
    --modelPath /mnt/data/jusang/mlvu/qwen_checkpoints \
    --max_new_tokens 128 \
    --temperature 0.1 \
    --top_p 0.9 \
    --fps 30 \
    --max_frames 8 \
