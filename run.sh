export CUDA_VISIBLE_DEVICES=0

export TOKENIZERS_PARALLELISM=false

python3 inference.py \
    --dataDir ./data \
    --outputDir ./output \
    --dataset test \
    --model qwen2.5-vl-3b \
    --max_new_tokens 512 \
    --temperature 0.1 \
    --top_p 0.9
