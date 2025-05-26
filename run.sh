export CUDA_VISIBLE_DEVICES=0

export TOKENIZERS_PARALLELISM=false

python3 inference.py \
    --dataDir ./data \
    --outputDir ./output \
    --dataset test2 \
    --model internvideo2_5_8b \
    --modelPath None \
    --context_exist True \
    --max_new_tokens 64 \
    --temperature 0.1 \
    --top_p 0.9 \
    --fps 30 \
    --max_frames 8 \
