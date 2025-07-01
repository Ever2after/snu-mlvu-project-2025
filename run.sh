export CUDA_VISIBLE_DEVICES=0



export TOKENIZERS_PARALLELISM=false

python3 inference.py \
    --dataDir ./dataset_gen/scene \
    --outputDir ./output_no_context \
    --dataset full1 \
    --datasetPath ./output_no_context/full1/qwen2.5-vl-3b-mlvu-2.json \
    --model qwen2.5-vl-3b \
    --modelPath /mnt/data/jusang/checkpoints/qwen2.5-vl-3b-mlvu-2 \
    --max_new_tokens 1024 \
    --temperature 0.1 \
    --top_p 0.9 \
    --fps 10 \
    --max_frames 50

#  --datasetPath ./data/qwen2.5-vl/concat/qa_test.json \
# --refer_context  \
    # --gen_context \
