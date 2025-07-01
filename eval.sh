export CUDA_VISIBLE_DEVICES=0

python3 evaluate.py \
    --resultPath ./output_no_context/full1/gpt-4o.json \
    --name gpt-4o-full1 \
    --metrics multi-choice \
    --outputDir ./eval \
    --detailed True
