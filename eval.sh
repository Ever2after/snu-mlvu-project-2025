export CUDA_VISIBLE_DEVICES=0

python3 evaluate.py \
    --resultDir ./output \
    --name test2_results \
    --metrics em,f1,rouge,bleu,meteor \
    --outputDir ./eval \
    --detailed True
