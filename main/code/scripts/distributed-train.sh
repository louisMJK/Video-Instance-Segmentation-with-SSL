torchrun --nproc_per_node=4 ../train.py \
    --predictor-dir '/scratch/yl10745/dl/gsta/model_best.pth' \
    --fcn-dir '/scratch/yl10745/dl/segmentation/output/fcn_resnet50_jaccard_0.1826/fcn_resnet50_best.pth' \
    --freeze-backbone --freeze-fcn-head \
    --optim adam \
    --lr-base 1e-4 \
    --sched cosine \
    -b 4 --workers 16 \
    --epochs 1 \
