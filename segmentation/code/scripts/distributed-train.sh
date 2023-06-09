torchrun --nproc_per_node=4 ../train.py \
    --model fcn_resnet50 \
    --freeze \
    --lr-base 2e-2 \
    -b 64 --workers 16 \
    --epochs 40 \
