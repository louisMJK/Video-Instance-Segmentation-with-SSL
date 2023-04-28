torchrun --nproc_per_node=4 ../train.py \
    --lr-base 1e-3 \
    --optim adam \
    -b 32 --workers 16 \
    --epochs 10 \
