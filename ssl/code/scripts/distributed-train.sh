torchrun --nproc_per_node=4 ../train.py \
    --backbone resnet50 \
    --optim lars \
    --sched cosine \
    --lr-base 1e-2 \
    -b 64 --workers 16 \
    --epochs 100 \
