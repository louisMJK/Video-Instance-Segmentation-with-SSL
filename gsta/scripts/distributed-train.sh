torchrun --nproc_per_node=4 ../train.py \
    --lr-base 1e-4 \
    --optim adam \
    -b 4 --workers 16 \
    --epochs 40 \
    --hid-S 64\
    --hid-T 512\
    --data-dir '../../dl/dataset/'\
    --amp
