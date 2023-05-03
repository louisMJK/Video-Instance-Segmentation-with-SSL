torchrun --nproc_per_node=4 ../train2.py \
    --lr-base 1e-5 \
    --optim adam \
    -b 4 --workers 16 \
    --epochs 30 \
    --hid-S 64\
    --hid-T 512\
    --data-dir '../../Dataset_Student/'\
    --best-model '../model_best.pth'\
    --drop-path 0.1 \
    --amp \
    --next-n-frame 8\