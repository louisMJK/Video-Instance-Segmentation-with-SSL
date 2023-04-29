torchrun --nproc_per_node=4 ../train.py \
    --lr-base 1e-5 \
    --optim adam \
    -b 16 --workers 16 \
    --epochs 3 \
    --hid-S 64\
    --hid-T 512\
    --data-dir '../../Dataset_Student/'\
    --best-model 'best_model.pth'\
    --amp
