torchrun --nproc_per_node=4 ../train.py \
    --lr-base 1e-3 \
    --optim adam \
    -b 4 --workers 16 \
    --epochs 10 \
    --hid-S 64\
    --hid-T 512\
    --data-dir '../../Dataset_Student/'\
    --best-model 'best_model.pth'
