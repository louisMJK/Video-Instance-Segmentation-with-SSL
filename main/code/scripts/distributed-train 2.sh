torchrun --nproc_per_node=4 ../train.py \
    --use-fcn-head \
    --freeze-backbone --freeze-fcn \
    --lr-base 1e-3 \
    --optim adam \
    -b 16 --workers 16 \
    --epochs 10 \
