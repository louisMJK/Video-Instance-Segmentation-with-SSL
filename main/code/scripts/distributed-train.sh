torchrun --nproc_per_node=4 ../train.py \
    --predictor-dir '' \
    --fcn-dir '' \
    --freeze-backbone --freeze-fcn-head \
    --optim adam \
    --lr-base 1e-4 \
    --sched cosine \
    -b 8 --workers 16 \
    --epochs 1 \
