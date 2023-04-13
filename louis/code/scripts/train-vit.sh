python ../train.py \
    --model vit_b_16_swag --layers-unfreeze 1 --fine-tune 2 \
    --img-size 384 --img-val-resize 384 \
    --optim adam \
    --sched exp --lr-decay 0.9 \
    --epochs 30 \

python ../train.py \
    --model vit_b_16_swag --layers-unfreeze 1 --fine-tune 2 \
    --img-size 384 --img-val-resize 384 \
    --optim adam \
    --sched step --step-size 10 --lr-decay 0.1 \
    --epochs 30 \
