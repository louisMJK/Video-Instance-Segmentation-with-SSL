python ../train.py \
    --model resnet50 --layers-unfreeze 1 --fine-tune 3 \
    --img-size 384 --img-val-resize 512 \
    --optim adam \
    --sched step --step-size 2 --lr-decay 0.9 \
    --epochs 30 \

