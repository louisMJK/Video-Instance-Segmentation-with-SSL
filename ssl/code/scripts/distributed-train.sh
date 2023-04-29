torchrun --nproc_per_node=4 ../train.py \
    --backbone resnet50 \
    --use-trained --model-dir '../../output/backbone-resnet50-0.9167/byol_best.pth' \
    --optim sgd \
    --sched exp --lr-decay 0.9 \
    --lr-base 1e-3 \
    -b 64 --workers 16 \
    --epochs 20 \
