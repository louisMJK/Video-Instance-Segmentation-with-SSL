torchrun --nproc_per_node=4 ../train.py \
    --backbone resnet50 \
    --use-trained --model-dir '../../output/backbone-0.9801/byol_best.pth' \
    --optim lars \
    --sched cosine \
    --lr-base 5e-3 \
    -b 64 --workers 16 \
    --epochs 20 \
