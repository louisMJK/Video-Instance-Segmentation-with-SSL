# Compute Node
ssh burst
cd /scratch/yl10745/

# Copy code ...

# CPU
srun --partition=interactive --account csci_ga_2572_2023sp_18 \
    --time=3:00:00 \
    --pty /bin/bash


# GPU
# 1 GPU
srun --partition=n1s8-v100-1 --gres=gpu:1 --account csci_ga_2572_2023sp_18 \
    --time=1:00:00 \
    --cpus-per-task=8 \
    --pty /bin/bash
source /etc/profile
cd /scratch/yl10745/


# 2 GPU
srun --partition=n1s16-v100-2 --gres=gpu:2 --account csci_ga_2572_2023sp_18 \
    --time=00:10:00 \
    --pty /bin/bash

# 4 GPU
srun --partition=n1c24m128-v100-4 --gres=gpu:4 --account csci_ga_2572_2023sp_18 --time=00:10:00 --pty /bin/bash


source /etc/profile
cd /scratch/yl10745/


# Singularity
# singularity exec --nv \
#     --bind /scratch \
#     --overlay /scratch/yl10745/src-kaggle/deep-learning.ext3:rw \
#     /scratch/yl10745/src-kaggle/cuda11.7.99-cudnn8.5-devel-ubuntu22.04.2.sif \
#     /bin/bash

singularity exec --nv \
    --bind /scratch \
    --overlay /scratch/yl10745/src/deep-learning.ext3:rw \
    /scratch/yl10745/src/cuda11.7.99-cudnn8.5-devel-ubuntu22.04.2.sif \
    /bin/bash

singularity exec --nv \
    --bind /scratch \
    --overlay /scratch/yb970/dl/my_pytorch.ext3:rw \
    /scratch/yb970/dl/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
    /bin/bash

#singluarity yb970
singularity exec --overlay my_pytorch.ext3:rw /scratch/yb970/dl/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif /bin/bash

source /ext3/env.sh
source /etc/profile


# Copy code from /scratch/yl10745
scp -rp greene-dtn:/scratch/yl10745/dl-video-instance-segmentation/ssl/code .


scp -rp greene-dtn:/scratch/yb970/dl/frame_pred .

scp -rp greene-dtn:/scratch/yl10745/dl-video-instance-segmentation/segmentation/code .


# Copy outputs to login node
scp -rp ./output/  greene-dtn:/scratch/yl10745/dl-video-instance-segmentation/ssl/output  


# unzip -o dataset.zip -x '**/.*' -x '**/__MACOSX'

