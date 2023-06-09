import torch
from torch.utils.data import Dataset

import os
import copy
from PIL import Image, ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True


class UnlabeledDataset(Dataset):
    def __init__(self, root='../../dataset/train/', transform=None):
        self.root = root
        self.transform = transform
        self.vid_list = sorted(os.listdir(root))
        self.img_list = ['image_' + str(i) + '.png' for i in range(22)]
    
    def __len__(self):
        return len(self.vid_list) * 22

    def __getitem__(self, idx):
        vid_idx = idx // 22
        img_idx = idx % 22
        # load image
        img_path = os.path.join(self.root, self.vid_list[vid_idx], self.img_list[img_idx])
        img = Image.open(img_path).convert("RGB")
        # transforms
        if self.transform is not None:
            img = self.transform(img)
        return img, 0


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    elif hasattr(args, "rank"):
        pass
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(f"| distributed init (rank {args.rank}): {args.dist_url}", flush=True)
    torch.distributed.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        pass



# shapes = ["cube", "sphere", "cylinder"]
# materials = ["metal", "rubber"]
# colors = ["gray", "red", "blue", "green", "brown", "cyan", "purple", "yellow"]

# def get_id(the_object):
#     color = the_object['color']
#     material = the_object['material']
#     shape = the_object['shape']
    
#     c_id = colors.index(color)
#     m_id = materials.index(material)
#     s_id = shapes.index(shape)
    
#     obj_id = s_id * 16 + m_id * 8 + c_id + 1
    
#     return obj_id
