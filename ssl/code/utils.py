from torch.utils.data import Dataset
import os
from PIL import Image
import copy
from torch import nn
from lightly.models.modules import BYOLPredictionHead, BYOLProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.utils.scheduler import cosine_schedule


class UnlabeledDataset(Dataset):
    def __init__(self, root='../../dataset/unlabeled/', transform=None):
        self.root = root
        self.transform = transform
        self.vid_list = sorted(os.listdir(root))
        # self.img_list = ['image_' + str(i) + '.png' for i in range(22)]
        self.data_dir = root
        self.image_list = []
        for subdir in self.vid_list:
            subdir_path = os.path.join(self.data_dir, subdir)
            for file in os.listdir(subdir_path):
                    self.image_list.append(os.path.join(subdir_path, file))
                    
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, 0
    



class BYOL(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        self.backbone = backbone
        self.projection_head = BYOLProjectionHead(512, 1024, 256)
        self.prediction_head = BYOLPredictionHead(256, 1024, 256)

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

    def forward(self, x):
        y = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(y)
        p = self.prediction_head(z)
        return p

    def forward_momentum(self, x):
        y = self.backbone_momentum(x).flatten(start_dim=1)
        z = self.projection_head_momentum(y)
        z = z.detach()
        return z
    
shapes = ["cube", "sphere", "cylinder"]
materials = ["metal", "rubber"]
colors = ["gray", "red", "blue", "green", "brown", "cyan", "purple", "yellow"]

def get_id(the_object):
    color = the_object['color']
    material = the_object['material']
    shape = the_object['shape']
    
    c_id = colors.index(color)
    m_id = materials.index(material)
    s_id = shapes.index(shape)
    
    obj_id = s_id * 16 + m_id * 8 + c_id + 1
    
    return obj_id