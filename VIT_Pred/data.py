from torch.utils.data import Dataset
import os
from PIL import Image, ImageFile
import torchvision.transforms as T
import torch
ImageFile.LOAD_TRUNCATED_IMAGES = True

class UnlabledtrainpredViT(Dataset):
    def __init__(self, root='../../dataset/unlabeled/', transform=T.Compose([T.ToTensor(), T.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])])):
        self.root = root
        self.transform = transform
        self.vid_list = sorted([f for f in os.listdir(root) if f.startswith('video')])
        self.img_list = ['image_' + str(i) + '.png' for i in range(11)]
        self.target_img = 'image_21.png' # 21 is the target image
        self.data_dir = root
                    
    def __len__(self):
        return len(self.vid_list)

    def __getitem__(self, idx):
        video = self.vid_list[idx]
        imgs = []
        for index in self.img_list:
            img_path  = os.path.join(self.data_dir, video, index)
            img = Image.open(img_path).convert("RGB")
            if self.transform is not None:
                img = self.transform(img)
            imgs.append(img)
        target_img_path = os.path.join(self.data_dir, video, self.target_img)
        target_img = Image.open(target_img_path).convert("RGB")
        if self.transform is not None:
            target_img = self.transform(target_img)
        return torch.stack(imgs), target_img


