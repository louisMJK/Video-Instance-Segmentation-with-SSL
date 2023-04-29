from torch.utils.data import Dataset
import os
from PIL import Image, ImageFile
from torchvision import transforms
import torch
ImageFile.LOAD_TRUNCATED_IMAGES = True

class UnlabledtrainpredSim(Dataset):
    def __init__(self, root='../../dataset/unlabeled/', transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])])):
        self.root = root
        self.transform = transform
        self.vid_list = sorted([f for f in os.listdir(root) if f.startswith('video')])
        # self.img_list = ['image_' + str(i) + '.png' for i in range(22)]
        self.data_dir = root
                    
    def __len__(self):
        return len(self.vid_list)

    def __getitem__(self, idx):
        video = self.vid_list[idx]
        imgs_train = []
        imgs_target = []
        for i in range(22):
            filename = 'image_' + str(i) + '.png'
            img_path  = os.path.join(self.data_dir, video, filename)
            img = Image.open(img_path).convert("RGB")
            if self.transform is not None:
                img = self.transform(img)
            if i < 11:
                imgs_train.append(img)
            else:
                imgs_target.append(img)
        assert len(imgs_train) == 11
        assert len(imgs_target) == 11
        return torch.stack(imgs_train), torch.stack(imgs_target)



