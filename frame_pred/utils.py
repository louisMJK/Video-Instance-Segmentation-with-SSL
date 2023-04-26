from torch.utils.data import Dataset
import os
from PIL import Image
import copy
from torch import nn
from torchvision import transforms
import torch
from lightly.models.modules import BYOLPredictionHead, BYOLProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.utils.scheduler import cosine_schedule

class Unlabledtrainpred(Dataset):
    def __init__(self, root='../../dataset/unlabeled/', transform=transforms.ToTensor()):
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



class ConvLSTM(nn.Module):
    def __init__(self, input_channel, num_filter, b_h_w, kernel_size, device, stride=1, padding=1, seq_len=11, ):
        super().__init__()
        self._conv = nn.Conv2d(in_channels=input_channel + num_filter,
                               out_channels=num_filter*4,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding)
        self._batch_size, self._state_height, self._state_width = b_h_w
        # if using requires_grad flag, torch.save will not save parameters in deed although it may be updated every epoch.
        # Howerver, if you use declare an optimizer like Adam(model.parameters()),
        # parameters will not be updated forever.
        self.Wci = nn.Parameter(torch.zeros(1, num_filter, self._state_height, self._state_width)).to(device)
        self.Wcf = nn.Parameter(torch.zeros(1, num_filter, self._state_height, self._state_width)).to(device)
        self.Wco = nn.Parameter(torch.zeros(1, num_filter, self._state_height, self._state_width)).to(device)
        self._input_channel = input_channel
        self._num_filter = num_filter
        self.seq_len = seq_len
        self.device = device

    # inputs and states should not be all none
    # inputs: S*B*C*H*W
    def forward(self, inputs=None, states=None):
        inputs = inputs.permute(1, 0, 2, 3, 4) #from B*S*C*H*W convert to S*B*C*H*W
        seq_len = self.seq_len if inputs is None else inputs.size(0)
        if states is None:
            c = torch.zeros((inputs.size(1), self._num_filter, self._state_height,
                                  self._state_width), dtype=torch.float).to(self.device)
            h = torch.zeros((inputs.size(1), self._num_filter, self._state_height,
                             self._state_width), dtype=torch.float).to(self.device)
        else:
            h, c = states

        outputs = []
        for index in range(seq_len):
            # initial inputs
            if inputs is None:
                x = torch.zeros((h.size(0), self._input_channel, self._state_height,
                                      self._state_width), dtype=torch.float).to(self.device)
            else:
                x = inputs[index, ...]
            cat_x = torch.cat([x, h], dim=1)
            conv_x = self._conv(cat_x)

            i, f, tmp_c, o = torch.chunk(conv_x, 4, dim=1)

            i = torch.sigmoid(i+self.Wci*c)
            f = torch.sigmoid(f+self.Wcf*c)
            c = f*c + i*torch.tanh(tmp_c)
            o = torch.sigmoid(o+self.Wco*c)
            h = o*torch.tanh(c)
            outputs.append(h)
        return torch.stack(outputs), (h, c)
    
class Predictor(nn.Module):
    def __init__(self, input_channel, num_filter, b_h_w, kernel_size,device, out_channels = 3, stride= 1, padding= 1):
        super().__init__()
        self.input_channel = input_channel
        self.num_filter = num_filter
        self.b_h_w = b_h_w
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.convlstm = ConvLSTM(input_channel, num_filter, b_h_w, kernel_size, device, stride, padding)
        self.conv = nn.Conv2d(in_channels=num_filter, out_channels = out_channels, kernel_size=1)
    def forward(self, inputs=None, states=None, seq_len=11):
        _, (h, _) = self.convlstm(inputs)
        output = self.conv(h)
        return output