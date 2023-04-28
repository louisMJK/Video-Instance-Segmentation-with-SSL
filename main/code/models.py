import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
import numpy as np


def create_model(args):

    model = VideoInstanceSeg()

    # load backbone
    resnet = models.resnet50(replace_stride_with_dilation=[False, True, True])
    backbone = nn.Sequential(*list(resnet.children())[:-2])
    backbone.load_state_dict(torch.load(args.backbone_dir, map_location='cpu'), strict=True)
    model.resnet_aux.load_state_dict(backbone[:-1].state_dict())
    model.layer4.load_state_dict(backbone[-1].state_dict())
    print(f'\nBackbone loaded from {args.backbone_dir}')

    del backbone, resnet

    # load predictors
    if args.use_predictor:
        model.predictor.load_state_dict(torch.load(args.predictor_dir, map_location='cpu'), strict=True)

    # load FCN heads
    if args.use_fcn_head:
        model.classifier.load_state_dict(torch.load(args.fcn_dir + 'classifier.pth', map_location='cpu'), strict=True)
        model.aux_classifier.load_state_dict(torch.load(args.fcn_dir + 'aux_classifier.pth', map_location='cpu'), strict=True)
        print(f'FCN heads loaded from {args.fcn_dir}')

    if args.freeze_backbone:
        for p in model.resnet_aux.parameters():
            p.requires_grad = False
        for p in model.layer4.parameters():
            p.requires_grad = False
    
    if args.freeze_predictor:
        for p in model.predictor.parameters():
            p.requires_grad = False
    
    if args.freeze_fcn:
        for p in model.classifier.parameters():
            p.requires_grad = False
        for p in model.aux_classifier.parameters():
            p.requires_grad = False

    return model



class VideoInstanceSeg(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(replace_stride_with_dilation=[False, True, True])
        self.resnet_aux = nn.Sequential(*list(resnet.children())[:-3])
        self.layer4 = list(resnet.children())[-3]
        self.predictor = Predictor()
        self.classifier = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Conv2d(512, 49, kernel_size=(1, 1), stride=(1, 1)),
        )
        self.aux_classifier = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Conv2d(256, 49, kernel_size=(1, 1), stride=(1, 1)),
        )
    
    def forward(self, x):
        # x shape: (B, 11, 3, H, W)
        B = x.shape[0]
        x = x.view(B * 11, 3, 160, 240)

        x1 = self.resnet_aux(x) # (B * 11, 1024, 20, 30)
        x2 = self.layer4(x1)    # (B * 11, 2048, 20, 30)

        x1 = x1.view(B, 11, 1024, 20, 30)
        x2 = x2.view(B, 11, 2048, 20, 30)

        x1 = self.predictor(x1)
        y = self.predictor(x2)
        y += x2.mean(dim=1)

        x1 = self.aux_classifier(x1)
        y = self.classifier(y)

        output = {}
        output['out'] = F.interpolate(y, size=(160, 240), mode="bilinear", align_corners=False)
        output['aux'] = F.interpolate(x1, size=(160, 240), mode="bilinear", align_corners=False)

        return output



class Predictor(nn.Module):
    def __init__(self, num_hiddens = 256, num_heads = 4, mlp_hiddens = 1024, dropout = 0.1, in_size = (11, 3072, 20, 30), num_layers = 1):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(11, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        # B * 11 * 2048 * 5 * 8 to B * 440 * 2048

        x = x.permute(0,2,3,4,1)

        x = self.fc(x).squeeze(-1)

        return x



class MultiHeadSelfAttention(nn.Module):
    def __init__(self, num_hiddens, num_heads, dropout=0.1):
        super().__init__()
        assert num_hiddens % num_heads == 0
        self.num_heads = num_heads
        self.dropout = dropout
        qkv_bias = False
        self.W_q = nn.LazyLinear(num_hiddens, bias=qkv_bias)
        self.W_k = nn.LazyLinear(num_hiddens, bias=qkv_bias)
        self.W_v = nn.LazyLinear(num_hiddens, bias=qkv_bias)
        self.W_h = nn.Linear(num_hiddens, num_hiddens)

    def dot_product_attention(self, Q, K, V):
        # input shape:  (batch_size, num_heads, num_patches, num_hiddens/num_heads)
        # output shape: (batch_size, num_heads, num_patches, num_hiddens/num_heads)
        d = Q.shape[-1]
        scores = torch.matmul(Q, K.transpose(-2,-1)) / np.sqrt(d)  # (batch_size, num_heads, num_patches, num_patches)
        A = nn.Softmax(dim=-1)(scores)
        H = torch.matmul(nn.Dropout(self.dropout)(A), V)  # (batch_size, num_heads, num_patches, num_hiddens/num_heads)
        return H
    
    def split_heads(self, X):
        # input:  (batch_size, num_patches, num_hiddens)
        # output: (batch_size, num_heads, num_patches, num_hiddens/num_heads)
        return X.reshape(X.shape[0], X.shape[1], self.num_heads, -1).transpose(1, 2)
    
    def concat_heads(self, X):
        # input:  (batch_size, num_heads, num_patches, num_hiddens/num_heads)
        # output: (batch_size, num_patches, num_hiddens)
        X = X.transpose(1,2)
        return X.reshape(X.shape[0], X.shape[1], -1)

    def forward(self, X):
        # input shape:  (batch_size, num_patches, in_hiddens)
        # return shape: (batch_size,)
        Q = self.split_heads(self.W_q(X))  # (batch_size, num_heads, num_patches, num_hiddens/num_heads)
        K = self.split_heads(self.W_k(X))
        V = self.split_heads(self.W_v(X))
        H = self.dot_product_attention(Q, K, V)  # (batch_size, num_heads, num_patches, num_hiddens/num_heads)
        H = self.W_h(self.concat_heads(H))  # (batch_size, num_patches, num_hiddens)
        return H

class TransformerBlock(nn.Module):
    def __init__(self, num_hiddens, num_heads, mlp_hiddens, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(num_hiddens)
        self.attention = MultiHeadSelfAttention(num_hiddens, num_heads, dropout)
        self.norm2 = nn.LayerNorm(num_hiddens)
        self.mlp = nn.Sequential(
            nn.Linear(num_hiddens, mlp_hiddens),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hiddens, num_hiddens),
            nn.Dropout(dropout)
        )
    
    def forward(self, X):
        X = X + self.attention(self.norm1(X))
        X = X + self.mlp(self.norm2(X))
        return X
