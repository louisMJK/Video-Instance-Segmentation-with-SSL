import torch
from torch import nn
from torchvision import models
import numpy as np


def create_model(args):

    model = VideoInstanceSeg()

    if args.freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad = False
    
    if args.freeze_predictor:
        for p in model.predictor.parameters():
            p.requires_grad = False
        for p in model.fc.parameters():
            p.requires_grad = False
    
    if args.freeze_fcn:
        params = [
            model.classifier.parameters(),
            model.aux_classifier.parameters(),
            model.conv_t_1.parameters(),
            model.conv_t_2.parameters()
        ]
        for p in params:
            p.requires_grad = False

    return model



class VideoInstanceSeg(nn.Module):
    def __init__(self):
        super().__init__()
        fcn_resnet = models.segmentation.fcn_resnet50(num_classes=49, aux_loss=True)
        self.backbone = fcn_resnet.backbone
        self.predictor = Predictor()
        self.classifier = fcn_resnet.classifier
        self.aux_classifier = fcn_resnet.aux_classifier
        self.conv_t_1 = nn.ConvTranspose2d(49, 49, kernel_size=16, padding=4, stride=8)
        self.conv_t_2 = nn.ConvTranspose2d(49, 49, kernel_size=16, padding=4, stride=8)
        self.fc = nn.Sequential(
            nn.Linear(11, 256), 
            nn.ReLU(), 
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        x = self.backbone(x)         
        x1 = x['out']  # (B * 11, 2048, 20, 30)
        x2 = x['aux']  # (B * 11, 1024, 20, 30)

        BT, C, H, W = x1.shape
        B = BT // 11
        x1 = x1.reshape(B, 11, C, H, W)
        x1 = self.predictor(x1)  # (B, 2048, 20, 30)
        x2 = self.fc(x2.reshape(B, 11, 1024, H, W).permute(0,2,3,4,1)).squeeze(-1)  # (B, 1024, 20, 30)

        x1 = self.classifier(x1)      # (B, 49, 20, 30)
        x2 = self.aux_classifier(x2)  # (B, 49, 20, 30)

        y = {}
        y['out'] = self.conv_t_1(x1)
        y['aux'] = self.conv_t_2(x2)
        return y


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
    
class Predictor(nn.Module): #input shape (11, 2048, 5, 8)
    def __init__(self, num_hiddens = 512, num_heads = 8, mlp_hiddens = 2048, dropout = 0.1, in_size = (11, 2048, 20, 30), num_layers = 2):
        super().__init__()
        self.num_hiddens = num_hiddens
        T_, C_, H_, W_ = in_size
        # self.transformer = nn.Sequential(
        #     TransformerBlock(num_hiddens, num_heads, mlp_hiddens, dropout),
        #     TransformerBlock(num_hiddens, num_heads, mlp_hiddens, dropout),
        #     TransformerBlock(num_hiddens, num_heads, mlp_hiddens, dropout),
        # )
        self.transformer = nn.Sequential()
        for i in range(num_layers):
            self.transformer.add_module(f"{i}", TransformerBlock(num_hiddens, num_heads, mlp_hiddens, dropout))
        self.linear1 = nn.Linear(C_, self.num_hiddens)
        self.linear2 = nn.Linear(self.num_hiddens, C_)
        self.linear3 = nn.Linear(T_*H_*W_, H_*W_)
        self.relu = nn.ReLU()
        self.pos_embedding = nn.Parameter(0.02 * torch.randn(1, T_*H_*W_, num_hiddens))


    def forward(self, X):
        B, T, C, H, W = X.shape
        # B * 11 * 2048 * 5 * 8 to B * 440 * 2048
        X = X.permute(0,2,1,3,4).reshape(B, C, T*H*W).permute(0,2,1)

        # B * 440 * 2048 to B * 440 * 512
        X = self.linear1(X)
        #add positional embedding
        X = X + self.pos_embedding
        X = self.transformer(X)

        # B * 440 * 512 to B * 440 * 2048
        X = self.linear2(X)
        X = self.relu(X)
        # B * 440 * 2048 to B * 2048 * 40
        X = X.permute(0,2,1)
        X = self.linear3(X)
        X = X.reshape(B, C, H, W)
        return X

