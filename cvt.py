from functools import partial
from math import sqrt
import torch
import torch.nn as nn

def trunc_normal_(tensor,mean=0,std=0.09):
    with torch.no_grad():
        size = tensor.shape
        tmp = tensor.new_empty(size+(4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.data.mul_(std).add_(mean)
        return tensor

class Rearrange(nn.Module):
    """[B, C, H, W] to [B, HW,C]"""
    def forward(self, x:torch.Tensor):
        assert len(x.shape) == 4, "x.shape is [B, C, H, W],({x.shape})"
        x = x.flatten(-2).transpose(-1,-2).contiguous()
        return x

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):
    def __init__(self, drop_prob=None, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ConvEmbed(nn.Module):
    def __init__(self,patch_size=7,in_chans=3,embed_dim=64,stride=4,padding=2,norm_layer=None):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding)
        self.norm = nn.Identity() if norm_layer is None else norm_layer(embed_dim)
    
    def forward(self, x:torch.Tensor):
        x = self.proj(x)
        B, C, H, W,=x.shape
        x = x.reshape(B, C, -1).transpose(-1,-2).contiguous()
        x = self.norm(x)
        return x, H, W

class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in,kernel_size,padding,stride):
        super().__init__()
        self.conv = nn.Conv2d(dim_in, dim_in, kernel_size,stride,padding,groups=dim_in)
        self.bn = nn.BatchNorm2d(dim_in)
        self.rearrage = Rearrange()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.rearrage(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim_in, num_heads, qkv_bias=True, attn_drop=0.,proj_drop=0.,with_cls_token=False):
        super().__init__()
        self.with_cls_token = with_cls_token
        self.num_heads = num_heads
        self.scale = dim_in ** -0.5
        self.conv_proj_q = DepthWiseConv2d(dim_in, 3, 1, 1)
        self.conv_proj_k = DepthWiseConv2d(dim_in, 3, 1, 2)
        self.conv_proj_v = DepthWiseConv2d(dim_in, 3, 1, 2)
        self.proj_q = nn.Linear(dim_in, dim_in, bias=qkv_bias)
        self.proj_k = nn.Linear(dim_in, dim_in, bias=qkv_bias)
        self.proj_v = nn.Linear(dim_in, dim_in, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_in, dim_in)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x:torch.Tensor,H, W):
        if self.with_cls_token:
            cls_token, x = x[:,0], x[:,1:]
        B, HW, C = x.shape
        x = x.transpose(-1,-2).contiguous().reshape(B, C, H, -1)
        q:torch.Tensor = self.conv_proj_q(x)
        k:torch.Tensor = self.conv_proj_k(x)
        v:torch.Tensor = self.conv_proj_v(x)
        if self.with_cls_token:
            q = torch.cat((cls_token[:,None], q), dim=1)
            k = torch.cat((cls_token[:,None], k), dim=1)
            v = torch.cat((cls_token[:,None], v), dim=1)
        B, T, _ = q.shape
        q = self.proj_q(q)
        q = q.reshape(B, T, self.num_heads, -1).transpose(-2, -3)
        B, T, _ = k.shape
        k = self.proj_k(k)
        k = k.reshape(B, T, self.num_heads, -1).transpose(-2, -3)
        v = self.proj_v(v)
        B, T, _ = v.shape
        v = v.reshape(B, T, self.num_heads, -1).transpose(-2, -3)
        attn:torch.Tensor = q.matmul(k.transpose(-1,-2)) * self.scale
        attn = self.attn_drop(attn.softmax(-1))
        x = attn.matmul(v)
        x = x.transpose(-2,-3).contiguous().reshape(B, -1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self,dim_in,num_heads,mlp_ratio=4.,qkv_bias=True, drop=0.,attn_drop=0.,
                drop_path=0.,act_layer=nn.GELU,norm_layer=nn.LayerNorm,with_cls_token=False):
        super().__init__()
        self.norm1 = norm_layer(dim_in)
        self.attn = Attention(dim_in, num_heads, qkv_bias, attn_drop, drop, with_cls_token)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim_in)
        dim_mlp_hidden = int(dim_in * mlp_ratio)
        self.mlp = Mlp(in_features=dim_in,hidden_features=dim_mlp_hidden,act_layer=act_layer,drop=drop)
    
    def forward(self, x, H, W):
        y = self.norm1(x)
        attn = self.attn(y, H, W)
        x = x+ self.drop_path(attn)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class VisionTransformer(nn.Module):
    def __init__(self,patch_size=16,patch_stride=16,patch_padding=0,in_chans=3,with_cls_token=False,
                 embed_dim=768,depth=12,num_heads=12,mlp_ratio=4.,qkv_bias=True,drop_rate=0.,
                 attn_drop_rate=0.,drop_path_rate=0.,act_layer=nn.GELU,norm_layer=nn.LayerNorm):
        super().__init__()
        self.patch_embed = ConvEmbed(patch_size, in_chans, embed_dim, patch_stride,patch_padding, norm_layer)
        if with_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.cls_token = None
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        blocks = []
        for j in range(depth):
            blocks.append(
                Block(dim_in=embed_dim,num_heads=num_heads,mlp_ratio=mlp_ratio,qkv_bias=qkv_bias,drop=drop_rate,attn_drop=attn_drop_rate,
                    drop_path=dpr[j],act_layer=act_layer,norm_layer=norm_layer,with_cls_token=with_cls_token))
        self.blocks = nn.ModuleList(blocks)
        if self.cls_token is not None:
            trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights_trunc_normal)
    
    def _init_weights_trunc_normal(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self,x:torch.Tensor):
        x, H, W = self.patch_embed(x)
        B, _, C =x.shape
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), 1)
        x = self.pos_drop(x)
        for i, blk in enumerate(self.blocks):
            x = blk(x, H, W)
        if self.cls_token is not None:
            cls_token, x = x[:,0], x[:,1:]
        x = x.transpose(-1, -2).contiguous().reshape(B, C, H, -1)
        if self.cls_token is not None:
            return x, cls_token
        return x
        

class ConvolutionalVisionTransformer(nn.Module):
    def __init__(self, large=False, in_chans=3, dim_embed=384, num_classes=1000, act_layer=nn.GELU,norm_layer=nn.LayerNorm):
        super().__init__()
        depth = [1, 4, 16] if large else [1, 2, 10]
        self.norm = norm_layer(dim_embed)
        self.head = nn.Linear(dim_embed, num_classes) if num_classes > 0 else nn.Identity()
        trunc_normal_(self.head.weight, std=0.02)
        self.stage0 = VisionTransformer(in_chans=in_chans,act_layer=act_layer,norm_layer=norm_layer,depth=depth[0],num_heads=1,
                            embed_dim=64,patch_padding=2,patch_size=7,patch_stride=4)
        self.stage1 = VisionTransformer(in_chans=64,act_layer=act_layer,norm_layer=norm_layer,depth=depth[1],num_heads=3,
                            embed_dim=192,patch_padding=1,patch_size=3,patch_stride=2)
        self.stage2 = VisionTransformer(in_chans=192,act_layer=act_layer,norm_layer=norm_layer,depth=depth[2],num_heads=6,
                            embed_dim=384,patch_padding=1,patch_size=3,patch_stride=2,drop_path_rate=0.1,with_cls_token=True)
    
    def forward(self, x):
        x = self.stage0(x)
        x = self.stage1(x)
        x, cls_tokens = self.stage2(x)
        x = self.norm(cls_tokens)
        x = self.head(x)
        return x

def create_cvt(model_name="CvT-13-224x224-IN-1k",num_classes=1000, pre_trained=True):
    if "-21-" in model_name:
        large=True
    else:
        large=False
    model = ConvolutionalVisionTransformer(large=large, act_layer=QuickGELU, norm_layer=partial(nn.LayerNorm,eps=1e-5), num_classes=num_classes)
    if pre_trained:
        state = model.state_dict()
        ckp = torch.load("weight/"+model_name+".pth","cpu")
        for k,v in ckp.items():
            if k in state.keys():
                if state[k].shape == v.shape:
                    state[k] = v
                else:
                    print("not equal  |  ", k)
            else:
                print(k)
        model.load_state_dict(state)
    out_features = model.head.out_features
    if out_features != num_classes:
        in_features = model.head.in_features
        model.head = nn.Linear(in_features,  out_features)
    return model

# if "__main__" == __name__:
#     model = ConvolutionalVisionTransformer(act_layer=QuickGELU, norm_layer=partial(nn.LayerNorm,eps=1e-5))
#     ckp = torch.load("weight/CvT-13-224x224-IN-1k.pth","cpu")
#     state = model.state_dict()
#     for k,v in ckp.items():
#         if k in state.keys():
#             print(k, "  |  ckp:",v.shape, "  |  state:",state[k].shape)
        # print(k)
    # print(model)
    # tensor = torch.randn((8, 3, 224, 224))
    # model(tensor)