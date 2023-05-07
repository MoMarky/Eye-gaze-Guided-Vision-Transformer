# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from timm.models.vision_transformer import VisionTransformer



class MaskViT(VisionTransformer):
    def __init__(self, layer=None, num_classes=1000):
        super().__init__(patch_size=16, embed_dim=384, depth=12, num_heads=6, num_classes=num_classes)
        if layer is None:
            self.layer=11
        else:
            self.layer=layer

        # self.head = nn.Linear()
        # if num_classes != 1000:
        #     self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        #     self.init_weights()

    def forward_with_mask(self,x,mask):
        x = self.patch_embed(x)
        B,num_patches=x.shape[0],x.shape[1]
        mask_size=mask.sum()//x.shape[0]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
            cls_token_nums=1
        else:
            cls_token_nums=2
        x = self.pos_drop(x + self.pos_embed)
        res_x = torch.masked_select(x[:,cls_token_nums:,:],mask.reshape(x.shape[0],x.shape[1]-cls_token_nums,1))
        res_x = res_x.reshape(x.shape[0],mask.sum()//x.shape[0],-1)
        res_x = torch.cat((cls_token,res_x),dim=1)
        for i in range(self.layer):
            res_x=self.blocks[i](res_x)
        index=torch.range(0,x.shape[0]*(x.shape[1]-cls_token_nums)-1,dtype=torch.int64,device=mask.device)
        index=index.reshape(x.shape[0],-1)
        index=torch.masked_select(index,mask.reshape(x.shape[0],-1))
        cls_token = res_x[:,:cls_token_nums,:]
        res_x = res_x[:,cls_token_nums:,:]
        res_x = res_x.reshape(B*mask_size,-1)
        x = x[:,cls_token_nums:,:]
        x = x.reshape(x.shape[0]*num_patches,-1)
        x = x.index_add_(0,index,res_x)
        x = x.reshape(B,num_patches,-1)
        x = torch.cat((cls_token,x),dim=1)
        for i in range(self.layer, len(self.blocks)):
            x = self.blocks[i](x)
        x = self.norm(x)
        return self.head(x[:,0])
