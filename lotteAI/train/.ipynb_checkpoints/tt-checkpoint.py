import torch
import torch.nn as nn
import timm
import math
from functools import reduce
from operator import mul

class VIT2(nn.Module):
    def __init__(self):
        super().__init__()
       
        self.model = timm.create_model('vit_base_patch8_224', num_classes=10, pretrained=True)
        # freeze - VIT
        for name, param in self.model.named_parameters():
            if 'head' not in name:
                param.requires_grad = False
                
        ##PROMPT
        self.prompt_tokens = 5  # number of prompted tokens
        self.prompt_dropout = nn.Dropout(0.0)
        self.prompt_dim = self.model.embed_dim
        self.prompt_embeddings = nn.Parameter(torch.zeros(1, self.prompt_tokens, self.prompt_dim))

        #initiate prompt:
        val = math.sqrt(6. / float(3 * reduce(mul, self.model.patch_embed.patch_size, 1) + self.prompt_dim))
        nn.init.uniform_(self.prompt_embeddings.data, -val, val)
        
    def incorporate_prompt(self, x, prompt_embeddings, n_prompt: int = 0):
        B = x.shape[0]
        
        # concat prompts: (batch size, cls_token + n_prompt + n_patches, hidden_dim)
        x = torch.cat((
            x[:, :1, :],
            self.prompt_dropout(prompt_embeddings.expand(B, -1, -1)),
            x[:, (1+n_prompt):, :]
        ), dim=1)
        
        return x

    def forward_features(self, x):
        x = self.model.patch_embed(x)
        x = self.model._pos_embed(x)
        x = self.model.norm_pre(x)
        
        # add prompts
        x = self.incorporate_prompt(x, self.prompt_embeddings)
        
        # if self.prompt_type == 'deep':
        #     # deep mode
        #     x = self.encoder.blocks[0](x)
        #     for i in range(1, self.total_d_layer):
        #         x = self.incorporate_prompt(x, self.deep_prompt_embeddings[i-1], self.prompt_tokens)
        #         x = self.encoder.blocks[i](x)
        # else:
        #     # shallow mode
        x = self.model.blocks(x)
            
        x = self.model.norm(x)
        return x

    def forward(self, x):
        # print('this is forw')
        x = self.forward_features(x)
        x = self.model.forward_head(x)
        # print(x.shape)
        return x
        
