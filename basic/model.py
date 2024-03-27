import timm
import torch.nn as nn
import torch
from operator import mul
import math
from functools import reduce

class VIT_Base_Deep(nn.Module):
    def __init__(self, modelname, num_classes, pretrained,prompt_tokens, prompt_dropout):
        super().__init__()

        self.model = timm.create_model('vit_base_patch8_224', num_classes=10, pretrained=True)


        #FREEZE-VIT
        for name, param in self.model.named_parameters():
            if 'head' not in name:
                param.requires_grad = False

        ##PROMPT
        self.prompt_type = 'deep'
        self.prompt_tokens = 5  # number of prompted tokens
        self.prompt_dropout = nn.Dropout(0.0)
        self.prompt_dim = self.model.embed_dim
        self.prompt_embeddings = nn.Parameter(torch.zeros(1, self.prompt_tokens, self.prompt_dim))

        #Initiate Prompt
        val = math.sqrt(6. / float(3 * reduce(mul, self.model.patch_embed.patch_size, 1) + self.prompt_dim))
        nn.init.uniform_(self.prompt_embeddings.data, -val, val)

        if self.prompt_type == 'deep':  # noqa
            self.total_d_layer = len(self.model.blocks)
            self.deep_prompt_embeddings = nn.Parameter(
                torch.zeros(self.total_d_layer-1, self.prompt_tokens, self.prompt_dim)
            ) #shape [11, 5, 768]
            # xavier_uniform initialization
            nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)
        print()

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
        x = self.model.patch_embed(x) #[1, 784, 768]
        x = self.model._pos_embed(x)
        x = self.model.norm_pre(x) #[1, 785, 768]

        # add prompts #self.prompt_embeddings.shape =(1,5,768)
        x = self.incorporate_prompt(x, self.prompt_embeddings)

        if self.prompt_type == 'deep':
            # deep mode
            x = self.model.blocks[0](x)
            for i in range(1, self.total_d_layer):
                x = self.incorporate_prompt(x, self.deep_prompt_embeddings[i-1], self.prompt_tokens)
                x = self.model.blocks[i](x)
        else:
            # shallow mode
            x = self.model.blocks(x)

        x = self.model.norm(x)
        return x #[4, 790, 768]

    def forward_deep_prompt(self, embedding_output):
        B = embedding_output.shape[0]

    def forward(self, x):

        x = self.forward_features(x)
        x = self.model.forward_head(x) #(4, 10) = (B, class)

        return x
# class VPT(nn.Module):
#
#     def __init__(self, modelname, num_classes, pretrained,prompt_tokens, prompt_dropout):
#         super().__init__()
#
#         self.model =timm.create_model('vit_base_patch8_224', num_classes=num_classes, pretrained=pretrained)
#         print()
#         # freeze - VIT
#         for n, p in self.model.named_parameters():
#             if 'head' not in n:
#                 p.requires_grad = False
#
#         # prompt
#         self.prompt_tokens = prompt_tokens  # number of prompted tokens
#         self.prompt_dropout = nn.Dropout(prompt_dropout)
#         self.prompt_dim = self.model.embed_dim
#
#         #initialize prompt
#         val = math.sqrt(6. / float(3 * reduce(mul, self.model.patch_embed.patch_size, 1) + self.prompt_dim))
#         self.prompt_embeddings = nn.Parameter(torch.zeros(1, self.prompt_tokens, self.prompt_dim))
#
#         # xavier_uniform initialization
#         nn.init.uniform_(self.prompt_embeddings.data, -val, val)
#
#     def __forward__(self, mode ):
#         print('this is vpt ')
#
#         pass
