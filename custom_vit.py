import torch
import torch.nn as nn
import timm
import math

class VIT_Base_Deep(nn.Module):
    def __init__(self, num_classes_product=2, num_classes_angle=3):
        super(VIT_Base_Deep, self).__init__()
       
        self.model = timm.create_model('vit_large_patch16_224', num_classes=10, pretrained=True)
        
        # Freeze all parameters except for the head
        for name, param in self.model.named_parameters():
            if 'head' not in name:
                param.requires_grad = False
                
        # Prompt initialization
        self.prompt_type = 'deep'
        self.prompt_tokens = 5  # Number of prompted tokens
        self.prompt_dropout = nn.Dropout(0.0)
        self.prompt_dim = self.model.embed_dim
        self.prompt_embeddings = nn.Parameter(torch.zeros(1, self.prompt_tokens, self.prompt_dim))

        # Initialize prompt embeddings
        val = math.sqrt(6. / float(3 * self.model.patch_embed.num_patches + self.prompt_dim))
        nn.init.uniform_(self.prompt_embeddings.data, -val, val)

        if self.prompt_type == 'deep':
            self.total_d_layer = len(self.model.blocks)
            self.deep_prompt_embeddings = nn.Parameter(
                torch.zeros(self.total_d_layer-1, self.prompt_tokens, self.prompt_dim)
            )
            # Xavier uniform initialization
            nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)
        
    def incorporate_prompt(self, x, prompt_embeddings, n_prompt=0):
        B = x.shape[0]
        # Concatenate prompts
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
        
        # Add prompts
        x = self.incorporate_prompt(x, self.prompt_embeddings)
        
        if self.prompt_type == 'deep':
            # Deep mode
            x = self.model.blocks[0](x)
            for i in range(1, self.total_d_layer):
                x = self.incorporate_prompt(x, self.deep_prompt_embeddings[i-1], self.prompt_tokens)
                x = self.model.blocks[i](x)
        else:
            # Shallow mode
            x = self.model.blocks(x)
            
        x = self.model.norm(x)
        return x
        
    def forward(self, x):
        x = self.forward_features(x)
        x = self.model.forward_head(x)
        return x
