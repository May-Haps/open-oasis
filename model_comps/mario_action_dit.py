from einops import rearrange
import torch
import torch.nn as nn

from model_comps.action_gated_dit import ActionGatedDiT
from model_comps.vae import AutoencoderKL

class MarioActionDiT(ActionGatedDiT):
    def __init__(
            self,
            input_h=60,
            input_w=64,
            patch_size=2,      
            in_channels=4,
            hidden_size=256,   
            depth=6,
            num_heads=8,
            max_frames=32,
            action_dim=8 # A, up, left, B, start, right, down, select
    ) -> None:
        super().__init__(
            input_h=input_h,
            input_w=input_w,
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            external_cond_dim=hidden_size, 
            max_frames=max_frames
        )

        self.action_embedding_table = nn.Parameter(torch.randn(action_dim, hidden_size))
        self.action_bias = nn.Parameter(torch.zeros(hidden_size))

        self.mario_action_initialize_weights()
        
    def forward(self, x, t, actions):
        external_cond = actions @ self.action_embedding_table + self.action_bias
        return super().forward(x, t, external_cond=external_cond)

    def mario_action_initialize_weights(self):
        nn.init.normal_(self.action_embedding_table, std=0.02)

def get_mario_vae():
    # 240 / 4 = 60
    # 256 / 4 = 64
    return AutoencoderKL(
        latent_dim=4,
        input_height=240,
        input_width=256,
        patch_size=4,
        enc_dim=256,
        enc_depth=4,
        enc_heads=8,
        dec_dim=256,
        dec_depth=4,
        dec_heads=8
    )