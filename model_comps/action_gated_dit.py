import torch
from torch import nn

from model_comps.dit import DiT, gate, modulate, SpatioTemporalDiTBlock

class ActionGatedDiTBlock(SpatioTemporalDiTBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_routing = nn.Sequential(
            nn.SiLU(),
            nn.Linear(kwargs['hidden_size'], 2)
        )

    def forward(self, x, c):
        B, T, H, W, D = x.shape

        ar_weights = self.action_routing(c)
        s_ar_gate = torch.sigmoid(ar_weights[:, :, 0]).view(B, T, 1, 1, 1)
        t_ar_gate = torch.sigmoid(ar_weights[:, :, 1]).view(B, T, 1, 1, 1)

        # spatial block
        s_shift_msa, s_scale_msa, s_gate_msa, s_shift_mlp, s_scale_mlp, s_gate_mlp = self.s_adaLN_modulation(c).chunk(6, dim=-1)
        s_res = gate(self.s_attn(modulate(self.s_norm1(x), s_shift_msa, s_scale_msa)), s_gate_msa)
        x = x + s_res * s_ar_gate
        x = x + gate(self.s_mlp(modulate(self.s_norm2(x), s_shift_mlp, s_scale_mlp)), s_gate_mlp)

        # temporal block
        t_shift_msa, t_scale_msa, t_gate_msa, t_shift_mlp, t_scale_mlp, t_gate_mlp = self.t_adaLN_modulation(c).chunk(6, dim=-1)
        t_res = gate(self.t_attn(modulate(self.t_norm1(x), t_shift_msa, t_scale_msa)), t_gate_msa)
        x = x + t_res * t_ar_gate
        x = x + gate(self.t_mlp(modulate(self.t_norm2(x), t_shift_mlp, t_scale_mlp)), t_gate_mlp)

        return x

class ActionGatedDiT(DiT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        new_blocks = []
        for i in range(len(self.blocks)):
            new_block = ActionGatedDiTBlock(
                hidden_size=self.blocks[i].s_norm1.normalized_shape[0],
                num_heads=self.num_heads,
                is_causal=True,
                spatial_rotary_emb=self.spatial_rotary_emb,
                temporal_rotary_emb=self.temporal_rotary_emb
            )
            new_block.load_state_dict(self.blocks[i].state_dict(), strict=False)
            new_blocks.append(new_block)
            
        self.blocks = nn.ModuleList(new_blocks)
        
    def initialize_action_layers(self):
        for block in self.blocks:
            nn.init.constant_(block.action_routing[-1].weight, 0)
            nn.init.constant_(block.action_routing[-1].bias, 0)
