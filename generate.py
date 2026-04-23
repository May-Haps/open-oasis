"""
Generate a Mario gameplay video from a prompt image and action sequence.

Run:
    python generate.py --ckpt runs/mario_v1/ckpt30.pt \
                       --prompt path/to/frame.png \
                       --actions path/to/actions.pt \
                       --output video.mp4
"""

import argparse
from pprint import pprint

import torch
from einops import rearrange
from PIL import Image
from torch import autocast
from torchvision.io import write_video
from torchvision.transforms.functional import to_tensor, resize
from tqdm import tqdm

from model.dit import MarioWorldModel
from model.utils import sigmoid_beta_schedule, action_int_to_bits

assert torch.cuda.is_available()
device = "cuda:0"

TARGET_SIZE = (256, 256)


def load_prompt(path: str, n_frames: int = 1) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    frame = to_tensor(img)
    frame = resize(frame, list(TARGET_SIZE))      # [3, 256, 256]
    # normalize to [-1, 1] for diffusion
    frame = frame * 2 - 1
    # [1, n_frames, 3, 256, 256]
    return frame.unsqueeze(0).unsqueeze(0).expand(1, n_frames, -1, -1, -1).clone()


def load_actions(path: str, offset: int = 0) -> torch.Tensor:
    """
    Load actions from a .pt file.
    Accepts either:
      - int tensor [T] of raw NES action bytes → converts to bits
      - float tensor [T, 8] already in bit form
    Returns [1, T+1, 8] with a zero prepended for the prompt frame.
    """
    actions = torch.load(path, weights_only=True)
    if actions.ndim == 1:
        actions = action_int_to_bits(actions[offset:])
    else:
        actions = actions[offset:].float()
    zero = torch.zeros(1, actions.shape[-1])
    actions = torch.cat([zero, actions], dim=0)   # prepend zero for prompt frame
    return actions.unsqueeze(0)                   # [1, T+1, 8]


def main(args):
    torch.manual_seed(0)

    model = MarioWorldModel()
    state = torch.load(args.ckpt, weights_only=True)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model = model.to(device).eval()
    print(f"Loaded model from {args.ckpt}")

    x = load_prompt(args.prompt).to(device)                        # [1,1,3,256,256] in [-1,1]
    actions = load_actions(args.actions, args.action_offset).to(device)  # [1,T,8]

    total_frames = args.num_frames
    n_prompt = args.n_prompt_frames
    ddim_steps = args.ddim_steps
    max_noise_level = 1000
    noise_abs_max = 20.0
    stabilization_level = 15

    noise_range = torch.linspace(-1, max_noise_level - 1, ddim_steps + 1)

    betas = sigmoid_beta_schedule(max_noise_level).float().to(device)
    alphas = 1.0 - betas
    alphas_cumprod = rearrange(torch.cumprod(alphas, dim=0), "T -> T 1 1 1")

    B = x.shape[0]

    for i in tqdm(range(n_prompt, total_frames)):
        chunk = torch.randn((B, 1, *x.shape[-3:]), device=device)
        chunk = torch.clamp(chunk, -noise_abs_max, noise_abs_max)
        x = torch.cat([x, chunk], dim=1)
        start_frame = max(0, i + 1 - model.max_frames)

        for noise_idx in reversed(range(1, ddim_steps + 1)):
            t_ctx  = torch.full((B, i), stabilization_level - 1, dtype=torch.long, device=device)
            t      = torch.full((B, 1), noise_range[noise_idx], dtype=torch.long, device=device)
            t_next = torch.full((B, 1), noise_range[noise_idx - 1], dtype=torch.long, device=device)
            t_next = torch.where(t_next < 0, t, t_next)
            t      = torch.cat([t_ctx, t], dim=1)
            t_next = torch.cat([t_ctx, t_next], dim=1)

            x_curr  = x[:, start_frame:]
            t_curr  = t[:, start_frame:]
            tn_curr = t_next[:, start_frame:]

            with torch.no_grad():
                with autocast("cuda", dtype=torch.bfloat16):
                    v = model(x_curr, t_curr, actions[:, start_frame : i + 1])

            x_start = alphas_cumprod[t_curr].sqrt() * x_curr - (1 - alphas_cumprod[t_curr]).sqrt() * v
            x_noise = ((1 / alphas_cumprod[t_curr]).sqrt() * x_curr - x_start) / (
                1 / alphas_cumprod[t_curr] - 1
            ).sqrt()

            alpha_next = alphas_cumprod[tn_curr]
            alpha_next[:, :-1] = torch.ones_like(alpha_next[:, :-1])
            if noise_idx == 1:
                alpha_next[:, -1:] = torch.ones_like(alpha_next[:, -1:])
            x[:, -1:] = alpha_next.sqrt() * x_start + x_noise * (1 - alpha_next).sqrt()

    # [-1,1] → [0,255] uint8
    out = (x.clamp(-1, 1) + 1) / 2
    out = rearrange(out, "b t c h w -> b t h w c")
    out = (out * 255).byte().cpu()
    write_video(args.output, out[0], fps=args.fps)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",          type=str, required=True)
    parser.add_argument("--prompt",        type=str, required=True, help="Path to prompt PNG frame")
    parser.add_argument("--actions",       type=str, required=True, help="Path to .pt action file")
    parser.add_argument("--output",        type=str, default="video.mp4")
    parser.add_argument("--num-frames",    type=int, default=64)
    parser.add_argument("--n-prompt-frames", type=int, default=1)
    parser.add_argument("--ddim-steps",   type=int, default=10)
    parser.add_argument("--fps",           type=int, default=30)
    parser.add_argument("--action-offset", type=int, default=0)
    args = parser.parse_args()
    pprint(vars(args))
    main(args)
