import torch


def sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=1, clamp_min=1e-5):
    """
    sigmoid schedule — https://arxiv.org/abs/2212.11972 Figure 8
    better for images > 64x64
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


# NES button order (MSB to LSB): A, up, left, B, start, right, down, select
ACTION_BUTTONS = ["A", "up", "left", "B", "start", "right", "down", "select"]
ACTION_DIM = 8


def action_int_to_bits(action_int: torch.Tensor) -> torch.Tensor:
    """Convert NES controller byte tensor [...] to [..., 8] binary float tensor."""
    bits = torch.zeros(*action_int.shape, ACTION_DIM, dtype=torch.float32)
    for i in range(ACTION_DIM):
        bits[..., i] = ((action_int >> (7 - i)) & 1).float()
    return bits
