import torch
from model_comps.utils import sigmoid_beta_schedule

class NoiseScheduler:
    def __init__(self, timesteps: int, device: str) -> None:
        self.timesteps = timesteps
        self.device = device

        self.beta_ts = sigmoid_beta_schedule(timesteps).to(device, dtype=torch.float32)
        alpha_ts = 1 - self.beta_ts
        alpha_bar_ts = torch.cumprod(alpha_ts, dim=0)
        self.sqrt_alpha_bar_ts = torch.sqrt(alpha_bar_ts)
        self.sqrt_1minus_alpha_bar_ts = torch.sqrt(1 - alpha_bar_ts)

    def noised_sample_and_velocity_target(
            self, 
            x0: torch.Tensor,
            t: torch.Tensor,
            noise: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sqrt_alpha_bar_t = self.sqrt_alpha_bar_ts[t].unsqueeze(dim=2).unsqueeze(dim=3).unsqueeze(dim=4)
        sqrt_1minus_alpha_bar_t = self.sqrt_1minus_alpha_bar_ts[t].unsqueeze(dim=2).unsqueeze(dim=3).unsqueeze(dim=4)
        xt = sqrt_alpha_bar_t * x0 + sqrt_1minus_alpha_bar_t * noise
        v_target = sqrt_alpha_bar_t * noise - sqrt_1minus_alpha_bar_t * x0
        return xt, v_target