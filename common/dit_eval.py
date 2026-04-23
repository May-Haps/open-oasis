import torch
import torch.nn as nn
from einops import rearrange
from torchvision.io import write_video

class MarioEvalModule:
    """Consolidated evaluation components: Metrics + Visualization."""
    
    @staticmethod
    def calculate_psnr(pred, target):
        mse = torch.mean((pred - target) ** 2)
        if mse == 0: return 100.0
        return 20 * torch.log10(1.0 / torch.sqrt(mse))

    @staticmethod
    def save_comparison_video(gt_pixels, pred_pixels, path, fps=60):
        """GT and Pred side-by-side. gt/pred shape: (T, C, H, W)"""
        # Combine horizontally: (T, 3, H, W*2)
        combined = torch.cat([gt_pixels, pred_pixels], dim=-1)
        # Convert to (T, H, W*2, 3) uint8 for saving
        video = (combined.permute(0, 2, 3, 1).clamp(0, 1) * 255).byte().cpu()
        write_video(path, video, fps=fps)