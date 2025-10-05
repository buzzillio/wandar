"""Wanda + Selectivity scoring for unstructured pruning.

Extends Wanda's |W_ij| * ||X_j||_2 with two cheap add-ons computed in one calibration pass:
1. IDF-style rarity: penalizes "always-on" channels via log(1/p_j) where p_j = active rate.
2. Top-quantile spikiness: rewards "peaky specialists" via (mu_top / mu).

Reuses Wanda's WrappedGPT hooks—just adds median tracking and top-q stats.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class SelectivityStatsLight:
    """Batched stats collector - processes ALL channels at once, not one-by-one."""

    def __init__(self, num_channels: int):
        self.num_channels = num_channels
        # Store a few random samples for each channel [num_samples, num_channels]
        self.samples = []
        self.max_samples = 200  # Total samples to keep across all batches

    def update(self, x: torch.Tensor):
        """Update with activation tensor of shape [batch*seq, num_channels].
        
        Args:
            x: Activation tensor [N, C] where C = num_channels
        """
        if x.numel() == 0:
            return
        
        # Randomly sample a few rows (tokens) from this batch
        n_rows = x.shape[0]
        if len(self.samples) < self.max_samples:
            # Need more samples - take up to 50 random rows
            n_to_sample = min(50, n_rows, self.max_samples - len(self.samples))
            if n_to_sample > 0:
                indices = torch.randperm(n_rows, device=x.device)[:n_to_sample]
                sampled = x[indices].cpu()  # [n_to_sample, num_channels]
                self.samples.append(sampled)
    
    def finalize(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute IDF and spikiness for all channels at once.
        
        Returns:
            idf_scores: [num_channels]
            spikiness_scores: [num_channels]
        """
        if len(self.samples) == 0:
            return torch.ones(self.num_channels), torch.ones(self.num_channels)
        
        # Stack all samples: [total_samples, num_channels]
        all_samples = torch.cat(self.samples, dim=0)  # [N, C]
        
        # Compute per-channel stats in batch
        mean_per_channel = all_samples.mean(dim=0)  # [C]
        median_per_channel = torch.median(all_samples, dim=0).values  # [C]
        
        # IDF: fraction above median per channel
        above_median = (all_samples > median_per_channel.unsqueeze(0)).float().mean(dim=0)  # [C]
        p_j = torch.clamp(above_median, 1e-6, 0.999)
        idf_scores = torch.log(1.0 / (p_j + 1e-9))
        idf_scores = torch.clamp(idf_scores, 0.1, 10.0)
        
        # Spikiness: top 10% mean / overall mean per channel
        q90_per_channel = torch.quantile(all_samples, 0.9, dim=0)  # [C]
        spikiness_scores = torch.ones(self.num_channels)
        
        for j in range(self.num_channels):
            top_vals = all_samples[all_samples[:, j] >= q90_per_channel[j], j]
            if len(top_vals) > 0:
                mu_top = top_vals.mean().item()
                mu = mean_per_channel[j].item()
                spikiness_scores[j] = mu_top / (mu + 1e-9)
        
        spikiness_scores = torch.clamp(spikiness_scores, 1.0, 20.0)
        
        return idf_scores, spikiness_scores


def collect_wanda_selectivity_stats(
    model,
    dataloader,
    device,
    quantile: float = 0.9,
) -> Dict[int, Dict[str, torch.Tensor]]:
    """Collect Wanda activation norms + selectivity stats (IDF, spikiness) in one pass.
    
    Returns:
        dict[layer_idx] -> {
            'input_norm': tensor of shape [in_features],  # ||X_j||_2 per input channel
            'idf': tensor of shape [in_features],          # log(1/p_j)
            'spikiness': tensor of shape [in_features],    # mu_top / mu
        }
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False
    
    layers = model.model.layers
    stats = {}
    hooks = []
    
    for layer_idx, layer in enumerate(layers):
        subset = {}
        for name, module in layer.named_modules():
            if isinstance(module, nn.Linear):
                subset[name] = module
        
        if not subset:
            continue
        
        # Initialize stats collectors per module
        layer_stats = {}
        for name, module in subset.items():
            in_features = module.weight.shape[1]
            layer_stats[name] = {
                'scaler_row': torch.zeros(in_features, device='cpu'),
                'nsamples': 0,
                'selectivity': SelectivityStatsLight(in_features),  # Single collector for ALL channels
            }
        
        def make_hook(name):
            def hook(module, inp, out):
                if inp is None or len(inp) == 0:
                    return
                x = inp[0].detach()
                
                # Flatten batch/seq dims
                if x.dim() == 3:
                    x = x.reshape(-1, x.shape[-1])  # [B*T, C]
                elif x.dim() == 2:
                    pass
                else:
                    return
                
                # Wanda norm: ||X_j||_2
                x_t = x.t()  # [C, B*T]
                norms = torch.norm(x_t, p=2, dim=1).cpu()
                
                tmp = x.shape[0]
                old_n = layer_stats[name]['nsamples']
                new_n = old_n + tmp
                
                layer_stats[name]['scaler_row'] = (
                    layer_stats[name]['scaler_row'] * old_n / new_n
                    + (norms ** 2) / new_n
                )
                layer_stats[name]['nsamples'] = new_n
                
                # Selectivity stats - pass entire batch at once
                # Only update every 4th batch to keep it fast
                if new_n % 4 == 0:
                    layer_stats[name]['selectivity'].update(x)
            
            return hook
        
        for name, module in subset.items():
            hooks.append(module.register_forward_hook(make_hook(name)))
        
        stats[layer_idx] = layer_stats
    
    # Run calibration
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (tuple, list)):
                input_ids = batch[0]
            else:
                input_ids = batch
            input_ids = input_ids.to(device)
            model(input_ids=input_ids)
    
    # Remove hooks
    for h in hooks:
        h.remove()
    
    # Finalize stats
    result = {}
    for layer_idx, layer_stats in stats.items():
        layer_result = {}
        for name, stat in layer_stats.items():
            input_norm = torch.sqrt(stat['scaler_row'])
            
            # Finalize returns tensors directly
            idf_scores, spikiness_scores = stat['selectivity'].finalize()
            
            layer_result[name] = {
                'input_norm': input_norm,
                'idf': idf_scores,
                'spikiness': spikiness_scores,
            }
        
        result[layer_idx] = layer_result
    
    model.config.use_cache = use_cache
    return result
