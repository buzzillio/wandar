import torch
import torch.nn as nn

# Define WrappedGPT class
class WrappedGPT:
    """
    This class wraps a GPT layer for specific operations.
    """

    def __init__(self, layer, layer_id=0, layer_name="none"):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]

        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.nsamples = 0

        self.layer_id = layer_id 
        self.layer_name = layer_name

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        self.scaler_row *= self.nsamples / (self.nsamples+tmp)
        self.nsamples += tmp

        inp = inp.type(torch.float32)
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2  / self.nsamples


# Enhanced wrapper that computes selectivity metrics
class WrappedGPTSelectivity:
    """
    Enhanced GPT layer wrapper that computes selectivity metrics:
    - L2 norm (Wanda's original metric)
    - IDF-style rarity (active rate per channel)
    - Top-quantile spikiness (peaked vs flat activations)
    """

    def __init__(self, layer, layer_id=0, layer_name="none", quantile=0.9, eps=1e-10):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]

        # Original Wanda metric: L2 norm
        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        
        # For IDF computation: track mean for threshold
        self.channel_sum = torch.zeros((self.columns), device=self.dev)
        self.channel_sq_sum = torch.zeros((self.columns), device=self.dev)
        self.active_count = torch.zeros((self.columns), device=self.dev)
        self.total_count = 0
        
        # For spikiness: collect values for quantile computation
        # We use a reservoir-like approach: keep track of sums above/below quantile
        self.quantile = quantile
        self.all_channel_values = []  # Will store batched values temporarily
        
        # Hyperparameters
        self.eps = eps
        self.nsamples = 0
        self.layer_id = layer_id 
        self.layer_name = layer_name
        
        # Final computed metrics
        self.idf_scores = None
        self.spiky_scores = None

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()  # shape: [columns, num_tokens]

        inp = inp.type(torch.float32)
        num_tokens = inp.shape[1]
        
        # Update sample count
        self.scaler_row *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        
        # 1. Original Wanda metric: L2 norm per channel
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2 / self.nsamples
        
        # 2. Accumulate statistics for IDF and spikiness
        self.channel_sum += inp.sum(dim=1)
        self.channel_sq_sum += (inp ** 2).sum(dim=1)
        self.total_count += num_tokens
        
        # Store values for quantile computation (we'll compute after all batches)
        self.all_channel_values.append(inp.detach().clone())

    def finalize_metrics(self, tau_percentile=0.6, idf_clip_max=10.0, spiky_clip_max=10.0):
        """
        Compute final IDF and spikiness scores after all batches have been processed.
        
        Args:
            tau_percentile: Percentile for active threshold (default 0.6 = 60th percentile)
            idf_clip_max: Maximum value for IDF scores
            spiky_clip_max: Maximum value for spikiness scores
        """
        if len(self.all_channel_values) == 0:
            raise ValueError("No batches added yet!")
        
        # Concatenate all values: shape [columns, total_tokens]
        all_values = torch.cat(self.all_channel_values, dim=1)
        
        # Compute channel means
        channel_mean = self.channel_sum / self.total_count
        
        # === IDF Computation ===
        # Compute per-channel threshold (tau_j = percentile of channel activations)
        tau_j = torch.quantile(all_values, tau_percentile, dim=1, keepdim=False)
        
        # Count active rate: fraction of activations above threshold
        active_counts = (all_values > tau_j.unsqueeze(1)).float().sum(dim=1)
        p_j = active_counts / self.total_count
        
        # IDF score: log(1 / (p_j + eps)), clipped
        idf_raw = torch.log(1.0 / (p_j + self.eps))
        self.idf_scores = torch.clamp(idf_raw, 0.0, idf_clip_max)
        
        # === Spikiness Computation ===
        # Compute q-quantile per channel
        Q_j = torch.quantile(all_values, self.quantile, dim=1, keepdim=False)
        
        # Compute mean of top activations (>= Q_j)
        mu_top_j = torch.zeros(self.columns, device=self.dev)
        for j in range(self.columns):
            top_mask = all_values[j] >= Q_j[j]
            if top_mask.sum() > 0:
                mu_top_j[j] = all_values[j][top_mask].mean()
            else:
                mu_top_j[j] = all_values[j].max()  # fallback
        
        # Spikiness: mu_top / mu_mean, clipped to [1, spiky_clip_max]
        spiky_raw = mu_top_j / (channel_mean + self.eps)
        self.spiky_scores = torch.clamp(spiky_raw, 1.0, spiky_clip_max)
        
        # Free memory
        self.all_channel_values = []
        
        return self.idf_scores, self.spiky_scores