from typing import Dict, Optional, Union

import torch
import torch.nn.functional as F


class TFIDFStats:
    """Track TF-IDF statistics for old NeuronRank formula.
    
    TF (Term Frequency): Average activation strength across all tokens
    IDF (Inverse Document Frequency): Selectivity based on how rarely neuron fires
    """

    def __init__(self, size: int, dtype: torch.dtype = torch.float32, device: Union[torch.device, str] = "cpu"):
        self.size = size
        self.dtype = dtype
        self.device = device
        self.total_tokens = 0
        # Sum of absolute activations for TF
        self.activation_sum = torch.zeros(size, dtype=dtype, device=device)
        # Count of tokens where neuron fired (activation > 0)
        self.active_count = torch.zeros(size, dtype=torch.long, device=device)

    def update(self, activations: torch.Tensor):
        """Update statistics with a batch of activations.
        
        Args:
            activations: Tensor of shape [batch_size, neuron_dim] or [batch*seq_len, neuron_dim]
        """
        if activations is None or activations.numel() == 0:
            return
        
        activations = activations.to(device=self.device, dtype=self.dtype)
        num_tokens = activations.shape[0]
        
        # TF: sum of absolute activations
        abs_act = torch.abs(activations)
        self.activation_sum += abs_act.sum(dim=0)
        
        # IDF: count where neuron is active (activation > 0)
        active_mask = abs_act > 0
        self.active_count += active_mask.sum(dim=0).to(torch.long)
        
        self.total_tokens += num_tokens

    def compute_tf_idf(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute TF and IDF scores.
        
        Returns:
            tf: Mean absolute activation across all tokens (higher = stronger)
            idf: Log-based selectivity score (higher = more selective/sparse)
        """
        if self.total_tokens == 0:
            return torch.zeros(self.size, device=self.device), torch.zeros(self.size, device=self.device)
        
        # TF: average activation strength
        tf = self.activation_sum / self.total_tokens
        
        # IDF: log((T + 1) / (n_active + 1)) + 1
        # Higher when neuron fires rarely (selective)
        idf = torch.log((self.total_tokens + 1) / (self.active_count.to(self.dtype) + 1)) + 1
        
        return tf, idf


class RunningStats:
    """Track mean and variance for high-dimensional tensors using Welford grouping."""

    def __init__(self, size: int, dtype: torch.dtype = torch.float32, device: Union[torch.device, str] = "cpu"):
        self.count = 0
        self.dtype = dtype
        self.device = device
        self.mean = torch.zeros(size, dtype=dtype, device=device)
        self.M2 = torch.zeros(size, dtype=dtype, device=device)

    def update(self, batch: torch.Tensor):
        if batch is None or batch.numel() == 0:
            return
        batch = batch.to(device=self.device, dtype=self.dtype)
        batch_count = batch.shape[0]
        if batch_count == 0:
            return
        batch_mean = batch.mean(dim=0)
        batch_delta = batch - batch_mean.unsqueeze(0)
        batch_M2 = (batch_delta * batch_delta).sum(dim=0)

        if self.count == 0:
            self.mean.copy_(batch_mean)
            self.M2.copy_(batch_M2)
            self.count = batch_count
            return

        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean += delta * batch_count / total
        self.M2 += batch_M2 + (delta * delta) * self.count * batch_count / total
        self.count = total

    def variance(self) -> torch.Tensor:
        if self.count < 2:
            return torch.zeros_like(self.mean)
        return self.M2 / (self.count - 1)


def _compute_top_token_ids(dataloader, tokenizer, max_classes=512):
    if max_classes is None or max_classes <= 0:
        return []

    vocab_size = getattr(tokenizer, "vocab_size", None)
    if vocab_size is None:
        return []

    counts = torch.zeros(vocab_size, dtype=torch.long)
    for batch in dataloader:
        input_ids = batch[0] if isinstance(batch, (tuple, list)) else batch
        ids = input_ids.reshape(-1).cpu()
        valid = (ids >= 0) & (ids < vocab_size)
        if not valid.any():
            continue
        ids = ids[valid]
        counts.index_add_(0, ids, torch.ones_like(ids, dtype=torch.long))

    nonzero = (counts > 0).nonzero(as_tuple=False).squeeze(-1)
    if nonzero.numel() == 0:
        return []

    k = min(max_classes, nonzero.numel())
    return torch.topk(counts, k).indices.tolist()


def collect_neuronrank_statistics(model, dataloader, tokenizer, device, max_classes=512):
    """Capture post-activation statistics for each LLaMA MLP gate projection."""

    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise ValueError("Expected the model to expose model.layers for NeuronRank scoring.")

    batches = list(dataloader)
    if not batches:
        raise ValueError("Calibration dataloader for NeuronRank produced no batches.")

    top_token_ids = _compute_top_token_ids(batches, tokenizer, max_classes=max_classes)
    vocab_size = getattr(tokenizer, "vocab_size", 0)
    class_index_lookup = torch.full((vocab_size,), -1, dtype=torch.long) if top_token_ids else None
    if class_index_lookup is not None:
        for idx, tok in enumerate(top_token_ids):
            if 0 <= tok < vocab_size:
                class_index_lookup[tok] = idx

    num_classes = len(top_token_ids)
    print(f"loading calibration data ({len(batches)} batches, tracking {num_classes} token classes)")

    layer_stats = {}
    hooks = []
    current_batch_token_ids = None
    class_lookup_cache = {}

    for layer_idx, layer in enumerate(model.model.layers):
        gate_proj = getattr(layer.mlp, "gate_proj", None)
        if gate_proj is None:
            continue

        layer_device = gate_proj.weight.device
        layer_dtype = torch.float32

        layer_stats[layer_idx] = {
            "sample": RunningStats(gate_proj.out_features, dtype=layer_dtype, device=layer_device),
            "token": RunningStats(gate_proj.out_features, dtype=layer_dtype, device=layer_device),
            "class_sum": torch.zeros((num_classes, gate_proj.out_features), dtype=layer_dtype, device=layer_device) if num_classes else None,
            "class_count": torch.zeros(num_classes, dtype=layer_dtype, device=layer_device) if num_classes else None,
        }

        def make_hook(idx):
            def hook(_module, _inputs, output):
                if output is None:
                    return
                act = F.silu(output)
                if act.dim() == 3:
                    per_sample = act.mean(dim=1)
                    per_token = act.flatten(0, 1)
                else:
                    per_sample = act
                    per_token = act

                per_sample_local = per_sample.detach().to(dtype=layer_dtype)
                per_token_local = per_token.detach().to(dtype=layer_dtype)

                layer_stats[idx]["sample"].update(per_sample_local)
                layer_stats[idx]["token"].update(per_token_local)

                if class_index_lookup is not None:
                    flat_ids = current_batch_token_ids
                    if flat_ids is None or flat_ids.numel() != per_token_local.shape[0]:
                        return
                    lookup = class_lookup_cache.get(per_sample_local.device)
                    if lookup is None:
                        lookup = class_index_lookup.to(per_sample_local.device)
                        class_lookup_cache[per_sample_local.device] = lookup
                    within_vocab = flat_ids < lookup.shape[0]
                    if not within_vocab.any():
                        return
                    masked_ids = flat_ids[within_vocab]
                    class_indices = lookup[masked_ids]
                    valid_mask = class_indices != -1
                    if not valid_mask.any():
                        return
                    idxs = class_indices[valid_mask]
                    token_values = per_token_local[within_vocab][valid_mask]
                    layer_stats[idx]["class_sum"].index_add_(0, idxs, token_values)
                    layer_stats[idx]["class_count"].index_add_(0, idxs, torch.ones(idxs.size(0), dtype=layer_dtype, device=token_values.device))

            return hook

        hooks.append(gate_proj.register_forward_hook(make_hook(layer_idx)))

    model.eval()
    with torch.no_grad():
        for batch in batches:
            if isinstance(batch, (tuple, list)):
                input_ids = batch[0]
            else:
                input_ids = batch
            input_ids = input_ids.to(device)
            attention_mask = torch.ones_like(input_ids, device=device)
            current_batch_token_ids = input_ids.reshape(-1)
            model(input_ids=input_ids, attention_mask=attention_mask)
            current_batch_token_ids = None

    for handle in hooks:
        handle.remove()

    stats = {}
    for idx, stat in layer_stats.items():
        sample_var = stat["sample"].variance().to(dtype=torch.float32, device="cpu")
        token_var = stat["token"].variance().to(dtype=torch.float32, device="cpu")
        class_var = None
        if stat.get("class_sum") is not None:
            counts = stat["class_count"]
            valid = counts > 0
            if valid.any():
                class_means = (stat["class_sum"][valid] / counts[valid].unsqueeze(1)).to(dtype=torch.float32, device="cpu")
                if class_means.size(0) > 1:
                    class_var = class_means.var(dim=0, unbiased=False)
                else:
                    class_var = torch.zeros_like(sample_var)
        stats[idx] = {
            "sample_variance": sample_var,
            "token_variance": token_var,
            "class_variance": class_var,
        }
    return stats


def _safe_scalar_pow(base: float, exponent: float) -> float:
    if exponent == 0.0:
        return 1.0
    if base <= 0.0:
        return 0.0
    return float(base) ** float(exponent)


def compute_neuronrank_scores(
    model,
    stats,
    token_weight=0.0,
    variance_exp=1.0,
    variance_multi=1.0,
    magnitude_multi=0.0,
) -> Dict[int, Dict[str, Optional[torch.Tensor]]]:
    scores: Dict[int, Dict[str, Optional[torch.Tensor]]] = {}
    for layer_idx, layer in enumerate(model.model.layers):
        gate_proj = getattr(layer.mlp, "gate_proj", None)
        if gate_proj is None:
            continue
        weight = gate_proj.weight.detach().to(dtype=torch.float32, device="cpu")
        row_norm = torch.norm(weight, p=2, dim=1)

        layer_stats = stats.get(layer_idx)
        if layer_stats is None:
            variance = torch.zeros_like(row_norm)
        else:
            variance = layer_stats["sample_variance"].to(row_norm.device)
            if token_weight > 0:
                variance = variance + token_weight * layer_stats["token_variance"].to(row_norm.device)
            class_variance = layer_stats.get("class_variance")
            if class_variance is not None:
                variance = variance + class_variance.to(row_norm.device)

        variance = variance.clamp(min=0.0)
        
        if variance_exp == 0.0:
            variance_term = torch.ones_like(variance)
        elif variance_exp == 1.0:
            variance_term = variance
        else:
            variance_term = torch.pow(variance.clamp(min=1e-12), variance_exp)
        
        variance_component = variance_term * variance_multi
        magnitude_component = row_norm * magnitude_multi
        channel_score = variance_component + magnitude_component

        scores[layer_idx] = {
            "channel": channel_score,
            "variance": variance if variance is not None else None,
        }
    return scores


def compute_neuronrank_class_scores(stats, min_value: float = 0.0):
    """Derive per-neuron scores directly from class-wise variance."""

    scores = {}
    for layer_idx, layer_stats in stats.items():
        class_variance = layer_stats.get("class_variance")
        if class_variance is None:
            continue
        scores[layer_idx] = class_variance.clamp_min(min_value)
    return scores


def apply_neuronrank_pruning(model, scores, sparsity_ratio):
    total_channels = 0
    total_pruned = 0

    for layer_idx, layer in enumerate(model.model.layers):
        layer_entry = scores.get(layer_idx)
        if not layer_entry:
            continue

        layer_score = layer_entry.get("channel") if isinstance(layer_entry, dict) else layer_entry
        if layer_score is None:
            continue
        gate_proj = getattr(layer.mlp, "gate_proj", None)
        up_proj = getattr(layer.mlp, "up_proj", None)
        down_proj = getattr(layer.mlp, "down_proj", None)
        if gate_proj is None:
            continue

        num_channels = layer_score.numel()
        total_channels += num_channels
        num_to_prune = int(num_channels * sparsity_ratio)
        if num_to_prune <= 0:
            continue

        prune_idx = torch.argsort(layer_score)[:num_to_prune]
        prune_idx_list = prune_idx.tolist()

        gate_proj.weight.data[prune_idx_list, :] = 0
        if gate_proj.bias is not None:
            gate_proj.bias.data[prune_idx_list] = 0

        if up_proj is not None:
            up_proj.weight.data[prune_idx_list, :] = 0
            if up_proj.bias is not None:
                up_proj.bias.data[prune_idx_list] = 0

        if down_proj is not None:
            down_proj.weight.data[:, prune_idx_list] = 0

        layer_pct = 100.0 * num_to_prune / num_channels
        print(f"[NeuronRank] layer {layer_idx}: pruned {num_to_prune}/{num_channels} channels ({layer_pct:.2f}%)")
        total_pruned += num_to_prune

    return total_pruned, total_channels


def collect_neuronrank_old_statistics(model, dataloader, device):
    """Collect TF-IDF statistics for old NeuronRank formula.
    
    For each MLP gate projection, tracks:
    - TF: Average absolute activation strength across all tokens
    - IDF: Selectivity measure based on how often neuron fires
    """
    
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise ValueError("Expected the model to expose model.layers for NeuronRank scoring.")

    batches = list(dataloader)
    if not batches:
        raise ValueError("Calibration dataloader for NeuronRank produced no batches.")

    print(f"Collecting TF-IDF statistics ({len(batches)} batches)")

    layer_stats = {}
    hooks = []

    for layer_idx, layer in enumerate(model.model.layers):
        gate_proj = getattr(layer.mlp, "gate_proj", None)
        if gate_proj is None:
            continue

        layer_device = gate_proj.weight.device
        layer_dtype = torch.float32

        layer_stats[layer_idx] = TFIDFStats(
            size=gate_proj.out_features,
            dtype=layer_dtype,
            device=layer_device
        )

        def make_hook(idx):
            def hook(_module, _inputs, output):
                if output is None:
                    return
                # Apply SiLU activation
                act = F.silu(output)
                
                # Flatten to [num_tokens, neuron_dim]
                if act.dim() == 3:
                    act_flat = act.flatten(0, 1)  # [batch*seq_len, neuron_dim]
                else:
                    act_flat = act
                
                # Update TF-IDF statistics
                layer_stats[idx].update(act_flat)
            
            return hook

        hooks.append(gate_proj.register_forward_hook(make_hook(layer_idx)))

    model.eval()
    with torch.no_grad():
        for batch in batches:
            if isinstance(batch, (tuple, list)):
                input_ids = batch[0]
            else:
                input_ids = batch
            input_ids = input_ids.to(device)
            attention_mask = torch.ones_like(input_ids, device=device)
            model(input_ids=input_ids, attention_mask=attention_mask)

    for handle in hooks:
        handle.remove()

    # Compute final TF and IDF scores
    stats = {}
    for idx, tfidf_stats in layer_stats.items():
        tf, idf = tfidf_stats.compute_tf_idf()
        stats[idx] = {
            "tf": tf.to(dtype=torch.float32, device="cpu"),
            "idf": idf.to(dtype=torch.float32, device="cpu"),
            "total_tokens": tfidf_stats.total_tokens,
        }
        print(f"Layer {idx}: TF range [{tf.min():.6f}, {tf.max():.6f}], "
              f"IDF range [{idf.min():.6f}, {idf.max():.6f}], "
              f"tokens={tfidf_stats.total_tokens}")
    
    return stats


def compute_neuronrank_old_scores(
    model,
    stats,
    weight_exp=1.0,
    tf_exp=1.0, 
    idf_exp=1.0,
) -> Dict[int, torch.Tensor]:
    """Compute scores using old NeuronRank formula: |W|^α × TF^β × IDF^γ
    
    Args:
        model: The model to score
        stats: TF-IDF statistics from collect_neuronrank_old_statistics
        weight_exp: Exponent α for weight magnitude
        tf_exp: Exponent β for term frequency (activation strength)
        idf_exp: Exponent γ for inverse document frequency (selectivity)
    
    Returns:
        Dictionary mapping layer_idx to neuron scores
    """
    scores = {}
    
    for layer_idx, layer in enumerate(model.model.layers):
        gate_proj = getattr(layer.mlp, "gate_proj", None)
        if gate_proj is None:
            continue
        
        weight = gate_proj.weight.detach().to(dtype=torch.float32, device="cpu")
        # Weight magnitude per neuron (L2 norm across input dimension)
        weight_magnitude = torch.norm(weight, p=2, dim=1)
        
        layer_stats = stats.get(layer_idx)
        if layer_stats is None:
            # No statistics available, use weight magnitude only
            scores[layer_idx] = torch.pow(weight_magnitude.clamp(min=1e-12), weight_exp)
            continue
        
        tf = layer_stats["tf"].to("cpu")
        idf = layer_stats["idf"].to("cpu")
        
        # Apply exponents: |W|^α × TF^β × IDF^γ
        if weight_exp == 0.0:
            weight_term = torch.ones_like(weight_magnitude)
        elif weight_exp == 1.0:
            weight_term = weight_magnitude
        else:
            weight_term = torch.pow(weight_magnitude.clamp(min=1e-12), weight_exp)
        
        if tf_exp == 0.0:
            tf_term = torch.ones_like(tf)
        elif tf_exp == 1.0:
            tf_term = tf
        else:
            tf_term = torch.pow(tf.clamp(min=1e-12), tf_exp)
        
        if idf_exp == 0.0:
            idf_term = torch.ones_like(idf)
        elif idf_exp == 1.0:
            idf_term = idf
        else:
            idf_term = torch.pow(idf.clamp(min=1e-12), idf_exp)
        
        # Combine: score = |W|^α × TF^β × IDF^γ
        score = weight_term * tf_term * idf_term
        scores[layer_idx] = score
        
        print(f"Layer {layer_idx} scores: min={score.min():.6e}, max={score.max():.6e}, mean={score.mean():.6e}")
    
    return scores
