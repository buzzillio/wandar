from typing import Dict, Optional, Union

import torch
import torch.nn.functional as F


class TFIDFStats:
    """Track TF-IDF statistics for old NeuronRank formula.
    
    TF (Term Frequency): Average activation strength across all tokens
    IDF (Inverse Document Frequency): Selectivity based on how rarely neuron fires
    """

    def __init__(self, size: int, dtype: torch.dtype = torch.float32, device: Union[torch.device, str] = "cpu", activation_threshold: float = 1e-6):
        self.size = size
        self.dtype = dtype
        self.device = device
        self.activation_threshold = activation_threshold
        self.total_tokens = 0
        # Sum of absolute activations for TF
        self.activation_sum = torch.zeros(size, dtype=dtype, device=device)
        # Count of tokens where neuron fired (activation > threshold)
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
        
        # IDF: count where neuron is active (activation > threshold)
        # Threshold avoids counting near-zero floating point values
        # This gives better selectivity signal than activation > 0
        active_mask = abs_act > self.activation_threshold
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
            "class_sq_sum": torch.zeros((num_classes, gate_proj.out_features), dtype=layer_dtype, device=layer_device) if num_classes else None,
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
                    layer_stats[idx]["class_sq_sum"].index_add_(0, idxs, token_values * token_values)
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
        sample_tracker: RunningStats = stat["sample"]
        token_tracker: RunningStats = stat["token"]

        sample_var = sample_tracker.variance().to(dtype=torch.float32, device="cpu")
        token_var = token_tracker.variance().to(dtype=torch.float32, device="cpu")
        token_mean = token_tracker.mean.to(dtype=torch.float32, device="cpu")
        token_count = float(token_tracker.count)

        class_var = None
        class_sum_cpu = None
        class_sq_sum_cpu = None
        class_count_cpu = None
        class_means_cpu = None
        class_vars_cpu = None

        class_sum = stat.get("class_sum")
        class_sq_sum = stat.get("class_sq_sum")
        class_count = stat.get("class_count")
        if class_sum is not None and class_sq_sum is not None and class_count is not None:
            class_sum_cpu = class_sum.to(dtype=torch.float32, device="cpu")
            class_sq_sum_cpu = class_sq_sum.to(dtype=torch.float32, device="cpu")
            class_count_cpu = class_count.to(dtype=torch.float32, device="cpu")
            valid = class_count_cpu > 0
            if valid.any():
                class_means_cpu = torch.zeros_like(class_sum_cpu)
                class_means_cpu[valid] = class_sum_cpu[valid] / class_count_cpu[valid].unsqueeze(1)
                # compute class variances (diagonal assumption)
                class_vars_cpu = torch.zeros_like(class_sum_cpu)
                class_vars_cpu[valid] = (class_sq_sum_cpu[valid] / class_count_cpu[valid].unsqueeze(1)) - class_means_cpu[valid] * class_means_cpu[valid]
                class_vars_cpu = torch.clamp(class_vars_cpu, min=0.0)
                nonzero_means = class_means_cpu[valid]
                if nonzero_means.size(0) > 1:
                    class_var = nonzero_means.var(dim=0, unbiased=False)
                else:
                    class_var = torch.zeros_like(sample_var)
            else:
                class_var = torch.zeros_like(sample_var)

        stats[idx] = {
            "sample_variance": sample_var,
            "token_variance": token_var,
            "class_variance": class_var,
            "token_mean": token_mean,
            "token_count": token_count,
            "class_sum": class_sum_cpu,
            "class_sq_sum": class_sq_sum_cpu,
            "class_count": class_count_cpu,
            "class_means": class_means_cpu,
            "class_vars": class_vars_cpu,
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


def compute_neuronrank_between_scores(stats, weighted: bool = True, eps: float = 1e-12):
    """Compute simple between-class variance per neuron using class means.

    If weighted=True, use class priors by counts:
        between_j = sum_k p_k (mu_kj - mu_bar_j)^2
    else (unweighted):
        between_j = mean_k (mu_kj - mu_bar_j)^2
    """
    scores = {}
    for layer_idx, layer_stats in stats.items():
        token_mean = layer_stats.get("token_mean")
        class_means = layer_stats.get("class_means")
        class_counts = layer_stats.get("class_count")

        if token_mean is None or class_means is None:
            continue

        means = class_means
        if not torch.is_tensor(means):
            continue

        if weighted:
            counts = class_counts
            if counts is None or not torch.is_tensor(counts):
                continue
            valid = counts > 0
            if not valid.any():
                continue
            counts_valid = counts[valid]
            means_valid = means[valid]
            total = counts_valid.sum()
            if total <= 0:
                continue
            priors = (counts_valid / total).unsqueeze(1)
            diff = means_valid - token_mean.unsqueeze(0)
            between = (priors * (diff * diff)).sum(dim=0)
        else:
            # unweighted: average across non-zero classes
            # mask rows with any nonzero count if available
            if class_counts is not None and torch.is_tensor(class_counts):
                valid = class_counts > 0
                if valid.any():
                    means = means[valid]
            diff = means - token_mean.unsqueeze(0)
            between = (diff * diff).mean(dim=0)

        between = torch.nan_to_num(between, nan=0.0, posinf=0.0, neginf=0.0)
        scores[layer_idx] = {"channel": between.clamp_min(0.0)}

    return scores

def compute_neuronrank_qda_scores(stats, eps: float = 1e-12):
    """Compute QDA-inspired per-neuron scores using class means and variances.

    We assume diagonal covariance per neuron. For each neuron j, compute
    score_j = sum_k p_k * (mu_kj - mu_bar_j)^2 / (var_kj + eps)
    where p_k are class priors (by counts), mu_bar is overall mean.
    """
    qda_scores: Dict[int, Dict[str, torch.Tensor]] = {}

    for layer_idx, layer_stats in stats.items():
        token_mean = layer_stats.get("token_mean")
        class_means = layer_stats.get("class_means")
        class_vars = layer_stats.get("class_vars")
        class_counts = layer_stats.get("class_count")

        if token_mean is None or class_means is None or class_vars is None or class_counts is None:
            continue

        counts = class_counts
        means = class_means
        vars_ = class_vars
        if not (torch.is_tensor(counts) and torch.is_tensor(means) and torch.is_tensor(vars_)):
            continue

        valid = counts > 1  # need at least 2 to estimate variance
        if not valid.any():
            continue

        counts_valid = counts[valid]
        means_valid = means[valid]
        vars_valid = vars_[valid]
        total = counts_valid.sum()
        if total <= 0:
            continue

        priors = (counts_valid / total).unsqueeze(1)  # [K,1]
        diff = means_valid - token_mean.unsqueeze(0)   # [K, D]
        denom = vars_valid + eps                       # [K, D]
        contrib = priors * (diff * diff) / denom       # [K, D]
        score = contrib.sum(dim=0)                     # [D]
        # sanitize and clamp
        score = torch.nan_to_num(score, nan=0.0, posinf=torch.finfo(score.dtype).max, neginf=0.0)
        qda_scores[layer_idx] = {"channel": score}

    return qda_scores


def compute_neuronrank_pca_qda_scores(stats, pca_components: int = 128, eps: float = 1e-12):
    """Compute PCA+QDA scores: project to PCA space, then apply QDA discriminant analysis.
    
    Steps:
    1. Compute PCA on class means (capturing main directions of class separation)
    2. Project class means and overall mean to PCA space
    3. Compute QDA discriminant score in PCA space
    4. Back-project scores to original neuron space as importance weights
    
    Args:
        stats: NeuronRank statistics with class means, vars, counts
        pca_components: Number of PCA components to retain (default 128)
        eps: Numerical stability constant
    
    Returns:
        Dictionary mapping layer_idx to {"channel": score_tensor}
    """
    pca_qda_scores: Dict[int, Dict[str, torch.Tensor]] = {}

    for layer_idx, layer_stats in stats.items():
        token_mean = layer_stats.get("token_mean")
        class_means = layer_stats.get("class_means")
        class_vars = layer_stats.get("class_vars")
        class_counts = layer_stats.get("class_count")

        if token_mean is None or class_means is None or class_vars is None or class_counts is None:
            continue

        counts = class_counts
        means = class_means
        vars_ = class_vars
        if not (torch.is_tensor(counts) and torch.is_tensor(means) and torch.is_tensor(vars_)):
            continue

        valid = counts > 1
        if not valid.any():
            continue

        counts_valid = counts[valid]
        means_valid = means[valid]
        vars_valid = vars_[valid]
        total = counts_valid.sum()
        if total <= 0:
            continue

        D = means_valid.shape[1]
        K = means_valid.shape[0]
        
        # Step 1: PCA on class means (centered by overall mean)
        centered_means = means_valid - token_mean.unsqueeze(0)  # [K, D]
        
        # Handle case where we have fewer classes than requested components
        n_components = min(pca_components, K, D)
        
        if K < 2 or D < 2:
            # Fallback to simple between-class variance
            priors = (counts_valid / total).unsqueeze(1)
            diff = means_valid - token_mean.unsqueeze(0)
            score = (priors * (diff * diff)).sum(dim=0)
            score = torch.nan_to_num(score, nan=0.0, posinf=torch.finfo(score.dtype).max, neginf=0.0)
            pca_qda_scores[layer_idx] = {"channel": score.clamp_min(0.0)}
            continue
        
        # Compute covariance matrix of centered class means
        # cov = (1/K) * X^T X where X is [K, D]
        cov_matrix = torch.mm(centered_means.t(), centered_means) / K  # [D, D]
        
        # Eigendecomposition
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
            # Sort by descending eigenvalue
            idx = torch.argsort(eigenvalues, descending=True)
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # Take top n_components
            principal_components = eigenvectors[:, :n_components]  # [D, n_components]
            
        except RuntimeError:
            # If eigendecomposition fails, fall back to simple approach
            priors = (counts_valid / total).unsqueeze(1)
            diff = means_valid - token_mean.unsqueeze(0)
            score = (priors * (diff * diff)).sum(dim=0)
            score = torch.nan_to_num(score, nan=0.0, posinf=torch.finfo(score.dtype).max, neginf=0.0)
            pca_qda_scores[layer_idx] = {"channel": score.clamp_min(0.0)}
            continue
        
        # Step 2: Project class means and variances to PCA space
        means_pca = torch.mm(means_valid, principal_components)  # [K, n_components]
        token_mean_pca = torch.mm(token_mean.unsqueeze(0), principal_components).squeeze(0)  # [n_components]
        
        # Project variances (diagonal approximation)
        # var_pca ≈ PC^T * diag(var) * PC, but we use squared projection as approximation
        vars_pca = torch.mm(vars_valid, principal_components ** 2)  # [K, n_components]
        
        # Step 3: QDA in PCA space
        priors = (counts_valid / total).unsqueeze(1)  # [K, 1]
        diff_pca = means_pca - token_mean_pca.unsqueeze(0)  # [K, n_components]
        denom_pca = vars_pca + eps  # [K, n_components]
        contrib_pca = priors * (diff_pca * diff_pca) / denom_pca  # [K, n_components]
        score_pca = contrib_pca.sum(dim=0)  # [n_components]
        
        # Step 4: Back-project to original space
        # Use principal component loadings weighted by PCA-QDA scores
        score = torch.mm(score_pca.unsqueeze(0), principal_components.t()).squeeze(0)  # [D]
        
        # Alternative: Use squared loadings weighted by importance
        # This emphasizes neurons that contribute most to discriminative PCA directions
        pc_squared = principal_components ** 2  # [D, n_components]
        neuron_importance = torch.mm(pc_squared, score_pca.unsqueeze(1)).squeeze(1)  # [D]
        
        # Combine both signals
        score = score * 0.5 + neuron_importance * 0.5
        
        # Sanitize
        score = torch.nan_to_num(score, nan=0.0, posinf=torch.finfo(score.dtype).max, neginf=0.0)
        pca_qda_scores[layer_idx] = {"channel": score.clamp_min(0.0)}

    return pca_qda_scores


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


def collect_neuronrank_old_statistics(model, dataloader, device, activation_threshold=1e-6):
    """Collect TF-IDF statistics for old NeuronRank formula.
    
    For each MLP gate projection, tracks:
    - TF: Average absolute activation strength across all tokens
    - IDF: Selectivity measure based on how often neuron fires
    
    Args:
        model: The model to collect statistics from
        dataloader: Calibration data
        device: Device to run on
        activation_threshold: Threshold for considering neuron "active" (default: 1e-6)
    """
    
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise ValueError("Expected the model to expose model.layers for NeuronRank scoring.")

    batches = list(dataloader)
    if not batches:
        raise ValueError("Calibration dataloader for NeuronRank produced no batches.")

    print(f"Collecting TF-IDF statistics ({len(batches)} batches, threshold={activation_threshold})")

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
            device=layer_device,
            activation_threshold=activation_threshold
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


# ---------- Doc-level TF-IDF stats (NR-TFIDF++) ----------
class DocTFIDFStats:
    """Document-level TF-IDF statistics for neurons.
    
    Treats each calibration sequence as a 'document' and computes:
    - TF: average |activation| across all tokens
    - DF: number of documents where neuron is active (above threshold)
    - IDF: log((N_docs + 1) / (DF + 1)) + 1
    """
    def __init__(self, device, dtype=torch.float32, q_active=0.60, eps=1e-12):
        self.device = device
        self.dtype = dtype
        self.q_active = q_active  # Percentile threshold for "active"
        self.eps = eps
        self.doc_count = 0
        self.channel_sum = {}      # layer_name -> [C] sum |act|
        self.doc_active = {}       # layer_name -> [C] number of docs where channel active
        self.doc_thresholds = {}   # layer_name -> [C] running q_active threshold estimate

    def _ensure(self, key, C):
        """Initialize tensors for a new layer."""
        if key not in self.channel_sum:
            self.channel_sum[key] = torch.zeros(C, device=self.device, dtype=self.dtype)
            self.doc_active[key] = torch.zeros(C, device=self.device, dtype=torch.int32)
            self.doc_thresholds[key] = torch.zeros(C, device=self.device, dtype=self.dtype)

    @torch.no_grad()
    def update_doc(self, layer_key: str, acts: torch.Tensor):
        """Update statistics with activations from one document (sequence).
        
        Args:
            layer_key: Layer identifier (e.g., "layer_5.mlp.gate_proj")
            acts: Activation tensor [tokens, channels]
        """
        T, C = acts.shape
        self._ensure(layer_key, C)
        A = acts.abs().to(device=self.device, dtype=self.dtype)

        # Per-channel active threshold from this doc (approximate quantile)
        k = max(1, int(self.q_active * T))
        thresh, _ = torch.kthvalue(A, k, dim=0)  # [C]
        self.doc_thresholds[layer_key] += thresh

        # TF: sum of activations across all tokens
        self.channel_sum[layer_key] += A.sum(dim=0)
        
        # DF: count documents where channel is active at least once
        active_mask = (A >= thresh.unsqueeze(0)).any(dim=0)
        self.doc_active[layer_key] += active_mask.to(torch.int32)
        
        self.doc_count += 1

    @torch.no_grad()
    def finalize(self):
        """Compute final TF and IDF tensors."""
        results = {}
        for key in self.channel_sum.keys():
            # TF: mean activation over all tokens across docs
            # Assume ~128 tokens/doc average
            total_tokens = self.doc_count * 128
            tf = self.channel_sum[key] / max(total_tokens, 1)
            
            # IDF: inverse document frequency
            df = self.doc_active[key].to(torch.float32)
            idf = torch.log((self.doc_count + 1.0) / (df + 1.0)) + 1.0
            
            # Sanitize
            tf = torch.nan_to_num(tf, nan=0.0, posinf=0.0, neginf=0.0)
            idf = torch.clamp(torch.nan_to_num(idf, nan=1.0), min=0.0, max=10.0)
            
            results[key] = (tf.to(self.dtype), idf.to(self.dtype))
        
        return results


# ---------- Topic-level TF-IDF stats ----------
def kmeans_assign(x, k, iters=10):
    """Simple cosine k-means clustering on device.
    
    Args:
        x: Input tensor [N, D]
        k: Number of clusters
        iters: Number of iterations
    
    Returns:
        labels: Cluster assignments [N]
    """
    x = torch.nn.functional.normalize(x, dim=1)
    N, D = x.shape
    
    # Initialize centers with random subset
    idx = torch.randperm(N, device=x.device)[:k]
    centers = x[idx].clone()
    
    for _ in range(iters):
        # Assign to nearest center
        sims = x @ centers.T  # [N, k]
        labels = sims.argmax(dim=1)  # [N]
        
        # Update centers
        for j in range(k):
            mask = labels == j
            if mask.any():
                centers[j] = torch.nn.functional.normalize(x[mask].mean(dim=0), dim=0)
    
    return labels


class TopicTFIDFStats(DocTFIDFStats):
    """Topic-level TF-IDF statistics using semantic clustering.
    
    Extends DocTFIDFStats by clustering tokens into K topics and computing
    TF-IDF at the topic level rather than document level.
    """
    def __init__(self, device, dtype=torch.float32, k_topics=64, proj_dim=128, **kw):
        super().__init__(device, dtype, **kw)
        self.k = k_topics
        self.proj = None
        self.proj_dim = proj_dim
        self.topic_tf_sum = {}   # layer_key -> [C]
        self.topic_df = {}       # layer_key -> [C]
        self.topic_count = 0

    @torch.no_grad()
    def update_doc_with_topics(self, layer_key: str, acts: torch.Tensor):
        """Update statistics with topic-clustered activations.
        
        Args:
            layer_key: Layer identifier
            acts: Activation tensor [tokens, channels]
        """
        T, C = acts.shape
        self._ensure(layer_key, C)
        A = acts.abs().to(device=self.device, dtype=self.dtype)

        # Lazy-init random projection for clustering tokens
        if self.proj is None:
            D = A.shape[1]
            self.proj = torch.randn(D, self.proj_dim, device=A.device, dtype=A.dtype) / (D ** 0.5)

        # Project tokens and cluster
        Z = A @ self.proj  # [tokens, proj_dim]
        labels = kmeans_assign(Z, self.k, iters=4)
        
        # Initialize accumulators if needed
        if layer_key not in self.topic_tf_sum:
            self.topic_tf_sum[layer_key] = torch.zeros(C, device=A.device, dtype=A.dtype)
            self.topic_df[layer_key] = torch.zeros(C, device=A.device, dtype=torch.int32)
        
        # For each topic, compute per-channel mean and active-any
        for t in range(self.k):
            mask = labels == t
            if not mask.any():
                continue
            
            chunk = A[mask]  # [tokens_in_topic, C]
            tf_t = chunk.mean(dim=0)  # Per-channel mean
            
            # Count as "active in topic" if any token crosses within-topic threshold
            k_val = max(1, int(0.60 * chunk.shape[0]))
            thr, _ = torch.kthvalue(chunk, k_val, dim=0)
            active = (chunk >= thr.unsqueeze(0)).any(dim=0)

            self.topic_tf_sum[layer_key] += tf_t
            self.topic_df[layer_key] += active.to(torch.int32)

        self.topic_count += 1

    @torch.no_grad()
    def finalize_topics(self):
        """Compute final topic-level TF and IDF tensors."""
        results = {}
        for key in self.topic_tf_sum.keys():
            # TF: average activation strength across topics
            tf_topic = self.topic_tf_sum[key] / max(self.topic_count, 1)
            
            # IDF: inverse topic frequency
            df_topic = self.topic_df[key].to(torch.float32)
            idf_topic = torch.log((self.k + 1.0) / (df_topic + 1.0)) + 1.0
            
            # Sanitize
            tf_topic = torch.nan_to_num(tf_topic, nan=0.0)
            idf_topic = torch.clamp(torch.nan_to_num(idf_topic, nan=1.0), min=0.0, max=10.0)
            
            results[key] = (tf_topic.to(self.dtype), idf_topic.to(self.dtype))
        
        return results


# ---------- Compute per-weight scores from TF-IDF tensors ----------
def broadcast_to_weights(weight: torch.Tensor, per_channel_vec: torch.Tensor, name: str):
    """Broadcast per-channel scores to per-weight scores.
    
    Args:
        weight: Weight tensor [out_features, in_features]
        per_channel_vec: Per-channel scores [channels]
        name: Module name (determines broadcast direction)
    
    Returns:
        Broadcasted tensor matching weight shape
    """
    # MLP projections
    if ("gate_proj" in name) or ("up_proj" in name):
        # Column-wise scores (output channels)
        return per_channel_vec.view(-1, 1).expand_as(weight)
    elif "down_proj" in name:
        # Row-wise scores (input channels)
        return per_channel_vec.view(1, -1).expand_as(weight)
    
    # Attention projections
    elif ("q_proj" in name) or ("k_proj" in name) or ("v_proj" in name):
        # Column-wise scores (output channels)
        return per_channel_vec.view(-1, 1).expand_as(weight)
    elif "o_proj" in name:
        # Row-wise scores (input channels from concat heads)
        return per_channel_vec.view(1, -1).expand_as(weight)
    
    else:
        # Fallback: column-wise
        return per_channel_vec.view(-1, 1).expand_as(weight)


def compute_tfidf_scores(weight: torch.Tensor,
                         name: str,
                         tf: torch.Tensor,
                         idf: torch.Tensor,
                         alpha=1.0,
                         beta=1.0,
                         gamma=1.0,
                         add_spikiness: torch.Tensor = None,
                         rho=0.0):
    """Compute per-weight importance scores using TF-IDF formula.
    
    Score = |W|^α × TF^β × IDF^γ × (spikiness^ρ if provided)
    
    Args:
        weight: Weight tensor
        name: Module name
        tf: Per-channel term frequency
        idf: Per-channel inverse document frequency
        alpha: Weight magnitude exponent
        beta: TF exponent
        gamma: IDF exponent
        add_spikiness: Optional per-channel spikiness scores
        rho: Spikiness exponent
    
    Returns:
        Per-weight importance scores
    """
    Wmag = weight.abs().to(torch.float32)
    
    # Per-channel score: TF^β × IDF^γ
    s_ch = (tf.clamp_min(0.0) ** beta) * (idf.clamp_min(0.0) ** gamma)
    
    # Optional spikiness multiplier
    if rho > 0.0 and (add_spikiness is not None):
        s_ch = s_ch * (add_spikiness.clamp_min(1e-6) ** rho)
    
    # Broadcast to per-weight and combine with magnitude
    S = broadcast_to_weights(Wmag, s_ch, name)
    return (Wmag ** alpha) * S
