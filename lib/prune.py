import time 
import heapq 
import torch 
import torch.nn as nn 
from .sparsegpt import SparseGPT 
from .layerwrapper import WrappedGPT
from .data import get_loaders 
from .neuronrank import (
    collect_neuronrank_statistics,
    compute_neuronrank_scores,
    compute_neuronrank_fisher_scores,
    apply_neuronrank_pruning,
)
from .wanda_selectivity import collect_wanda_selectivity_stats

from .ablate import AblateGPT 

def should_prune_layer(args, layer_idx, total_layers):
    """
    Determine if a layer should be pruned based on pruning_last flag.
    
    Args:
        args: Command line arguments
        layer_idx: Index of the current layer (0-based)
        total_layers: Total number of layers in the model
        
    Returns:
        bool: True if layer should be pruned, False otherwise
    """
    if args.pruning_last is None:
        return True  # Prune all layers if no restriction
    
    # Only prune the last X layers
    return layer_idx >= (total_layers - args.pruning_last)

def should_prune_module(args, layer_idx, total_layers, module_name):
    """
    Determine if a specific module in a layer should be pruned.
    
    Args:
        args: Command line arguments
        layer_idx: Index of the current layer (0-based)
        total_layers: Total number of layers in the model
        module_name: Name of the module (e.g., 'mlp.gate_proj', 'self_attn.q_proj')
        
    Returns:
        bool: True if module should be pruned, False otherwise
    """
    # First check if we should prune this layer at all
    if not should_prune_layer(args, layer_idx, total_layers):
        return False
    
    # If pruning_last is specified, only prune MLP modules
    if args.pruning_last is not None:
        return 'mlp' in module_name
    
    # Otherwise, follow normal pruning rules
    return True

def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def check_sparsity(model):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    layers = model.model.layers
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            
            # Handle meta tensors - skip them as they have no actual data
            if W.device.type == 'meta':
                print(f"Warning: Skipping layer {i} {name} (meta tensor)")
                continue
            
            # Move to CPU if needed to avoid device issues
            if W.device.type == 'cuda':
                zero_count = (W == 0).sum().cpu().item()
                param_count = W.numel()
            else:
                zero_count = (W == 0).sum().item()
                param_count = W.numel()
            
            count += zero_count
            total_params += param_count
            sub_count += zero_count
            sub_params += param_count

        if sub_params > 0:
            print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache 
    
    if total_params == 0:
        print("Warning: No parameters found, returning 0 sparsity")
        return 0.0
    
    return float(count)/total_params 

def prepare_calibration_input(model, dataloader, device):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    device_map = getattr(model, "hf_device_map", None)
    if device_map and "model.embed_tokens" in device_map:
        device = device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((128, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass 
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids 

def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha 
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1,1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True)-1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask==True).sum() / W_mask.numel()
    return W_mask, cur_sparsity


def _apply_unstructured_mask(weight_tensor: torch.Tensor, metric_tensor: torch.Tensor, ratio: float):
    """Zero-out the lowest-scoring weights according to the metric."""
    numel = weight_tensor.numel()
    if numel == 0:
        return 0, 0

    num_to_prune = int(ratio * numel)
    if num_to_prune <= 0:
        return 0, numel

    flat_metric = metric_tensor.reshape(-1)
    k = min(max(num_to_prune, 1), flat_metric.numel())
    threshold = torch.kthvalue(flat_metric, k).values

    mask = metric_tensor <= threshold
    pruned = mask.sum().item()
    weight_tensor[mask] = 0
    return pruned, numel

def prune_magnitude(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    layers = model.model.layers 
    total_layers = len(layers)

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            # Check if we should prune this module based on pruning_last flag
            if not should_prune_module(args, i, total_layers, name):
                if args.pruning_last is not None:
                    print(f"[Magnitude] Skipping layer {i} module {name} (not in last {args.pruning_last} MLP layers)")
                continue
            
            print(f"[Magnitude] Pruning layer {i} module {name}")
            W = subset[name].weight.data
            
            # Handle meta tensors by getting the actual device from device_map
            if W.device.type == 'meta':
                if hasattr(model, 'hf_device_map'):
                    # Find the actual device for this layer
                    layer_key = f"model.layers.{i}"
                    if layer_key in model.hf_device_map:
                        actual_device = model.hf_device_map[layer_key]
                        print(f"Warning: Layer {i} {name} on meta device, expected device: {actual_device}")
                        print(f"Skipping - model may not be fully loaded. Try running without device_map='auto'")
                    else:
                        print(f"ERROR: Layer {i} {name} is on meta device and no device mapping found!")
                else:
                    print(f"ERROR: Layer {i} {name} is on meta device - model not fully loaded!")
                continue
            
            W_metric = torch.abs(W).to(torch.float32)
            if prune_n != 0:
                W_mask = (torch.zeros_like(W)==1)
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
                W[W_mask] = 0
                pruned = W_mask.sum().item()
                print(f"[Magnitude] Pruned layer {i} module {name}: {pruned}/{W.numel()} weights")
            else:
                pruned, numel = _apply_unstructured_mask(W, W_metric, args.sparsity_ratio)
                print(f"[Magnitude] Pruned layer {i} module {name}: {pruned}/{numel} weights")
    
    # Clean up memory after pruning
    torch.cuda.empty_cache()
    import gc
    gc.collect()

def prune_wanda(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    # Get Wanda hyperparameters
    w_alpha = getattr(args, 'w_alpha', 1.0)
    w_beta = getattr(args, 'w_beta', 1.0)
    print(f"🎯 Wanda hyperparameters: α={w_alpha} (weight multiplier), β={w_beta} (activation multiplier)")
    print(f"   Formula: Score = |W| * {w_alpha} × |X| * {w_beta}")

    print("loading calibdation data")
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)

    layers = model.model.layers
    total_layers = len(layers)
    device_map = getattr(model, "hf_device_map", None)
    
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        
        # Filter subset based on pruning_last flag
        filtered_subset = {}
        for name in subset:
            if should_prune_module(args, i, total_layers, name):
                filtered_subset[name] = subset[name]
            elif args.pruning_last is not None:
                print(f"Skipping layer {i} module {name} (not in last {args.pruning_last} MLP layers)")
        
        # Skip layer entirely if no modules to prune
        if not filtered_subset:
            continue
            
        subset = filtered_subset

        if device_map and f"model.layers.{i}" in device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            weight = subset[name].weight.data
            scaler = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1))).to(weight.device, dtype=torch.float32)
            
            # Apply Wanda hyperparameters and normalize to avoid numerical instability
            weight_component = torch.abs(weight).to(torch.float32) * w_alpha
            activation_component = scaler * w_beta
            W_metric = weight_component * activation_component

            # Normalize to prevent small values from causing precision issues
            # This makes the ranking robust to the scale of alpha and beta
            if W_metric.max() > 0:
                W_metric = W_metric / W_metric.max()

            W_mask = None
            if prune_n != 0:
                # structured n:m sparsity
                W_mask = (torch.zeros_like(weight, dtype=torch.bool))
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
            else:
                if args.use_variant:
                    sort_res = torch.sort(W_metric, dim=-1, stable=True)
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    while (torch.abs(cur_sparsity - args.sparsity_ratio) > 0.001) and (alpha_hist[1] - alpha_hist[0] >= 0.001):
                        if cur_sparsity > args.sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    pruned, numel = _apply_unstructured_mask(weight, W_metric, args.sparsity_ratio)
                    print(f"[Wanda] layer {i} name {name} pruned {pruned}/{numel}")
                    continue

            if W_mask is not None:
                weight[W_mask] = 0

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()


def prune_neuronrank_unstructured(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    if prune_n != 0 or prune_m != 0:
        raise ValueError("NeuronRank unstructured pruning does not support N:M sparsity.")

    if args.pruning_last is not None:
        print(f"Note: --pruning_last {args.pruning_last} is set. Only MLP modules in the last {args.pruning_last} transformer blocks will be pruned.")
    elif args.magnitude_multi == 0.0 and args.nr_include_attention:
        print("Note: magnitude_multi=0.0. Attention modules will be skipped because they lack variance statistics.")

    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("Loading calibration data for NeuronRank...")
    dataloader, _ = get_loaders(
        "c4",
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=model.seqlen,
        tokenizer=tokenizer,
    )

    print("Collecting activation statistics...")
    stats = collect_neuronrank_statistics(
        model,
        dataloader,
        tokenizer,
        device,
        max_classes=args.neuronrank_max_classes,
    )
    print(f"Collected statistics for {len(stats)} layers")

    scores = compute_neuronrank_scores(
        model,
        stats,
        token_weight=args.neuronrank_token_weight,
        variance_exp=args.variance_exp,
        variance_multi=args.variance_multi,
        magnitude_multi=args.magnitude_multi,
    )
    print(
        f"Computed scores for {len(scores)} layers (variance_exp={args.variance_exp}, "
        f"variance_multi={args.variance_multi}, magnitude_multi={args.magnitude_multi})"
    )

    total_pruned = 0
    total_weights = 0

    layers = model.model.layers
    total_layers = len(layers)

    for i, layer in enumerate(layers):
        subset = find_layers(layer)
        layer_entry = scores.get(i)
        layer_variance = layer_entry.get("variance") if isinstance(layer_entry, dict) else None

        for name, module in subset.items():
            if not should_prune_module(args, i, total_layers, name):
                if args.pruning_last is not None:
                    print(f"[NeuronRank] Skipping layer {i} module {name} (not in last {args.pruning_last} MLP layers)")
                continue

            if not args.nr_include_attention and "mlp" not in name:
                print(f"[NeuronRank] Skipping layer {i} module {name} (nr_include_attention=False)")
                continue

            weight = module.weight.data
            if args.sparsity_ratio <= 0:
                continue

            metric = None
            if args.magnitude_multi != 0.0:
                metric = torch.abs(weight).to(torch.float32) * float(args.magnitude_multi)

            if layer_variance is not None and "mlp" in name and args.variance_multi != 0.0:
                variance_vec = layer_variance.to(weight.device, dtype=torch.float32)
                if args.variance_exp == 0.0:
                    variance_term = torch.ones_like(variance_vec)
                elif args.variance_exp == 1.0:
                    variance_term = variance_vec
                else:
                    variance_term = torch.pow(variance_vec.clamp(min=1e-12), args.variance_exp)

                variance_component = variance_term * float(args.variance_multi)
                if "gate_proj" in name or "up_proj" in name:
                    addition = variance_component.view(-1, 1)
                elif "down_proj" in name:
                    addition = variance_component.view(1, -1)
                else:
                    addition = None

                if addition is not None:
                    addition = addition.to(weight.device, dtype=torch.float32)
                    if metric is None:
                        metric = torch.zeros_like(weight, dtype=torch.float32)
                    metric = metric + addition

            if metric is None:
                if "mlp" not in name:
                    if args.magnitude_multi == 0.0:
                        print(f"[NeuronRank] Skipping layer {i} module {name} (no metric available)")
                        continue
                    metric = torch.abs(weight).to(torch.float32)
                else:
                    print(f"[NeuronRank] Warning: layer {i} module {name} has no variance stats; using magnitude only")
                    metric = torch.abs(weight).to(torch.float32)

            if not torch.isfinite(metric).all():
                metric = torch.nan_to_num(metric, nan=0.0, posinf=0.0, neginf=0.0)

            if torch.count_nonzero(metric).item() == 0:
                continue

            pruned, numel = _apply_unstructured_mask(weight, metric, args.sparsity_ratio)
            total_pruned += pruned
            total_weights += numel

            print(f"[NeuronRank-Unstructured] layer {i} module {name} pruned {pruned}/{numel}")

    if args.nr_prune_lm_head and hasattr(model, "lm_head") and isinstance(model.lm_head, nn.Linear):
        weight = model.lm_head.weight.data
        metric = torch.abs(weight).to(torch.float32)
        pruned, numel = _apply_unstructured_mask(weight, metric, args.sparsity_ratio)
        total_pruned += pruned
        total_weights += numel
        print(f"[NeuronRank-Unstructured] lm_head pruned {pruned}/{numel}")

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

    if total_weights:
        pct = 100.0 * total_pruned / total_weights
        print(f"[NeuronRank-Unstructured] global pruning ratio {pct:.2f}% ({total_pruned}/{total_weights})")


def prune_neuronrank_fisher(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    if prune_n != 0 or prune_m != 0:
        raise ValueError("NeuronRank Fisher pruning does not support N:M sparsity.")

    if args.sparsity_type != "unstructured":
        raise ValueError("NeuronRank Fisher pruning currently supports only unstructured sparsity type.")

    if args.neuronrank_max_classes <= 0:
        raise ValueError("NeuronRank Fisher pruning requires --neuronrank-max-classes > 0 to collect class statistics.")

    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("Loading calibration data for NeuronRank-Fisher...")
    dataloader, _ = get_loaders(
        "c4",
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=model.seqlen,
        tokenizer=tokenizer,
    )

    print("Collecting activation statistics for Fisher scoring...")
    stats = collect_neuronrank_statistics(
        model,
        dataloader,
        tokenizer,
        device,
        max_classes=args.neuronrank_max_classes,
    )

    fisher_scores = compute_neuronrank_fisher_scores(stats)
    if not fisher_scores:
        raise RuntimeError("NeuronRank Fisher pruning could not compute any scores; ensure calibration data and class tracking are available.")

    # Unstructured pruning using Fisher LDA scores (per-weight masking)
    total_pruned = 0
    total_weights = 0
    layers = model.model.layers
    total_layers = len(layers)
    for i, layer in enumerate(layers):
        subset = find_layers(layer)
        # apply --pruning_last filter at module level
        for name, module in subset.items():
            if not should_prune_module(args, i, total_layers, name):
                if args.pruning_last is not None:
                    print(f"[NeuronRank-Fisher] Skipping layer {i} module {name} (not in last {args.pruning_last} MLP layers)")
                continue
            if "mlp" not in name:
                continue
            weight = module.weight.data
            if args.sparsity_ratio <= 0:
                continue
            # get neuron-level Fisher scores for this layer
            neuron_scores = fisher_scores.get(i, {}).get("channel")
            if neuron_scores is None:
                continue
            var = neuron_scores.to(weight.device, dtype=torch.float32)
            # broadcast scores to weight matrix
            if "gate_proj" in name or "up_proj" in name:
                metric = var.view(-1, 1).expand_as(weight)
            elif "down_proj" in name:
                metric = var.view(1, -1).expand_as(weight)
            else:
                metric = var.view(-1, 1).expand_as(weight)  # fallback
            # sanitize metric
            metric = torch.nan_to_num(metric, nan=0.0, posinf=0.0, neginf=0.0)
            if torch.count_nonzero(metric).item() == 0:
                continue
            pruned, numel = _apply_unstructured_mask(weight, metric, args.sparsity_ratio)
            total_pruned += pruned
            total_weights += numel
            print(f"[NeuronRank-Fisher] layer {i} module {name} pruned {pruned}/{numel}")
    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
    if total_weights:
        pct = 100.0 * total_pruned / total_weights
        print(f"[NeuronRank-Fisher] global pruning ratio {pct:.2f}% ({total_pruned}/{total_weights})")
    else:
        print("[NeuronRank-Fisher] No weights were pruned.")


def prune_neuronrank_old(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    """Old NeuronRank formula: score = |W|^α × TF^β × IDF^γ
    
    Supports both structured (per-neuron) and unstructured (per-weight) pruning.
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("Loading calibration data for NeuronRank-OLD")
    dataloader, _ = get_loaders(
        "c4",
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=model.seqlen,
        tokenizer=tokenizer,
    )

    print("Collecting TF-IDF statistics")
    from lib.neuronrank import collect_neuronrank_old_statistics, compute_neuronrank_old_scores
    
    # Get activation threshold from args (default to 1e-6)
    activation_threshold = getattr(args, 'activation_threshold', 1e-6)
    
    stats = collect_neuronrank_old_statistics(model, dataloader, device, activation_threshold=activation_threshold)
    
    # Get exponents from args (default to 1.0 if not specified)
    weight_exp = getattr(args, 'weight_exp', 1.0)
    tf_exp = getattr(args, 'tf_exp', 1.0)
    idf_exp = getattr(args, 'idf_exp', 1.0)
    
    print(f"Computing scores with exponents: weight={weight_exp}, TF={tf_exp}, IDF={idf_exp}")
    scores = compute_neuronrank_old_scores(
        model,
        stats,
        weight_exp=weight_exp,
        tf_exp=tf_exp,
        idf_exp=idf_exp,
    )

    layers = model.model.layers
    total_layers = len(layers)
    total_pruned = 0
    total_weights = 0

    if args.sparsity_type == "unstructured":
        # Unstructured pruning: prune individual weights
        for i, layer in enumerate(layers):
            subset = find_layers(layer)
            layer_scores = scores.get(i)
            
            for name, module in subset.items():
                # Check pruning_last flag
                if not should_prune_module(args, i, total_layers, name):
                    if args.pruning_last is not None:
                        print(f"Skipping layer {i} module {name} (not in last {args.pruning_last} MLP layers)")
                    continue
                
                if "mlp" not in name:
                    continue

                weight = module.weight.data
                if args.sparsity_ratio <= 0:
                    continue

                abs_weight = torch.abs(weight).to(torch.float32)

                # Build metric from neuron scores
                if layer_scores is None:
                    metric = abs_weight
                else:
                    neuron_scores = layer_scores.to(weight.device, dtype=torch.float32)

                    if "gate_proj" in name or "up_proj" in name:
                        metric = abs_weight * neuron_scores.view(-1, 1)
                    elif "down_proj" in name:
                        metric = abs_weight * neuron_scores.view(1, -1)
                    else:
                        metric = abs_weight

                if not torch.isfinite(metric).all():
                    metric = torch.nan_to_num(metric, nan=0.0, posinf=0.0, neginf=0.0)

                if torch.count_nonzero(metric).item() == 0:
                    continue

                pruned, numel = _apply_unstructured_mask(weight, metric, args.sparsity_ratio)
                total_pruned += pruned
                total_weights += numel

                print(f"[NeuronRank-OLD] layer {i} {name} pruned {pruned}/{numel}")

    else:
        # Structured pruning: prune entire neurons
        for i, layer in enumerate(layers):
            # Check pruning_last flag
            if args.pruning_last is not None:
                layers_to_prune = total_layers - args.pruning_last
                if i < layers_to_prune:
                    print(f"Skipping layer {i} (not in last {args.pruning_last} layers)")
                    continue

            layer_score = scores.get(i)
            if layer_score is None:
                continue

            gate_proj = getattr(layer.mlp, "gate_proj", None)
            up_proj = getattr(layer.mlp, "up_proj", None)
            down_proj = getattr(layer.mlp, "down_proj", None)
            if gate_proj is None:
                continue

            num_channels = layer_score.numel()
            total_weights += num_channels
            num_to_prune = int(num_channels * args.sparsity_ratio)
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

            total_pruned += num_to_prune
            layer_pct = 100.0 * num_to_prune / num_channels
            print(f"[NeuronRank-OLD] layer {i}: pruned {num_to_prune}/{num_channels} channels ({layer_pct:.2f}%)")

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

    if total_weights:
        pct = 100.0 * total_pruned / total_weights
        print(f"[NeuronRank-OLD] global pruning ratio {pct:.2f}% ({total_pruned}/{total_weights})")


def prune_neuronrank_last(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    """NeuronRank-LAST: Exact implementation of the NeuronRank formula
    
    Score = |W|^α × TF^β × IDF^γ
    
    Where:
    - |W|^α: Weight magnitude (absolute value)
    - TF^β: Term Frequency (average activation strength)
    - IDF^γ: Inverse Document Frequency (activation selectivity/sparsity)
    
    TF = (1/T) × Σ|activation_t| where T = total tokens
    IDF = log((T+1)/(n_active+1)) + 1 where n_active = tokens where activation > 0
    
    Only supports unstructured pruning of MLP layers.
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("=" * 60)
    print("🎯 NeuronRank-LAST: |W|^α × TF^β × IDF^γ")
    print("=" * 60)
    
    # Get exponents from args (default to 1.0)
    weight_exp = getattr(args, 'weight_exp', 1.0)
    tf_exp = getattr(args, 'tf_exp', 1.0)
    idf_exp = getattr(args, 'idf_exp', 1.0)
    
    print(f"Exponents: α={weight_exp} (weight), β={tf_exp} (TF), γ={idf_exp} (IDF)")
    print(f"Sparsity target: {args.sparsity_ratio:.1%}")
    
    if args.sparsity_type != "unstructured":
        raise ValueError("neuronrank_last only supports unstructured pruning")

    print("\nLoading calibration data...")
    dataloader, _ = get_loaders(
        "c4",
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=model.seqlen,
        tokenizer=tokenizer,
    )

    print("Collecting TF-IDF statistics...")
    from lib.neuronrank import collect_neuronrank_old_statistics, compute_neuronrank_old_scores
    
    # Get activation threshold from args (default to 1e-6)
    activation_threshold = getattr(args, 'activation_threshold', 1e-6)
    
    stats = collect_neuronrank_old_statistics(model, dataloader, device, activation_threshold=activation_threshold)
    
    print("\nComputing neuron importance scores...")
    scores = compute_neuronrank_old_scores(
        model,
        stats,
        weight_exp=weight_exp,
        tf_exp=tf_exp,
        idf_exp=idf_exp,
    )

    layers = model.model.layers
    total_layers = len(layers)
    total_pruned = 0
    total_weights = 0

    print("\nApplying unstructured pruning to MLP layers...")
    
    for i, layer in enumerate(layers):
        subset = find_layers(layer)
        layer_scores = scores.get(i)
        
        for name, module in subset.items():
            # Check pruning_last flag
            if not should_prune_module(args, i, total_layers, name):
                if args.pruning_last is not None and i == total_layers - args.pruning_last - 1:
                    print(f"Skipping layer {i} module {name} (not in last {args.pruning_last} MLP layers)")
                continue
            
            # Only prune MLP modules
            if "mlp" not in name:
                continue

            weight = module.weight.data
            if args.sparsity_ratio <= 0:
                continue

            abs_weight = torch.abs(weight).to(torch.float32)

            # Build metric from neuron scores
            if layer_scores is None:
                metric = abs_weight
            else:
                neuron_scores = layer_scores.to(weight.device, dtype=torch.float32)

                if "gate_proj" in name or "up_proj" in name:
                    # Column projection: broadcast neuron scores across rows
                    metric = abs_weight * neuron_scores.view(-1, 1)
                elif "down_proj" in name:
                    # Row projection: broadcast neuron scores across columns
                    metric = abs_weight * neuron_scores.view(1, -1)
                else:
                    metric = abs_weight

            # Sanitize metric
            if not torch.isfinite(metric).all():
                metric = torch.nan_to_num(metric, nan=0.0, posinf=0.0, neginf=0.0)

            if torch.count_nonzero(metric).item() == 0:
                continue

            pruned, numel = _apply_unstructured_mask(weight, metric, args.sparsity_ratio)
            total_pruned += pruned
            total_weights += numel

            if i < 3 or i >= total_layers - 3:
                print(f"  Layer {i:2d} {name:20s}: pruned {pruned:7d}/{numel:7d} weights")

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    if total_weights:
        pct = 100.0 * total_pruned / total_weights
        print(f"✅ NeuronRank-LAST: Pruned {total_pruned:,}/{total_weights:,} weights ({pct:.2f}%)")
    else:
        print("⚠️  No weights were pruned!")
    print("=" * 60)


def prune_neuronrank_tfidf(args, model, tokenizer, device, prune_n=0, prune_m=0):
    """
    NeuronRank TF-IDF++ pruning with doc-level or topic-level IDF.
    
    Provides more sophisticated selectivity signals than token-level IDF by
    treating calibration sequences as documents or clustering tokens into topics.
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False
    
    print(f"🔬 Loading calibration data for NeuronRank TF-IDF++ ({args.nr_tfidf_mode} mode)...")
    dataloader, _ = get_loaders(
        "c4",
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=model.seqlen,
        tokenizer=tokenizer,
    )
    
    # Import the new TF-IDF stats classes
    from lib.neuronrank import DocTFIDFStats, TopicTFIDFStats, compute_tfidf_scores
    
    # Initialize statistics collector
    if args.nr_tfidf_mode == 'topic':
        print(f"📊 Using topic-level TF-IDF with {args.nr_tfidf_k} topics")
        stats_collector = TopicTFIDFStats(
            device=device,
            dtype=torch.float32,
            k_topics=args.nr_tfidf_k,
            proj_dim=128,
            q_active=args.nr_q_active,
        )
    else:
        print(f"📊 Using document-level TF-IDF")
        stats_collector = DocTFIDFStats(
            device=device,
            dtype=torch.float32,
            q_active=args.nr_q_active,
        )
    
    # Collect activations
    print("📈 Collecting TF-IDF statistics...")
    layers = model.model.layers
    
    # Register hooks to collect activations
    handles = []
    
    def make_hook(layer_idx, module_name):
        def hook(module, input, output):
            # Output is the activation tensor
            if isinstance(output, tuple):
                acts = output[0]
            else:
                acts = output
            
            # acts shape: [batch, seq_len, hidden_dim]
            if acts.dim() == 3:
                acts = acts.reshape(-1, acts.size(-1))  # [batch*seq_len, hidden_dim]
            
            layer_key = f"layer_{layer_idx}.{module_name}"
            
            if args.nr_tfidf_mode == 'topic':
                stats_collector.update_doc_with_topics(layer_key, acts)
            else:
                stats_collector.update_doc(layer_key, acts)
        
        return hook
    
    # Register hooks on MLP modules (always)
    for i, layer in enumerate(layers):
        if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'gate_proj'):
            h = layer.mlp.gate_proj.register_forward_hook(make_hook(i, 'mlp.gate_proj'))
            handles.append(h)
    
    # Register hooks on attention modules (if enabled)
    if args.nr_include_attention:
        print("📊 Including attention layers (collecting q_proj, k_proj, v_proj, o_proj statistics)")
        for i, layer in enumerate(layers):
            if hasattr(layer, 'self_attn'):
                attn = layer.self_attn
                if hasattr(attn, 'q_proj'):
                    h = attn.q_proj.register_forward_hook(make_hook(i, 'self_attn.q_proj'))
                    handles.append(h)
                if hasattr(attn, 'k_proj'):
                    h = attn.k_proj.register_forward_hook(make_hook(i, 'self_attn.k_proj'))
                    handles.append(h)
                if hasattr(attn, 'v_proj'):
                    h = attn.v_proj.register_forward_hook(make_hook(i, 'self_attn.v_proj'))
                    handles.append(h)
                if hasattr(attn, 'o_proj'):
                    h = attn.o_proj.register_forward_hook(make_hook(i, 'self_attn.o_proj'))
                    handles.append(h)
    else:
        print("⏭️  Skipping attention layers (--nr-skip-attention)")
    
    print(f"✅ Registered {len(handles)} forward hooks")
    
    # Run calibration data through model
    model.eval()
    for batch_idx, batch in enumerate(dataloader):
        try:
            inp = batch[0].to(device)
            with torch.no_grad():
                model(inp)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1}/{len(dataloader)} calibration batches")
        except Exception as e:
            print(f"Warning: Error processing batch {batch_idx}: {e}")
            continue
    
    # Remove hooks
    for h in handles:
        h.remove()
    
    # Finalize statistics
    print("🧮 Finalizing TF-IDF statistics...")
    if args.nr_tfidf_mode == 'topic':
        tfidf_stats = stats_collector.finalize_topics()
    else:
        tfidf_stats = stats_collector.finalize()
    
    print(f"✅ Collected TF-IDF statistics for {len(tfidf_stats)} modules")
    if len(tfidf_stats) > 0:
        print(f"   Sample keys: {list(tfidf_stats.keys())[:3]}")
    else:
        print(f"   ⚠️  WARNING: No statistics collected! Check hooks and data flow.")
    
    # Prune each layer
    print(f"✂️  Applying TF-IDF++ pruning (α={args.weight_exp}, β={args.tf_exp}, γ={args.idf_exp})...")
    total_layers = len(layers)
    total_pruned = 0
    total_weights = 0
    
    skipped_no_stats = 0
    skipped_not_mlp = 0
    skipped_should_not_prune = 0
    
    for i in range(total_layers):
        layer = layers[i]
        subset = find_layers(layer)
        
        for name in subset:
            if not should_prune_module(args, i, total_layers, name):
                skipped_should_not_prune += 1
                continue
            
            # Check if we should skip attention modules
            is_attention = 'self_attn' in name
            is_mlp = 'mlp' in name
            
            if not is_mlp and not is_attention:
                skipped_not_mlp += 1
                continue
            
            if is_attention and not args.nr_include_attention:
                skipped_not_mlp += 1
                continue
            
            # Get weight tensor
            W = subset[name].weight.data
            
            # Determine which statistics to use
            if is_mlp:
                # For MLP modules, use gate_proj statistics
                layer_key = f"layer_{i}.mlp.gate_proj"
            else:
                # For attention modules, use the specific projection's statistics
                layer_key = f"layer_{i}.{name}"
            
            # Check if we have TF-IDF stats for this layer
            if layer_key not in tfidf_stats:
                if i < 3:  # Only print for first few layers to avoid spam
                    print(f"  Warning: No TF-IDF statistics for {layer_key}, skipping layer {i} {name}")
                skipped_no_stats += 1
                continue
            
            tf, idf = tfidf_stats[layer_key]
            
            # Move to weight's device
            tf = tf.to(W.device)
            idf = idf.to(W.device)
            
            # Compute per-weight scores using TF-IDF formula
            metric = compute_tfidf_scores(
                weight=W,
                name=name,
                tf=tf,
                idf=idf,
                alpha=args.weight_exp,
                beta=args.tf_exp,
                gamma=args.idf_exp,
                rho=args.nr_spikiness_exp,
            )
            
            # Sanitize metric
            if not torch.isfinite(metric).all():
                metric = torch.nan_to_num(metric, nan=0.0, posinf=0.0, neginf=0.0)
            
            if torch.count_nonzero(metric).item() == 0:
                print(f"  Warning: All-zero metric for layer {i} {name}, skipping")
                continue
            
            # Apply unstructured mask
            pruned, numel = _apply_unstructured_mask(W, metric, args.sparsity_ratio)
            total_pruned += pruned
            total_weights += numel
            
            if i < 2 or i >= total_layers - 2:  # Print first 2 and last 2 layers
                print(f"  [NeuronRank-TFIDF] layer {i:2d} {name:20s}: pruned {pruned:7d}/{numel:7d} weights")
    
    # Optional: Prune LM head using magnitude
    if args.nr_prune_lm_head and hasattr(model, 'lm_head'):
        print("📝 Pruning LM head using magnitude...")
        lm_head_weight = model.lm_head.weight.data
        lm_head_metric = lm_head_weight.abs().to(torch.float32)
        
        # Sanitize
        lm_head_metric = torch.nan_to_num(lm_head_metric, nan=0.0, posinf=0.0, neginf=0.0)
        
        if torch.count_nonzero(lm_head_metric).item() > 0:
            pruned, numel = _apply_unstructured_mask(lm_head_weight, lm_head_metric, args.sparsity_ratio)
            total_pruned += pruned
            total_weights += numel
            print(f"  [NeuronRank-TFIDF] lm_head: pruned {pruned:7d}/{numel:7d} weights")
        else:
            print(f"  Warning: LM head has all-zero metric, skipping")
    
    model.config.use_cache = use_cache
    
    if total_weights > 0:
        actual_sparsity = total_pruned / total_weights
        print(f"🎯 NeuronRank TF-IDF++ ({args.nr_tfidf_mode}): pruned {total_pruned}/{total_weights} weights ({actual_sparsity:.2%} sparsity)")
    else:
        print(f"⚠️  Warning: No weights were pruned!")
        print(f"   Debug: skipped_should_not_prune={skipped_should_not_prune}, skipped_not_mlp={skipped_not_mlp}, skipped_no_stats={skipped_no_stats}")
    
    # Clean up
    torch.cuda.empty_cache()
    import gc
    gc.collect()


def prune_hybrid(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    """
    Hybrid pruning: Wanda for attention, NeuronRank (TF-IDF or OLD) for MLPs.
    
    This method combines the strengths of different pruning approaches:
    - Wanda for attention layers (magnitude × activation)
    - NeuronRank TF-IDF/OLD for MLP layers (semantic selectivity)
    """
    if prune_n != 0 or prune_m != 0:
        raise ValueError("Hybrid pruning does not support N:M sparsity.")
    
    use_cache = model.config.use_cache
    model.config.use_cache = False
    
    print("=" * 60)
    print("🔀 HYBRID PRUNING MODE")
    print("=" * 60)
    print(f"  Attention: Wanda (magnitude × activation)")
    print(f"  MLP:       {args.hybrid_mlp_method}")
    print("=" * 60)
    
    # Load calibration data (shared for both methods)
    print("📂 Loading calibration data...")
    dataloader, _ = get_loaders(
        "c4",
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=model.seqlen,
        tokenizer=tokenizer,
    )
    
    layers = model.model.layers
    total_layers = len(layers)
    
    # ========================================
    # PART 1: Collect Wanda statistics for ATTENTION
    # ========================================
    print("\n" + "=" * 60)
    print("🎯 PART 1: Wanda Statistics for Attention Layers")
    print("=" * 60)
    
    from .layerwrapper import WrappedGPT
    
    wrapped_layers = {}
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        wrapped_layers[i] = {}
        
        for name in subset:
            # Only wrap attention modules for Wanda
            if 'self_attn' in name:
                wrapped_layers[i][name] = WrappedGPT(subset[name])
    
    def add_batch(name):
        def tmp(_, inp, out):
            wrapped_layers[name[0]][name[1]].add_batch(inp[0].data, out.data)
        return tmp
    
    handles = []
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        for name in subset:
            if 'self_attn' in name:
                handles.append(subset[name].register_forward_hook(add_batch((i, name))))
    
    # Run calibration through model for Wanda
    for j, batch in enumerate(dataloader):
        with torch.no_grad():
            model(batch[0].to(device))
        if (j + 1) % 10 == 0:
            print(f"  Processed {j + 1}/{len(dataloader)} batches for Wanda")
    
    for h in handles:
        h.remove()
    
    print(f"✅ Collected Wanda statistics for attention layers")
    
    # ========================================
    # PART 2: Collect NeuronRank statistics for MLPs
    # ========================================
    print("\n" + "=" * 60)
    print(f"🧠 PART 2: {args.hybrid_mlp_method.upper()} Statistics for MLP Layers")
    print("=" * 60)
    
    # IMPORTANT: Reload calibration data (dataloader is exhausted after first pass)
    print("📂 Reloading calibration data for MLP statistics...")
    dataloader, _ = get_loaders(
        "c4",
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=model.seqlen,
        tokenizer=tokenizer,
    )
    
    mlp_stats = {}
    
    if args.hybrid_mlp_method == 'neuronrank_tfidf':
        # Use TF-IDF++ statistics collection
        from lib.neuronrank import DocTFIDFStats, TopicTFIDFStats, compute_tfidf_scores
        
        # Initialize statistics collector
        if args.nr_tfidf_mode == 'topic':
            print(f"📊 Using topic-level TF-IDF with {args.nr_tfidf_k} topics")
            stats_collector = TopicTFIDFStats(
                device=device,
                dtype=torch.float32,
                k_topics=args.nr_tfidf_k,
                proj_dim=128,
                q_active=args.nr_q_active,
            )
        else:
            print(f"📊 Using document-level TF-IDF")
            stats_collector = DocTFIDFStats(
                device=device,
                dtype=torch.float32,
                q_active=args.nr_q_active,
            )
        
        # Register hooks for MLP gate_proj
        mlp_handles = []
        
        def make_mlp_hook(layer_idx, module_name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    acts = output[0]
                else:
                    acts = output
                
                if acts.dim() == 3:
                    acts = acts.reshape(-1, acts.size(-1))
                
                layer_key = f"layer_{layer_idx}.{module_name}"
                
                if args.nr_tfidf_mode == 'topic':
                    stats_collector.update_doc_with_topics(layer_key, acts)
                else:
                    stats_collector.update_doc(layer_key, acts)
            
            return hook
        
        for i, layer in enumerate(layers):
            if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'gate_proj'):
                h = layer.mlp.gate_proj.register_forward_hook(make_mlp_hook(i, 'mlp.gate_proj'))
                mlp_handles.append(h)
        
        print(f"✅ Registered {len(mlp_handles)} MLP hooks")
        
        # Run calibration through model for TF-IDF
        for j, batch in enumerate(dataloader):
            with torch.no_grad():
                model(batch[0].to(device))
            if (j + 1) % 10 == 0:
                print(f"  Processed {j + 1}/{len(dataloader)} batches for TF-IDF")
        
        for h in mlp_handles:
            h.remove()
        
        # Finalize TF-IDF statistics
        if args.nr_tfidf_mode == 'topic':
            mlp_stats = stats_collector.finalize_topics()
        else:
            mlp_stats = stats_collector.finalize()
        
        print(f"✅ Collected TF-IDF statistics for {len(mlp_stats)} MLP modules")
    
    elif args.hybrid_mlp_method == 'neuronrank_old':
        # Use neuronrank_old statistics collection
        from lib.neuronrank import TFIDFStats, compute_neuronrank_old_scores
        import torch.nn.functional as F
        
        # Get activation threshold from args
        activation_threshold = getattr(args, 'activation_threshold', 1e-6)
        
        # Create per-layer statistics collectors
        layer_stats = {}
        mlp_handles = []
        
        # Register hooks for each layer's gate_proj
        for layer_idx, layer in enumerate(layers):
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

            mlp_handles.append(gate_proj.register_forward_hook(make_hook(layer_idx)))
        
        print(f"✅ Registered {len(mlp_handles)} MLP hooks")
        
        # Run calibration through model
        for j, batch in enumerate(dataloader):
            with torch.no_grad():
                model(batch[0].to(device))
            if (j + 1) % 10 == 0:
                print(f"  Processed {j + 1}/{len(dataloader)} batches for NeuronRank-OLD")
        
        for h in mlp_handles:
            h.remove()
        
        # Compute final TF and IDF scores
        mlp_stats = {}
        for idx, tfidf_stats in layer_stats.items():
            tf, idf = tfidf_stats.compute_tf_idf()
            mlp_stats[idx] = {
                "tf": tf.to(dtype=torch.float32, device="cpu"),
                "idf": idf.to(dtype=torch.float32, device="cpu"),
                "total_tokens": tfidf_stats.total_tokens,
            }
            print(f"  Layer {idx}: TF range [{tf.min():.6f}, {tf.max():.6f}], "
                  f"IDF range [{idf.min():.6f}, {idf.max():.6f}], "
                  f"tokens={tfidf_stats.total_tokens}")
        
        print(f"✅ Collected NeuronRank-OLD statistics for {len(mlp_stats)} MLP modules")
    
    # ========================================
    # PART 3: Apply pruning
    # ========================================
    print("\n" + "=" * 60)
    print("✂️  PART 3: Applying Hybrid Pruning")
    print("=" * 60)
    
    total_pruned = 0
    total_weights = 0
    attn_pruned = 0
    attn_weights = 0
    mlp_pruned = 0
    mlp_weights = 0
    
    # Debug counters
    mlp_skipped_no_stats = 0
    mlp_skipped_zero_metric = 0
    mlp_attempted = 0
    
    for i in range(total_layers):
        layer = layers[i]
        subset = find_layers(layer)
        
        for name in subset:
            if not should_prune_module(args, i, total_layers, name):
                continue
            
            W = subset[name].weight.data
            is_attention = 'self_attn' in name
            is_mlp = 'mlp' in name
            
            if not is_attention and not is_mlp:
                continue
            
            # Debug: Track MLP attempts
            if is_mlp:
                mlp_attempted += 1
                if i == 0:  # Debug first layer
                    print(f"  [DEBUG] Attempting MLP layer {i} {name}")
            
            # ===== ATTENTION: Use Wanda =====
            if is_attention:
                wrapped = wrapped_layers[i].get(name)
                if wrapped is None:
                    continue
                
                # Compute Wanda metric
                W_metric = torch.abs(W)
                activation_metric = torch.sqrt(wrapped.scaler_row.reshape((1, -1)))
                metric = W_metric * activation_metric
                
                # Sanitize
                metric = torch.nan_to_num(metric, nan=0.0, posinf=0.0, neginf=0.0)
                
                if torch.count_nonzero(metric).item() == 0:
                    continue
                
                pruned, numel = _apply_unstructured_mask(W, metric, args.sparsity_ratio)
                attn_pruned += pruned
                attn_weights += numel
                
                if i < 2 or i >= total_layers - 2:
                    print(f"  [Wanda-Attn] layer {i:2d} {name:20s}: pruned {pruned:7d}/{numel:7d} weights")
            
            # ===== MLP: Use NeuronRank TF-IDF or OLD =====
            elif is_mlp:
                if args.hybrid_mlp_method == 'neuronrank_tfidf':
                    layer_key = f"layer_{i}.mlp.gate_proj"
                    
                    if layer_key not in mlp_stats:
                        if i == 0:  # Debug
                            print(f"  [DEBUG] No stats for {layer_key}, available: {list(mlp_stats.keys())[:3]}")
                        mlp_skipped_no_stats += 1
                        continue
                    
                    tf, idf = mlp_stats[layer_key]
                    tf = tf.to(W.device)
                    idf = idf.to(W.device)
                    
                    # Debug first layer
                    if i == 0:
                        print(f"  [DEBUG] Layer {i} {name}: tf shape={tf.shape}, range=[{tf.min():.6f}, {tf.max():.6f}]")
                        print(f"  [DEBUG] Layer {i} {name}: idf shape={idf.shape}, range=[{idf.min():.6f}, {idf.max():.6f}]")
                        print(f"  [DEBUG] Layer {i} {name}: W shape={W.shape}")
                    
                    # Debug exponents
                    if i == 0:
                        print(f"  [DEBUG] Exponents: α={args.weight_exp}, β={args.tf_exp}, γ={args.idf_exp}, ρ={args.nr_spikiness_exp}")
                    
                    # Compute TF-IDF metric
                    metric = compute_tfidf_scores(
                        weight=W,
                        name=name,
                        tf=tf,
                        idf=idf,
                        alpha=args.weight_exp,
                        beta=args.tf_exp,
                        gamma=args.idf_exp,
                        rho=args.nr_spikiness_exp,
                    )
                    
                    # Debug metric
                    if i == 0:
                        print(f"  [DEBUG] Layer {i} {name}: metric shape={metric.shape}, range=[{metric.min():.6f}, {metric.max():.6f}], nonzero={torch.count_nonzero(metric).item()}")
                
                elif args.hybrid_mlp_method == 'neuronrank_old':
                    # Get neuron scores for this layer
                    layer_stats_dict = mlp_stats.get(i)
                    
                    if layer_stats_dict is None:
                        mlp_skipped_no_stats += 1
                        continue
                    
                    # Extract TF and IDF
                    tf = layer_stats_dict["tf"].to(W.device, dtype=torch.float32)
                    idf = layer_stats_dict["idf"].to(W.device, dtype=torch.float32)
                    
                    # Compute neuron scores: |W|^α × TF^β × IDF^γ
                    # For gate_proj: weight shape is [intermediate_size, hidden_size]
                    # We need per-neuron (output) scores
                    if "gate_proj" in name or "up_proj" in name:
                        # Column projection: weight shape [out_features, in_features]
                        # Compute weight magnitude per output neuron (L2 norm across input dim)
                        weight_magnitude = torch.norm(W, p=2, dim=1)
                    elif "down_proj" in name:
                        # Row projection: weight shape [out_features, in_features]
                        # Compute weight magnitude per input neuron (L2 norm across output dim)
                        weight_magnitude = torch.norm(W, p=2, dim=0)
                    else:
                        # Fallback
                        weight_magnitude = torch.abs(W)
                    
                    # Apply exponents
                    weight_term = torch.pow(weight_magnitude.clamp(min=1e-12), args.weight_exp)
                    tf_term = torch.pow(tf.clamp(min=1e-12), args.tf_exp)
                    idf_term = torch.pow(idf.clamp(min=1e-12), args.idf_exp)
                    
                    # Compute neuron scores
                    neuron_scores = weight_term * tf_term * idf_term
                    
                    # Broadcast to weight matrix
                    abs_weight = torch.abs(W).to(torch.float32)
                    if "gate_proj" in name or "up_proj" in name:
                        metric = abs_weight * neuron_scores.view(-1, 1)
                    elif "down_proj" in name:
                        metric = abs_weight * neuron_scores.view(1, -1)
                    else:
                        metric = abs_weight
                
                # Sanitize
                metric = torch.nan_to_num(metric, nan=0.0, posinf=0.0, neginf=0.0)
                
                if torch.count_nonzero(metric).item() == 0:
                    if i == 0:  # Debug
                        print(f"  [DEBUG] Zero metric for {name}")
                    mlp_skipped_zero_metric += 1
                    continue
                
                pruned, numel = _apply_unstructured_mask(W, metric, args.sparsity_ratio)
                mlp_pruned += pruned
                mlp_weights += numel
                
                if i < 2 or i >= total_layers - 2:
                    method_tag = "TFIDF" if args.hybrid_mlp_method == 'neuronrank_tfidf' else "OLD"
                    print(f"  [NR-{method_tag}-MLP] layer {i:2d} {name:20s}: pruned {pruned:7d}/{numel:7d} weights")
    
    total_pruned = attn_pruned + mlp_pruned
    total_weights = attn_weights + mlp_weights
    
    model.config.use_cache = use_cache
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 HYBRID PRUNING SUMMARY")
    print("=" * 60)
    
    if attn_weights > 0:
        attn_sparsity = attn_pruned / attn_weights
        print(f"  Attention (Wanda):   {attn_pruned:,}/{attn_weights:,} weights ({attn_sparsity:.2%} sparsity)")
    
    if mlp_weights > 0:
        mlp_sparsity = mlp_pruned / mlp_weights
        print(f"  MLP ({args.hybrid_mlp_method}): {mlp_pruned:,}/{mlp_weights:,} weights ({mlp_sparsity:.2%} sparsity)")
    else:
        print(f"  ⚠️  MLP: NO WEIGHTS PRUNED!")
        print(f"     Debug: mlp_attempted={mlp_attempted}, skipped_no_stats={mlp_skipped_no_stats}, skipped_zero_metric={mlp_skipped_zero_metric}")
    
    if total_weights > 0:
        total_sparsity = total_pruned / total_weights
        print(f"  TOTAL:               {total_pruned:,}/{total_weights:,} weights ({total_sparsity:.2%} sparsity)")
    else:
        print("  ⚠️  Warning: No weights were pruned!")
    
    print("=" * 60)
    
    # Clean up
    torch.cuda.empty_cache()
    import gc
    gc.collect()


def prune_wanda_idf(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    """Wanda × IDF: S_ij = |W_ij| * ||X_j||_2 * log(1/p_j)"""
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("loading calibration data")
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed, seqlen=model.seqlen, tokenizer=tokenizer)
    print("collecting Wanda + IDF stats")
    
    stats = collect_wanda_selectivity_stats(model, dataloader, device, quantile=0.9)
    
    layers = model.model.layers
    total_layers = len(layers)
    
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        layer_stats = stats.get(i, {})

        for name in subset:
            # Check if we should prune this module based on pruning_last flag
            if not should_prune_module(args, i, total_layers, name):
                if args.pruning_last is not None:
                    print(f"Skipping layer {i} module {name} (not in last {args.pruning_last} MLP layers)")
                continue
                
            print(f"pruning layer {i} name {name}")
            W = subset[name].weight.data
            module_stats = layer_stats.get(name)
            
            if module_stats is None:
                W_metric = torch.abs(W)
            else:
                input_norm = module_stats['input_norm'].to(W.device, dtype=W.dtype)
                idf = module_stats['idf'].to(W.device, dtype=W.dtype)
                
                # Wanda × IDF
                scale = input_norm * idf
                W_metric = torch.abs(W) * scale.reshape((1, -1))

            W_mask = torch.zeros_like(W, dtype=torch.bool)
            if prune_n != 0:
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii:(ii+prune_m)].float()
                        W_mask.scatter_(1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)
                indices = sort_res[1][:, :int(W_metric.shape[1] * args.sparsity_ratio)]
                W_mask.scatter_(1, indices, True)

            W[W_mask] = 0

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


def prune_wanda_spiky(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    """Wanda × Spikiness: S_ij = |W_ij| * ||X_j||_2 * (mu_top / mu)"""
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("loading calibration data")
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed, seqlen=model.seqlen, tokenizer=tokenizer)
    print("collecting Wanda + Spikiness stats")
    
    stats = collect_wanda_selectivity_stats(model, dataloader, device, quantile=0.9)
    
    layers = model.model.layers
    total_layers = len(layers)
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        layer_stats = stats.get(i, {})

        for name in subset:
            # Check if we should prune this module based on pruning_last flag
            if not should_prune_module(args, i, total_layers, name):
                if args.pruning_last is not None:
                    print(f"Skipping layer {i} module {name} (not in last {args.pruning_last} MLP layers)")
                continue
                
            print(f"pruning layer {i} name {name}")
            W = subset[name].weight.data
            module_stats = layer_stats.get(name)
            
            if module_stats is None:
                W_metric = torch.abs(W)
            else:
                input_norm = module_stats['input_norm'].to(W.device, dtype=W.dtype)
                spikiness = module_stats['spikiness'].to(W.device, dtype=W.dtype)
                
                # Wanda × Spikiness
                scale = input_norm * spikiness
                W_metric = torch.abs(W) * scale.reshape((1, -1))

            W_mask = torch.zeros_like(W, dtype=torch.bool)
            if prune_n != 0:
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii:(ii+prune_m)].float()
                        W_mask.scatter_(1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)
                indices = sort_res[1][:, :int(W_metric.shape[1] * args.sparsity_ratio)]
                W_mask.scatter_(1, indices, True)

            W[W_mask] = 0

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


def prune_wanda_selective(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    """Wanda × IDF × Spikiness: S_ij = |W_ij| * ||X_j||_2 * log(1/p_j) * (mu_top / mu)"""
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("loading calibration data")
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed, seqlen=model.seqlen, tokenizer=tokenizer)
    print("collecting Wanda + Selectivity stats")
    
    stats = collect_wanda_selectivity_stats(model, dataloader, device, quantile=0.9)
    
    layers = model.model.layers
    total_layers = len(layers)
    
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        layer_stats = stats.get(i, {})

        for name in subset:
            # Check if we should prune this module based on pruning_last flag
            if not should_prune_module(args, i, total_layers, name):
                if args.pruning_last is not None:
                    print(f"Skipping layer {i} module {name} (not in last {args.pruning_last} MLP layers)")
                continue
                
            print(f"pruning layer {i} name {name}")
            W = subset[name].weight.data
            module_stats = layer_stats.get(name)
            
            if module_stats is None:
                W_metric = torch.abs(W)
            else:
                input_norm = module_stats['input_norm'].to(W.device, dtype=W.dtype)
                idf = module_stats['idf'].to(W.device, dtype=W.dtype)
                spikiness = module_stats['spikiness'].to(W.device, dtype=W.dtype)
                
                # Wanda × IDF × Spikiness
                scale = input_norm * idf * spikiness
                W_metric = torch.abs(W) * scale.reshape((1, -1))

            W_mask = torch.zeros_like(W, dtype=torch.bool)
            if prune_n != 0:
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii:(ii+prune_m)].float()
                        W_mask.scatter_(1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)
                indices = sort_res[1][:, :int(W_metric.shape[1] * args.sparsity_ratio)]
                W_mask.scatter_(1, indices, True)

            W[W_mask] = 0

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


@torch.no_grad()
def prune_sparsegpt(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    device_map = getattr(model, "hf_device_map", None)
    if device_map and "model.embed_tokens" in device_map:
        dev = device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    total_layers = len(layers)
    
    for i in range(len(layers)):
        layer = layers[i]
        if device_map and f"model.layers.{i}" in device_map:
            dev = device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        subset = find_layers(layer)
        
        # Filter subset based on pruning_last flag
        filtered_subset = {}
        for name in subset:
            if should_prune_module(args, i, total_layers, name):
                filtered_subset[name] = subset[name]
            elif args.pruning_last is not None:
                print(f"Skipping layer {i} module {name} (not in last {args.pruning_last} MLP layers)")
        
        # Skip layer entirely if no modules to prune
        if not filtered_subset:
            continue
            
        subset = filtered_subset

        gpts = {}
        for name in subset:
            gpts[name] = SparseGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')

            gpts[name].fasterprune(args.sparsity_ratio, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()



@torch.no_grad()
def prune_ablate(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    device_map = getattr(model, "hf_device_map", None)
    if device_map and "model.embed_tokens" in device_map:
        dev = device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    total_layers = len(layers)
    
    for i in range(len(layers)):
        layer = layers[i]
        if device_map and f"model.layers.{i}" in device_map:
            dev = device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        subset = find_layers(layer)
        
        # Filter subset based on pruning_last flag
        filtered_subset = {}
        for name in subset:
            if should_prune_module(args, i, total_layers, name):
                filtered_subset[name] = subset[name]
            elif args.pruning_last is not None:
                print(f"Skipping layer {i} module {name} (not in last {args.pruning_last} MLP layers)")
        
        # Skip layer entirely if no modules to prune
        if not filtered_subset:
            continue
            
        subset = filtered_subset

        gpts = {}
        for name in subset:
            gpts[name] = AblateGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')

            if args.prune_method == "ablate_wanda_seq":
                prune_mask = gpts[name].get_wanda_mask(args.sparsity_ratio, prune_n, prune_m)
            elif args.prune_method == "ablate_mag_seq":
                prune_mask = gpts[name].get_mag_mask(args.sparsity_ratio, prune_n, prune_m)
            elif "iter" in args.prune_method:
                prune_mask = None 

            gpts[name].fasterprune(args, args.sparsity_ratio, mask=prune_mask, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()