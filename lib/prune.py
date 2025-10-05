import time 
import heapq 
import torch 
import torch.nn as nn 
from .sparsegpt import SparseGPT 
from .layerwrapper import WrappedGPT
from .data import get_loaders 
from .neuronrank import collect_neuronrank_statistics, compute_neuronrank_scores
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
            count += (W==0).sum().item()
            total_params += W.numel()

            sub_count += (W==0).sum().item()
            sub_params += W.numel()

        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache 
    return float(count)/total_params 

def prepare_calibration_input(model, dataloader, device):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    # dev = model.hf_device_map["model.embed_tokens"]
    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

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
                    print(f"Skipping layer {i} module {name} (not in last {args.pruning_last} MLP layers)")
                continue
            W = subset[name].weight.data 
            W_metric = torch.abs(W)
            if prune_n != 0:
                W_mask = (torch.zeros_like(W)==1)
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                # Memory-efficient thresholding using kthvalue instead of sorting
                with torch.no_grad():
                    flat = W_metric.detach().reshape(-1)  # stays on GPU
                    k = int(flat.numel() * args.sparsity_ratio)
                    if k > 0 and k < flat.numel():
                        thresh = flat.kthvalue(k).values  # GPU scalar
                    elif k == 0:
                        thresh = flat.min() - 1  # prune nothing
                    else:
                        thresh = flat.max() + 1  # prune everything
                    W_mask = (W_metric <= thresh)

            W[W_mask] = 0

def prune_wanda(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    print("loading calibdation data")
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)

    layers = model.model.layers
    total_layers = len(layers)
    
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

        if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
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
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                if args.use_variant:
                    # wanda variant 
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    while (torch.abs(cur_sparsity - args.sparsity_ratio)>0.001) and (alpha_hist[1]-alpha_hist[0]>=0.001):
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
                    # unstructured pruning
                    indices = sort_res[1][:,:int(W_metric.shape[1]*args.sparsity_ratio)]
                    W_mask.scatter_(1, indices, True)

            subset[name].weight.data[W_mask] = 0  ## set weights to zero 

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()


def prune_neuronrank_unstructured(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    if prune_n != 0 or prune_m != 0:
        raise ValueError("NeuronRank unstructured pruning does not support N:M sparsity.")

    def _apply_unstructured_mask(weight_tensor, metric_tensor, ratio):
        numel = metric_tensor.numel()
        num_to_prune = int(numel * ratio)
        if num_to_prune <= 0:
            return 0, numel

        flat_metric = metric_tensor.reshape(-1)
        threshold_idx = min(num_to_prune, flat_metric.numel() - 1)
        threshold = torch.sort(flat_metric)[0][threshold_idx]
        mask = metric_tensor <= threshold
        weight_tensor[mask] = 0
        return mask.sum().item(), numel

    use_cache = model.config.use_cache
    model.config.use_cache = False

    dataloader, _ = get_loaders(
        "c4",
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=model.seqlen,
        tokenizer=tokenizer,
    )

    stats = collect_neuronrank_statistics(
        model,
        dataloader,
        tokenizer,
        device,
        max_classes=args.neuronrank_max_classes,
    )
    scores = compute_neuronrank_scores(
        model,
        stats,
        token_weight=args.neuronrank_token_weight,
        variance_exp=args.variance_exp,
        variance_multi=args.variance_multi,
        magnitude_multi=args.magnitude_multi,
    )

    total_pruned = 0
    total_weights = 0

    magnitude_scale = float(args.magnitude_multi)

    layers = model.model.layers
    total_layers = len(layers)
    
    for i, layer in enumerate(layers):
        subset = find_layers(layer)
        layer_entry = scores.get(i)
        layer_variance = None
        if isinstance(layer_entry, dict):
            layer_variance = layer_entry.get("variance")

        for name, module in subset.items():
            # Check pruning_last flag first
            if not should_prune_module(args, i, total_layers, name):
                if args.pruning_last is not None:
                    print(f"Skipping layer {i} module {name} (not in last {args.pruning_last} MLP layers)")
                continue
                
            if not args.nr_include_attention and "mlp" not in name:
                continue

            weight = module.weight.data
            if args.sparsity_ratio <= 0:
                continue

            metric = None
            if magnitude_scale != 0.0:
                metric = torch.abs(weight).to(torch.float32) * magnitude_scale

            if (
                layer_variance is not None
                and "mlp" in name
            ):
                variance_vec = layer_variance.to(weight.device, dtype=torch.float32)
                if args.variance_exp == 0.0:
                    variance_term = torch.ones_like(variance_vec)
                elif args.variance_exp == 1.0:
                    variance_term = variance_vec
                else:
                    variance_term = torch.pow(variance_vec.clamp(min=1e-12), args.variance_exp)

                variance_component = variance_term * args.variance_multi
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
                continue

            if not torch.isfinite(metric).all():
                metric = torch.nan_to_num(metric, nan=0.0, posinf=0.0, neginf=0.0)

            if torch.count_nonzero(metric).item() == 0:
                continue

            pruned, numel = _apply_unstructured_mask(weight, metric, args.sparsity_ratio)
            total_pruned += pruned
            total_weights += numel

            print(f"[NeuronRank-Unstructured] layer {i} name {name} pruned {pruned} / {numel}")

    if args.nr_prune_lm_head and hasattr(model, "lm_head") and isinstance(model.lm_head, nn.Linear):
        weight = model.lm_head.weight.data
        metric = torch.abs(weight)
        pruned, numel = _apply_unstructured_mask(weight, metric, args.sparsity_ratio)
        total_pruned += pruned
        total_weights += numel
        print(f"[NeuronRank-Unstructured] lm_head pruned {pruned} / {numel}")

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

    if total_weights:
        pct = 100.0 * total_pruned / total_weights
        print(f"[NeuronRank-Unstructured] global pruning ratio {pct:.2f}% ({total_pruned}/{total_weights})")


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
    
    stats = collect_neuronrank_old_statistics(model, dataloader, device)
    
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
        def _apply_unstructured_mask(weight_tensor, metric_tensor, ratio):
            numel = metric_tensor.numel()
            num_to_prune = int(numel * ratio)
            if num_to_prune <= 0:
                return 0, numel

            flat_metric = metric_tensor.reshape(-1)
            threshold_idx = min(num_to_prune, flat_metric.numel() - 1)
            threshold = torch.sort(flat_metric)[0][threshold_idx]
            mask = metric_tensor <= threshold
            weight_tensor[mask] = 0
            return mask.sum().item(), numel

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

                # Build metric from neuron scores
                if layer_scores is None:
                    metric = torch.abs(weight).to(torch.float32)
                else:
                    neuron_scores = layer_scores.to(weight.device, dtype=torch.float32)
                    
                    # Broadcast neuron scores to weight dimensions
                    if "gate_proj" in name or "up_proj" in name:
                        metric = neuron_scores.view(-1, 1).expand_as(weight)
                    elif "down_proj" in name:
                        metric = neuron_scores.view(1, -1).expand_as(weight)
                    else:
                        metric = torch.abs(weight).to(torch.float32)

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

    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

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
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
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

    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

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
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
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