import torch

from torch import nn
import itertools

import numpy as np

# Hooks and metrics
# TODO: make this generic


# https://gist.github.com/airalcorn2/50ec06517ce96ecc143503e21fa6cb91
def patch_attention(m):
    forward_orig = m.forward

    def wrap(*args, **kwargs):
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = False

        return forward_orig(*args, **kwargs)

    m.forward = wrap


class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out[1])

    def clear(self):
        self.outputs = []

    def get_metrics(self):
        data = self.outputs[0]
        bsz = data.shape[0]
        nhead = data.shape[1]

        return [
            get_matrix_metrics(data[i, j])
            for (i, j) in itertools.product(range(bsz), range(nhead))
        ]


def unpack_weights_transformer(layer):
    k = layer.attn.W_K.data
    q = layer.attn.W_Q.data
    v = layer.attn.W_V.data
    in_proj = layer.attn.W_O.data
    ffn_in = layer.mlp.W_in.data
    ffn_out = layer.mlp.W_out.data

    return k, q, v, in_proj, ffn_in, ffn_out


def unpack_biases_transformer(layer):
    b_in = layer.mlp.b_in.data
    b_out = layer.mlp.b_out.data

    return b_in, b_out


def unpack_weights_mlp(layer):
    return layer._parameters["weight"]


def unpack_weights_hf_transformer(layer):
    k = layer.attention.self.key.weight.data
    q = layer.attention.self.query.weight.data
    v = layer.attention.self.value.weight.data
    in_proj = layer.attention.output.dense.weight.data
    ffn_in = layer.intermediate.dense.weight.data
    ffn_out = layer.output.dense.weight.data

    return k, q, v, in_proj, ffn_in, ffn_out


def unpack_biases_hf_transformer(layer):
    b_in = layer.intermediate.dense.bias.data
    b_out = layer.output.dense.bias.data

    return b_in, b_out


def concatenate_matrices(matrices):
    return torch.cat([m.flatten() for m in matrices])


def get_layer_metrics_transformer(layer, nheads):
    k, q, v, in_proj, ffn_in, ffn_out = unpack_weights_transformer(layer)

    return (
        [get_matrix_metrics(k[i, :, :]) for i in range(nheads)],
        [get_matrix_metrics(q[i, :, :]) for i in range(nheads)],
        [get_matrix_metrics(v[i, :, :]) for i in range(nheads)],
        get_matrix_metrics(in_proj),
        get_matrix_metrics(ffn_in),
        get_matrix_metrics(ffn_out),
    )


def get_layer_metrics_hf_transformer(layer):
    k, q, v, in_proj, ffn_in, ffn_out = unpack_weights_hf_transformer(layer)

    return (
        # [get_matrix_metrics(k[i, :, :]) for i in range(nheads)],
        # [get_matrix_metrics(q[i, :, :]) for i in range(nheads)],
        # [get_matrix_metrics(v[i, :, :]) for i in range(nheads)],
        # NB: we might have separate into individual heads, but that's an empirical question
        get_matrix_metrics(k),
        get_matrix_metrics(q),
        get_matrix_metrics(v),
        get_matrix_metrics(in_proj),
        get_matrix_metrics(ffn_in),
        get_matrix_metrics(ffn_out),
    )


def get_layer_metrics_mlp(layer):
    weight = layer.weight
    # bias = layer.bias
    return get_matrix_metrics(weight)  # , get_matrix_metrics(bias)


def get_matrix_metrics(X):
    if torch.isnan(X).any():
        return
    if torch.isinf(X).any():
        return
    if torch.isneginf(X).any():
        return

    def get_flattened_l1_norm(x):
        return torch.linalg.vector_norm(x, ord=1)

    def get_flattened_l2_norm(x):
        return torch.linalg.vector_norm(x, ord=2)

    def get_spectral_norm(X):
        return torch.linalg.matrix_norm(X, ord=2)

    l1 = get_flattened_l1_norm(X).item()
    l2 = get_flattened_l2_norm(X).item()

    trace = torch.trace(X).item()
    spectral = get_spectral_norm(X).item()
    singular_vals = torch.svd(X, compute_uv=False).S
    singular_vals[singular_vals < 1e-5] = 0.0
    mean = torch.mean(singular_vals).item()
    var = torch.var(singular_vals).item()

    return {
        "l1": l1,
        "l2": l2,
        "trace": trace,
        "spectral": spectral,
        "code_sparsity": l1 / l2,
        "computational_sparsity": trace / spectral,
        "mean_singular_value": mean,
        "var_singular_value": var,
        "singular_values": singular_vals.tolist(),
    }


# assuming a 4D tensor
def get_tensor_metrics(X):
    def get_average_l1_norm(x):
        return torch.flatten(torch.linalg.vector_norm(x, ord=1, dim=(2, 3)))

    def get_average_l2_norm(x):
        return torch.flatten(torch.linalg.vector_norm(x, ord=2, dim=(2, 3)))

    l1s = get_average_l1_norm(X)
    l2s = get_average_l2_norm(X)
    code_sparsities = l1s / l2s

    return {
        "l1": torch.mean(l1s).item(),
        "l2": torch.mean(l2s).item(),
        "code_sparsity": torch.mean(code_sparsities).item(),
    }


@torch.no_grad()
def get_metrics_resnet18(model):
    data_dict = {
        "w": [],
    }
    weights = []
    biases = []

    data_dict["w"].append(get_tensor_metrics(model.conv1.weight))
    weights.append(model.conv1.weight.flatten())

    for layer in [model.layer1, model.layer2, model.layer3, model.layer4]:
        for block in layer:
            data_dict["w"].append(get_tensor_metrics(block.conv1.weight))
            data_dict["w"].append(get_tensor_metrics(block.conv2.weight))
            weights.append(block.conv1.weight.flatten())
            weights.append(block.conv2.weight.flatten())

    data_dict["w"].append(get_matrix_metrics(model.fc.weight))
    weights.append(model.fc.weight.flatten())
    biases.append(model.fc.bias.flatten())

    data_dict["w_all"] = get_distribution_stats(torch.cat(weights))
    data_dict["b_all"] = get_distribution_stats(torch.cat(biases))

    return data_dict


@torch.no_grad()
def get_metrics_lenet5(model):
    data_dict = {
        "w": [],
    }
    weights = []
    biases = []

    data_dict["w"].append(get_tensor_metrics(model.layer1[0].weight))
    weights.append(model.layer1[0].weight.flatten())
    biases.append(model.layer1[0].bias.flatten())

    data_dict["w"].append(get_tensor_metrics(model.layer2[0].weight))
    weights.append(model.layer2[0].weight.flatten())
    biases.append(model.layer2[0].bias.flatten())

    data_dict["w"].append(get_matrix_metrics(model.fc1.weight))
    weights.append(model.fc1.weight.flatten())
    biases.append(model.fc1.bias.flatten())

    data_dict["w"].append(get_matrix_metrics(model.fc2.weight))
    weights.append(model.fc2.weight.flatten())
    biases.append(model.fc2.bias.flatten())

    data_dict["w"].append(get_matrix_metrics(model.fc3.weight))
    weights.append(model.fc3.weight.flatten())
    biases.append(model.fc3.bias.flatten())

    data_dict["w_all"] = get_distribution_stats(torch.cat(weights))
    data_dict["b_all"] = get_distribution_stats(torch.cat(biases))

    return data_dict


@torch.no_grad()
def get_metrics_transformer(model, nheads):
    data_dict = {
        "k": [],
        "q": [],
        "v": [],
        "in_proj": [],
        "ffn_in": [],
        "ffn_out": [],
    }
    for i, layer in enumerate(model.blocks):
        k, q, v, in_proj, ffn_in, ffn_out = get_layer_metrics_transformer(layer, nheads)
        data_dict["k"].append({i: k})
        data_dict["q"].append({i: q})
        data_dict["v"].append({i: v})
        data_dict["in_proj"].append({i: in_proj})
        data_dict["ffn_in"].append({i: ffn_in})
        data_dict["ffn_out"].append({i: ffn_out})

    weights = torch.cat(
        [
            concatenate_matrices(unpack_weights_transformer(layer))
            for layer in model.blocks
        ]
    )
    biases = torch.cat(
        [
            concatenate_matrices(unpack_biases_transformer(layer))
            for layer in model.blocks
        ]
    )

    data_dict["w_all"] = get_distribution_stats(weights)
    data_dict["b_all"] = get_distribution_stats(biases)

    return data_dict


def get_distribution_stats(X):
    mean = torch.mean(X).item()
    var = torch.var(X).item()
    median = torch.median(X).item()

    return {"mean": mean, "var": var, "median": median}


@torch.no_grad()
def get_metrics_mlp(model):
    data_dict = {
        "w": [],
    }

    weights = []
    biases = []

    for i, layer in enumerate(model.layers):
        if isinstance(layer, nn.Linear):
            w = get_layer_metrics_mlp(layer)
            data_dict["w"].append({i: w})
            weights.append(layer.weight.flatten())
            biases.append(layer.bias.flatten())

    weights = torch.cat(weights)
    biases = torch.cat(biases)

    data_dict["w_all"] = get_distribution_stats(weights)
    data_dict["b_all"] = get_distribution_stats(biases)

    return data_dict


@torch.no_grad()
def get_metrics_hf_transformer(model):
    data_dict = {
        "k": [],
        "q": [],
        "v": [],
        "in_proj": [],
        "ffn_in": [],
        "ffn_out": [],
    }

    for i, layer in enumerate(model.bert.encoder.layer):
        k, q, v, in_proj, ffn_in, ffn_out = get_layer_metrics_hf_transformer(layer)
        data_dict["k"].append({i: k})
        data_dict["q"].append({i: q})
        data_dict["v"].append({i: v})
        data_dict["in_proj"].append({i: in_proj})
        data_dict["ffn_in"].append({i: ffn_in})
        data_dict["ffn_out"].append({i: ffn_out})

    weights = torch.cat(
        [
            concatenate_matrices(unpack_weights_hf_transformer(layer))
            for layer in model.bert.encoder.layer
        ]
    )
    biases = torch.cat(
        [
            concatenate_matrices(unpack_biases_hf_transformer(layer))
            for layer in model.bert.encoder.layer
        ]
    )

    data_dict["w_all"] = get_distribution_stats(weights)
    data_dict["b_all"] = get_distribution_stats(biases)

    return data_dict


@torch.no_grad()
def get_lm_loss_hf_transformer(model, train_dataloader, device):
    data_dict = {
        "train_loss": [],
    }

    train_loss = []

    for batch in train_dataloader:
        input_ids, attention_masks, labels = (
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
            batch["labels"].to(device),
        )
        # print(input_ids.shape, attention_masks.shape, labels.shape)
        loss = model(input_ids, attention_mask=attention_masks, labels=labels).loss
        train_loss.append(loss.item())

    data_dict["train_loss"] = np.mean(train_loss)

    return data_dict
