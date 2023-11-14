import math
import os
import numpy as np
import einops

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.data import dataset

from torchvision.models.resnet import BasicBlock, ResNet, conv1x1


def custom_weight_init(m, init_scaling):
    if hasattr(m, "weight"):
        m.weight.data *= init_scaling

        if m.bias is not None:
            m.bias.data *= init_scaling


# Defining the convolutional neural network
class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc1 = nn.Linear(400, 120)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu1(out)
        out = self.fc3(out)
        return out


class MyBasicBlock(BasicBlock):
    def __init__(self, use_batch_norm, use_residual, **kwargs):
        self.use_batch_norm = use_batch_norm
        self.use_residual = use_residual
        super().__init__(**kwargs)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        if self.use_batch_norm:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.use_batch_norm:
            out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        if self.use_residual:
            out += identity

        out = self.relu(out)

        return out


class MyResNet(ResNet):
    def __init__(self, use_batch_norm, use_residual, **kwargs):
        self.use_batch_norm = use_batch_norm
        self.use_residual = use_residual
        super().__init__(**kwargs)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.use_batch_norm,
                self.use_residual,
                inplanes=self.inplanes,
                planes=planes,
                stride=stride,
                downsample=downsample,
                groups=self.groups,
                base_width=self.base_width,
                dilation=previous_dilation,
                norm_layer=norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.use_batch_norm,
                    self.use_residual,
                    inplanes=self.inplanes,
                    planes=planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        if self.use_batch_norm:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class MLP(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dims,
        output_dim,
        init_scaling: float = 2.0,
        kaiming_uniform: bool = True,
        use_batch_norm: bool = False,
        use_layer_norm: bool = False,
    ):
        super(MLP, self).__init__()
        self.init_scaling = init_scaling
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm

        if self.use_batch_norm and self.use_layer_norm:
            raise ValueError(
                "Only one of use_batch_norm and use_layer_norm can be True"
            )

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        if self.use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dims[0]))
        if self.use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dims[0]))
        layers.append(nn.ReLU())

        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))
            if self.use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dims[i + 1]))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        # layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)

        if kaiming_uniform:
            self.layers.apply(self._init_weights_kaiming)
        else:
            for layer in self.layers:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

    @torch.no_grad()
    def _init_weights_kaiming(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight)
            module.weight *= self.init_scaling

    def forward(self, x):
        x = self.layers(x)
        return x


class HookPoint(nn.Module):
    """A helper class to get access to intermediate activations (inspired by Garcon)
    It's a dummy module that is the identity function by default
    I can wrap any intermediate activation in a HookPoint and get a convenient way to add PyTorch hooks
    """

    def __init__(self):
        super().__init__()
        self.fwd_hooks = []
        self.bwd_hooks = []

    def give_name(self, name):
        # Called by the model at initialisation
        self.name = name

    def add_hook(self, hook, dir="fwd"):
        # Hook format is fn(activation, hook_name)
        # Change it into PyTorch hook format (this includes input and output,
        # which are the same for a HookPoint)
        def full_hook(module, module_input, module_output):
            return hook(module_output, name=self.name)

        if dir == "fwd":
            handle = self.register_forward_hook(full_hook)
            self.fwd_hooks.append(handle)
        elif dir == "bwd":
            handle = self.register_backward_hook(full_hook)
            self.bwd_hooks.append(handle)
        else:
            raise ValueError(f"Invalid direction {dir}")

    def remove_hooks(self, dir="fwd"):
        if (dir == "fwd") or (dir == "both"):
            for hook in self.fwd_hooks:
                hook.remove()
            self.fwd_hooks = []
        if (dir == "bwd") or (dir == "both"):
            for hook in self.bwd_hooks:
                hook.remove()
            self.bwd_hooks = []
        if dir not in ["fwd", "bwd", "both"]:
            raise ValueError(f"Invalid direction {dir}")

    def forward(self, x):
        return x


# %% ../transformer.ipynb 6
class Embed(nn.Module):
    """Define network architecture
    I defined my own transformer from scratch so I'd fully understand each component
    - I expect this wasn't necessary or particularly important, and a bunch of this replicates existing Pyt functionality
    """

    def __init__(self, d_vocab, d_model):
        super().__init__()
        self.W_E = nn.Parameter(torch.randn(d_model, d_vocab) / np.sqrt(d_model))

    def forward(self, x):
        return torch.einsum("dbp -> bpd", self.W_E[:, x])


# | export
class Unembed(nn.Module):
    def __init__(self, d_vocab, d_model):
        super().__init__()
        self.W_U = nn.Parameter(torch.randn(d_model, d_vocab) / np.sqrt(d_vocab))

    def forward(self, x):
        return x @ self.W_U


# | export
class PosEmbed(nn.Module):
    def __init__(self, max_ctx, d_model):
        super().__init__()
        self.W_pos = nn.Parameter(torch.randn(max_ctx, d_model) / np.sqrt(d_model))

    def forward(self, x):
        return x + self.W_pos[: x.shape[-2]]


# | export
class Attention(nn.Module):
    def __init__(self, d_model, num_heads, d_head, n_ctx, model):
        super().__init__()
        self.model = model
        self.W_K = nn.Parameter(
            torch.randn(num_heads, d_head, d_model) / np.sqrt(d_model)
        )
        self.W_Q = nn.Parameter(
            torch.randn(num_heads, d_head, d_model) / np.sqrt(d_model)
        )
        self.W_V = nn.Parameter(
            torch.randn(num_heads, d_head, d_model) / np.sqrt(d_model)
        )
        self.W_O = nn.Parameter(
            torch.randn(d_model, d_head * num_heads) / np.sqrt(d_model)
        )
        self.register_buffer("mask", torch.tril(torch.ones((n_ctx, n_ctx))))
        self.d_head = d_head
        self.hook_k = HookPoint()
        self.hook_q = HookPoint()
        self.hook_v = HookPoint()
        self.hook_z = HookPoint()
        self.hook_attn = HookPoint()
        self.hook_attn_pre = HookPoint()

    def forward(self, x):
        k = self.hook_k(torch.einsum("ihd,bpd->biph", self.W_K, x))
        q = self.hook_q(torch.einsum("ihd,bpd->biph", self.W_Q, x))
        v = self.hook_v(torch.einsum("ihd,bpd->biph", self.W_V, x))
        attn_scores_pre = torch.einsum("biph,biqh->biqp", k, q)
        attn_scores_masked = torch.tril(attn_scores_pre) - 1e10 * (
            1 - self.mask[: x.shape[-2], : x.shape[-2]]
        )
        attn_matrix = self.hook_attn(
            F.softmax(
                self.hook_attn_pre(attn_scores_masked / np.sqrt(self.d_head)), dim=-1
            )
        )
        z = self.hook_z(torch.einsum("biph,biqp->biqh", v, attn_matrix))
        z_flat = einops.rearrange(z, "b i q h -> b q (i h)")
        out = torch.einsum("df,bqf->bqd", self.W_O, z_flat)
        return out


# | export
class FF(nn.Module):
    def __init__(self, d_model, d_mlp, act_type, model):
        super().__init__()
        self.model = model
        self.W_in = nn.Parameter(torch.randn(d_mlp, d_model) / np.sqrt(d_model))
        self.b_in = nn.Parameter(torch.zeros(d_mlp))
        self.W_out = nn.Parameter(torch.randn(d_model, d_mlp) / np.sqrt(d_model))
        self.b_out = nn.Parameter(torch.zeros(d_model))
        self.act_type = act_type
        self.hook_pre = HookPoint()
        self.hook_post = HookPoint()
        assert act_type in ["ReLU", "GeLU"]

    def forward(self, x):
        x = self.hook_pre(torch.einsum("md,bpd->bpm", self.W_in, x) + self.b_in)
        if self.act_type == "ReLU":
            x = F.relu(x)
        elif self.act_type == "GeLU":
            x = F.gelu(x)
        x = self.hook_post(x)
        x = torch.einsum("dm,bpm->bpd", self.W_out, x) + self.b_out
        return x


# export
class TransformerBlock(nn.Module):
    def __init__(
        self, d_model, d_mlp, d_head, num_heads, n_ctx, act_type, model, use_ln=False
    ):
        super().__init__()
        self.model = model
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = Attention(d_model, num_heads, d_head, n_ctx, model=self.model)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = FF(d_model, d_mlp, act_type, model=self.model)
        self.hook_attn_out = HookPoint()
        self.hook_mlp_out = HookPoint()
        self.hook_resid_pre = HookPoint()
        self.hook_resid_mid = HookPoint()
        self.hook_resid_post = HookPoint()
        self.use_ln = use_ln

    def forward(self, x):
        x = self.hook_resid_mid(
            x + self.hook_attn_out(self.attn((self.hook_resid_pre(x))))
        )
        if self.use_ln:
            x = self.ln1(x)
        x = self.hook_resid_post(x + self.hook_mlp_out(self.mlp((x))))
        if self.use_ln:
            x = self.ln2(x)
        return x


# | export
class Transformer(nn.Module):
    def __init__(
        self,
        d_model,
        d_head,
        d_vocab,
        num_heads,
        num_layers,
        n_ctx,
        act_type="ReLU",
        use_ln=False,
    ):
        """this function could be augmented to contain more options for creating different architectures"""
        super().__init__()
        self.embed = Embed(d_vocab=d_vocab, d_model=d_model)
        self.pos_embed = PosEmbed(max_ctx=n_ctx, d_model=d_model)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    d_mlp=d_model * 4,
                    d_head=d_head,
                    num_heads=num_heads,
                    n_ctx=n_ctx,
                    act_type=act_type,
                    model=[self],
                    use_ln=use_ln,
                )
                for i in range(num_layers)
            ]
        )
        self.unembed = Unembed(d_vocab=d_vocab, d_model=d_model)

        for name, module in self.named_modules():
            if type(module) == HookPoint:
                module.give_name(name)

    def forward(self, x):
        x = self.embed(x)
        x = self.pos_embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.unembed(x)
        return x

    def hook_points(self):
        return [module for name, module in self.named_modules() if "hook" in name]

    def remove_all_hooks(self):
        for hp in self.hook_points():
            hp.remove_hooks("fwd")
            hp.remove_hooks("bwd")

    def cache_all(self, cache, incl_bwd=False):
        # Caches all activations wrapped in a HookPoint
        def save_hook(tensor, name):
            cache[name] = tensor.detach()

        def save_hook_back(tensor, name):
            cache[name + "_grad"] = tensor[0].detach()

        for hp in self.hook_points():
            hp.add_hook(save_hook, "fwd")
            if incl_bwd:
                hp.add_hook(save_hook_back, "bwd")


# from https://github.com/ejmichaud/grokking-squared/blob/0229df94de69b8384e560367280a43a238112bf5/notebooks/erics-implementation.ipynb
class ToyModel(nn.Module):
    def __init__(
        self,
        digit_rep_dim,
        internal_rep_dim,
        encoder_width=50,
        encoder_depth=3,
        decoder_width=50,
        decoder_depth=3,
        activation=nn.Tanh,
        device="cpu",
    ):
        """A toy model for grokking with an encoder, a exact addition operation, and a decoder.

        Arguments:
            digit_rep_dim (int): Dimension of vectors representing symbols in binary op table
            internal_rep_dim (int): Dimension of encoded representation (usually 1 or 2)
            encoder_width (int): Width of MLP for the encoder
            encoder_depth (int): Depth of MLP for the encoder (a depth of 2 is 1 hidden layer)
            decoder_width (int): Width of MLP for the decoder
            decoder_depth (int): Depth of MLP for the decoder (a depth of 2 is 1 hidden layer)
            activation: PyTorch class for activation function to use for encoder/decoder MLPs
            device: device to put the encoder and decoder on
        """
        super(ToyModel, self).__init__()
        self.digit_rep_dim = digit_rep_dim

        # ------ Create Encoder ------
        encoder_layers = []
        for i in range(encoder_depth):
            if i == 0:
                encoder_layers.append(nn.Linear(digit_rep_dim, encoder_width))
                encoder_layers.append(activation())
            elif i == encoder_depth - 1:
                encoder_layers.append(nn.Linear(encoder_width, internal_rep_dim))
            else:
                encoder_layers.append(nn.Linear(encoder_width, encoder_width))
                encoder_layers.append(activation())
        self.encoder = nn.Sequential(*encoder_layers)

        # ------ Create Decoder ------
        decoder_layers = []
        for i in range(decoder_depth):
            if i == 0:
                decoder_layers.append(nn.Linear(internal_rep_dim, decoder_width))
                decoder_layers.append(activation())
            elif i == decoder_depth - 1:
                decoder_layers.append(nn.Linear(decoder_width, digit_rep_dim))
            else:
                decoder_layers.append(nn.Linear(decoder_width, decoder_width))
                decoder_layers.append(activation())
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x1, x2):
        """Runs the toy model on input `x`.

        `x` must contain vectors of dimension 2 * `digit_rep_dim`, since it represents a pair
        of symbols that we want to compute our binary operation between.
        """
        return self.decoder(self.encoder(x1) + self.encoder(x2))
