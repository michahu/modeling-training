import torch
from torch.utils.data import Dataset
import numpy as np
import random

from itertools import product


def distance(x, y):
    assert x.shape == y.shape
    return torch.norm(x - y, 2)


class ModularArithmetic(Dataset):
    @property
    def fns_dict(self):
        return {
            "add": lambda x, y, z: (x + y) % z,
            "sub": lambda x, y, z: (x - y) % z,
            "mul": lambda x, y, z: (x * y) % z,
            "div": lambda x, y, z: (x / y) % z,
        }

    def __init__(self, operation, num_to_generate: int = 113):
        """Generate train and test split"""
        result_fn = self.fns_dict[operation]
        self.x = [
            [i, j, num_to_generate]
            for i in range(num_to_generate)
            for j in range(num_to_generate)
        ]
        self.y = [result_fn(i, j, k) for (i, j, k) in self.x]

    def __getitem__(self, index):
        return torch.tensor(self.x[index]), self.y[index]

    def __len__(self):
        return len(self.x)


class SparseParity(Dataset):
    def __init__(self, num_samples, total_bits, parity_bits):
        self.x = torch.randint(0, 2, (num_samples, total_bits)) * 2 - 1.0
        # self.x = torch.tensor([[random.choice([-1, 1]) for j in range(total_bits)] for i in range(num_samples)])
        # self.s = np.random.choice(total_bits, parity_bits, replace=False)
        self.y = torch.prod(self.x[:, :parity_bits], dim=1)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]


class Addition(Dataset):
    def __init__(self, rep_dim, p):
        self.symbol_to_reps = {}
        for i in range(2 * p - 1):
            vec = torch.randn((1, rep_dim))
            symbol_reps[i] = vec / torch.norm(vec, 2)

        self.table = dict()
        self.pairs = [(i, j) for (i, j) in product(range(p), range(p)) if i <= j]
        for (i, j) in self.pairs:
            self.table[(i, j)] = i + j

    def __getitem__(self, index):
        i, j = self.pairs[index]
        sum = self.table((i, j))
        return (
            i,
            j,
            sum,
            self.symbol_to_reps[i],
            self.symbol_to_reps[j],
            self.symbol_to_reps[sum],
        )

    def get_i_from_rep(self, rep):
        for i, candidate_rep in self.symbol_to_reps.items():
            if torch.all(rep.to(candidate_rep.device) == candidate_rep):
                return i
        raise ValueError("No symbol found for rep")

    def get_closest_i_from_rep(self, rep):
        closest_rep = min(
            self.symbol_reps.values(),
            key=lambda pos_rep: distance(pos_rep.to(device), rep),
        )
        pred_i = get_i_from_rep(closest_rep)
        return pred_i
