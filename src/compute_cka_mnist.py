# this calculation doesn't use parallelization because the models are much smaller

import torch
from torchvision import datasets, transforms

import glob
import numpy as np

from torch_cka import CKA

from src.model import MLP
from tqdm import tqdm

import argparse

# MNIST
# note: this analysis can't be done for modular arithmetic and sparse parities,
# because the datasets for these tasks are randomly generated,
# and so the representations are necessarily different

# hack to suppress labels
class XOnlyDataset(torch.utils.data.Dataset):
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset
    
    def __getitem__(self, index):
        x, y = self.original_dataset[index]
        return x
    
    def __len__(self):
        return len(self.original_dataset)

def compute_cka_mnist(pth_to_models, save_pth, bsz):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.flatten(x)),
        ]
    )
    # compare on the validation set
    data = datasets.MNIST(
        "./data", train=False, download=True, transform=transform
    )
    dataloader = torch.utils.data.DataLoader(
        XOnlyDataset(data), batch_size=bsz, shuffle=False
    )
    torch.manual_seed(0)

    model1 = MLP(input_dim=784, hidden_dims=[800, 800], output_dim=10)

    model2 = MLP(input_dim=784, hidden_dims=[800, 800], output_dim=10)

    data_arr = np.empty((587, 3, 40, 40))

    for i in tqdm(range(0, 2350, 4)):

        for seed_i in range(40):
            for seed_j in range(0, seed_i):
                model1.load_state_dict(torch.load(
                    glob.glob(pth_to_models + f'/*seed{seed_i}*/model_{i}.pt')[0]
                ))
                model2.load_state_dict(torch.load(
                    glob.glob(pth_to_models + f'/*seed{seed_j}*/model_{i}.pt')[0]
                ))

                cka = CKA(model1, model2,
                        model1_name="MLP1",
                        model2_name="MLP2",
                        model1_layers=['layers.0', 'layers.1', 'layers.2'],
                        model2_layers=['layers.0', 'layers.1', 'layers.2'],
                        device='cuda'
                )
                
                with torch.no_grad():
                    cka.compare(dataloader, only_compare_diagonals=True)
                    cka_out = cka.export()
                
                data_arr[i//4, :, seed_i, seed_j] = cka_out['CKA'].diag().numpy()
    
    np.save(save_pth, data_arr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pth_to_models', type=str, default='./data/raw/mnist_v3')
    parser.add_argument('--save_pth', type=str, default='./data/evals/cka_mnist.npy')
    parser.add_argument('--batch_size', type=int, default=1024)
    args = parser.parse_args()

    compute_cka_mnist(args.pth_to_models, args.save_pth, args.batch_size)


if __name__ == '__main__':
    main()



    
