import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


def get_dataloaders(cuda, batch_size):
    kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                    transform=transforms.ToTensor()),
                    batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=False, **kwargs)
    return train_loader, test_loader


def plot_elbo(train_elbo_ls, test_elbo_ls):
    plt.plot(train_elbo_ls, label='train')
    plt.plot(test_elbo_ls, label='test')
    plt.xlabel('Epoch')
    plt.ylabel('ELBO')
    plt.legend()
    plt.title('Discrete VAE with 4 latents and 8 categories')
    plt.savefig('figures/elbo.png')
    plt.close()