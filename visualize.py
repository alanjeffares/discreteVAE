import torch
import matplotlib.pyplot as plt

from model import VAE
from helpers import get_dataloaders

# load the trained model with weights
model = VAE(ncat=4, nlat=8)
model.load_state_dict(torch.load('vae.pt'))


# visualize some of the reconstructions
_, test_loader = get_dataloaders(False, 5)
fig, ax = plt.subplots(5, 4, figsize=(4, 5)) 
x_input_probs, _ = next(iter(test_loader))
x_input_binarized = torch.bernoulli(x_input_probs)
x_input = x_input_binarized.view(-1, 784)
p = model.encode(x_input)
z = model.sample(p)
x_recon_probs = model.decode(z)
x_recon_binarized = torch.bernoulli(x_recon_probs)
for i in range(4):
    for j in range(5):
        if i == 0:
            ax[j,i].imshow(x_input_probs[j].view(28, 28).detach().numpy(), cmap='gray')
        elif i == 1:
            ax[j,i].imshow(x_input_binarized[j].view(28, 28).detach().numpy(), cmap='gray')
        elif i == 2:
            ax[j,i].imshow(x_recon_probs[j].view(28, 28).detach().numpy(), cmap='gray')
        else:
            ax[j,i].imshow(x_recon_binarized[j].view(28, 28).detach().numpy(), cmap='gray')
        ax[j, i].set_xticks([])
        ax[j, i].set_yticks([]) 
# add column labels
ax[0, 0].set_title('Input\nprobs')
ax[0, 1].set_title('Input\nsample')
ax[0, 2].set_title('VAE recon\nprobs')
ax[0, 3].set_title('VAE recon\nsample')
# add row labels
for i in range(5):
    ax[i, 0].set_ylabel(f'Img {i+1}')     
plt.tight_layout()
plt.savefig('figures/reconstructions.png')
plt.close()
            

# visualize some of the latent space by fixing values and varying one latent at a time 
fig, ax = plt.subplots(8, 2*4, figsize=(2*4, 8))
for lat in range(4):
    for cat in range(8):
        z = torch.zeros(1, 4, 8)
        z[0, :, 0] = 1
        z[0, lat, 0] = 0
        z[0, lat, cat] = 1
        x_recon = model.decode(z)
        x_recon = torch.round(x_recon.view(28, 28)).detach().numpy()
        ax[cat, int(2*lat+1)].imshow(x_recon, cmap='gray')
        ax[cat, int(2*lat+1)].set_xticks([])
        ax[cat, int(2*lat+1)].set_yticks([])   
        ax[cat, int(2*lat)].imshow(z.view(4, 8).detach().numpy(), cmap='gray')
        ax[cat, int(2*lat)].set_xticks([])
        ax[cat, int(2*lat)].set_yticks([])
# add column labels
for i in range(4):# adjust vertical alignment manually
    ax[0, int(2*i)].set_title(f'Latent {i+1}', pad=25)
    ax[0, int(2*i+1)].set_title(f'Decoded\nsample')
# add row labels
for i in range(8):
    ax[i, 0].set_ylabel(f'Cat {i+1}')
plt.tight_layout()
plt.savefig('figures/latents.png')
plt.close()
        
        
        
        