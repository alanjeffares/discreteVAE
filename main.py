import torch
import torch.nn.functional as F
import math

from model import VAE
from helpers import get_dataloaders, plot_elbo

BATCH_SIZE = 100
EPOCHS = 160
CUDA = True
          

def train(epoch, model, device, train_loader, optimizer):
    model.train()
    ELBO_train = 0
    for x_input, _ in train_loader:
        x_input = x_input.to(device).view(-1, 784)
        x_input = torch.bernoulli(x_input)  # sample a binarized input from the input probabilities
        optimizer.zero_grad()
        p = model.encode(x_input)
        z = model.sample(p)
        x_recon = model.decode(z)
        
        # to calculate gradients (eqns 84, 85) we will need to assemble the respective expressions
        # and then allow pytorch autograd to get the gradients for us using .backward()
        
        # start with the decoder parameters, theta, which only need BCE
        BCE = F.binary_cross_entropy(x_recon, x_input, reduce=False).sum(-1)
        loss_decoder = - BCE.sum()  # eqn 85
        
        # for the encoder we need two things, the entropy and the expectation terms
        entropy = torch.distributions.Categorical(p).entropy().sum()  # eqn 70
        
        p_sampled = torch.masked_select(p, z == 1.).reshape(p.shape[0], -1)  # probabilites corresponding to sampled z
        B = 125  # this is a baseline, which simply reduces the variance of the gradient estimator without changing the expected value
        expectation = torch.sum((BCE - B).detach() * torch.log(p_sampled).sum(-1))  # eqn 81
        
        loss_encoder = entropy - expectation  # eqn 84
        
        # put the loss together for backpropogation
        loss = loss_encoder + loss_decoder
        
        # also calculate the ELBO to track performance (eqn 86)
        ELBO = entropy - model.nlat * math.log(model.ncat) - BCE.sum()
                
        loss.backward()
        ELBO_train += ELBO.item()
        optimizer.step()

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, ELBO_train / len(train_loader.dataset)))
    return ELBO_train / len(train_loader.dataset)


    
def test(model, device, test_loader):
    with torch.no_grad():
        model.eval()
        running_KL = 0
        for x_input, _ in test_loader:
            x_input = x_input.to(device).view(-1, 784)
            x_input = torch.bernoulli(x_input)
            p = model.encode(x_input)
            z = model.sample(p)
            x_recon = model.decode(z)
            
            BCE = F.binary_cross_entropy(x_recon, x_input, reduce=False).sum()
            entropy = torch.distributions.Categorical(p).entropy().sum() 
            
            KL = entropy - model.nlat * math.log(model.ncat) - BCE
            running_KL += KL.item()
        
        print('====> Test set loss: {:.4f}'.format(running_KL / len(test_loader.dataset)))
    return running_KL / len(test_loader.dataset)

def main(nlat, ncat):
    device = torch.device("cuda" if CUDA else "cpu")
    model = VAE(nlat, ncat).to(device)
    train_loader, test_loader = get_dataloaders(CUDA, BATCH_SIZE)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, maximize=True)
    train_elbo_ls = []; test_elbo_ls = []
    for epoch in range(1, EPOCHS + 1):
        train_elbo_i = train(epoch, model, device, train_loader, optimizer)
        test_elbo_i = test(model, device, test_loader)
        train_elbo_ls.append(train_elbo_i); test_elbo_ls.append(test_elbo_i)
    torch.save(model.state_dict(), 'vae.pt')
    plot_elbo(train_elbo_ls, test_elbo_ls)
    
    
if __name__ == '__main__':
    main(4, 8)
