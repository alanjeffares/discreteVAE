from torch import nn
from torch.nn import functional as F
from torch.distributions import OneHotCategorical

class VAE(nn.Module):
    def __init__(self, nlat=4, ncat=8):
        super(VAE, self).__init__()
        self.nlat = nlat
        self.ncat = ncat
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, nlat*ncat)
        self.fc4 = nn.Linear(nlat*ncat, 256)
        self.fc5 = nn.Linear(256, 512)
        self.fc6 = nn.Linear(512, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        logits = self.fc3(h2).view(len(x), self.nlat, self.ncat)
        return F.softmax(logits, -1)
    
    def decode(self, z):
        h4 = F.relu(self.fc4(z.view(len(z), self.nlat*self.ncat)))
        h5 = F.relu(self.fc5(h4))
        return F.sigmoid(self.fc6(h5))
    
    def sample(self, p):
        m = OneHotCategorical(p)
        return m.sample()