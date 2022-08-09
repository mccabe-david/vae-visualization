"""
Import necessary libraries to create a variational autoencoder
The code is mainly developed using the PyTorch library
"""
import wfdb
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchvision import datasets, transforms
# from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
import random
# from scipy.io import loadmat

cwd = os.getcwd()
# wfdb.io.dl_database('mimic3wdb-matched/p04/p040239', os.getcwd())
record = wfdb.rdrecord('3458887_0008')

PPG_Data = []
# Load Data
for filename in os.listdir(cwd):
    if filename.endswith('.hea'):
        with open(cwd+'\\'+filename, "r") as f:
            if 'layout' in filename: continue
            for idx, line in enumerate(f.readlines()):
                if "PLETH" in line:
                    record = wfdb.rdrecord(filename[:-4])
                    if type(record.p_signal) == None:
                        PPG_Data.append(record.d_signal[:, idx - 1])
                    else:
                        PPG_Data.append(record.p_signal[:, idx - 1])
print(len(PPG_Data))
for sample in PPG_Data:
    print(len(sample))
raise


"""
Determine if any GPUs are available
"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


"""
Initialize Hyperparameters
"""
batch_size = 12
learning_rate = 1e-3
num_epochs = 10


"""
Create dataloaders to feed data into the neural network
Default MNIST dataset is used and standard train/test split is performed
"""
train_loader = torch.utils.data.DataLoader(
    recordings,
    batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    recordings,
    batch_size=1)


"""
A Convolutional Variational Autoencoder
"""
class VAE(nn.Module):
    def __init__(self, imgChannels=12, featureDim=12*32*1992, zDim=256):
        super(VAE, self).__init__()

        # Initializing the 2 convolutional layers and 2 full-connected layers for the encoder
        self.encConv1 = nn.Conv1d(imgChannels, 16, 5)
        self.encConv2 = nn.Conv1d(16, 32, 5)
        self.encFC1 = nn.Linear(featureDim, zDim)
        self.encFC2 = nn.Linear(featureDim, zDim)

        # Initializing the fully-connected layer and 2 convolutional layers for decoder
        self.decFC1 = nn.Linear(zDim, featureDim)
        self.decConv1 = nn.ConvTranspose1d(32, 16, 5)
        self.decConv2 = nn.ConvTranspose1d(16, imgChannels, 5)

    def encoder(self, x):

        # Input is fed into 2 convolutional layers sequentially
        # The output feature map are fed into 2 fully-connected layers to predict mean (mu) and variance (logVar)
        # Mu and logVar are used for generating middle representation z and KL divergence loss
        
        x = F.relu(self.encConv1(x))
        x = F.relu(self.encConv2(x))
        x = x.view(-1, 12*32*1992)
        mu = self.encFC1(x)
        logVar = self.encFC2(x)
        return mu, logVar

    def reparameterize(self, mu, logVar):

        #Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar/2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decoder(self, z):

        # z is fed back into a fully-connected layers and then into two transpose convolutional layers
        # The generated output is the same size of the original input
        x = F.relu(self.decFC1(z))
        x = x.view(-1, 12, 32, 1992)
        x = x.squeeze()
        x = F.relu(self.decConv1(x))
        x = torch.sigmoid(self.decConv2(x))
        return x

    def forward(self, x):

        # The entire pipeline of the VAE: encoder -> reparameterization -> decoder
        # output, mu, and logVar are returned for loss computation
        mu, logVar = self.encoder(x)
        z = self.reparameterize(mu, logVar)
        out = self.decoder(z)
        return out, mu, logVar

"""
Initialize the network and the Adam optimizer
"""
net = VAE().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

"""
Training the network for a given number of epochs
The loss after every epoch is printed
"""
for epoch in range(num_epochs):
    for idx, data in enumerate(train_loader, 0):

        imgs = data.float()
        imgs = imgs.to(device)
        
        print(idx, imgs.shape, max(np.asarray(imgs).flatten()))
        
        # Feeding a batch of images into the network to obtain the output image, mu, and logVar
        out, mu, logVar = net(imgs)

        # The loss is the BCE loss combined with the KL divergence to ensure the distribution is learnt
        kl_divergence = -0.5 * torch.sum(1 + logVar - mu.pow(2) - logVar.exp())
        loss = F.binary_cross_entropy(out, imgs, reduction='sum') + kl_divergence

        # Backpropagation based on the loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Epoch {}: Loss {}'.format(epoch, loss))



"""
The following part takes a random image from test loader to feed into the VAE.
Both the original image and generated image from the distribution are shown.
"""

net.eval()
with torch.no_grad():
    for data in random.sample(list(test_loader), 1):
        imgs, _ = data
        imgs = imgs.to(device)
        img = np.transpose(imgs[0].cpu().numpy(), [1,2,0])
        plt.subplot(121)
        plt.imshow(np.squeeze(img))
        out, mu, logVAR = net(imgs)
        outimg = np.transpose(out[0].cpu().numpy(), [1,2,0])
        plt.subplot(122)
        plt.imshow(np.squeeze(outimg))