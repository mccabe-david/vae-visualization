"""
Import necessary libraries to create a variational autoencoder
The code is mainly developed using the PyTorch library
"""
import wfdb
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import random
from bs4 import BeautifulSoup
import requests

def get_url_paths(url, folder = 'p05/'):
    response = requests.get(url + folder)
    if response.ok:
        response_text = response.text
    else:
        return response.raise_for_status()
    soup = BeautifulSoup(response_text, 'html.parser')
    temp = 'mimic3wdb-matched/' + folder
    parent = [temp + node.get('href')[:-1] for node in soup.find_all('a')]
    return parent

url = 'https://physionet.org/static/published-projects/mimic3wdb-matched/1.0/'
patients = get_url_paths(url, folder = 'p04/') # optionally specify folder here as an arg (e.g. folder = 'p07/')

cwd = os.getcwd()
patients_to_use = 1
PPG_Data = []
for m, patient in enumerate(random.sample(patients, patients_to_use)):
    
    print("Loading Patient " + str(m) + "'s data")
    # Turn off print statements
    sys.stdout = open(os.devnull, 'w')

    wfdb.io.dl_database(patient, cwd)
    
    # Turn print statements back on
    sys.stdout = sys.__stdout__

    # Load Patient PLETH Data
    for filename in os.listdir(cwd):
        if filename.endswith('.hea'):
            with open(cwd+'\\'+filename, "r") as f:
                if 'layout' not in filename: 
                    try: 
                        for idx, line in enumerate(f.readlines()):
                            if "PLETH" in line:
                                record = wfdb.rdrecord(filename[:-4])
                                if type(record.p_signal) != None and record.fs == 125:
                                    PPG_Data.append(record.p_signal[:, idx - 1])
                                break 
                        os.remove(filename[:-4]+ '.dat')
                    except:
                        pass
            os.remove(filename)
        if filename.endswith('n.dat'):
            os.remove(filename)

# Split data into 10-12 second intervals from each array
# (sampling rate is 125x per second)
recordings = []
sample_length = 1250
for patient in PPG_Data:
    temp = 0
    length = len(patient) - sample_length
    while(length > temp):
        temp2 = patient[temp:temp+sample_length]
        if not np.isnan(temp2).any(): recordings.append(temp2)
        temp += sample_length
del PPG_Data


"""
Determine if any GPUs are available
"""
raise
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


"""
Initialize Hyperparameters
"""
batch_size = 100
learning_rate = 1e-3
num_epochs = 10

"""
Final Input Processing
"""
random.shuffle(recordings)
recordings = recordings[:len(recordings)-(len(recordings)%batch_size)]
print("Total Samples: " + str(len(recordings)))
cutoff = int(0.9*(len(recordings)//batch_size))*batch_size

"""
Create dataloaders to feed data into the neural network
"""
train_loader = torch.utils.data.DataLoader(
    recordings[:cutoff],
    batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    recordings[cutoff:],
    batch_size=batch_size)


"""
A Convolutional Variational Autoencoder
"""
class VAE(nn.Module):
    def __init__(self, imgChannels=1, featureDim=batch_size*(sample_length-57), zDim=256):
        super(VAE, self).__init__()

        # Initializing the 2 convolutional layers and 2 full-connected layers for the encoder
        self.encConv1 = nn.Conv1d(imgChannels, 10, 40)
        self.encConv2 = nn.Conv1d(10, 5, 10)
        self.encConv3 = nn.Conv1d(5, 1, 10)
        self.encFC1 = nn.Linear(featureDim, zDim)
        self.encFC2 = nn.Linear(featureDim, zDim)

        # Initializing the fully-connected layer and 2 convolutional layers for decoder
        self.decFC1 = nn.Linear(zDim, featureDim)
        self.decConv1 = nn.ConvTranspose1d(1, 5, 10)
        self.decConv2 = nn.ConvTranspose1d(5, 10, 10)
        self.decConv3 = nn.ConvTranspose1d(10, 1, 40)

    def encoder(self, x):

        # Input is fed into 2 convolutional layers sequentially
        # The output feature map are fed into 2 fully-connected layers to predict mean (mu) and variance (logVar)
        # Mu and logVar are used for generating middle representation z and KL divergence loss
        x = torch.unsqueeze(x, 1)
        # print(x.shape)
        x = F.relu(self.encConv1(x))
        # print(x.shape)
        x = F.relu(self.encConv2(x))
        # print(x.shape)
        x = F.relu(self.encConv3(x))
        # print(x.shape)
        x = x.view(-1, batch_size*(sample_length-57))
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
        x = x.view(-1, batch_size, sample_length-57)
        x = torch.squeeze(x)
        x = torch.unsqueeze(x, 1)
        # print(x.shape)
        x = F.relu(self.decConv1(x))
        x = F.relu(self.decConv2(x))
        x = torch.sigmoid(self.decConv3(x))
        return x

    def forward(self, x):

        # The entire pipeline of the VAE: encoder -> reparameterization -> decoder
        # output, mu, and logVar are returned for loss computation
        mu, logVar = self.encoder(x)
        z = self.reparameterize(mu, logVar)
        out = self.decoder(z)
        out = torch.squeeze(out)
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
        # Feeding a batch of images into the network to obtain the output image, mu, and logVar
        out, mu, logVar = net(imgs)
        # The loss is the BCE loss combined with the KL divergence to ensure the distribution is learnt
        kl_divergence = -0.5 * torch.sum(1 + logVar - mu.pow(2) - logVar.exp())
        # if idx % 1000: print(kl_divergence)
        loss = F.mse_loss(out, imgs) # + kl_divergence
        if loss < 0:
            if kl_divergence < 0: 
                print("KL")
            else:
                print("MSE")
            raise
        
        # Backpropagation based on the loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

print('Epoch {}: Loss {}'.format(epoch, loss))
with torch.no_grad():
    for data in random.sample(list(test_loader), 1):
        plt.gca().set_prop_cycle(color=['red', 'green'])
        imgs = data.float()
        imgs = imgs.to(device)
        img = imgs[0]
        plt.plot(img)
        out, mu, logVAR = net(imgs)
        outimg = out[0]
        plt.plot(outimg)
        plt.show()

"""
The following part takes a random image from test loader to feed into the VAE.
Both the original image and generated image from the distribution are shown.
"""

net.eval()