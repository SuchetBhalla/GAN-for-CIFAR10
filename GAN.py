#MODULES
import matplotlib.pyplot as plt
import numpy as np

from sys import exit

from pathlib import Path

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision.datasets as datasets
from torch.utils.data import DataLoader

#CLASSES
#This NN classifies images as real or fake
class Discriminator(nn.Module):
    
    def __init__(self):
        
        super().__init__()
        
        
        self.conv1 = nn.Conv2d(in_channels= 3, out_channels= 64,
                               kernel_size= 3, stride= 1,
                               padding= 1)
        self.conv2 = nn.Conv2d(64, 128, 3, 2, 1)
        self.conv3 = nn.Conv2d(128, 128, 3, 2, 1)
        self.conv4 = nn.Conv2d(128, 256, 3, 2, 1)
        
        self.norm1 = nn.BatchNorm2d(128)
        self.norm2 = nn.BatchNorm2d(128)
        self.norm3 = nn.BatchNorm2d(256)
        
        self.fc1 = nn.Linear(256*4*4, 1)
    
    #forward  pass
    def forward(self, x):
        
        x= self.conv1(x)
        x= F.leaky_relu(x, negative_slope= 0.2)
        
        x= self.conv2(x)
        x= F.leaky_relu(self.norm1(x), negative_slope= 0.2)
        
        x= self.conv3(x)
        x= F.leaky_relu(self.norm2(x), negative_slope= 0.2)
        
        x= self.conv4(x)
        x= F.leaky_relu(self.norm3(x), negative_slope= 0.2)
        
        x = x.view(-1, 256*4*4)
        x = F.dropout(x, p= 0.4)
        
        x = self.fc1(x)
        x = torch.sigmoid(x)
        
        return x
    
#This NN generates fake images
class Generator( nn.Module ):
    
    def __init__(self, noise_dim):
        super().__init__()
        
        self.lin1 = nn.Linear( noise_dim, 256 * 4 * 4 )
        
        self.ct1 = nn.ConvTranspose2d(in_channels= 256, out_channels= 128,
                                      kernel_size= 4, stride= 2, padding= 1)
        self.ct2 = nn.ConvTranspose2d( 128, 128, 4, 2, 1 )
        self.ct3 = nn.ConvTranspose2d( 128, 128, 4, 2, 1 )
        
        self.conv = nn.Conv2d( 128, 3, 3, padding= 1 )
        
    #The sequential-network
    def forward(self, x):
        
        # Pass random-noise as input
        x = F.leaky_relu( self.lin1(x), negative_slope= 0.2 )
        
        x = x.view(-1, 256, 4, 4)

        x = F.leaky_relu( self.ct1(x), negative_slope= 0.2 )
        x = F.leaky_relu( self.ct2(x), negative_slope= 0.2 )
        x = F.leaky_relu( self.ct3(x), negative_slope= 0.2 )
        
        x = torch.tanh( self.conv(x) )
        
        return x

#FUNCTION DEFINITIONS
#This functions saves images in the folder '/kaggle/working/results/'
    # mode= 0 for fake, 1 for real
def matplotlib_imshow(img, step, mode):

    img = img / 2 + 0.5     # unnormalize
    
    npimg = img.numpy()
    
    if mode == 0:
        plt.imsave("/kaggle/working/results/fake" + str(step) +".png",
                   np.transpose(npimg, (1, 2, 0)) )
    if step == 0 and mode == 1:
        plt.imsave("/kaggle/working/results/real" + str(step) +".png",
                   np.transpose(npimg, (1, 2, 0)) )

#MAIN
#Creates a diretory to store the outputs of the generator
Path("/kaggle/working/results").mkdir(exist_ok=True)

# Hyperparameters
lr = 3e-4
noise_dim = 100
batch_size = 512
num_epochs = 200

#Creates the NNs
disc = Discriminator().cuda()
gen = Generator( noise_dim ).cuda()

#Optimizers
opt_disc = optim.Adam(disc.parameters(), lr=lr)
opt_gen = optim.Adam(gen.parameters(), lr=lr)

#Loss
criterion = nn.BCELoss()

#To view progress in learning
fixed_noise = torch.randn( (batch_size, noise_dim) ).cuda()

#To transform the dataset
transformations = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5 ,0.5), (0.5, 0.5 ,0.5)),
    ]
)

#Downloads the dataset
dataset = datasets.CIFAR10(root="dataset/", transform= transformations, download=True)

#Prepares it for usage
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#Training
step = 0

for epoch in range(num_epochs):
    for batch_idx, (real_imgs, _) in enumerate(loader):
        batch_size = real_imgs.shape[0]
        real_imgs= real_imgs.cuda()
        
        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        noise = torch.randn( batch_size, noise_dim ).cuda()
        fake_imgs = gen( noise )
        
        disc_real = disc( real_imgs ).view(-1)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))
        
        disc_fake = disc( fake_imgs ).view(-1)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        
        lossD = (lossD_real + lossD_fake) / 2
        
        disc.zero_grad()
        lossD.backward(retain_graph=True)
        opt_disc.step()

        ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        # where the second option of maximizing doesn't suffer from
        # saturating gradients
        
        output = disc( fake_imgs ).view(-1)
        
        lossG = criterion(output, torch.ones_like(output))
        
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()

        if batch_idx == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}] Batch {batch_idx}/{len(loader)} \
                      Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
            )

            with torch.no_grad():
                fake = gen( fixed_noise )
                
                #new
                # get some random training images
                dataiter = iter(loader)
                images, labels = next(dataiter)
                
                # create grid of images
                img_grid_fake = torchvision.utils.make_grid(fake[:64, :, :, :])
                img_grid_real = torchvision.utils.make_grid(images[:64, :, :, :])
                #img_grid_real = torchvision.utils.make_grid( real_imgs , normalize=True)
                
                # show images
                matplotlib_imshow(img_grid_fake.cpu(), step, 0)
                matplotlib_imshow(img_grid_real.cpu(), step, 1)
                
                
                step += 1
                
#clean up
del plt
del np
del exit

del torch
del torchvision
del transforms

del nn
del optim
del F

del datasets
del DataLoader

del disc, gen, opt_disc, opt_gen, criterion, fixed_noise, transformations

del dataset, loader