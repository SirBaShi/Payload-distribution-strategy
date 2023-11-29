import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from UT-GAN import Generator,Discrimnator

def doubletanh(x):
    return torch.where(x <= -np.pi/2, -torch.ones_like(x), torch.where(x >= np.pi/2, torch.ones_like(x), torch.tanh(x)))

lr = 0.0002 
bs = 16 
epochs = 100 
lambda1 = 0.01 
lambda2 = 0.01 

dataset = load_dataset()
train_set, test_set = split_dataset(dataset, ratio=0.8)
train_loader = DataLoader(train_set, batch_size=bs, shuffle=True)

generator = Generator()
discriminator = Discriminator()
double_tanh = DoubleTanh()

optimizer_G = Adam(generator.parameters(), lr=lr)
optimizer_D = Adam(discriminator.parameters(), lr=lr)

bce_loss = BCELoss() 
mse_loss = MSELoss() 
ce_loss = CELoss() 

for epoch in range(epochs):
    for cover, secret in train_loader:
        prob, stego = generator(cover, secret)
        embed = double_tanh(cover, prob, secret)
        real_prob = discriminator(cover)
        fake_prob = discriminator(embed)

        loss_G_adv = bce_loss(fake_prob, torch.ones(bs))
        loss_G_emb = mse_loss(embed, cover)
        loss_G_cls = ce_loss(stego, secret)
        loss_G = loss_G_adv + lambda1 * loss_G_emb + lambda2 * loss_G_cls

        loss_D_real = bce_loss(real_prob, torch.ones(bs))
        loss_D_fake = bce_loss(fake_prob, torch.zeros(bs))
        loss_D = (loss_D_real + loss_D_fake) / 2

        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

    print(f"Epoch {epoch}, Loss_G: {loss_G}, Loss_D: {loss_D}")

save_model(generator, discriminator
evaluate_model(generator, discriminator, test_set)
