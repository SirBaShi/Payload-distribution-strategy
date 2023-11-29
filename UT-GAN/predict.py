import torch
import torchvision
import numpy as np
from PIL import Image

def embed(X, M):
    X = torch.from_numpy(np.array(X)).float()
    M = torch.from_numpy(np.array(M)).float()
    X = X.unsqueeze(0)
    M = M.unsqueeze(0)
    X = (X - 127.5) / 127.5
    M = (M - 0.5) / 0.5
    P = generator(X)
    E = double_tanh(P * M)
    Y = X + E
    Y = Y * 127.5 + 127.5
    Y = Y.squeeze(0)
    Y = Image.fromarray(Y.numpy().astype(np.uint8))
    return Y

generator = torch.load('utgan.pth')
generator.eval()
X = Image.open('cover.pgm')
M = np.random.randint(0, 2, size=(8,))
Y = embed(X, M)
Y.save('stego.png')


def doubletanh(x):
    return torch.where(x <= -np.pi/2, -torch.ones_like(x), torch.where(x >= np.pi/2, torch.ones_like(x), torch.tanh(x)))
