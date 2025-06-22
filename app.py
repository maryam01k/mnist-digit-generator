
import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Constants
latent_dim = 100
num_classes = 10
image_size = 28
model_path = "generator.pt"

# Generator class (same as used in training)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.label_embed = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, image_size * image_size),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        x = torch.cat([noise, self.label_embed(labels)], 1)
        img = self.model(x)
        return img.view(img.size(0), 1, image_size, image_size)

# Load model
device = torch.device("cpu")
G = Generator().to(device)
G.load_state_dict(torch.load(model_path, map_location=device))
G.eval()

# UI
st.title("MNIST Digit Generator")
digit = st.number_input("Select a digit (0–9)", min_value=0, max_value=9, step=1)

if st.button("Generate"):
    z = torch.randn(5, latent_dim)
    labels = torch.full((5,), digit, dtype=torch.long)
    with torch.no_grad():
        generated = G(z, labels).squeeze().cpu()

    # Plot results
    fig, axs = plt.subplots(1, 5, figsize=(10, 2))
    for i in range(5):
        axs[i].imshow(generated[i], cmap="gray")
        axs[i].axis("off")
    st.pyplot(fig)
