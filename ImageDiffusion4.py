import os
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid GUI issues
import matplotlib.pyplot as plt
import einops
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, ToTensor, Lambda
import imageio
from PIL import Image

# Set environment variables to handle OpenMP conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
np.seterr(all='raise')  # Raise errors for numerical issues

# Setting reproducibility
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Parameters
no_train = False
batch_size = 128
n_epochs = 75
lr = 0.0005
beta1 = 0.9
beta2 = 0.999
store_path = "ddpm_digits.pt"
data_folder = r"C:/Users/Ron/Documents/2025/NN/Image_Diffuision_Model/DigitsData"

# Custom Dataset for DigitsData
class DigitsDataDataset(Dataset):
    def __init__(self, root_dir, transform=None, valid_extensions=('.png', '.jpg', '.jpeg')):
        self.root_dir = root_dir
        self.transform = transform
        self.valid_extensions = valid_extensions
        self.image_files = []
        self.labels = []

        for subdir in os.listdir(root_dir):
            subdir_path = os.path.join(root_dir, subdir)
            if os.path.isdir(subdir_path):
                for f in os.listdir(subdir_path):
                    if f.lower().endswith(valid_extensions):
                        img_path = os.path.join(subdir_path, f)
                        img_num = int(f.replace('image', '').replace('.png', '').replace('.jpg', '').replace('.jpeg', ''))
                        self.image_files.append(img_path)
                        if 1 <= img_num <= 1000:
                            label = 1
                        elif 1001 <= img_num <= 2000:
                            label = 2
                        elif 2001 <= img_num <= 3000:
                            label = 3
                        elif 3001 <= img_num <= 4000:
                            label = 4
                        elif 4001 <= img_num <= 5000:
                            label = 5
                        elif 5001 <= img_num <= 6000:
                            label = 6
                        elif 6001 <= img_num <= 7000:
                            label = 7
                        elif 7001 <= img_num <= 8000:
                            label = 8
                        elif 8001 <= img_num <= 9000:
                            label = 9
                        elif 9001 <= img_num <= 10000:
                            label = 0
                        else:
                            raise ValueError(f"Unexpected image number: {img_num} in {img_path}")
                        self.labels.append(label)

        sorted_pairs = sorted(zip(self.image_files, self.labels), key=lambda x: int(os.path.basename(x[0]).replace('image', '').split('.')[0]))
        self.image_files, self.labels = zip(*sorted_pairs) if sorted_pairs else ([], [])

        print(f"Found {len(self.image_files)} images in {root_dir}")

        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {root_dir}. Check the folder structure and file extensions.")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('L')
        if self.transform:
            image = self.transform(image)
        return image, label

# Define data transforms
transform = Compose([
    ToTensor(),
    Lambda(lambda x: (x - 0.5) * 2)
])

# Load dataset
try:
    dataset = DigitsDataDataset(root_dir=data_folder, transform=transform, valid_extensions=('.png', '.jpg', '.jpeg'))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
except ValueError as e:
    print(e)
    exit()

def show_images(images, title="", save_path=None):
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy()
    fig = plt.figure(figsize=(8, 8))
    rows = int(len(images) ** 0.5)
    cols = (len(images) + rows - 1) // rows
    for idx, img in enumerate(images):
        if idx < len(images):
            plt.subplot(rows, cols, idx + 1)
            plt.imshow(img[0], cmap="gray")
            plt.axis("off")
    fig.suptitle(title, fontsize=16)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

class MyBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, stride=1, padding=1):
        super(MyBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.activation = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)
        return out

def sinusoidal_embedding(n_steps, dim):
    embedding = torch.zeros(n_steps, dim)
    wk = torch.tensor([1 / 10000 ** (2 * j / dim) for j in range(dim)])
    wk = wk.reshape((1, dim))
    t = torch.arange(n_steps).reshape((n_steps, 1))
    embedding[:, ::2] = torch.sin(t * wk[:, ::2])
    embedding[:, 1::2] = torch.cos(t * wk[:, ::2])
    return embedding

class MyUNet(nn.Module):
    def __init__(self, n_steps=500, time_emb_dim=64):
        super(MyUNet, self).__init__()
        self.time_embed = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.requires_grad_(False)

        # Encoder
        self.te1 = self._make_te(time_emb_dim, 1)
        self.enc1 = MyBlock(1, 16)  # 28x28 -> 28x28
        self.pool1 = nn.Conv2d(16, 16, 3, stride=2, padding=1)  # 28x28 -> 14x14
        self.te2 = self._make_te(time_emb_dim, 16)
        self.enc2 = MyBlock(16, 32)  # 14x14 -> 14x14
        self.pool2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)  # 14x14 -> 7x7
        self.te3 = self._make_te(time_emb_dim, 32)
        self.enc3 = MyBlock(32, 64)  # 7x7 -> 7x7
        self.pool3 = nn.Conv2d(64, 64, 3, stride=2, padding=1)  # 7x7 -> 4x4

        # Bottleneck
        self.te_mid = self._make_te(time_emb_dim, 64)
        self.b_mid = MyBlock(64, 128)  # 4x4 -> 4x4

        # Decoder
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0)  # 4x4 -> 8x8
        self.crop1 = nn.Conv2d(64, 64, kernel_size=2, stride=1, padding=0)  # 8x8 -> 7x7
        self.te4 = self._make_te(time_emb_dim, 128)
        self.dec1 = MyBlock(128, 64)  # 7x7 -> 7x7
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, padding=0)  # 7x7 -> 14x14
        self.te5 = self._make_te(time_emb_dim, 64)
        self.dec2 = MyBlock(64, 32)  # 14x14 -> 14x14
        self.up3 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2, padding=0)  # 14x14 -> 28x28
        self.te6 = self._make_te(time_emb_dim, 32)
        self.dec3 = MyBlock(32, 16)  # 28x28 -> 28x28
        self.out = nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)  # 28x28 -> 28x28

    def _make_te(self, dim_in, dim_out):
        return nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.ReLU(),
            nn.Linear(dim_out, dim_out)
        )

    def forward(self, x, t):
        t = self.time_embed(t)
        n = len(x)
        out1 = self.enc1(x + self.te1(t).reshape(n, -1, 1, 1))
        out2 = self.enc2(self.pool1(out1) + self.te2(t).reshape(n, -1, 1, 1))
        out3 = self.enc3(self.pool2(out2) + self.te3(t).reshape(n, -1, 1, 1))
        out_mid = self.b_mid(self.pool3(out3) + self.te_mid(t).reshape(n, -1, 1, 1))
        up1 = self.up1(out_mid)
        up1 = self.crop1(up1)
        dec1 = self.dec1(torch.cat((out3, up1), dim=1))
        dec2 = self.dec2(torch.cat((out2, self.up2(dec1)), dim=1))
        dec3 = self.dec3(torch.cat((out1, self.up3(dec2)), dim=1))
        out = self.out(dec3)
        return out

class DDPM(nn.Module):
    def __init__(self, network, n_steps=500, min_beta=1e-4, max_beta=0.02, device=None, image_chw=(1, 28, 28)):
        super(DDPM, self).__init__()
        self.n_steps = n_steps
        self.device = device
        self.image_chw = image_chw
        self.network = network.to(device)
        self.betas = torch.linspace(min_beta, max_beta, n_steps).to(device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0).to(device)

    def forward(self, x0, t, eta=None):
        n, c, h, w = x0.shape
        a_bar = self.alpha_bars[t]
        if eta is None:
            eta = torch.randn(n, c, h, w).to(self.device)
        noisy = a_bar.sqrt().reshape(n, 1, 1, 1) * x0 + (1 - a_bar).sqrt().reshape(n, 1, 1, 1) * eta
        return noisy

    def backward(self, x, t):
        return self.network(x, t)

def generate_new_images(ddpm, n_samples=9, device=None, frames_per_gif=10, gif_name="sampling.gif", c=1, h=28, w=28):
    frame_idxs = np.linspace(0, ddpm.n_steps, frames_per_gif, endpoint=False).astype(np.int64)
    frames = []

    with torch.no_grad():
        if device is None:
            device = ddpm.device
        x = torch.randn(n_samples, c, h, w).to(device)
        for idx, t in enumerate(range(ddpm.n_steps - 1, -1, -1)):
            time_tensor = torch.full((n_samples,), t, device=device, dtype=torch.long)
            eta_theta = ddpm.backward(x, time_tensor)
            alpha_t = ddpm.alphas[t]
            alpha_t_bar = ddpm.alpha_bars[t]
            x = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta)
            if t > 0:
                z = torch.randn(n_samples, c, h, w).to(device)
                beta_t = ddpm.betas[t]
                sigma_t = beta_t.sqrt()
                x = x + sigma_t * z
            if idx in frame_idxs or t == 0:
                normalized = x.clone()
                for i in range(len(normalized)):
                    min_val, max_val = normalized[i].min(), normalized[i].max()
                    if max_val > min_val:  # Avoid division by zero
                        normalized[i] = (normalized[i] - min_val) / (max_val - min_val)
                        normalized[i] *= 255
                    else:
                        normalized[i] = torch.zeros_like(normalized[i])  # Handle edge case
                frame = einops.rearrange(normalized, "(b1 b2) c h w -> (b1 h) (b2 w) c", b1=int(n_samples ** 0.5))
                frame = frame.cpu().numpy().astype(np.uint8)
                frame = frame.squeeze(-1)  # Convert [84, 84, 1] to [84, 84] for grayscale
                frames.append(frame)
    with open(gif_name, "wb") as f:
        imageio.mimsave(f, frames, "GIF", duration=0.5)
    return x

def training_loop(ddpm, loader, n_epochs, optim, device, display_frequency=10, store_path="ddpm_digits.pt"):
    mse = nn.MSELoss()
    best_loss = float("inf")
    n_steps = ddpm.n_steps
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for batch in loader:
            x0 = batch[0].to(device)
            n = len(x0)
            eta = torch.randn_like(x0).to(device)
            t = torch.randint(0, n_steps, (n,), device=device)
            noisy_imgs = ddpm(x0, t, eta)
            eta_theta = ddpm.backward(noisy_imgs, t)
            loss = mse(eta_theta, eta)
            optim.zero_grad()
            loss.backward()
            optim.step()
            epoch_loss += loss.item() * len(x0) / len(loader.dataset)
        if (epoch + 1) % display_frequency == 0:
            generated = generate_new_images(ddpm, n_samples=9, device=device, gif_name=f"epoch_{epoch + 1}.gif")
            save_path = f"generated_epoch_{epoch + 1}.png"
            show_images(generated, f"Images generated at epoch {epoch + 1}", save_path=save_path)
        log_string = f"Epoch {epoch + 1}/{n_epochs}, Loss: {epoch_loss:.3f}"
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(ddpm.state_dict(), store_path)
            log_string += " --> Best model saved"
        print(log_string)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize model
net = MyUNet(n_steps=500)
ddpm = DDPM(net, n_steps=500, min_beta=1e-4, max_beta=0.02, device=device, image_chw=(1, 28, 28))

# Load pretrained model if not training
if no_train:
    try:
        ddpm.load_state_dict(torch.load(store_path, map_location=device))
        print(f"Loaded pretrained model from {store_path}")
    except FileNotFoundError:
        print(f"Pretrained model not found at {store_path}. Please train the model first.")
else:
    optim = Adam(ddpm.parameters(), lr=lr, betas=(beta1, beta2))
    training_loop(ddpm, loader, n_epochs, optim, device, display_frequency=10, store_path=store_path)

# Generate and display images
generated = generate_new_images(ddpm, n_samples=9, device=device, frames_per_gif=10, gif_name="final_sampling.gif")
show_images(generated, "Final generated images", save_path="final_generated.png")