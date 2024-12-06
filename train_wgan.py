import torch
import os
from tqdm import tqdm
import argparse
from torchvision import datasets, transforms
import torch.optim as optim
import matplotlib.pyplot as plt
from model import Generator, Critic
from utils import save_models
import torch.distributions as MN
from torchvision.utils import make_grid

def gradient_penalty(critic, real, fake, device):
    batch_size, C = real.shape
    epsilon = torch.rand(batch_size, 1).repeat(1, C).to(device)
    interpolated_images = real * epsilon + fake * (1 - epsilon)
    interpolated_images.requires_grad_(True)

    mixed_scores = critic(interpolated_images)
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True
    )[0]
    gradient = gradient.view(batch_size, -1)
    gradient_norm = gradient.norm(2, dim=1)
    return torch.mean((gradient_norm - 1) ** 2)

def train_critic(x, G, D, D_optimizer, lambda_gp, device, z_dim):
    x = x.to(device)
    z = torch.randn(x.size(0), z_dim).to(device)
    x_fake = G(z).detach()

    # Critic outputs
    D_real = D(x).mean()
    D_fake = D(x_fake).mean()

    # Loss
    gp = gradient_penalty(D, x, x_fake, device)
    D_loss = D_fake - D_real + lambda_gp * gp

    # Backpropagation
    D_optimizer.zero_grad()
    D_loss.backward()
    D_optimizer.step()

    return D_loss.item()

def train_generator(G, D, G_optimizer, device, z_dim, batch_size):
    z = torch.randn(batch_size, z_dim).to(device)
    G_fake = G(z)
    G_loss = -D(G_fake).mean()

    # Backpropagation
    G_optimizer.zero_grad()
    G_loss.backward()
    G_optimizer.step()

    return G_loss.item()

def plot_losses(G_losses, D_losses, epoch):
    plt.figure(figsize=(10, 5))
    plt.plot(G_losses, label='Generator Loss')
    plt.plot(D_losses, label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'Losses up to Epoch {epoch}')
    plt.savefig(f'losses_up_to_epoch_{epoch}.png')
    plt.close()

def main(args, continue_training=False, plot = False, plot_fake = False):
    
    
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    z_dim = 100
    mnist_dim = 784
    lambda_gp = args.lambda_gp
    n_critic = args.n_critic
    
    loc = torch.zeros(z_dim).to(device)
    scale = torch.ones(z_dim).to(device)
    normal = MN.Normal(loc, scale*2)
    diagn = MN.Independent(normal, 1)
    zs = diagn.sample([100]).to(device)
    
    


    # Dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])
    train_dataset = datasets.MNIST(root='data/MNIST/', train=True, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)

    if continue_training:
        G = Generator(g_output_dim=mnist_dim).to(device)
        D = Critic(mnist_dim).to(device)
        G.load_state_dict(torch.load("checkpoints/G.pth"))
        D.load_state_dict(torch.load("checkpoints/D.pth"))
        G.eval()
        D.eval()
    
    else:
        # Model Initialization
        G = Generator(g_output_dim=mnist_dim).to(device)
        D = Critic(mnist_dim).to(device)

    # Optimizers
    G_optimizer = optim.Adam(G.parameters(), lr=args.lr)
    D_optimizer = optim.Adam(D.parameters(), lr=args.lr)

    # Lists to collect losses
    G_losses = []
    D_losses = []
    

    # Training loop
    for epoch in tqdm(range(1, args.epochs + 1)):
        G_loss_epoch, D_loss_epoch = 0, 0
        for x, _ in tqdm(train_loader):
            x = x.view(-1, mnist_dim)
            for _ in range(n_critic):
                D_loss_epoch += train_critic(x, G, D, D_optimizer, lambda_gp, device, z_dim)
            # Train Generator
            G_loss_epoch += train_generator(G, D, G_optimizer, device, z_dim, x.size(0))

        


        # Average losses for the epoch
        G_loss_epoch /= len(train_loader)
        D_loss_epoch /= len(train_loader)*n_critic
        
        
        if len(G_losses) > 1:
            if G_loss_epoch > G_losses[-1]:
                n_critic -= 1
                if n_critic < 1:
                    n_critic = 1
            
        # Collect losses
        G_losses.append(G_loss_epoch)
        D_losses.append(D_loss_epoch)

        print(f"Epoch {epoch}/{args.epochs}, D Loss: {D_loss_epoch:.4f}, G Loss: {G_loss_epoch:.4f}")

        # Save checkpoints and plot losses every 10 epochs
        if epoch % 1 == 0:
            save_models(G, D, "checkpoints_wgan_adam", epoch)
            
            if plot:
                plot_losses(G_losses, D_losses, epoch)
                
            if plot_fake:
                plot_fake_datas(G, zs, z_dim, device, n=100, grid_size=(10, 10), save_path=f'fake_data_grid_epoch_{epoch}.png')
            
            
def plot_fake_datas(G, zs, z_dim, device, n=100, grid_size=(10, 10), save_path='fake_data_grid.png'):
    with torch.no_grad():
        x = G(zs)
        x = x.reshape(-1, 1, 28, 28)  # Reshape to (batch_size, channels, height, width)
        
        # Create a grid of images
        grid = make_grid(x, nrow=grid_size[1], normalize=True, padding=2)
        
        # Plot the grid
        plt.figure(figsize=(grid_size[1], grid_size[0]))
        plt.imshow(grid.permute(1, 2, 0).cpu(), cmap='gray')
        plt.axis('off')
        plt.title("Generated Images", fontsize=20)
        
        # Save the plot
        plt.savefig(save_path)
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train WGAN-GP.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs for training.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for training.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument("--lambda_gp", type=float, default=10, help="Gradient penalty lambda.")
    parser.add_argument("--n_critic", type=int, default=7, help="Number of Critic updates per Generator update.")
    args = parser.parse_args()

    main(args, continue_training=False, plot_fake=True, plot=True)
