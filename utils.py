import os
import torch
import datetime
import model as m
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.utils import make_grid



def plot_fake_data(fake_data, X_train, eps, epoch, path_to_save, sampling_mode = 'Langevin'):
    plt.figure(figsize=(8, 8))
    plt.xlim(-2., 2.)
    plt.ylim(-2., 2.)
    title = fr"Training data and {sampling_mode}"
    plt.title(title, fontsize=20)
    plt.scatter(X_train[:,:1], X_train[:,1:], alpha=0.5, color='gray', 
                marker='o', label = 'training samples')
    plt.scatter(fake_data[:,:1], fake_data[:,1:], alpha=0.5, color='blue', 
                marker='o', label = 'samples by G')
    plt.legend()
    plt.grid(True)
    if path_to_save is not None:
        cur_time = datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
        plot_name = cur_time + f'_wgan_sampling_{epoch}_epoch.jpg'
        path_to_plot = os.path.join(path_to_save, plot_name)
        plt.savefig(path_to_plot)
    else:
        plt.show()




def save_models(G, D, folder, epoch):
    torch.save(G.state_dict(), os.path.join(folder,f'G.pth_{epoch}'))
    torch.save(D.state_dict(), os.path.join(folder,f'D.pth_{epoch}'))


def load_model(model, folder, epoch=None):
    if isinstance(model, m.Generator):
        ckpt_path = os.path.join(folder, f"G.pth" if epoch is None else f"G.pth_{epoch}")
    elif isinstance(model, m.Discriminator):
        ckpt_path = os.path.join(folder, f"D.pth" if epoch is None else f"D.pth_{epoch}")
    else:
        raise ValueError("Model type not recognized. Expected m.Generator or m.Discriminator.")

    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
    model.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
    return model
def imshow(img, title=None):
    npimg = img.numpy()
    if npimg.ndim == 2:  # Image en niveaux de gris
        plt.imshow(npimg, cmap='gray')
    else:  # Image en couleur
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if title is not None:
        plt.title(title, y=-0.05, fontsize=15)
    plt.axis('off')
    plt.show()

def visualize_samples(folder, title, n_samples=225, grid_size=(15, 15)):
    index_random = np.random.randint(0, 10000, n_samples)
    image_files = sorted(os.listdir(folder))
    selected_files = [image_files[i] for i in index_random]
    images = []
    transform = transforms.ToTensor()
    
    for file in selected_files:
        img = Image.open(os.path.join(folder, file)).convert("L")  # Convertir en niveaux de gris
        img_tensor = transform(img)  # Convertir en tenseur
        images.append(img_tensor)
    
    # Créer une grille
    grid = make_grid(images, nrow=grid_size[1], normalize=True, padding=2)
    

    # Afficher la grille
    plt.figure(figsize=(15, 15))
    plt.imshow(grid.permute(1, 2, 0), cmap='gray')  # Permuter pour HWC
    plt.axis('off')
    plt.title(title, fontsize=20)
    plt.show()


def plot_evolution(directory: str = "evolution_ddls", grid_size=(5, 10), max = 100):
    images = []
    transform = transforms.ToTensor()
    for n in range(max + 1):
        img = Image.open(os.path.join(directory,f"{n}.png")).convert("L")
        img_tensor = transform(img)
        images.append(img_tensor)
    
    # # Créer une grille
    grid = make_grid(images, nrow=grid_size[1], normalize=True, padding=2)
    
    # Afficher la grille
    plt.figure(figsize=(grid_size[1], grid_size[0]))
    plt.imshow(grid.permute(1, 2, 0), cmap='gray')
    plt.axis('off')
    plt.title("Evolution of the generated images", fontsize=20)
    plt.show()



    
if __name__ == '__main__':
    visualize_samples('samples', "WGAN-GP DDLS t = 82 steps = 800 lr = 0.0009")
    # plot_evolution(max= 500)