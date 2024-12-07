import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_ENABLE_CUDA_FALLBACK"] = "1"


import torch 
import torch.mps
import torchvision
import os
import argparse
from tqdm import tqdm
from torch import autograd
import torch.distributions as MN
from model import Generator, Discriminator
from utils import load_model


def cal_deriv(inputs, outputs, device):
    grads = autograd.grad(outputs=outputs,
                          inputs=inputs,
                          grad_outputs=torch.ones(outputs.size()).to(device),
                          create_graph=True,
                          retain_graph=True,
                          only_inputs=True)[0]
    return grads

def langevin_sampling(zs, z_dim, generator, discriminator, batch_size, 
                      langevin_rate=0.001, langevin_noise_std=0.1,
                      langevin_steps=500, t=None, device='cuda', 
                      store_prev=False, each_it_save=0,
                      decay=False):

    zs_init = zs.clone()
    
    # Prior distribution initialization
    mean = torch.zeros(z_dim, device=device)
    prior_std = torch.eye(z_dim, device=device)
    lgv_std = prior_std * langevin_noise_std
    prior = MN.MultivariateNormal(loc=mean, covariance_matrix=prior_std)
    
    if store_prev:
        history = [zs.clone()]
    
    lgv_prior = MN.MultivariateNormal(loc=mean, covariance_matrix=lgv_std)
    for i in tqdm(range(langevin_steps)):
        zs = autograd.Variable(zs, requires_grad=True)
        fake_images = generator(zs)
        fake_dict = discriminator(fake_images)

        # Compute energy
 
        energy = -prior.log_prob(zs) - fake_dict
        
        # Compute gradients
        z_grads = cal_deriv(inputs=zs, outputs=energy, device=device)

        if t is not None and t < z_dim:
            mask = torch.zeros_like(z_grads)
            mask[:, :t] = 1.0 
            z_grads = z_grads * mask
        
        if decay:
            langevin_rate = langevin_rate / (1 + i / langevin_steps)

        # Update latent
        zs = zs - 0.5 * langevin_rate * z_grads + (langevin_rate**0.5) * lgv_prior.sample([batch_size])
        
        
        if store_prev and i % each_it_save == 0:
            history.append(zs.clone())
            
    if store_prev:
        return torch.stack(history, dim=0)
    else:
        return zs



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Images with DDLS.')
    parser.add_argument("--batch_size", type=int, default=200,
                      help="The batch size to use for training.")
    parser.add_argument("--langevin_steps", type=int, default=1000,
                      help="Number of Langevin steps for latent refinement.")
    args = parser.parse_args()

    # Set device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print('Loading Models...')
    mnist_dim = 784

    # Generator
    generator = Generator(g_output_dim=mnist_dim).to(device)
    generator = load_model(generator, 'checkpoints')
    generator = torch.nn.DataParallel(generator).to(device)
    generator.eval()

    # Discriminator
    discriminator = Discriminator(d_input_dim=mnist_dim).to(device)
    discriminator = load_model(discriminator, 'checkpoints')
    discriminator = torch.nn.DataParallel(discriminator).to(device)
    discriminator.eval()

    print('Models loaded.')

    print('Start Generating...')
    os.makedirs('samples', exist_ok=True)

    n_samples = 0
    batch_size_ = 200
    z_dim = 100
    loc = torch.zeros(z_dim).to(device)
    scale = torch.ones(z_dim).to(device)
    normal = MN.Normal(loc, scale*1.3)
    diagn = MN.Independent(normal, 1)
    

  
    while n_samples < 10000:
        zs = diagn.sample([batch_size_]).to(device)
        # Refine latents with DDLS
        zs_refined = langevin_sampling(
            zs, z_dim, generator, discriminator, 
            batch_size=batch_size_,
            langevin_steps=args.langevin_steps,
            t = 82,
            decay= False,
            store_prev=False,
            each_it_save=100,
            device = device
            )
            
        with torch.no_grad():
            x = generator(zs_refined)
            x = x.reshape(batch_size_, 28, 28)
            for k in range(x.shape[0]):
                if n_samples < 10000:
                    torchvision.utils.save_image(x[k:k+1], os.path.join('samples', f'{n_samples}.png'))         
                    n_samples += 1

