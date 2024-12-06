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

torch.autograd.set_detect_anomaly(True)


def cal_deriv(inputs, outputs, device):
    grads = autograd.grad(outputs=outputs,
                          inputs=inputs,
                          grad_outputs=torch.ones(outputs.size()).to(device),
                          create_graph=True,
                          retain_graph=True,
                          only_inputs=True)[0]
    return grads

def new_langevin_sampling(zs, z_dim, generator, discriminator, batch_size, 
                      langevin_rate=0.01, langevin_noise_std=0.1,
                      langevin_steps=500, t=None, device="mps", 
                      store_prev=False, each_it_save=0,
                      decay = False, grad_threshold=1e-3, rate_decay_factor=0.5):
    
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
        GAN_part = -discriminator(fake_images)
        latent_part = -diagn.log_prob(zs)
        
        
        # Compute total energy
        energy = GAN_part + latent_part
        
        zs.requires_grad_(True)
        
        z_grads = cal_deriv(inputs=zs, outputs=energy, device=device)
        
        grad_norm = torch.norm(z_grads, p=2)
        if grad_norm < grad_threshold:
            langevin_rate *= rate_decay_factor
        
        
        if t is not None and t < z_dim:
            mask = torch.zeros_like(z_grads)
            mask[:, :t] = 1.0 
            z_grads = z_grads * mask
            
        
        if decay :
            langevin_rate = langevin_rate / (1 + i / langevin_steps)
        
        
        zs = zs - 0.5 * langevin_rate * z_grads + (langevin_rate**0.5) * lgv_prior.sample([batch_size])
    

        if store_prev and i % each_it_save == 0:
            history.append(zs.clone())
            
    if store_prev:
        return torch.stack(history, dim=0)
    else:
        return zs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Images with DDLS.')
    parser.add_argument("--batch_size", type=int, default=64,
                      help="The batch size to use for training.")
    parser.add_argument("--langevin_steps", type=int, default=750,
                      help="Number of Langevin steps for latent refinement.")
    parser.add_argument("--eps", type=float, default=0.01, help="The learning rate for Langevin sampling.")
    
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
    
    # model to use
    
    # 47 45 41 28 26 25 22 20

    # Generator
    generator = Generator(g_output_dim=mnist_dim).to(device)
    generator = load_model(generator, 'checkpoints_wgan_adam', epoch=20)
    generator = torch.nn.DataParallel(generator).to(device)
    generator.eval()

    # Discriminator
    discriminator = Discriminator(d_input_dim=mnist_dim).to(device)
    discriminator = load_model(discriminator, 'checkpoints_wgan_adam', epoch=20)
    discriminator = torch.nn.DataParallel(discriminator).to(device)
    discriminator.eval()
    
    
    for p in discriminator.parameters():  
        p.requires_grad = False
    for p in generator.parameters():  
        p.requires_grad = False

    print('Models loaded.')

    print('Start Generating...')
    os.makedirs('samples', exist_ok=True)

    n_samples = 0
    z_dim = 100
    loc = torch.zeros(z_dim).to(device)
    scale = torch.ones(z_dim).to(device)
    normal = MN.Normal(loc, scale*2)
    diagn = MN.Independent(normal, 1)

    batch_size = args.batch_size
    
    
  
    while n_samples < 10000:
        zs = diagn.sample([batch_size]).to(device)
        # Refine latents with DDLS
        zs_refined = new_langevin_sampling(zs, z_dim, generator, discriminator, batch_size,
                                             langevin_rate=args.eps, langevin_noise_std=0.1,
                                             langevin_steps=args.langevin_steps,
                                             t = 40,
                                             device=device,
                                             store_prev=False,
                                             each_it_save=1,
                                             decay= True
                                             )
        with torch.no_grad():
            x = generator(zs)
            x = x.reshape(-1, 28, 28)
            for k in range(x.shape[0]):
                if n_samples < 10000:
                    torchvision.utils.save_image(x[k:k+1], os.path.join('samples_wgan', f'{n_samples}.png'))         
                    n_samples += 1
