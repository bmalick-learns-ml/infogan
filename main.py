import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as T
from torch.nn import functional as F
from torchvision.utils import make_grid

from src.model import Generator, Discriminator
from src.train import train_infogan

if __name__=="__main__":

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5,),(0.5,))
        ])

    train_data = torchvision.datasets.MNIST(root="data/mnist", train=True, download=True,
                                            transform=transform)

    train_batch_size = 128
    num_workers = 2
    train_dataloader = torch.utils.data.DataLoader(dataset=train_data, batch_size=train_batch_size, num_workers=num_workers, shuffle=True)

    lr_D = 2.e-4
    lr_G = 1e-3
    lambd = 1.

    num_epochs = 50
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    latent_dim = 62
    num_codes = 2
    num_classes = 10
    discriminator = Discriminator(init_channels=64, num_classes=num_classes, num_codes=num_codes).to(device)
    generator = Generator(init_channels=1024, latent_dim=latent_dim, num_classes=num_classes, num_codes=num_codes).to(device)
        
    fixed_noise = torch.normal(0., 1., size=(10, latent_dim))
    fixed_label = F.one_hot(torch.arange(10)).float()
    fixed_code = 2. * torch.rand(10, num_codes) - 1.
    train_infogan(
        D=discriminator, G=generator, lr_D=lr_D, lr_G=lr_G, lambd=lambd,
        latent_dim=latent_dim, num_codes=num_codes,dataloader=train_dataloader, num_epochs=num_epochs, 
        device=device, fixed_noise=fixed_noise, fixed_label=fixed_label, fixed_code=fixed_code, 
        visualize=True)

    num_examples = 12
    labels = F.one_hot(torch.arange(10)).float().repeat_interleave(num_examples,dim=0).to(device)
    noise = torch.normal(0., 1., size=(num_examples, latent_dim)).to(device)
    code_vary_c2 = torch.cat(
        (2. * torch.rand(num_examples, 1) - 1., torch.linspace(-2,2,num_examples).reshape(-1,1)),
        dim=1).to(device)
    code_vary_c3 = torch.cat(
        (torch.linspace(-2,2,num_examples).reshape(-1,1), 2. * torch.rand(num_examples, 1) - 1.),
        dim=1).to(device)

    def vary_ci(ci):
        if ci=="c2":
            code = code_vary_c2
        elif ci=="c3":
            code = code_vary_c3
        else: return

        generator.eval()

        gen_imgs = []

        for i in range(10):
            with torch.no_grad():
                fake_data = generator(noise, labels[i*num_examples:(i+1)*num_examples], code).detach().cpu()
                gen_imgs.append(fake_data)
        gen_imgs = torch.cat(gen_imgs, dim=0)
        fig, ax = plt.subplots(figsize=(19.2,10.8))
        ax.imshow(T.ToPILImage()(make_grid(gen_imgs, nrow=num_examples)))
        ax.axis("off")
        plt.savefig(f"vary_{ci}.png", bbox_inches="tight", pad_inches=0)
        plt.close()

    vary_ci("c2")
    vary_ci("c3")


