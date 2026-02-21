import os
import matplotlib.pyplot as plt
import torch
import torchvision
import numpy as np
from torch import nn
import torchvision.transforms as T
from torch.nn import functional as F
from torchvision.utils import make_grid
from torchvision.utils import make_grid

def init_weights(w):
    if isinstance(w, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        torch.nn.init.normal_(w.weight.data, 0., 0.02)
        if w.bias is not None:
            torch.nn.init.constant_(w.bias.data, 0.0)
    elif isinstance(w, (nn.BatchNorm2d, nn.BatchNorm1d)):
        torch.nn.init.normal_(w.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(w.bias.data, 0.0)


def update_discriminator(x, z, c1, c2, D, G, criterion, trainer_D):
    trainer_D.zero_grad()
    batch_size = x.shape[0]
    zeros = torch.zeros((batch_size,), device=x.device)
    ones = torch.ones((batch_size,), device=x.device)
    real_y, _, _ = D(x)
    fake_x = G(z, c1, c2)
    # Do not need to compute gradient for G, detach it from computing gradients.
    fake_y, _, _ = D(fake_x.detach())
    loss_D = (criterion(real_y, ones.reshape(real_y.shape)) +
        criterion(fake_y, zeros.reshape(fake_y.shape))) / 2
    loss_D.backward()
    trainer_D.step()
    return loss_D.item()

def update_generator(z, c1, c2, D, G, criterion, trainer_GQ, lambd):
    trainer_GQ.zero_grad()
    batch_size = z.shape[0]
    ones = torch.ones((batch_size,), device=z.device)
    fake_x = G(z, c1, c2)
    fake_y, fake_c1, fake_c2 = D(fake_x)
    loss_G = criterion(fake_y, ones.reshape(fake_y.shape))

    loss_cat = nn.CrossEntropyLoss()(fake_c1, torch.argmax(c1, dim=1))
    loss_cont = lambd * nn.MSELoss()(fake_c2, c2)
    loss_info =  loss_cat + loss_cont
    loss_total = loss_G + loss_info
    loss_total.backward()
    trainer_GQ.step()
    return loss_G.item(), loss_info.item()

def train_infogan(D, G, lr_D, lr_G, latent_dim, dataloader, num_codes, num_epochs, device,
                  fixed_noise, fixed_label, fixed_code, lambd=1., visualize=True):
    print(f"Device: {device}")
    criterion = nn.BCEWithLogitsLoss()

    G = G.to(device)
    D = D.to(device)

    D.apply(init_weights)
    G.apply(init_weights)

    trainer_D = torch.optim.Adam([
        {"params": D.head.parameters()},
        {"params": D.fc_shared.parameters()},
        {"params": D.conv.parameters()}],
        lr=lr_D, betas=(0.5, 0.999))
    trainer_GQ = torch.optim.Adam([
        {"params": G.parameters()},
        {"params": D.q_net.parameters()},
        {"params": D.q_c1.parameters()},
        {"params": D.q_c2.parameters()}],
        lr=lr_G, betas=(0.5, 0.999))

    metrics = []

    fixed_noise = fixed_noise.to(device)
    fixed_label = fixed_label.to(device)
    fixed_code = fixed_code.to(device)

    for epoch in range(num_epochs):
        G.train()
        D.train()
        loss_D_epoch = 0.
        loss_G_epoch = 0.
        loss_info_epoch = 0.
        num_instances = 0
        for step_num, batch in enumerate(dataloader):
            x,_ = batch
            x = x.to(device)
            batch_size = x.shape[0]
            num_instances += batch_size


            # # update discriminator
            # trainer_D.zero_grad()
            # zeros = torch.zeros((batch_size,), device=device)
            # ones = torch.ones((batch_size,), device=device)
            # real_y, _, _ = D(x)
            # c1 = F.one_hot(torch.randint(0, 10, (batch_size,))).to(device).float()
            # c2 = 2. * torch.rand(batch_size, num_codes, device=device) - 1.
            # z = torch.normal(0., 1., size=(batch_size, latent_dim), device=device)
            # fake_x = G(z, c1, c2)
            # # Do not need to compute gradient for G, detach it from computing gradients.
            # fake_y, _, _ = D(fake_x.detach())
            # loss_D = (criterion(real_y, ones.reshape(real_y.shape)) +
            #     criterion(fake_y, zeros.reshape(fake_y.shape))) / 2
            # loss_D.backward()
            # trainer_D.step()
            # loss_D_epoch += loss_D.item() * batch_size
            

            # # update generator and information loss
            # trainer_GQ.zero_grad()
            # c1 = F.one_hot(torch.randint(0, 10, (batch_size,))).to(device).float()
            # c2 = 2. * torch.rand(batch_size, num_codes, device=device) - 1.
            # z = torch.normal(0., 1., size=(batch_size, latent_dim), device=device)
            # fake_x = G(z, c1, c2)
            # fake_y, fake_c1, fake_c2 = D(fake_x)
            # loss_G = criterion(fake_y, ones.reshape(fake_y.shape))

            # loss_cat = nn.CrossEntropyLoss()(fake_c1, torch.argmax(c1, dim=1))
            # loss_cont = lambd * nn.MSELoss()(fake_c2, c2)
            # loss_info =  loss_cat + loss_cont
            # loss_total = loss_G + loss_info
            # loss_total.backward()
            # trainer_GQ.step()
            # loss_G_epoch += loss_G.item() * batch_size
            # loss_info_epoch += loss_info.item() * batch_size

            c1 = F.one_hot(torch.randint(0, 10, (batch_size,))).to(device).float()
            c2 = 2. * torch.rand(batch_size, num_codes, device=device) - 1.
            z = torch.normal(0., 1., size=(batch_size, latent_dim), device=device)
            loss_D = update_discriminator(x=x, z=z, c1=c1, c2=c2, D=D, G=G, criterion=criterion, trainer_D=trainer_D)
            loss_G, loss_info = update_generator(z=z, c1=c1, c2=c2, D=D, G=G, criterion=criterion, trainer_GQ=trainer_GQ, lambd=lambd)

            loss_D_epoch += loss_D * batch_size
            loss_G_epoch += loss_G * batch_size
            loss_info_epoch += loss_info * batch_size

        loss_D_epoch /= num_instances
        loss_G_epoch /= num_instances
        loss_info_epoch /= num_instances
        metrics.append([loss_D_epoch, loss_G_epoch, loss_info_epoch])
        print(f"[Epoch {epoch}/{num_epochs}] loss_D: {loss_D_epoch:.4f}, loss_G: {loss_G_epoch:.4f}, loss_info: {loss_info_epoch:.4f}")

        G.eval()
        D.eval()
    
        if visualize:
            os.makedirs("visualizations", exist_ok=True)
            with torch.no_grad():
                fake_data = G(fixed_noise, fixed_label, fixed_code).detach().cpu()
                fig, ax = plt.subplots(figsize=(19.2,10.8))
                ax.imshow(T.ToPILImage()(make_grid(fake_data, nrow=5)))
                ax.axis("off")
                ax.set_title(f"Epoch {epoch:02d}")
                plt.savefig(f"visualizations/generated-{epoch:02d}.png", bbox_inches="tight", pad_inches=1)
                plt.close()

    metrics = np.array(metrics)
    plt.semilogy(metrics[:, 0], label="loss_D")
    plt.semilogy(metrics[:, 1], label="loss_G")
    plt.semilogy(metrics[:, 2], label="loss_info")
    plt.legend()
    plt.show()
