import torch
from models import AutoEncoder

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import torchvision.utils as vutils
from utils import to_one_hot_vector

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sample_size = 64

    ckpt = torch.load("ckpts/recent.pth")
    model = AutoEncoder(ckpt["nc"], ckpt["ngf"]).to(device)
    model.load_state_dict(ckpt["netG"])

    img_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = MNIST('./data/MNIST', transform=img_transform)
    dataloader = DataLoader(dataset, batch_size=sample_size)

    imgs, label = next(iter(dataloader))

    new_imgs = reconstruct(model, imgs, label, device)
    vutils.save_image(imgs, "./inference_img/original.png")
    vutils.save_image(new_imgs, "./inference_img/new_img.png")

def reconstruct(model, imgs, label, device):
    imgs = imgs.to(device)
    label = to_one_hot_vector(10, label).to(device)
    new_imgs = model(imgs, label, label)
    return new_imgs.to("cpu")


def change_cond(model, imgs, label, device, target=0):
    imgs = imgs.to(device)
    new_label = label.clone().fill_(label[target])
    new_imgs = model(imgs, to_one_hot_vector(10, label).to(device), to_one_hot_vector(10, new_label).to(device))
    return new_imgs.to("cpu")


if __name__ == "__main__":
    main()