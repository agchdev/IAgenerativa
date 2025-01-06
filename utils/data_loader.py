import torch
import torchvision
import torchvision.transforms as transforms

def get_data_loader(batch_size=64):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalización a [-1, 1]
    ])

    dataset = torchvision.datasets.CIFAR10(
        root='data', 
        train=True, 
        download=True, 
        transform=transform
    )

    data_loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True
    )

    return data_loader
