import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from utility.cutout import Cutout


class Cifar2:
    def __init__(self, batch_size, threads):
        mean, std = self._get_statistics()

        train_transform = transforms.Compose([
            torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            Cutout()
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

        self.train = torch.utils.data.DataLoader(train_set, batch_size=batch_size, num_workers=threads,sampler=train_sampler)
        self.test = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=threads)
        self.train_sampler = train_sampler

        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def _get_statistics(self):
        train_set = torchvision.datasets.CIFAR10(root='./cifar', train=True, download=True, transform=transforms.ToTensor())

        data = torch.cat([d[0] for d in DataLoader(train_set)])
        return data.mean(dim=[0, 2, 3]), data.std(dim=[0, 2, 3])
