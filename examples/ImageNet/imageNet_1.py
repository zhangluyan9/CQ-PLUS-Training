from __future__ import print_function
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import argparse
import torch
from torch.utils.data import Subset
import numpy as np
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from catSNN import spikeLayer, transfer_model, load_model, max_weight, normalize_weight, SpikeDataset , fuse_bn_recursively
#from utils import to_tensor

from torch.utils.data.sampler import SubsetRandomSampler

import torchvision.models as models

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def quantize_to_bit(x, nbit):
    if nbit == 32:
        return x
    return torch.mul(torch.round(torch.div(x, 2.0**(1-nbit))), 2.0**(1-nbit))
def transfer_model(src, dst, quantize_bit=32):
    src_dict = src.state_dict()
    dst_dict = dst.state_dict()
    reshape_dict = {}
    for (k, v) in src_dict.items():
        if k in dst_dict.keys():
            reshape_dict[k] = nn.Parameter(quantize_to_bit(v.reshape(dst_dict[k].shape), quantize_bit))
    dst.load_state_dict(reshape_dict, strict=False)
def data_loader(batch_size=128, workers=1, pin_memory=True):
    traindir = os.path.join('../../../../ImageNet/imagenet_raw/train')
    valdir = os.path.join('../../../../ImageNet/imagenet_raw/val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomRotation(10),
            transforms.Resize(256),
            #transforms.Resize(480),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            AddGaussianNoise(std=0.01),
            #normalize
        ])
    )
    train_dataset1 = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomRotation(15),
            transforms.Resize(256),
            #transforms.Resize(480),
            transforms.CenterCrop(224),
            #transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            AddGaussianNoise(std=0.01),
            #normalize
        ])
    )
    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            #transforms.Resize(480),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            #normalize
        ])
    )
    #val_dataset_100 = Subset(val_dataset, range(0,15000))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
        sampler=None
    )
    num_training_samples = 10 
    train_sampler = SubsetRandomSampler(torch.arange(1000, 1100))
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=600,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
        #sampler=train_sampler
    )
    
    return train_loader, val_loader, val_dataset

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        #onehot = torch.nn.functional.one_hot(target, 10)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def change_shape(feature):
    datashape = feature.shape
    for i in range(datashape[0]):
        for j in range(datashape[1]):
            feature[i][j] = torch.Tensor(0.25*np.ones((2,2)))
    return nn.Parameter(feature, requires_grad=True)
def test_(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            print(pred.eq(target.view_as(pred)).sum().item())
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return correct

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return correct
def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Cifar10 LeNet Example')
    parser.add_argument('--batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1e-5, metavar='LR',
                        help='learning rate (default: 1)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--resume', type=str, default=None, metavar='RESUME',
                        help='Resume model from checkpoint')
    parser.add_argument('--T', type=int, default=500, metavar='N',
                        help='SNN time window')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    #device = torch.device("cpu")
    device = torch.device("cuda" if use_cuda else "cpu")

    train_loader, val_loader , val_dataset= data_loader()

    snn_dataset = SpikeDataset(val_dataset, T = args.T,theta = 0.999)
    snn_loader = torch.utils.data.DataLoader(snn_dataset, batch_size=1, shuffle=False)

    from vgg_imagenet import CatVGG_t,VGG_t
    #model1 = models.vgg11_bn(pretrained=True)
    #torch.save(model1.state_dict(), "imagevgg11bn_o.pt")
    #for param_tensor in model1.state_dict():
    #    print(param_tensor, "\t", model1.state_dict()[param_tensor].size())
    model = VGG_t('VGG11',bias = True).to(device)
    snn_model = CatVGG_t('VGG11', args.T, bias = True).to(device)
    #model.load_state_dict(torch.load("imageNetvgg11.pt"), strict=False)
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    correct_ = 0
    optimizer = optim.SGD(model.parameters(), lr=args.lr,momentum = 0.9,weight_decay= 1e-4)
    #optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    k = 0
    for epoch in range(1, args.epochs + 1):
        #if k<10:
        #    optimizer = optim.SGD(model.parameters(), lr=1e-3,momentum = 0.9,weight_decay= 0.0001)
        #    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
        #elif k<20:
        #    optimizer = optim.SGD(model.parameters(), lr=1e-4,momentum = 0.9,weight_decay= 0.0001)
        #    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
        #else:
        #    optimizer = optim.SGD(model.parameters(), lr=1e-5,momentum = 0.9,weight_decay= 0.0001)
        #    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

        train(args, model, device, train_loader, optimizer, epoch)
        correct = test(model, device, val_loader)
        if correct>correct_:
            correct_ = correct
            torch.save(model.state_dict(), "imageNmybn11_c_d_2.pt")
        k+=1
        scheduler.step()
    #torch.save(model.state_dict(), "imageNmybn11_c_d.pt")
    #correct = test_(model, device, val_loader)

    model = fuse_bn_recursively(model)
    for param_tensor in model.state_dict():
            print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    for param_tensor in snn_model.state_dict():
        print(param_tensor, "\t", snn_model.state_dict()[param_tensor].size())

    transfer_model(model, snn_model)
    print("successful transfer")
    test_(snn_model, device, snn_loader)

    #test(snn_model, device, snn_loader)
    #if args.save_model
    #torch.save(model.state_dict(), "YOUR MOERL HERE.pt")


if __name__ == '__main__':
    main()
