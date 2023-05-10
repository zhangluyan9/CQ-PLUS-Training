from __future__ import print_function
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from catSNN import spikeLayer, transfer_model, SpikeDataset ,load_model, fuse_module

class Quantization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, constant=100):
        ctx.constant = constant
        return torch.div(torch.floor(torch.mul(tensor, constant)), constant)

    @staticmethod
    def backward(ctx, grad_output):
        #print(grad_output)
        return F.hardtanh(grad_output), None 

Quantization_ = Quantization.apply

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1,1, bias=True)
        self.Bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3, 1,1, bias=True)
        self.Bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 1,bias=True)
        self.Bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(128, 128, 1, 1,0, bias=True)
        self.Bn4 = nn.BatchNorm2d(128)
        
        self.dropout1 = nn.Dropout2d(0.15)
        self.dropout2 = nn.Dropout2d(0.15)
        self.dropout3 = nn.Dropout2d(0.15)
        self.dropout4 = nn.Dropout2d(0.15)

        self.fc1 = nn.Linear(6272, 128, bias=True)
        self.fc2 = nn.Linear(128, 10, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.Bn1(x)
        x = torch.clamp(x, min=0, max=1)
        x = Quantization_(x,30)

        
        x = self.dropout1(x)
        y1 = x # 32

        x = self.conv2(x)
        x = self.Bn2(x)
        x = torch.clamp(x, min=0, max=1)
        x = Quantization_(x,30)

        x = torch.cat((x, y1), 1)
        y2 = x #32+32 = 64
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.Bn3(x)
        x = torch.clamp(x, min=0, max=1)
        x = Quantization_(x,30)

        x = self.dropout3(x)
        x = torch.cat((x, y2), 1)
        #64+64 =128

        x = self.conv4(x)
        x = self.Bn4(x)
        x = torch.clamp(x, min=0, max=1)
        x = Quantization_(x,30)

        self.dropout4 = nn.Dropout2d(0.15)

        x = F.avg_pool2d(x, 4)
        x = Quantization_(x,30)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.clamp(x, min=0, max=1)
        x = Quantization_(x,30)

        output = self.fc2(x)

        return output


class CatNet(nn.Module):

    def __init__(self, T):
        super(CatNet, self).__init__()
        self.T = T
        snn = spikeLayer(T)
        self.snn=snn

        self.conv1 = snn.conv(1, 32, 3, 1,1,bias=True)
        self.conv2 = snn.conv(32, 32, 3, 1,1,bias=True)
        self.conv3 = snn.conv(64, 64, 3,1,1, bias=True)
        self.conv4 = snn.conv(128, 128, 1, 1,0,bias=True)
        self.pool1 = snn.pool(4)

        self.fc1 = snn.dense((7,7,128), 128, bias=True)
        self.fc2 = snn.dense(128, 10, bias=True)


    def forward(self, x):
        x = self.snn.spike(self.conv1(x),theta = 0.999)
        y1=x
        x = self.snn.spike(self.conv2(x),theta = 0.999)
        x = torch.cat((x, y1), 1)
        y2=x
        x = self.snn.spike(self.conv3(x),theta = 0.999)
        x = torch.cat((x, y2), 1)
        
        x = self.snn.spike(self.conv4(x),theta = 0.999)
        x = self.snn.spike(self.pool1(x),theta = 0.999)
        x = self.snn.spike(self.fc1(x),theta = 0.999)
        x = self.fc2(x)
        return self.snn.sum_spikes(x)/self.T

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        onehot = torch.nn.functional.one_hot(target, 10)
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, onehot.type(torch.float))
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            onehot = torch.nn.functional.one_hot(target, 10)
            output = model(data)
            test_loss += F.mse_loss(output, onehot.type(torch.float), reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            #print(pred.eq(target.view_as(pred)).sum().item())
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=200, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=3, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1, metavar='LR',
                        help='learning rate (default: 1.0)')
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
    parser.add_argument('--T', type=int, default=30, metavar='N',
                        help='SNN time window')
    parser.add_argument('--resume', type=str, default=None, metavar='RESUME',
                        help='Resume model from checkpoint')
                        
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    #device = torch.device("cpu")
    kwargs = {'batch_size': args.batch_size}
    if use_cuda:
        kwargs.update({'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True},
                     )

    transform=transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    for i in range(0):

        im_aug = transforms.Compose([
        #transforms.RandomRotation(15),
        transforms.RandomCrop(28, padding = 6),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        AddGaussianNoise(std=0.01),
        ])
        dataset1 = dataset1 + datasets.MNIST('../data', train=True, download=True,
                       transform=im_aug)       
              
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)

    snn_dataset = SpikeDataset(dataset2, T = args.T,theta = 0.9)
    #print(type(dataset1[0][0]))
    train_loader = torch.utils.data.DataLoader(dataset1, batch_size=256, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset2, **kwargs)
    #print(test_loader[0])
    snn_loader = torch.utils.data.DataLoader(snn_dataset, **kwargs)

    model = Net().to(device)
    model.load_state_dict(torch.load("mnist_denseblock_.pt"), strict=False)
    snn_model = CatNet(args.T).to(device)

    if args.resume != None:
        load_model(torch.load(args.resume), model)
    for param_tensor in snn_model.state_dict():
            print(param_tensor, "\t", snn_model.state_dict()[param_tensor].size())
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)


    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()
    fuse_module(model)
    #test(model, device, train_loader)
    test(model, device, test_loader)
    #torch.save(model.state_dict(), "mnist_denseblock_.pt")
    #fuse_module(model)
    transfer_model(model, snn_model)
    test(snn_model, device, snn_loader)

    #if args.save_model:



if __name__ == '__main__':
    main()
