from __future__ import print_function
import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = "0"
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from utils import to_tensor
import time
from catSNN import spikeLayer, transfer_model, load_model, max_weight, normalize_weight, SpikeDataset ,fuse_bn_recursively, fuse_module
#from models.vgg_threshold_training import VGG_o, CatVGG_o
from models.vgg_ import CatVGG,VGG_

import catSNN
import catCuda
timestep_ = 64
load_t = "cifar100_vggo_7765.pt"

f_store = "cifar100_ctt_cifaro_32.npz"
def create_spike_input_cuda(input,T):
    spikes_data = [input for _ in range(T)]
    out = torch.stack(spikes_data, dim=-1).type(torch.FloatTensor).cuda() #float
    out = catCuda.getSpikes(out, 1-0.001)
    return out

def change_shape(feature):
    datashape = feature.shape
    for i in range(datashape[0]):
        for j in range(datashape[1]):
            feature[i][j] = torch.Tensor(0.25*np.ones((2,2)))
    return nn.Parameter(feature, requires_grad=True)

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def Initialize_trainable_pooling(feature):
    datashape = feature.shape
    for i in range(datashape[0]):
        for j in range(datashape[1]):
            feature[i][j] = torch.Tensor(0.25*np.ones((2,2)))
    return nn.Parameter(feature, requires_grad=True)

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
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
            #print(pred.eq(target.view_as(pred)).sum().item())
            correct += pred.eq(target.view_as(pred)).sum().item()
            
    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return correct

def spike_channel(input,Threshold):
    input = input.transpose(-4, -5)
    output = input.clone().cuda()
    for i in range(input.shape[0]):
        output[i] = catCuda.getSpikes(input[i].clone(), Threshold[i])
    output = output.transpose(-4, -5)
    return output

class AddQuantization(object):
    def __init__(self, min=0., max=1.):
        self.min = min
        self.max = max
        
    def __call__(self, tensor):
        #return torch.div(torch.floor(torch.mul(tensor, timestep_f)), timestep_f)
        return torch.clamp(torch.div(torch.floor(torch.mul(tensor, timestep_)), timestep_),min=0, max=1)

def threshold_training_snn(input, y_origin,threshold,T,index):
    #print(index)
    #print(threshold[0])
    threshold_pre_1 = threshold
    mul =  input.shape[0] *input.shape[2] *input.shape[3] *input.shape[4]  
    y_new = spike_channel(input.clone(),threshold_pre_1)
    y = torch.clamp(torch.div(torch.floor(torch.mul(y_origin, timestep_)), timestep_),min=0, max=1)
    #print(y_origin.shape,y.shape,y_new.shape)
    #print(y_origin[0][0][0][0])
    #print(torch.sum(y[0][0][0][0]))
    #print(torch.sum(y_new[0][0][0][0]))
    threshold_1 =  threshold_pre_1
    #print(y_new.transpose(-4, -5).shape)
    #print(torch.sum(y_new[0])/100)
    #print(torch.sum(y[0]))
    #print(y.transpose(-3, -4).shape,y_new.shape,y_new.transpose(-4, -5).shape)
    #print(y.shape[0],y_new.shape[1])

    for i in range (y_new.shape[1]):
        j = 0 
        diff = (torch.sum(y.transpose(-3, -4)[i])*timestep_-torch.sum(y_new.transpose(-4, -5)[i]))/mul
        #print("before",diff)
        #print(diff)
        #diff_ = torch.sum(torch.logical_xor(y.transpose(-4, -5)[i],y_new.transpose(-4, -5)[i]))/mul
        threshold_1[i] = threshold_1[i] - 0.1*diff
    #print(threshold_1)
    threshold_pre_1 =  threshold_1 
    return y_new,y


class CatNet_training(nn.Module):

    def __init__(self, T):
        super(CatNet_training, self).__init__()
        self.T = T
        snn = spikeLayer(T)
        self.snn=snn
        self.conv1 = snn.conv(3, 128, kernelSize=3, padding=1,bias=True)
        self.conv1_ = nn.Conv2d(3, 128, kernel_size=3, padding=1, bias=True)
        self.conv2 = snn.conv(128, 128, kernelSize=3, padding=1,bias=True)
        self.conv2_ = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True)
        self.pool1 = snn.pool(2)
        self.pool1_ = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv3= snn.conv(128, 256, kernelSize=3, padding=1,bias=True)
        self.conv3_ = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=True)
        self.conv4= snn.conv(256, 256, kernelSize=3, padding=1,bias=True)
        self.conv4_ = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True)
        self.pool2 = snn.pool(2)
        self.pool2_ = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv5= snn.conv(256, 512, kernelSize=3, padding=1,bias=True)
        self.conv5_ = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=True)
        self.conv6 = snn.conv(512, 512, kernelSize=3, padding=1,bias=True)
        self.conv6_ = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.pool3 = snn.pool(2)
        self.pool3_ = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv7 = snn.conv(512, 1024, kernelSize=3, padding=0,bias=True)
        self.conv7_ = nn.Conv2d(512, 1024, kernel_size=3, padding=0, bias=True)

        self.pool4 = snn.pool(2)
        self.pool4_ = nn.AvgPool2d(kernel_size=2, stride=2)

        
        self.classifier1 = snn.dense((1,1,1024), 100, bias=True)
        self.classifier1_ =  nn.Linear(1024, 100, bias=True)


    def forward(self, x,threshold_pre_1,threshold_pre_2,threshold_pre_3,threshold_pre_4,threshold_pre_5,threshold_pre_6,threshold_pre_7,p1,p2,p3,p4):
        x_ = torch.sum(x,dim=-1)/timestep_
        x,y = threshold_training_snn(self.conv1(x),self.conv1_(x_),threshold_pre_1,self.T,1)
        x,y = threshold_training_snn(self.conv2(x),self.conv2_(y),threshold_pre_2,self.T,2)
        x,y = threshold_training_snn(self.pool1(x),self.pool1_(y),p1,self.T,3)
        x,y = threshold_training_snn(self.conv3(x),self.conv3_(y),threshold_pre_3,self.T,4)
        x,y = threshold_training_snn(self.conv4(x),self.conv4_(y),threshold_pre_4,self.T,5)
        x,y = threshold_training_snn(self.pool2(x),self.pool2_(y),p2,self.T,6)
        x,y = threshold_training_snn(self.conv5(x),self.conv5_(y),threshold_pre_5,self.T,7)
        x,y = threshold_training_snn(self.conv6(x),self.conv6_(y),threshold_pre_6,self.T,8)
        x,y = threshold_training_snn(self.pool3(x),self.pool3_(y),p3,self.T,9)
        x,y = threshold_training_snn(self.conv7(x),self.conv7_(y),threshold_pre_7,self.T,10)
        x,y = threshold_training_snn(self.pool4(x),self.pool4_(y),p4,self.T,11)
        #out = y.view(y.size(0), -1)
        x = self.classifier1(x)
        return self.snn.sum_spikes(x)/self.T,threshold_pre_1,threshold_pre_2,threshold_pre_3,threshold_pre_4,threshold_pre_5,threshold_pre_6,threshold_pre_7,p1,p2,p3,p4


def test_snn(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    # 78 77 77
    #9 9654 8 9697 7 9712 6 9734
    """
    f = np.load(f_store)
    threshold_pre_1 = f['threshold_pre_1']
    threshold_pre_2 = f['threshold_pre_2']
    threshold_pre_3 = f['threshold_pre_3']
    threshold_pre_4 = f['threshold_pre_4']
    threshold_pre_5 = f['threshold_pre_5']
    threshold_pre_6 = f['threshold_pre_6']
    threshold_pre_7 = f['threshold_pre_7']

    p1 = f['p1']
    p2 = f['p2']
    p3 = f['p3']
    p4 = f['p4']
    """

    fac =  0.999#T_reduce/timestep 
    threshold_pre_1 = np.ones(128)*fac
    threshold_pre_2 = np.ones(128)*fac
    threshold_pre_3 = np.ones(256)*fac
    threshold_pre_4 = np.ones(256)*fac
    threshold_pre_5 = np.ones(512)*fac
    threshold_pre_6 = np.ones(512)*fac
    threshold_pre_7 = np.ones(1024)*fac

    p1 = np.ones(128)*fac
    p2 = np.ones(256)*fac
    p3 = np.ones(512)*fac
    p4 = np.ones(1024)*fac
    
    with torch.no_grad():
        i = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output,threshold_pre_1,threshold_pre_2,threshold_pre_3,threshold_pre_4,threshold_pre_5,threshold_pre_6,threshold_pre_7,p1,p2,p3,p4 = model(data,threshold_pre_1,threshold_pre_2,threshold_pre_3,threshold_pre_4,threshold_pre_5,threshold_pre_6,threshold_pre_7,p1,p2,p3,p4)
            test_loss +=  F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            print(pred.eq(target.view_as(pred)).sum().item())
            np.savez(f_store,threshold_pre_1=threshold_pre_1,threshold_pre_2=threshold_pre_2,threshold_pre_3=threshold_pre_3,threshold_pre_4=threshold_pre_4,threshold_pre_5=threshold_pre_5,threshold_pre_6=threshold_pre_6,threshold_pre_7=threshold_pre_7,p1=p1,p2=p2,p3=p3,p4=p4)
    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset),100. * correct / len(test_loader.dataset)))
    

    return correct

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Cifar100 LeNet Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
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
    parser.add_argument('--T', type=int, default=timestep_, metavar='N',
                        help='SNN time window')
    parser.add_argument('--k', type=int, default=10, metavar='N',
                        help='Data augmentation')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'batch_size': args.batch_size}
    if use_cuda:
        kwargs.update({'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True},
                     )
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform_train = transforms.Compose([
        
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        AddGaussianNoise(std=0.01),
        AddQuantization()
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        AddQuantization()
        ])

    trainset = datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform_train)
    train_loader_ = torch.utils.data.DataLoader(trainset, batch_size=512, shuffle=True)
       
    for i in range(args.k):

        im_aug = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomCrop(32, padding = 6),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        AddGaussianNoise(std=0.01),
        AddQuantization()
        ])
        trainset = trainset + datasets.CIFAR100(root='./data', train=True, download=True, transform=im_aug)

    for i in range(args.k):
        im_aug = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomCrop(32, padding = 6),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        AddGaussianNoise(std=0.01),
        AddQuantization()
        ])
        trainset = trainset + datasets.CIFAR100(root='./data', train=True, download=True, transform=im_aug)
   

    
    train_loader = torch.utils.data.DataLoader(
       trainset, batch_size=512+256, shuffle=True)

    testset = datasets.CIFAR100(
        root='./data', train=False, download=False, transform=transform_test)

    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=15*8, shuffle=False,pin_memory = True)


    snn_dataset = SpikeDataset(trainset, T = args.T,theta = 0.9)
    snn_loader = torch.utils.data.DataLoader(snn_dataset, batch_size=50, shuffle=False, pin_memory = True)

    
    model = VGG_('o', clamp_max=1.0, bias = True).to(device)
    model.load_state_dict(torch.load(load_t), strict=False)
    snn_model = CatVGG('o', args.T, is_noise=False, bias = True).to(device)


    snn_model_training = CatNet_training(args.T).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    test(model, device, test_loader)
    correct_ = 0

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, train_loader_)
        correct = test(model, device, test_loader)
        if correct>correct_:
            correct_ = correct
        scheduler.step()

    model = fuse_bn_recursively(model)
    #for param_tensor in model.state_dict():
    #    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    transfer_model(model, snn_model)
    #test(snn_model, device, snn_loader)

    #for param_tensor in snn_model_training.state_dict():
    #    print(param_tensor, "\t", snn_model_training.state_dict()[param_tensor].size())
    
    model_dict = snn_model_training.state_dict()
    pre_dict = snn_model.state_dict()
    pre_dict_ = model.state_dict()

    reshape_dict = {}
    reshape_dict['conv1.weight'] = nn.Parameter(pre_dict['features.0.weight'],requires_grad=False)
    reshape_dict['conv1.bias'] = nn.Parameter(pre_dict['features.0.bias'],requires_grad=False)

    reshape_dict['conv2.weight'] = nn.Parameter(pre_dict['features.3.weight'],requires_grad=False)
    reshape_dict['conv2.bias'] = nn.Parameter(pre_dict['features.3.bias'],requires_grad=False)

    reshape_dict['conv3.weight'] = nn.Parameter(pre_dict['features.8.weight'],requires_grad=False)
    reshape_dict['conv3.bias'] = nn.Parameter(pre_dict['features.8.bias'],requires_grad=False)

    reshape_dict['conv4.weight'] = nn.Parameter(pre_dict['features.11.weight'],requires_grad=False)
    reshape_dict['conv4.bias'] = nn.Parameter(pre_dict['features.11.bias'],requires_grad=False)

    reshape_dict['conv5.weight'] = nn.Parameter(pre_dict['features.16.weight'],requires_grad=False)
    reshape_dict['conv5.bias'] = nn.Parameter(pre_dict['features.16.bias'],requires_grad=False)

    reshape_dict['conv6.weight'] = nn.Parameter(pre_dict['features.19.weight'],requires_grad=False)
    reshape_dict['conv6.bias'] = nn.Parameter(pre_dict['features.19.bias'],requires_grad=False)

    reshape_dict['conv7.weight'] = nn.Parameter(pre_dict['features.24.weight'],requires_grad=False)
    reshape_dict['conv7.bias'] = nn.Parameter(pre_dict['features.24.bias'],requires_grad=False)

    reshape_dict['classifier1.weight']=nn.Parameter(pre_dict['classifier1.weight'],requires_grad=False)
    reshape_dict['classifier1.bias']=nn.Parameter(pre_dict['classifier1.bias'],requires_grad=False)

    reshape_dict['conv1_.weight'] = nn.Parameter(pre_dict_['features.0.weight'],requires_grad=False)
    reshape_dict['conv1_.bias'] = nn.Parameter(pre_dict_['features.0.bias'],requires_grad=False)

    reshape_dict['conv2_.weight'] = nn.Parameter(pre_dict_['features.3.weight'],requires_grad=False)
    reshape_dict['conv2_.bias'] = nn.Parameter(pre_dict_['features.3.bias'],requires_grad=False)

    reshape_dict['conv3_.weight'] = nn.Parameter(pre_dict_['features.8.weight'],requires_grad=False)
    reshape_dict['conv3_.bias'] = nn.Parameter(pre_dict_['features.8.bias'],requires_grad=False)

    reshape_dict['conv4_.weight'] = nn.Parameter(pre_dict_['features.11.weight'],requires_grad=False)
    reshape_dict['conv4_.bias'] = nn.Parameter(pre_dict_['features.11.bias'],requires_grad=False)

    reshape_dict['conv5_.weight'] = nn.Parameter(pre_dict_['features.16.weight'],requires_grad=False)
    reshape_dict['conv5_.bias'] = nn.Parameter(pre_dict_['features.16.bias'],requires_grad=False)

    reshape_dict['conv6_.weight'] = nn.Parameter(pre_dict_['features.19.weight'],requires_grad=False)
    reshape_dict['conv6_.bias'] = nn.Parameter(pre_dict_['features.19.bias'],requires_grad=False)

    reshape_dict['conv7_.weight'] = nn.Parameter(pre_dict_['features.24.weight'],requires_grad=False)
    reshape_dict['conv7_.bias'] = nn.Parameter(pre_dict_['features.24.bias'],requires_grad=False)

    reshape_dict['classifier1_.weight']=nn.Parameter(pre_dict_['classifier1.weight'],requires_grad=False)
    reshape_dict['classifier1_.bias']=nn.Parameter(pre_dict_['classifier1.bias'],requires_grad=False)

    model_dict.update(reshape_dict)
    snn_model_training.load_state_dict(model_dict)

    #test(snn_model, device, snn_loader)
    
    test_snn(snn_model_training, device, snn_loader)


if __name__ == '__main__':
    main()
