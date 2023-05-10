
'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn.functional as F
import torch.nn as nn
import catSNN
import catCuda
#vgg19_pretrain_11_0429t1
timestep_f = 500
time_s = 500
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'VGG19_': [64, 64, (64,64), 128, 128, (128,128), 256, 256, 256, 256, (256,256), 512, 512, 512, 512, (512,512), 512, 512, 512, 512, (512,512)],
    'Mynetwork':[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512,512,512, 'M']
}
def create_spike_input_cuda(input,T):
    spikes_data = [input for _ in range(T)]
    out = torch.stack(spikes_data, dim=-1).type(torch.FloatTensor).cuda() #float
    out = catCuda.getSpikes(out, 1.0)
    return out

class NewSpike(nn.Module):
    def __init__(self, T = 16):
        super(NewSpike, self).__init__()
        self.T = T

    def forward(self, x):
        x = (torch.sum(x, dim=4))/self.T
        x = create_spike_input_cuda(x, self.T)
        return x
class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.div(torch.floor(torch.mul(input, timestep_f)), timestep_f)

    @staticmethod
    def backward(ctx, grad_output):
        #print(grad_output)
        return F.hardtanh(grad_output)

quan_my = STEFunction.apply
class Clamp_q_(nn.Module):
    def __init__(self, min=0.0, max=1.0):
        super(Clamp_q_, self).__init__()
        self.min = min
        self.max = max

    def forward(self, x):
        x = torch.clamp(x, min=self.min, max=self.max)
        x = quan_my(x)
        #print(x.grad)
        #x = torch.div(torch.floor(torch.mul(x, timestep_f)), timestep_f)
        return x

class Clamp(nn.Module):
    def __init__(self, min=0.0, max=1):
        super(Clamp, self).__init__()
        self.min = min
        self.max = max

    def forward(self, x):
        return torch.clamp(x, min=self.min, max=self.max)

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            # nn.init.normal_(m.weight, 0, 0.1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            # nn.init.normal_(m.weight, 0, 0.1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)



class VGG_t(nn.Module):
    def __init__(self, vgg_name, quantize_factor=-1, clamp_max = 1.0, bias=False, quantize_bit=32):
        super(VGG_t, self).__init__()
        self.quantize_factor=quantize_factor
        self.clamp_max = clamp_max
        self.bias = bias
        self.features = self._make_layers(cfg[vgg_name], quantize_bit=quantize_bit)
        self.classifier1 = nn.Linear(512 * 7 * 7, 1000, bias=True)
        self.features.apply(initialize_weights)


    def forward(self, x):
        #with torch.no_grad():
        out = self.features(x)
        #out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier1(out)
        return out
 


    def _make_layers(self, cfg, quantize_bit=32):
        layers = []
        in_channels = 3
        for x in cfg:
            # [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
            if x == 'M':
                layers += [nn.AvgPool2d(kernel_size=2, stride=2),Clamp_q_()]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1, bias=self.bias),nn.BatchNorm2d(x),
                           Clamp_q_()]#catSNN.Clamp(max = self.clamp_max)
                #if self.quantize_factor!=-1:
                #    layers += [catSNN.Quantize(self.quantize_factor)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)




class CatVGG_t(nn.Module):
    def __init__(self, vgg_name, T, bias=True):
        super(CatVGG_t, self).__init__()
        self.snn = catSNN.spikeLayer(T)
        self.T = T
        self.bias=bias

        self.features = self._make_layers(cfg[vgg_name])
        #self.classifier = nn.Linear(512, 10, bias=False)
        #((1,1,512),10)
        self.classifier1 = self.snn.dense((7,7,512),1000, bias=True)
        #self.classifier2 = self.snn.dense(4096,4096, bias=self.bias)
        #self.classifier3 = self.snn.dense(4096,1000, bias=self.bias)

    def forward(self, x):
        #print(x)
        out = self.features(x)
        #print(torch.sum(out))
        #out = self.snn.sum_spikes(out)/self.T
        #out = out.view(out.size(0), -1)
        out = self.classifier1(out)
        #out = self.classifier2(out)
        #out = self.classifier3(out)
        out = self.snn.sum_spikes(out)/self.T
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [self.snn.pool(2),self.snn.spikeLayer()]
            else:
                layers += [self.snn.conv(in_channels, x, kernelSize=3, padding=1, bias=self.bias),
                        self.snn.spikeLayer()]
                in_channels = x
        # layers += [self.snn.sum_spikes_layer()]
        return nn.Sequential(*layers)
        








def test():
    net = CatVGG('VGG11', 60)
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    print(y.size())

# test()
