from typing import Type, Any, Callable, Union, List, Optional

import torch
import torch.nn as nn
import scipy.stats
from torch import Tensor
import numpy as np
import torch.nn.functional as F
from utils import logger

bit_error_rate = 1e-8
algo_prepare = True
dump_featuremap = False
dump_dir = 'dump'
threshold = 1e-3
batch_id, layer_id = 0, 0

def FI_input(x_in: Tensor):
    assert x_in.dtype == torch.float32

    x = x_in.clone().reshape(-1)

    num = x.numel()

    bitlen = 32
    bitnum = num * bitlen
    
    n_inject = bitnum * bit_error_rate
    n_inject=scipy.stats.poisson.rvs(n_inject)
    # print(x_in.shape, bitnum, bitnum * bit_error_rate, n_inject)

    if n_inject == 0:
        return x_in
    
    index = torch.randint(0, bitnum, (n_inject,))
    
    vauleindex = torch.div(index, bitlen, rounding_mode='floor')
    bitindex = index%bitlen
    bitmask = 1<<bitindex.to(torch.int32)

    fi_values = x[vauleindex]

    np_value = fi_values.cpu().numpy()
    np_value.dtype = np.int32 # trickly change type

    np_value ^= bitmask.cpu().numpy()

    np_value.dtype = np.float32

    fi_values = torch.tensor(np_value, device=x_in.device)

    x[vauleindex] = fi_values
    return x.reshape_as(x_in)

class WeihgtFI:
    def __init__(self, conv_layer) -> None:
        self.weight = conv_layer.weight

    def __enter__(self):
        w_in = self.weight
        assert w_in.dtype == torch.float32

        w = w_in.view(-1)

        num = w.numel()

        bitlen = 32
        bitnum = num * bitlen
        
        n_inject = bitnum * bit_error_rate
        n_inject=scipy.stats.poisson.rvs(n_inject)
        print(w_in.shape, bitnum, bitnum * bit_error_rate, n_inject)

        if n_inject == 0:
            self.value_index = None
            return
        
        index = torch.randint(0, bitnum, (n_inject,))
        
        vauleindex = torch.div(index, bitlen, rounding_mode='floor')
        bitindex = index%bitlen
        bitmask = 1<<bitindex.to(torch.int32)

        values = w[vauleindex]
        self.value_index = vauleindex
        self.orig_values = values

        np_value = values.cpu().numpy()
        np_value.dtype = np.int32 # trickly change type

        np_value ^= bitmask.cpu().numpy()

        np_value.dtype = np.float32

        fi_values = torch.tensor(np_value, device=w.device)

        w[vauleindex] = fi_values

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.value_index is not None:
            self.weight.view(-1)[self.value_index] = self.orig_values

class Conv2d_Raw(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(Conv2d_Raw, self).__init__(*args, **kwargs)
    
    def forward(self, input):
        global batch_id, layer_id

        if dump_featuremap:
            logger.logger.info('dumping batch %d layer %d'%(batch_id, layer_id))
            torch.save(input, dump_dir+'/'+'batch%d_layer%d'%(batch_id, layer_id)+'.pth')
        logger.logger.debug('run batch %d layer %d'%(batch_id, layer_id))
        
        out = super(Conv2d_Raw, self).forward(input)    
        
        logger.logger.debug('done batch %d layer %d'%(batch_id, layer_id))
        layer_id += 1

        return out

class Conv2d_Raw_FI_activation(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(Conv2d_Raw_FI_activation, self).__init__(*args, **kwargs)
    
    def forward(self, input):
        global batch_id, layer_id
        
        input = FI_input(input)

        if dump_featuremap:
            logger.logger.info('dumping batch %d layer %d'%(batch_id, layer_id))
            torch.save(input, dump_dir+'/'+'batch%d_layer%d'%(batch_id, layer_id)+'.pth')
        logger.logger.debug('run batch %d layer %d'%(batch_id, layer_id))
        
        out = super(Conv2d_Raw_FI_activation, self).forward(input)
        
        logger.logger.debug('done batch %d layer %d'%(batch_id, layer_id))
        layer_id += 1

        return out

class ProtectedConv2d_Threshold_FI_activation(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(ProtectedConv2d_Threshold_FI_activation, self).__init__(*args, **kwargs)
        self.min = -np.inf
        self.max = np.inf
    
    def forward(self, input):
        if algo_prepare:
            out = super(ProtectedConv2d_Threshold_FI_activation, self).forward(input)
            self.min = max(self.min, torch.min(out))
            self.max = min(self.max, torch.max(out))
            return out

        input = FI_input(input)
        out = super(ProtectedConv2d_Threshold_FI_activation, self).forward(input)
        #out[torch.where(torch.isnan(out))] = 0
        return torch.clip(out, self.min, self.max)
        
class ProtectedConv2d_Threshold_set0_FI_activation(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(ProtectedConv2d_Threshold_set0_FI_activation, self).__init__(*args, **kwargs)
        self.min = -np.inf
        self.max = np.inf
    
    def forward(self, input):
        if algo_prepare:
            out = super(ProtectedConv2d_Threshold_set0_FI_activation, self).forward(input)
            self.min = max(self.min, torch.min(out))
            self.max = min(self.max, torch.max(out))
            return out

        input = FI_input(input)
        out = super(ProtectedConv2d_Threshold_set0_FI_activation, self).forward(input)
        out[torch.where((out>self.max) | (out<self.min) | torch.isnan(out))] = 0
        return out

class ProtectedConv2d_TMR_FI_weight(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(ProtectedConv2d_TMR_FI_weight, self).__init__(*args, **kwargs)
        self.conv2 = nn.Conv2d(*args, **kwargs)
        self.conv3 = nn.Conv2d(*args, **kwargs)
        self.init_state = False

    def init_weight(self):
        self.conv2.weight.copy_(self.weight)
        self.conv3.weight.copy_(self.weight)
    
    def forward(self, input):
        if not self.init_state:
            self.init_weight()
            self.init_state = True

        with WeihgtFI(self):
            out1 = super(ProtectedConv2d, self).forward(input)
        with WeihgtFI(self.conv2):
            out2 = self.conv2.forward(input)
        with WeihgtFI(self.conv3):
            out3 = self.conv3.forward(input)

        diffpos = torch.where(out1!=out2)
        ref = out3[diffpos]
        alldiffpos = torch.where((out1[diffpos]!=ref) & (out2[diffpos]!=ref))
        ref[alldiffpos] = 0
        out1[diffpos] = ref
        # out1[torch.where(torch.isnan(out1) | torch.isinf(out1))] = 0
        return out1
        return (out1+out2+out3)/3

class ProtectedConv2d_TMR_FI_activation(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(ProtectedConv2d_TMR_FI_activation, self).__init__(*args, **kwargs)
        self.conv2 = nn.Conv2d(*args, **kwargs)
        self.conv3 = nn.Conv2d(*args, **kwargs)
        self.init_state = False

    def init_weight(self):
        self.conv2.weight.copy_(self.weight)
        self.conv3.weight.copy_(self.weight)
    
    def forward(self, input):
        if not self.init_state:
            self.init_weight()
            self.init_state = True

        out1 = super(ProtectedConv2d_TMR_FI_activation, self).forward(FI_input(input))
        out2 = self.conv2.forward(FI_input(input))
        out3 = self.conv3.forward(FI_input(input))

        diffpos = torch.where(out1!=out2)
        if len(diffpos[0]):
            logger.logger.info('TMR find mismatch %d pos in %s output'%(len(diffpos[0]), str(out1.shape)))
            
        ref = out3[diffpos]
        #alldiffpos = torch.where((out1[diffpos]!=ref) & (out2[diffpos]!=ref))
        #ref[alldiffpos] = 0
        out1[diffpos] = ref
        #out1[torch.where(torch.isnan(out1) | torch.isinf(out1))] = 0
        return out1
        return (out1+out2+out3)/3


class ProtectedConv2d_TMR(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(ProtectedConv2d_TMR, self).__init__(*args, **kwargs)
        self.conv2 = nn.Conv2d(*args, **kwargs)
        self.conv3 = nn.Conv2d(*args, **kwargs)
        self.init_state = False

    def init_weight(self):
        self.conv2.weight.copy_(self.weight)
        self.conv3.weight.copy_(self.weight)
    
    def forward(self, input):
        global batch_id, layer_id
        if not self.init_state:
            logger.logger.info('init TMR weight')
            self.init_weight()
            self.init_state = True

        logger.logger.debug('run batch %d layer %d'%(batch_id, layer_id))
        out1 = super(ProtectedConv2d_TMR, self).forward(input)
        out2 = self.conv2.forward(input)
        out3 = self.conv3.forward(input)

        diffpos = torch.where(out1!=out2)
        
        if len(diffpos[0]):
            logger.logger.info('TMR find mismatch %d pos in %s output'%(len(diffpos[0]), str(out1.shape)))

        ref = out3[diffpos]
        #alldiffpos = torch.where((out1[diffpos]!=ref) & (out2[diffpos]!=ref))
        #ref[alldiffpos] = 0
        out1[diffpos] = ref
        #out1[torch.where(torch.isnan(out1) | torch.isinf(out1))] = 0

        logger.logger.debug('done batch %d layer %d'%(batch_id, layer_id))
        layer_id += 1

        return out1
        return (out1+out2+out3)/3

def tensorclose(input, other, rtol=1e-3, atol=1e-6):
    return torch.abs(input-other) < atol + rtol * torch.abs(other)

class ProtectedConv2d_ABED_Raw(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(ProtectedConv2d_ABED_Raw, self).__init__(*args, **kwargs)
        self.init_state = False

    def init_weight(self):
        self.sum_weight = torch.sum(self.weight, dim=0).reshape(1, *self.weight.shape[1:])
    
    def forward(self, input):
        if not self.init_state:
            self.init_weight()
            self.init_state = True

        input_sum  = torch.sum(input, dim=0).reshape(1, *input.shape[1:])

        out = super(ProtectedConv2d_ABED_Raw, self).forward(input)
        out_old = out
        
        where = torch.where(out!=out_old)
        out_sum_check = F.conv2d(input_sum, self.sum_weight, stride=self.stride, padding=self.padding)
        out_sum_real = torch.sum(out, dim=(0,1)).reshape(1,1,*out.shape[2:])

        close = tensorclose(out_sum_real, out_sum_check, threshold)

        if not close.all():
            where = torch.where(~close)
            print(out_sum_real[where], out_sum_check[where], where)
            print('Detected corruption')

        return out

class ProtectedConv2d_ABED_FI_activation(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(ProtectedConv2d_ABED_FI_activation, self).__init__(*args, **kwargs)
        self.init_state = False

    def init_weight(self):
        self.sum_weight = torch.sum(self.weight, dim=0).reshape(1, *self.weight.shape[1:])
    
    def forward(self, input):
        if not self.init_state:
            self.init_weight()
            self.init_state = True

        input_sum  = torch.sum(input, dim=0).reshape(1, *input.shape[1:])
        input = FI_input(input)

        out = super(ProtectedConv2d_ABED_FI_activation, self).forward(input)
        out_old = out
        
        where = torch.where(out!=out_old)
        out_sum_check = F.conv2d(input_sum, self.sum_weight, stride=self.stride, padding=self.padding)
        out_sum_real = torch.sum(out, dim=(0,1)).reshape(1,1,*out.shape[2:])

        close = tensorclose(out_sum_real, out_sum_check, threshold)

        if not close.all():
            where = torch.where(~close)
            #print(out_sum_real[where], out_sum_check[where])
            #print('Detected corruption')

        return out

class ProtectedConv2d_ABED_Recomp(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(ProtectedConv2d_ABED_Recomp, self).__init__(*args, **kwargs)
        self.conv2 = nn.Conv2d(*args, **kwargs)
        self.conv3 = nn.Conv2d(*args, **kwargs)
        self.init_state = False

    def init_weight(self):
        self.sum_weight = torch.sum(self.weight, dim=0).reshape(1, *self.weight.shape[1:])
        self.conv2.weight.copy_(self.weight)
        self.conv3.weight.copy_(self.weight)
    
    def forward(self, input):
        if not self.init_state:
            self.init_weight()
            self.init_state = True

        input_sum  = torch.sum(input, dim=0).reshape(1, *input.shape[1:])

        out1 = super(ProtectedConv2d_ABED_Recomp, self).forward(input)
        
        out_sum_check = F.conv2d(input_sum, self.sum_weight, stride=self.stride, padding=self.padding)
        out_sum_real = torch.sum(out1, dim=(0,1)).reshape(1,1,*out1.shape[2:])

        close = tensorclose(out_sum_real, out_sum_check, threshold)

        if not close.all():
            logger.logger.info('ABED find %d mismatch in checksum %s, output %s'%(torch.count_nonzero(~close), str(close.shape), str(out1.shape)))
            out2 = self.conv2.forward(input)
            out_sum_real = torch.sum(out2, dim=(0,1)).reshape(1,1,*out2.shape[2:])
            
            close = tensorclose(out_sum_real, out_sum_check, threshold)

            if not close.all():
                logger.logger.info('ABED 2nd find %d mismatch in checksum %s, output %s'%(torch.count_nonzero(~close), str(close.shape), str(out1.shape)))

                out3 = self.conv3.forward(input)
                diffpos = torch.where(out1!=out2)
                ref = out3[diffpos]
                
                out1[diffpos] = ref
                return out1                

            return out2
            
        return out1

class ProtectedConv2d_ABED_Recomp_FI_activation(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(ProtectedConv2d_ABED_Recomp_FI_activation, self).__init__(*args, **kwargs)
        self.conv2 = nn.Conv2d(*args, **kwargs)
        self.conv3 = nn.Conv2d(*args, **kwargs)
        self.init_state = False

    def init_weight(self):
        self.sum_weight = torch.sum(self.weight, dim=0).reshape(1, *self.weight.shape[1:])
        self.conv2.weight.copy_(self.weight)
        self.conv3.weight.copy_(self.weight)
    
    def forward(self, input):
        if not self.init_state:
            self.init_weight()
            self.init_state = True

        input_sum  = torch.sum(input, dim=0).reshape(1, *input.shape[1:])

        out1 = super(ProtectedConv2d_ABED_Recomp_FI_activation, self).forward(FI_input(input))
        
        out_sum_check = F.conv2d(input_sum, self.sum_weight, stride=self.stride, padding=self.padding)
        out_sum_real = torch.sum(out1, dim=(0,1)).reshape(1,1,*out1.shape[2:])

        close = tensorclose(out_sum_real, out_sum_check, threshold)

        if not close.all():
            logger.logger.info('ABED find %d mismatch in checksum %s, output %s'%(torch.count_nonzero(~close), str(close.shape), str(out1.shape)))
            out2 = self.conv2.forward(FI_input(input))
            out_sum_real = torch.sum(out2, dim=(0,1)).reshape(1,1,*out2.shape[2:])
            
            close = tensorclose(out_sum_real, out_sum_check, threshold)

            if not close.all():
                logger.logger.info('ABED 2nd find %d mismatch in checksum %s, output %s'%(torch.count_nonzero(~close), str(close.shape), str(out1.shape)))
                
                out3 = self.conv3.forward(FI_input(input))
                diffpos = torch.where(out1!=out2)
                ref = out3[diffpos]
                
                out1[diffpos] = ref
                return out1                

            return out2
            
        return out1

class ProtectedConv2d_ABED_Recomp_FI_weight(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(ProtectedConv2d_ABED_Recomp_FI_weight, self).__init__(*args, **kwargs)
        self.conv2 = nn.Conv2d(*args, **kwargs)
        self.conv3 = nn.Conv2d(*args, **kwargs)
        self.init_state = False

    def init_weight(self):
        self.sum_weight = torch.sum(self.weight, dim=0).reshape(1, *self.weight.shape[1:])
        self.conv2.weight.copy_(self.weight)
        self.conv3.weight.copy_(self.weight)
    
    def forward(self, input):
        if not self.init_state:
            self.init_weight()
            self.init_state = True

        input_sum  = torch.sum(input, dim=0).reshape(1, *input.shape[1:])

        with WeihgtFI(self):
            out1 = super(ProtectedConv2d_ABED_Recomp_FI_weight, self).forward(input)
        
        out_sum_check = F.conv2d(input_sum, self.sum_weight, stride=self.stride, padding=self.padding)
        out_sum_real = torch.sum(out1, dim=(0,1)).reshape(1,1,*out1.shape[2:])

        close = tensorclose(out_sum_real, out_sum_check)

        if not close.all():
            with WeihgtFI(self.conv2):
                out2 = self.conv2.forward(input)
            out_sum_real = torch.sum(out2, dim=(0,1)).reshape(1,1,*out2.shape[2:])
            
            close = tensorclose(out_sum_real, out_sum_check)

            if not close.all():
                
                with WeihgtFI(self.conv3):
                    out3 = self.conv3.forward(input)
                diffpos = torch.where(out1!=out2)
                ref = out3[diffpos]
                
                out1[diffpos] = ref
                return out1                

            return out2
            
        return out1

ProtectedConv2d = Conv2d_Raw

class ThreshReLU(nn.Module):
    def __init__(self, **kwargs):
        super(ThreshReLU, self).__init__()
        self.register_buffer('max_act', torch.Tensor([0]))
    
    def forward(self, input: Tensor) -> Tensor:
        # On validation set
        # self.max_act.copy_(torch.max(torch.max(input).to(device=self.max_act.device), self.max_act))

        # Testing
        input *= (input<=self.max_act.to(device = input.device)).float()
        
        input[torch.where(torch.isnan(input) | torch.isinf(input))] = 0
        
        return F.relu(input)
    
ActivationFunction = nn.ReLU

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> ProtectedConv2d:
    """3x3 convolution with padding"""
    return ProtectedConv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> ProtectedConv2d:
    """1x1 convolution"""
    return ProtectedConv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu1 = ActivationFunction(inplace=True)
        self.relu2 = ActivationFunction(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu2(out)

        return out

class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu1 = ActivationFunction(inplace=True)
        self.relu2 = ActivationFunction(inplace=True)
        self.relu3 = ActivationFunction(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)

        return out

class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = ProtectedConv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu1 = ActivationFunction(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    **kwargs: Any,
) -> ResNet:

    model = ResNet(block, layers, **kwargs)

    return model

def resnet18(**kwargs: Any) -> ResNet:
    return _resnet(BasicBlock, [2, 2, 2, 2], **kwargs)