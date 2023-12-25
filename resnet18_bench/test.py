import torch 
import torchvision
from torch2trt import torch2trt

batch_size = 1
## export resnet50  input: batchsize 3 224 224
net = torchvision.models.resnet50(pretrained=True).cuda()

input_data = torch.rand((batch_size, 3, 224, 224), dtype=torch.float).cuda()

## convert to TensorRT int8 model
model_trt_int8 = torch2trt(net.eval(), [input_data], max_batch_size=batch_size)

out_trt = model_trt_int8(input_data)