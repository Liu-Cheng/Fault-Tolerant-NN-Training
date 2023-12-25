import torchvision.models as models
import torch
import torch.jit
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.utils.benchmark as benchmark
import torch.utils.data
import torch.utils
import torch.backends.cudnn
import torch.quantization.quantize_fx as quantize_fx
import copy
import timeit
import time
import argparse

from torch.fx import symbolic_trace

Model=models.resnet18

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-n', '--number', default=10000, type=int)
arg_parser.add_argument('-r', '--realdata', action='store_true')
arg_parser.add_argument('--cpu', action='store_true')
opt = arg_parser.parse_args()

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
model=Model(weights="ResNet18_Weights.IMAGENET1K_V1")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if opt.cpu:
    device = 'cpu'
model.to(device)
model.eval()

if opt.realdata:
    normalize=transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

    tf=transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])

    testset=datasets.ImageFolder('~/dataset/val', tf)
else:
    class Dataset(torch.utils.data.Dataset):
        def __init__(self):
            self.fake_img = torch.randn((3,224,224), device=device)
        def __len__():
            return 50000
        def __getitem__(self, index):
            return self.fake_img, 0

    testset=Dataset()

batch_size = 128

testdata = torch.utils.data.Subset(testset, range(opt.number))
testloader = torch.utils.data.DataLoader(testdata, batch_size=batch_size, shuffle=False)
torch.set_num_threads(16)

# with torch.inference_mode():
#     jitm = torch.jit.trace(model, torch.zeros((128,3,224,224), device=device))
# model = torch.jit.freeze(jitm)
# print(model.code)

def run_test_epoch():
    acc, len_data = 0, 0
    t0 = time.time()
    with torch.inference_mode():
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)
            out = model(images)

            acc+=torch.count_nonzero(torch.argmax(out, dim=1)==labels)
            len_data += len(labels)
    
    print("acc: %.2f"%(acc/len_data*100), "(%d/%d)"%(acc, len_data) , "tt: %.2fs"%(time.time()-t0))

timer = benchmark.Timer('run_test_epoch()', 'from __main__ import run_test_epoch')
#run_test_epoch()

print(timer.timeit(2))
