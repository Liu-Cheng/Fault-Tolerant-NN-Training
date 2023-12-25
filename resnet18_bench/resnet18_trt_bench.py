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
from torch2trt import torch2trt
import timeit

from torch.fx import symbolic_trace

Model=models.resnet18

realdata = False

model=Model(pretrained=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

if realdata:
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

testdata = torch.utils.data.Subset(testset, range(10000))
testloader = torch.utils.data.DataLoader(testdata, batch_size=batch_size, shuffle=False)
torch.set_num_threads(16)

print('done preload')

with torch.inference_mode():
    model_trt = torch2trt(model, [torch.zeros((batch_size, 3, 224, 224), dtype=torch.float, device=device)])

print('done transfer')

def run_test_epoch():
    acc, len_data = 0, 0

    for images, labels in testloader:
        images = images.to(device)
        labels = labels.to(device)
        out = model_trt(images)

        acc+=torch.count_nonzero(torch.argmax(out, dim=1)==labels)
        len_data += len(labels)
    
    print("acc: %.2f"%(acc/len_data*100), "(%d/%d)"%(acc, len_data))

timer = benchmark.Timer('run_test_epoch()', 'from __main__ import run_test_epoch')

print(timer.timeit(2))
