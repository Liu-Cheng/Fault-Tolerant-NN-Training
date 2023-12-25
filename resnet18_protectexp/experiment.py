from collections import OrderedDict
import time
time_entry = time.time()
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
import sys
import numpy as np
import os
import random
import argparse
import traceback
from utils import current_time, logger, setlogger

import resnet18_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled = True
#torch.backends.cudnn.benchmark = True
#torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()

# Datset Options
parser.add_argument("--data", type=str, default='~/dataset/val',
                    help="path to Dataset")

parser.add_argument("--all", action='store_true')
parser.add_argument("--policy", type=str, default='Conv2d_Raw', help="protect policy")
parser.add_argument("--threshold", type=float, default=1e-2, help="threshold argument")
parser.add_argument("--rate", type=float, default=1e-8, help="bit error rate when FI enabled")
parser.add_argument("--dump", action='store_true')
parser.add_argument("--repeat", type=int, default=1, help="repeat experiment times")
parser.add_argument("--datasize", type=str, default="512", help="128/512/1000/10000")
parser.add_argument("--genchecksum", action='store_true')
parser.add_argument("--genthresh", action='store_true') # when gensthresh, also modify code in model.ThreshReLU
parser.add_argument("--BER", action='store_true')
parser.add_argument("--act_thresh", action='store_true') # BER test use act thresh

opt = parser.parse_args()

if opt.all:
    assert opt.datasize == "512"
    assert opt.policy == 'Conv2d_Raw'

setlogger(opt.policy)
logger.logger.info('logging start, program start loading at %s'%(time.asctime(time.localtime(time_entry))))

protect_policys = {
    'Conv2d_Raw': resnet18_model.Conv2d_Raw,
    'Conv2d_Raw_FI_activation': resnet18_model.Conv2d_Raw_FI_activation,
    'ProtectedConv2d_ABED_Raw': resnet18_model.ProtectedConv2d_ABED_Raw,
    'ProtectedConv2d_TMR': resnet18_model.ProtectedConv2d_TMR,
    'ProtectedConv2d_ABED_Recomp': resnet18_model.ProtectedConv2d_ABED_Recomp,
    'ProtectedConv2d_TMR_FI_activation': resnet18_model.ProtectedConv2d_TMR_FI_activation,
    'ProtectedConv2d_ABED_Recomp_FI_activation': resnet18_model.ProtectedConv2d_ABED_Recomp_FI_activation,
}
resnet18_model.bit_error_rate = opt.rate
resnet18_model.threshold = opt.threshold
resnet18_model.ProtectedConv2d = protect_policys[opt.policy]
model=resnet18_model.resnet18() 

if opt.data != 'fake':
    normalize=transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

    tf=transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])

    testset=datasets.ImageFolder(opt.data, tf)

    #torchmodel = models.resnet18(weights="ResNet18_Weights.IMAGENET1K_V1")
    torchmodel = models.resnet18(pretrained = True)

    logger.logger.info('weight loaded')
    
    model.load_state_dict(torchmodel.state_dict(), strict = False)

    del torchmodel

else:
    class Dataset(torch.utils.data.Dataset):
        def __init__(self):
            self.fake_img = torch.randn((3,224,224), device=device)
        def __len__():
            return 50000
        def __getitem__(self, index):
            return self.fake_img, 0

    testset=Dataset()

model.to(device)
model.eval()
batch_size = 128

testdata = torch.utils.data.Subset(testset, range(10000))
testloader10000 = torch.utils.data.DataLoader(testdata, batch_size=batch_size, shuffle=False)
torch.set_num_threads(16)

# with torch.inference_mode():
#     jitm = torch.jit.trace(model, torch.zeros((128,3,224,224), device=device))
# model = torch.jit.freeze(jitm)
# print(model.code)

def tensordiff(input, other, rtol=1e-4, atol=1e-6):
    return torch.abs(input-other) > atol + rtol * torch.abs(other)

def check_input(checksum, images, device = 'cpu'):
    imagessum = torch.sum(images, axis = 3)
    diff = tensordiff(checksum, imagessum)
    count = torch.count_nonzero(diff)
    if count:
        logger.logger.warning(device + ' input checksum found %d different rows in %d rows'%(count, checksum.numel()))
    else:
        logger.logger.info(device + ' input checksum pass')

def run_test_epoch(testloader, repeat):
    acc, len_data, len_input = 0, 0, 0
    t0 = time.time()
    logger.logger.warning('start run')
    logger.logger.warning(str(opt))

    input_checksum = torch.load('input_checksum.pth')
    input_checksum_cuda = torch.load('input_checksum_cuda.pth')
    logger.logger.info('input checksum loaded')

    with torch.inference_mode():
        resnet18_model.batch_id = 0
        for images, labels in testloader:
            logger.logger.info('running batch %d'%resnet18_model.batch_id)
            check_input(input_checksum[len_input:len_input+len(labels)], images)
            images = images.to(device)
            check_input(input_checksum_cuda[len_input:len_input+len(labels)], images, 'gpu')
            labels = labels.to(device)

            for i in range(repeat):
                out = model(images)
                acc+=torch.count_nonzero(torch.argmax(out, dim=1)==labels)
                len_data += len(labels)

            len_input += len(labels)
            logger.logger.info("done batch %d, (%d/%d)"%(resnet18_model.batch_id, acc, len_data))
            resnet18_model.batch_id += 1
    
    logger.logger.warning("acc: %.2f"%(acc/len_data*100))
    logger.logger.warning("(%d/%d)"%(acc, len_data))
    logger.logger.warning("%.2fs"%(time.time()-t0))
    return acc, len_data

def run_test_epoch_dump(testloader):
    acc, len_data, len_input = 0, 0, 0
    resnet18_model.dump_featuremap=True
    resnet18_model.dump_dir = 'dump' + current_time()
    os.mkdir(resnet18_model.dump_dir)
    logger.logger.warning('start run')
    logger.logger.warning(str(opt))
    
    input_checksum = torch.load('input_checksum.pth')
    input_checksum_cuda = torch.load('input_checksum_cuda.pth')
    logger.logger.info('input checksum loaded')
    
    t0 = time.time()
    with torch.inference_mode():
        resnet18_model.batch_id = 0
        for images, labels in testloader:
            logger.logger.info('batch %d'%resnet18_model.batch_id)
            check_input(input_checksum[len_input:len_input+len(labels)], images)
            images = images.to(device)
            check_input(input_checksum_cuda[len_input:len_input+len(labels)], images, 'gpu')
            labels = labels.to(device)
            resnet18_model.layer_id = 0
            torch.save((images, labels), resnet18_model.dump_dir + '/'+'batch%d_input'%resnet18_model.batch_id+'.pth')
            out = model(images)
            torch.save(out, resnet18_model.dump_dir + '/'+'batch%d_out'%resnet18_model.batch_id+'.pth')

            acc+=torch.count_nonzero(torch.argmax(out, dim=1)==labels)
            len_data += len(labels)
            len_input += len(labels)
            logger.logger.info("done batch %d, (%d/%d)"%(resnet18_model.batch_id, acc, len_data))
            resnet18_model.batch_id += 1
    
    logger.logger.warning("acc: %.2f"%(acc/len_data*100))
    logger.logger.warning("%.2fs"%(time.time()-t0))
    return acc, len_data

def FI_experiment_BER(testloader = testloader10000):
    np.random.seed(0)
    torch.manual_seed(0)
    min_BER, max_BER = -9, -6
    n_datapoint = (max_BER-min_BER) * 5 + 1
    BERs = np.logspace(min_BER, max_BER, n_datapoint)
    accs = np.zeros(len(BERs))

    t0 = time.time()
    len_data=0
    model_local = model

    if opt.act_thresh:
        act_threshs = torch.load('resnet18_act_threshs.pt')

        resnet18_model.ActivationFunction = resnet18_model.ThreshReLU
        model_t = resnet18_model.resnet18().to(device).eval()
        model_t.load_state_dict(model_local.state_dict(), strict=False)
        model_t.load_state_dict(act_threshs, strict = False)
        model_local = model_t
    
    with torch.inference_mode():
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)
            for i, ber in enumerate(BERs):
                resnet18_model.bit_error_rate = ber
                #print('BER:', ber)
                out = model_local(images)
                accs[i]+=torch.count_nonzero(torch.argmax(out, dim=1)==labels)

            len_data += len(labels)
            sys.stdout.flush()

    print("%0.2fs"%(time.time()-t0))
    for i in range(len(BERs)):
        print("%0.4e"%BERs[i], end='\t')
        print("%0.2f%%"%(accs[i]/len_data*100))
    sys.stdout.flush()

def FI_experiment_BER_repeat(testloader = testloader10000, n_repeat = 16):
    np.random.seed(0)
    torch.manual_seed(0)
    min_BER, max_BER = -9, -7
    n_datapoint = (max_BER-min_BER) * 5 + 1
    BERs = np.logspace(min_BER, max_BER, n_datapoint)

    with torch.inference_mode():
        for i, ber in enumerate(BERs):
            acc = 0
            t0 = time.time()
            len_data = 0
            resnet18_model.bit_error_rate = ber

            for images, labels in testloader:
                images = images.to(device)
                labels = labels.to(device)
                for r in range(n_repeat):
                    out = model(images)
                    acc+=torch.count_nonzero(torch.argmax(out, dim=1)==labels)

                len_data += len(labels)

            print("%0.4e"%ber, end='\t')
            print("%0.2f%%"%(acc/len_data/n_repeat*100), end = '\t')
            print("%0.2fs"%(time.time()-t0))
            sys.stdout.flush()

def FI_experiment_BER_with_prepare(testloader = testloader10000):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    min_BER, max_BER = -8, -5
    n_datapoint = (max_BER-min_BER) * 5 + 1
    BERs = np.logspace(min_BER, max_BER, n_datapoint)
    accs = np.zeros(len(BERs))

    resnet18_model.algo_prepare = True

    with torch.inference_mode():
        for images, labels in testloader:
            images = images.to(device)
            out = model(images)
    print('prepare done')

    resnet18_model.algo_prepare = False
    t0 = time.time()
    len_data=0

    with torch.inference_mode():
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)
            for i, ber in enumerate(BERs):
                resnet18_model.bit_error_rate = ber
                #print('BER:', ber)
                out = model(images)
                out[torch.where(torch.isnan(out))] = 0
                accs[i]+=torch.count_nonzero(torch.argmax(out, dim=1)==labels)

            len_data += len(labels)
            sys.stdout.flush()

    print("%0.2fs"%(time.time()-t0))
    for i in range(len(BERs)):
        print("%0.4e"%BERs[i], end='\t')
        print("%0.2f%%"%(accs[i]/len_data*100))
    sys.stdout.flush()

def benchmark_time():
    timer = benchmark.Timer('run_test_epoch()', 'from __main__ import run_test_epoch')
    print(timer.timeit(2))

testloader128 = torch.utils.data.DataLoader(torch.utils.data.Subset(testset, range(128)), batch_size=batch_size, shuffle=False)
testloader512 = torch.utils.data.DataLoader(torch.utils.data.Subset(testset, range(512)), batch_size=batch_size, shuffle=False)
testloader1000 = torch.utils.data.DataLoader(torch.utils.data.Subset(testset, range(1000)), batch_size=batch_size, shuffle=False)

if opt.datasize == "128":
    testloader = testloader128
elif opt.datasize == "512":
    testloader = testloader512
elif opt.datasize == "1000":
    testloader = testloader1000
elif opt.datasize == "10000":
    testloader = testloader10000
else:
    assert False, "invalid datasize"

def generate_checksum():
    checksum = torch.zeros((1000,3,224))
    checksum_cuda = torch.zeros((1000,3,224), device=device)
    len_data = 0
    for images, labels in testloader:
        checksum[len_data:len_data+len(labels),:,:] = torch.sum(images, dim=3)
        checksum_cuda[len_data:len_data+len(labels),:,:] = torch.sum(images.cuda(), dim=3)
        len_data += len(labels)
    
    torch.save(checksum, 'input_checksum.pth')
    torch.save(checksum_cuda, 'input_checksum_cuda.pth')


def run_model(model, images_data, labels_data):
    acc, len_data, len_input = 0, 0, 0
    t0 = time.time()

    with torch.inference_mode():
        for i in (0,128,256,384):
            images = images_data[i:i+128]
            labels = labels_data[i:i+128]

            resnet18_model.layer_id = 0
            out = model(images)
            acc+=torch.count_nonzero(torch.argmax(out, dim=1)==labels)
            len_data += len(labels)

            logger.logger.info("done batch %d, (%d/%d)"%(resnet18_model.batch_id, acc, len_data))
            resnet18_model.batch_id += 1
    
    logger.logger.warning("acc: %.2f"%(acc/len_data*100))
    logger.logger.warning("%.2fs"%(time.time()-t0))


def run_all(testloader):
    logger.logger.warning('begin run all policy')
    logger.logger.warning(str(opt))

    model_raw = model
    resnet18_model.ProtectedConv2d = resnet18_model.ProtectedConv2d_TMR
    model_tmr = resnet18_model.resnet18()
    model_tmr.load_state_dict(model_raw.state_dict(), strict=False)
    model_tmr.eval().to(device)

    resnet18_model.ProtectedConv2d = resnet18_model.ProtectedConv2d_ABED_Recomp
    model_abed = resnet18_model.resnet18()
    model_abed.load_state_dict(model_raw.state_dict(), strict=False)
    model_abed.eval().to(device)
    
    resnet18_model.ProtectedConv2d = resnet18_model.Conv2d_Raw
    resnet18_model.ActivationFunction = resnet18_model.ThreshReLU
    model_act_thresh = resnet18_model.resnet18()
    model_act_thresh.load_state_dict(model_raw.state_dict(), strict=False)
    logger.logger.info('loading resnet18_act_threshs.pt')
    act_threshs = torch.load('resnet18_act_threshs.pt')
    model_act_thresh.load_state_dict(act_threshs, strict=False)
    model_act_thresh.eval().to(device)

    logger.logger.info('all model loaded. loading all images.')

    images_data = torch.zeros((512, 3, 224, 224), device=device)
    labels_data = torch.zeros((512), dtype=torch.int, device=device)
    
    input_checksum = torch.load('input_checksum.pth')
    input_checksum_cuda = torch.load('input_checksum_cuda.pth')
    logger.logger.info('input checksum loaded')

    len_input = 0

    for images, labels in testloader:
        logger.logger.info('load images %d:%d'%(len_input, len_input+len(labels)))
        check_input(input_checksum[len_input:len_input+len(labels)], images)
        images = images.to(device)
        check_input(input_checksum_cuda[len_input:len_input+len(labels)], images, 'gpu')
        labels = labels.to(device)
        images_data[len_input:len_input+len(labels)] = images
        labels_data[len_input:len_input+len(labels)] = labels
        len_input += len(images)
    
    logger.logger.info('input data loaded')

    for repeat in range(opt.repeat):
        logger.logger.warning('start repeat id %d'%repeat)
        
        logger.logger.info('run Conv2d_Raw')
        run_model(model_raw, images_data, labels_data)

        logger.logger.info('run ProtectedConv2d_TMR')
        run_model(model_tmr, images_data, labels_data)

        logger.logger.info('run ProtectedConv2d_ABED_Recomp thresh = 1e-2')
        resnet18_model.threshold = 1e-2
        run_model(model_abed, images_data, labels_data)

        logger.logger.info('run ProtectedConv2d_ABED_Recomp thresh = 1e-4')
        resnet18_model.threshold = 1e-4
        run_model(model_abed, images_data, labels_data)

        logger.logger.info('run Conv2d_Raw + ActThreshold')
        run_model(model_act_thresh, images_data, labels_data)

def genthresh():
    resnet18_model.ActivationFunction = resnet18_model.ThreshReLU
    model_thresh = resnet18_model.resnet18()
    resnet18_model.ActivationFunction = torch.nn.ReLU
    model_thresh.load_state_dict(model.state_dict(), strict=False)
    model_thresh.to(device)

    with torch.inference_mode():
        for images, labels in testloader1000:
            images = images.to(device)
            out = model_thresh(images)

    act_threshs = OrderedDict()
    for k, v in model_thresh.state_dict().items():
        if 'max_act' in k:
            act_threshs[k]=v
    print(act_threshs)
    torch.save(act_threshs, 'resnet18_act_threshs.pt')

if opt.BER:
    FI_experiment_BER(testloader)
    exit(0)

if opt.genchecksum:
    generate_checksum()
    exit(0)

if opt.genthresh:
    genthresh()
    exit(0)

if opt.all:
    try:
        run_all(testloader)
    except (Exception, KeyboardInterrupt) as e:
        logger.logger.fatal(traceback.format_exc())
    exit(0)

try:
    if opt.dump:
        run_test_epoch_dump(testloader)
    else:
        run_test_epoch(testloader, opt.repeat)
except (Exception, KeyboardInterrupt) as e:
    logger.logger.fatal(traceback.format_exc())

# benchmark_time()
