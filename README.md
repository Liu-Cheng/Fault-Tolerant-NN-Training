## Introduction
Neural networks that typically output discrete classification results rather than continuous values are 
not sensitive to computing variations caused by either hardware errors or quantization. Hence, it makes 
it possible to investigate the inherent fault tolerance of neural networks to protect against soft errors
in the underlying computing fabrics, which can be essentially incorporated in model paramters through 
training or fine-tuning.

In this project, we mainly investigate two different approaches to enhance the fault tolerance capability
of neural networks with training. 

First, we observe that soft errors that are typically abstracted as bit-flip 
errors pose more negative influence to the overall model accuracy when the values of weights and activations are larger. This is somehow as expected because bit-flip errors can induce larger computing variations when the data value is larger. With this observation, we seek to depress the data range in neural network processing during quantization as long as the accuracy meets the design constraints. Particularly, we may further shrink the quantization region such that more data can be quantized with lower bound while retaining the model accuracy. This approach is implemented along with the model quantization and does not need to change the model architecture.

Second, instead of restricting the data range of neural network processing, we try to fix the computing errors. Basically, the model accuracy drop is mostly attributed to large faulty data values, so we add an additional output filters to ignore exceptional outputs. The threshold that determines whether a data is exceptional can be simply obtained with profiling or sampling. However, it may also filter out some correct data and leads to model accuracy loss, we incorporate the filter with Relu functions widely used in neural networks and learn the threshold along during training such that both fault tolerance and model accuracy can be guaranteed.


The fault-tolerant approaches are implemented on GPU and compared to other typical fault tolerant approaches including TMR and algorithm based fault tolerant approach. The initial results are also compared as follows.


<img title="Comparison of baseline, TMR, and ABFT. Resnet18 trained on ImageNet is utilized." src="result1.jpg" style="height: 346px; width:396px;"/>
<center> Comparison of baseline, TMR, and ABFT. Resnet18 trained on ImageNet is utilized. </center>


<img title="Comparison of baseline, basic quantization bound, and optimized quantization bound. Resnet18 trained on CIFAR10 is utilized." src="result2.jpg" style="height: 303px; width:396px;"/>
<center>Comparison of baseline, basic quantization bound, and optimized quantization bound. Resnet18 trained on CIFAR10 is utilized.</center>

## Usage
Here is the usage of the program.
All the different fault-tolerant approaches are integrated in file
`experiment.py`, and its location is `./resnet18_protectexp/experiment.py`.


With the following command, you can reset the path of the dataset. Its default dataset path is `~/dataset/val`.
```shell
python3 experiment.py --data '/path/to/dataset'
```

With the following command, you can select the protection approaches inclusing TMR and ABED. Of course, you may also just use the raw computing without any protection.

```shell
python3 experiment.py--policy Conv2d Raw --repeat
```

With the following command, you can invoke TMR protection.
```shell
python3 experiment .py--policy protectedConv2d TMR --repeat 5
```

With the following command, you can invoke ABFT protection.
```shell
python3 experiment .py--policy protectedConv2d ABED Recomp --repeat 5Kpython3 experiment .py--policy protectedConv2d ABED Recomp --threshold 1e-4 --repeat 5
```

With the following command, you can dump events of the errors captured by the fault tolerant approaches.
```shell 
python3 experiment.py--dump
```

More detailed commands such as ABFT threshold, datasize, and bit error rate can be found in the argument setup code in the `experiment.py` file.

```python
parser.add_argument("--data", type=str, default='', help="path to Dataset")
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
```

## License

Copyright Released under the [MIT License](https://opensource.org/licenses/MIT).
