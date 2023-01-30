# Quantization Aware Training
Model : mobilenetv2    
Data : Cifar10    
    
activation is using Histogram   
weight is using perChannel MinMax    
training wtih fake quantization adn float32
## Training model    
```
$ bash try.sh
```
or    
```
$ python3 train.py --result --torchQAT
```
train.py option
```
QAT training test

optional arguments:
  -h, --help           show this help message and exit
  --model MODEL        select model. Default mobilenet
  --tiny               select tiny model.
  --weights WEIGHTS    load weights file name.
  --pretrain           using cifar10 pretrain weights.
  --device DEVICE      select device number
  --lr LR              initial lr. Default 1e-3.
  --seed SEED          select random seed. Default 42.
  --epoch EPOCH        train epoch. Default 200.
  --act                using layer.Quant_ReLU6 with activation output quant.
  --torchQAT           using pytorch Quantization Aware Training.
  --input-fake         fake quantization for input.
  --weight-fake        fake quantization for weight.
  --fake               fake quantization both input and weight.
  --savename SAVENAME  Name to store the model. Default test.pt.
  --result             export log csv.
```   

## Convert Model
```
python3 convert_test.py --weight models/weights/mobilenetv2_cifar10.pt
```
convert_test.py option
```
QAT training test

optional arguments:
  -h, --help           show this help message and exit
  --model MODEL        select model. Default mobilenet
  --random RANDOM      select random seed. Default 42
  --weight WEIGHT      load weights file name
  --tiny               select tiny model. Default False
  --only               test only one post quant
  --path PATH          convert test file path
  --savename SAVENAME  save scv file and the file name
  --loop LOOP          try loop time convert
```