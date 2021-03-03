# Gecko: Reconciling Privacy, Accuracy and Efficiency in Embedded Neural Networks

Summary: The paper is one of the first works which focusses on membership privacy by design where the goal is to design and train a model to inherently minimize membership privacy risk. We explore the design space of low capacity models and we find binarized models with XNOR operations to provide high privacy but at the cost of poor accuracy. We use knowledge distillation to train to boost the accuracy of the low precision model and find that it is indeed possible to have models with good accuracy, privacy while guaranteeing efficiency for low powered embedded devices.

## Experiments

The experiments for 

```bash
temp
```


## Credits

The code for binarization has been adapted from https://github.com/itayhubara/BinaryNet.pytorch and https://github.com/jiecaoyu/XNOR-Net-PyTorch.
