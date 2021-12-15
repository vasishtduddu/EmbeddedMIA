# Towards Privacy Aware Deep Learning for Embedded Systems (ACM SAC'22, PPML@NeurIPS'20)

This is the code repository for the paper "Towards Privacy Aware Deep Learning for Embedded Systems".

## Experiments

All the code is in Jupyter notebooks for easy reproducibility.
Following are the folders and their contents:

- Quantization: Contains the privacy risk analysis for binarization and XNOR networks.
- StdArchitectures: Contains the privacy risk analysis for standard deep learning architectures designed for efficiency such as SqueezeNet and MobileNet.
- Pruning: Contains the privacy risk analysis of pruning the models followed by retraining. "Sparsity" folder provides an alternate implementation in a different ML library.
- Defences: Contains the code for blackbox defences such as adversarial regularization and differential privacy for comparison with Gecko models.
- Knowledge Distillation: Contains the code for homogeneous and heterogenous knowledge distillation of quantized models which forms the Gecko training methodology.

## Credits

The code for binarization has been adapted from https://github.com/itayhubara/BinaryNet.pytorch and https://github.com/jiecaoyu/XNOR-Net-PyTorch.
