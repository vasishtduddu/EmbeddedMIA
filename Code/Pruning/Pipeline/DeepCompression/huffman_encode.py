import argparse
import math

import numpy as np
import torch
from torch.nn import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F

import torch

from util import huffman_encode_model
from model import *
from inference_attack import *


#Huffman encode a quantized model
no_cuda=False
modelpth = "saves/model_after_weight_sharing.ptmodel"

use_cuda = not no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else 'cpu')

model = torch.load(modelpth)
huffman_encode_model(model)

output_train, output_test, train_label, test_label = classifier_performance(model, trainloader, testloader)
inference_accuracy=inference_via_confidence(output_train, output_test, train_label, test_label)
print("Maximum Accuracy:",inference_accuracy)
