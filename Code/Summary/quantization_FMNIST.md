# Membership Privacy and Efficieny



### FashionMNIST

CNN Architecture:

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_3 (Conv2D)            (None, 26, 26, 32)        320       
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 24, 24, 64)        18496     
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 12, 12, 64)        0         
_________________________________________________________________
flatten_2 (Flatten)          (None, 9216)              0         
_________________________________________________________________
dense_3 (Dense)              (None, 128)               1179776   
_________________________________________________________________
dense_4 (Dense)              (None, 10)                1290      
=================================================================
```



Number of Parameters = 1,199,882(FP); 1,200,840 (Binarized); 1,626,824 (XNOR)



Accuracy:

Full Precision CNN: Train-100% Test-92.35%; Inference Accuracy: 57.46%; (dtype=float32)

Binarized CNN:  Train-88.68; Test: 86.9%; Inference Accuracy: 55.45%

XNOR Binarized CNN: Train-87.195% Test: 85.68% Inference Accuracy: 51.05%



MLP Architecture:

```
Layer (type)                 Output Shape              Param #   
=================================================================
dense_13 (Dense)             (None, 512)               401920    
_________________________________________________________________
dense_14 (Dense)             (None, 512)               262656    
_________________________________________________________________
dense_15 (Dense)             (None, 512)               262656    
_________________________________________________________________
dense_16 (Dense)             (None, 10)                5130      
=================================================================
```

Number of Parameters: 932,362 (FP); 937,000(Binarized); 937,000(XNOR)



Accuracy

FP MLP: Train: 99.34%; Test: 89.88%; Inference Accuracy: 54.86%

Binarized MLP: Train: 97.61%; Test: 89.60%; Inference Accuracy: 54.30%

XNOR MLP: Train: 92.67 Test: 86.68% Inference Accuracy: 51.74%



