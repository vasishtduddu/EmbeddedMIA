# Pruning + Quantization



## FashionMNIST



Original:

```
model accuracy for training and test- (0.97905, 0.8958)
membership inference accuracy is: 0.5450333333333335
```

After Pruning:

```
model accuracy for training and test- (0.42668333333333336, 0.4138)
membership inference accuracy is: 0.509425
```

After Retraining:

```
model accuracy for training and test- (0.9979, 0.8837)
membership inference accuracy is: 0.5657833333333333
```

Quantization

8bit

```
model accuracy for training and test- (0.9979666666666667, 0.8824)
membership inference accuracy is: 0.56545
```

4bit

```
model accuracy for training and test- (0.9868666666666667, 0.8765)
membership inference accuracy is: 0.556775
```

3bit

```
model accuracy for training and test- (0.9472333333333334, 0.8644)
membership inference accuracy is: 0.5424916666666666
```

2bit

```
model accuracy for training and test- (0.8854166666666666, 0.8349)
membership inference accuracy is: 0.526475
```



## Purchase100



Original:

```
model accuracy for training and test- (0.9867407489311514, 0.8641830713095184)
membership inference accuracy is: 0.595884509006184
```

Pruning:

```
model accuracy for training and test- (0.16550567595459237, 0.15970313978107123)
membership inference accuracy is: 0.5069659023500434
```

After Retraining:

```
model accuracy for training and test- (0.9951348960636887, 0.7925469615748457)
membership inference accuracy is: 0.620895495249081
```



8bit:

```
model accuracy for training and test- (0.9950888250036857, 0.7919613496103428)
membership inference accuracy is: 0.6205852889793717
```

4bit:

```
model accuracy for training and test- (0.961374023293528, 0.7756430469840984)
membership inference accuracy is: 0.5986268031921089
```

3bit:

```
model accuracy for training and test- (0.8812840925844022, 0.7369926573269067)
membership inference accuracy is: 0.5727563948316482
```

2bit: 

```
model accuracy for training and test- (0.6294965354562878, 0.5649578809856299)
membership inference accuracy is: 0.5338231478763951
```