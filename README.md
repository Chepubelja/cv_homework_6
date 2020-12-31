# Mini-project summary

|          |          |
|----------|----------|
|Start Date|2020-12-29|
|End Date  |2020-12-31|

## Description

### Background
For this mini-project, I have chosen CIFAR-10 dataset. The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. The classes are completely mutually exclusive. For example, there is no overlap between automobiles and trucks.
<br/><br/>
<p align="center"><img width="500" alt="cifar_10" src="https://user-images.githubusercontent.com/22610398/103415758-43cdd680-4b8c-11eb-8a8b-589c4d73a094.PNG"></p>

### Research questions
I have defined __2 main research questions__ that need to be tested:
- *__To check whether__ some more complex model (such as __ResNet18__) __can perform better than__ smaller models (such as __AlexNet__) on such small dataset.*
- *To test __how much we can improve plain AlexNet model__ by iteratively conducting experiments and building final model on top of the results of these experiments.*

### Plan of experiments
1. *Testing __few baseline models__ with different complexity.*
2. *Testing with / without __Early stopping__.*
3. *Testing with / without __Data Augmentation__.*
4. *Testing with / without __Batch Normalization__.*
5. *Comparison of different __learning rates__.*
6. *Comparison of different __batch sizes__.*
7. *Comparison of different __activation functions__.*
8. *Comparison of different __weights initialization methods__.*
9. *Testing with / without additional __input data normalization__.*
10. *Comparison of different __optimizers__.*
11. *Testing with / without __Learning rate scheduler__.*
12. *Testing with / without __Dropout__.*
13. *__Forming final model__, based on the results of all previous experiments.*

## Experiment journal

## Experiment №1 [e0000-e0002] Baseline models with different complexity

### Motivation

To test how well perform models with different complexities on CIFAR-10.

### Training config

- 30 epochs
- Batch size - 128
- Optimizer - Adam
- Learning rate - 0.0003
 
### Interpretation & Conclusion

| experiments |     diff                    | test acc. |
|-------------|-------------                |----------------|
|e0000        |  Baseline model __\*__      |     68.72  |
|__e0001__        |  __Vanilla AlexNet__            |     __71.21__  |
|e0002        |  Vanilla ResNet18           |     62.13  |

*__\*__ This baseline model consists of 3 conv blocks (conv + relu + maxpool) and 3 fully-connected layers at the end.*

<p align="center">
  <img width="600" src="https://user-images.githubusercontent.com/22610398/103421006-d2e5e900-4ba2-11eb-814d-7f98131f9d67.png">
</p>

### Observations:

- *Vanilla AlexNet outperforms other 2 models.*
- *Resnet18 is too complex model for this dataset, thus rapidly overfits on train and performs poorly on test. It possibly may be used for CIFAR, but needs more regularization (both implicit and explicit).*

*__Since AlexNet is the winner among 3 models, we would stick with this model for the further experiments__.*

<!--- ###################################################### --->

## Experiment №2 [e0001, e0003] With / without Early stopping

### Motivation

To test whether Early stopping can speed-up training and improve model's performance.

### Training config

- Same as in Experiment №1, but epochs - 35.

### Interpretation & Conclusion

| experiments |     diff                    | test acc. |
|-------------|-------------                |----------------|
|e0001    |      no early stopping          |  71.21     |
|__e0003__    |      __with early stopping__        |  __71.63__      |



<p float="left">
  <img width="469" src="https://user-images.githubusercontent.com/22610398/103419176-91514000-4b9a-11eb-9217-8bfcd546e3a2.png">
  <img width="470" src="https://user-images.githubusercontent.com/22610398/103419179-93b39a00-4b9a-11eb-983c-262800edffcb.png">
</p>

### Observations:

- *EarlyStopping doesn't affect much model's performance on so few epochs, need to be tested on more epochs for further conclusions.*

## Experiment №3 [e0001, e0004] With / without Data Augmentation

### Motivation

To test whether Data Augmentation can improve model's performance and introduce some implicit regularization by increasing amount of training data (increasing data variance).

### Training config

- Same as in Experiment №1.
- Data Augmentation:
  - Horizontal flip
  - Width shift 3 / 32
  - Height shift 3 / 32

### Interpretation & Conclusion

| experiments |     diff                    | test acc. |
|-------------|-------------                |----------------|
|e0001        |  no data augmentation            |  71.21     |
|__e0004__    |  __with data augmentation__            |  __78.46__     |

<p float="left">
  <img width="470" src="https://user-images.githubusercontent.com/22610398/103419828-5b618b00-4b9d-11eb-9d3e-d07cfad14eee.png">
  <img width="470" src="https://user-images.githubusercontent.com/22610398/103419841-61f00280-4b9d-11eb-88f5-d6d59127bc21.png">
</p>

### Observations:

- *__Data augmentation is probably the most important stage__, and carefully selected types of augmentation can improve performance a lot.*

## Experiment №4 [e0004-e0005] With / without Batch Normalization

### Motivation

To test whether Batch Normalization can stabilize training process and reduce amount of epochs needed for training a network.

### Training config

- Same as in Experiment №3.

### Interpretation & Conclusion

| experiments |     diff                    | test acc. |
|-------------|-------------                |----------------|
|  e0005      |  no batch normalization            | 76.71      |
|__e0004__    |  __with batch normalization__      | __78.46__      |

<p float="left">
  <img width="470" src="https://user-images.githubusercontent.com/22610398/103419906-b7c4aa80-4b9d-11eb-9a06-6af55d654521.png">
  <img width="470" src="https://user-images.githubusercontent.com/22610398/103419910-b98e6e00-4b9d-11eb-8165-caa9aa13fab9.png">
</p>

### Observations:
- *Batch normalization helps model to learn faster and thus by adding adding BN we got a little improvement in performance as well*.

## Experiment №5 [e0004, e0006-e0008] Different learning rates

### Motivation

To test what learning rate is the best for our set-up.

### Training config

- Same as in Experiment №3, but with different learning rates.

### Interpretation & Conclusion

| experiments |     diff                    | test acc. |
|-------------|-------------                |----------------|
|e0007        |  LR: 0.003            | 78.35      |
|__e0006__    |  __LR: 0.001__            | __79.58__      |
|e0004        |  LR: 0.0003            | 78.46      |
|e0008        |  LR: 0.0001            | 72.95      |

<p align="center">
  <img width="600" src="https://user-images.githubusercontent.com/22610398/103421143-8f3faf00-4ba3-11eb-9aef-b8a78bf8c7f5.png">
</p>

### Observations:
- *The default Adam learning rate works the best. Possibly, need to be further reduced with learning rate scheduler on later training epochs.*

## Experiment №6 [e0006, e0009-e0011] Different batch sizes

### Motivation

To test whether how much the correct choice of batch size affects model's performance.

### Training config

- Same as in Experiment №5, but with different batch sizes and lr - 0.001.

### Interpretation & Conclusion

| experiments |     diff                    | test acc. |
|-------------|-------------                |----------------|
|e0010        |  Batch size: 32            | 77.61      |
|e0009        |  Batch size: 64            | 77.33      |
|__e0006__    |  __Batch size: 128__            | __79.58__      |
|e0011        |  Batch size: 256            | 78.36      |

<p align="center">
  <img width="600" src="https://user-images.githubusercontent.com/22610398/103421204-cf069680-4ba3-11eb-9ce3-505d3306610f.png">
</p>

### Observations:
- *To my surprise, batch size can also significantly affect the model's performance (+2% from worst to best on test set).*

## Experiment №7 [e0006, e0012-e0014] Different activation functions

### Motivation

To compare and find out what activations works best for our set-up.

### Training config

- Same as in Experiment №6.

### Interpretation & Conclusion

| experiments |     diff                    | test acc. |
|-------------|-------------                |----------------|
|e0012        |  tanh            | 70.13      |
|__e0006__    |  __ReLU__            | __79.58__      |
|e0013        |  SeLU            | 78.43      |
|e0014        |  eLU            | 78.17      |

<p align="center">
  <img width="600" src="https://user-images.githubusercontent.com/22610398/103421262-04ab7f80-4ba4-11eb-8d0f-8be7e55bea7b.png">
</p>

### Observations:
- *The correct selection of activation function can also drastically change the final result (+9% from worst to best on test set).*

## Experiment №8 [e0006, e0015] Different weights initialization methods

### Motivation

To find out which weights initializer works best for our set-up.

### Training config

- Same as in Experiment №6.

### Interpretation & Conclusion

| experiments |     diff                    | test acc. |
|-------------|-------------                |----------------|
|e0015        |  He init            | 78.45      |
|__e0006__    |  __Xavier init__            |  __79.58__     |

<p float="left">
  <img width="470" src="https://user-images.githubusercontent.com/22610398/103420341-5998c700-4b9f-11eb-9837-120945816b12.png">
  <img width="470" src="https://user-images.githubusercontent.com/22610398/103420343-5b628a80-4b9f-11eb-953f-a2a91e1f4fcb.png">
</p>

### Observations:
- *There is no much difference between choosing weight initializer in our case, but maybe in other setups it could play bigger role.*

## Experiment №9 [e00019, e0020] With / without input data normalization + mean image subtraction

### Motivation

To check whether these additional pre-processing steps can speed-up training and model's convergence.

### Training config

- Same as in Experiment №6.

### Interpretation & Conclusion

| experiments |     diff                    | test acc. |
|-------------|-------------                |----------------|
|e0019        | No input data pre-processing             |  79.42     |
|__e0020__    | __With data normalization and mean image subtraction__  | __79.89__      |

<p float="left">
  <img width="470" src="https://user-images.githubusercontent.com/22610398/103420050-3588b600-4b9e-11eb-808d-ad0bfff1a6c8.png">
  <img width="470" src="https://user-images.githubusercontent.com/22610398/103420053-37527980-4b9e-11eb-9a63-65fd717a8952.png">
</p>

### Observations:
- *As could be seen from the plots, data normalization and mean image subtraction definely influence the model's training flow, it is much more stable and model is learning faster.*

## Experiment №10 [e0006, e0016-e0018] Different optimizers

### Motivation

To check how much optimizer affects the model's learning process.

### Training config

- Same as in Experiment №6, but with different optimizers.

### Interpretation & Conclusion

| experiments |     diff                    | test acc. |
|-------------|-------------                |----------------|
|e0017        |  SGD            | 54.21      |
|__e0006__    |  __Adam__            | __79.58__      |
|e0018        |  AdamW            | 11.23      |
|e0016        |  RMSProp            | 78.72      |

<p align="center">
  <img width="600" src="https://user-images.githubusercontent.com/22610398/103421307-50f6bf80-4ba4-11eb-9219-a869ca0d9957.png">
</p>

### Observations:
- *The correct selection of optimizer can also drastically change the final result. I believe Stohastic Gradient Descent didn't work, because it requires more hyperparameters tuning, than other 2 optimizers (Adam, RMSProp) which showed good results.*

## Experiment №11 [e0020-e0021] With / without Learning rate scheduler

### Motivation

To check if learning rate scheduler can improve performance by gradually reducing learning rate through the training flow.

### Training config

- Same as in Experiment №6.

### Interpretation & Conclusion

| experiments |     diff                    | test acc. |
|-------------|-------------                |----------------|
|e0020        |  no scheduler            | 81.74      |
|__e0021__    |  __with scheduler__            |  __79.58__     |

<p float="left">
  <img width="470" src="https://user-images.githubusercontent.com/22610398/103419993-07a37180-4b9e-11eb-87f0-8619d9aa7c5c.png">
  <img width="470" src="https://user-images.githubusercontent.com/22610398/103419995-096d3500-4b9e-11eb-83cb-9893b7ee2319.png">
</p>

### Observations:
- *Even on such small amount of epochs as 30, learning rate scheduler shows that decreasing learning rate over time is a must-have to improve the model's learning on the later stages.*

## Experiment №12 [e0021-e0022] With / without Dropout

### Motivation

To check if additional regularization would improve model's performance on unseen data (val, test).

### Training config

- Same as in Experiment №6.

### Interpretation & Conclusion

| experiments |     diff                    | test acc. |
|-------------|-------------                |----------------|
|e0021        |  no dropout            | 81.74      |
|__e0022__    |  __with dropout__            |  __82.92__     |

<p float="left">
  <img width="470" src="https://user-images.githubusercontent.com/22610398/103420105-72ed4380-4b9e-11eb-9cec-c881d7fc71be.png">
  <img width="470" src="https://user-images.githubusercontent.com/22610398/103420107-74b70700-4b9e-11eb-8313-634c766d5f96.png">
</p>

### Observations:
- *Since Dropout works as an regulizer for our model, the model is not so prone to overfit on the training data, and thus shows an improvement on the unseen (val and test sets).*

## Experiment №13 [e0023-e0025] Forming final model, based on the results of all previous experiments

### Motivation

To test how much we can improve our test accuracy against the results of the vanilla AlexNet model.

### Training config

#### Final model config 1

- 70 epochs
- Batch size - 128
- Optimizer - Adam
- Learning rate - 0.001
- Early stopping - 10 epochs no improvement
- LR Scheduler - drop 0.5 every 10 epochs

#### Final model config 2
- The same as Final model config 1, but 150 epochs and LR Scheduler - drop 0.5 every 20 epochs

#### Final model config 3
- The same as Final model config 2, but LR Scheduler - drop 0.5 every 15 epochs

### Interpretation & Conclusion

| experiments |     diff                    | test acc. |
|-------------|-------------                |----------------|
|e0023        |  Final model config 1                    | 83.02      |
|e0024    |      Final model config 2                    | 83.86      |
|__e0025__        |  __Final model config 3__            | __84.14__      |

<p align="center">
  <img width="600" src="https://user-images.githubusercontent.com/22610398/103421346-8c918980-4ba4-11eb-9eb5-b38fa9bcf97a.png">
</p>

### Observations:
- *Selecting the right number of epochs to drop learning rate and drop rate are also important while improving model's performance.*

## Summary

During conducting all these experiments, __I have found out few things__:
- *I have found __what parts of model tweaking and data processing influence the model's performance__ on the unseen data and to the which extent.*
- *I have __successfuly improved__ the vanilla AlexNet __from 71.21% to 84.14% on test set (+ 13%)__ by iteratevely building final model taking into an account the results of the previous experiments.*
- *__I have learned something new for myself__ during all these experiments.*
