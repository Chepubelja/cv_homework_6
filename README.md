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

[IMAGE]

### Observations:

- Vanilla AlexNet outperforms other 2 models.
- Resnet18 is too complex model for this dataset, thus rapidly overfits on train and performs poorly on test. It possibly may be used for CIFAR, but needs more regularization (both implicit and explicit).

*__Since AlexNet is the winner among 3 models, we would stick with this model for the further experiments__.*

<!--- ###################################################### --->

## Experiment №2 [e0001, e0003] With / without Early stopping

<p float="left">
  <img width="470" src="https://user-images.githubusercontent.com/22610398/103419176-91514000-4b9a-11eb-9217-8bfcd546e3a2.png">
  <img width="470" src="https://user-images.githubusercontent.com/22610398/103419179-93b39a00-4b9a-11eb-983c-262800edffcb.png">
</p>

## Experiment №3 [e0001, e0004] With / without Data Augmentation

<p float="left">
  <img width="470" src="https://user-images.githubusercontent.com/22610398/103419828-5b618b00-4b9d-11eb-9d3e-d07cfad14eee.png">
  <img width="470" src="https://user-images.githubusercontent.com/22610398/103419841-61f00280-4b9d-11eb-88f5-d6d59127bc21.png">
</p>

## Experiment №4 [e0004-e0005] With / without Batch Normalization

<p float="left">
  <img width="470" src="https://user-images.githubusercontent.com/22610398/103419906-b7c4aa80-4b9d-11eb-9a06-6af55d654521.png">
  <img width="470" src="https://user-images.githubusercontent.com/22610398/103419910-b98e6e00-4b9d-11eb-8165-caa9aa13fab9.png">
</p>

## Experiment №5 [e0004, e0006-e0008] Different learning rates

## Experiment №6 [e0006, e0009-e0011] Different batch sizes

## Experiment №7 [e0006, e0012-e0014] Different activation functions

## Experiment №8 [e0006, e0015] Different weights initialization methods

<p float="left">
  <img width="470" src="https://user-images.githubusercontent.com/22610398/103420341-5998c700-4b9f-11eb-9837-120945816b12.png">
  <img width="470" src="https://user-images.githubusercontent.com/22610398/103420343-5b628a80-4b9f-11eb-953f-a2a91e1f4fcb.png">
</p>

## Experiment №9 [e00019, e0020] With / without input data normalization

<p float="left">
  <img width="470" src="https://user-images.githubusercontent.com/22610398/103420050-3588b600-4b9e-11eb-808d-ad0bfff1a6c8.png">
  <img width="470" src="https://user-images.githubusercontent.com/22610398/103420053-37527980-4b9e-11eb-9a63-65fd717a8952.png">
</p>

## Experiment №10 [e0006, e0016-e0018] Different optimizers

## Experiment №11 [e0020-e0021] With / without Learning rate scheduler

<p float="left">
  <img width="470" src="https://user-images.githubusercontent.com/22610398/103419993-07a37180-4b9e-11eb-87f0-8619d9aa7c5c.png">
  <img width="470" src="https://user-images.githubusercontent.com/22610398/103419995-096d3500-4b9e-11eb-83cb-9893b7ee2319.png">
</p>

## Experiment №12 [e0021-e0022] With / without Dropout

<p float="left">
  <img width="470" src="https://user-images.githubusercontent.com/22610398/103420105-72ed4380-4b9e-11eb-9cec-c881d7fc71be.png">
  <img width="470" src="https://user-images.githubusercontent.com/22610398/103420107-74b70700-4b9e-11eb-8313-634c766d5f96.png">
</p>

## Experiment №13 [e0023-e0025] Forming final model, based on the results of all previous experiments

## Summary
