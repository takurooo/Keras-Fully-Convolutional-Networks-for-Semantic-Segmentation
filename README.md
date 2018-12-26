# Fully Convolutional Networks for semantic segmentation

Keras implementation of 
[Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038).

# Usage
## Training
1 Write training config in "args.json".  
2 `python train.py`  
3 Start training.

## Prediction
1 Write predict config in "args.json".  
2 `python predict.py`  
3 Start prediction and show result images.


# Results
Left : Input image  
Center : GT  
Right : Predicted image  
![prediction](https://user-images.githubusercontent.com/35373553/50434073-4521f400-091f-11e9-885b-4fe26e994fc9.png)

# Accuracy
![trn_acc](https://user-images.githubusercontent.com/35373553/50434096-5f5bd200-091f-11e9-9e1c-7f9530e10a88.png)

# Loss
![trn_loss](https://user-images.githubusercontent.com/35373553/50434098-608cff00-091f-11e9-99aa-0720d720c9bb.png)

# model(FCN8s)
![model](https://user-images.githubusercontent.com/35373553/50434663-7ac7dc80-0921-11e9-8ece-64d19476a42a.png)