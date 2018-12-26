# Fully Convolutional Networks for semantic segmentation

Keras implementation of 
[Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038).

# Usage
## Training
1 Create a new folder and put training and validation data.  
2 Write training config in "args.json".  
3 `python train.py`  
4 Start training.

## Prediction
1 Create a new folder and put test data.  
2 Write predict config in "args.json".  
3 `python predict.py`  
4 Start prediction and show result images.


# Results
Left : Input image  
Center : GT  
Right : Predicted image  
![prediction1](https://user-images.githubusercontent.com/35373553/50434073-4521f400-091f-11e9-885b-4fe26e994fc9.png)

![prediction2](https://user-images.githubusercontent.com/35373553/50443929-e6717000-0948-11e9-9127-6c36ac71aecc.png)

![prediction3](https://user-images.githubusercontent.com/35373553/50443984-28021b00-0949-11e9-960e-1024c55954dd.png)


# Accuracy
![trn_acc](https://user-images.githubusercontent.com/35373553/50443918-d0fc4600-0948-11e9-8146-355d67651e4b.png)

# Loss
![trn_loss](https://user-images.githubusercontent.com/35373553/50443921-d35ea000-0948-11e9-999a-a46ac75b06d5.png)

# model(FCN8s)
![model](https://user-images.githubusercontent.com/35373553/50434663-7ac7dc80-0921-11e9-8ece-64d19476a42a.png)