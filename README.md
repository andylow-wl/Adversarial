# Adversarial_Attack
Training an image classification model to detect adversarial attacks 

My notebook:
https://www.kaggle.com/andylow1704/image-multiclass-pytorch-with-adversarial-attack

Image classification dataset:
https://www.kaggle.com/puneet6060/intel-image-classification

Image classification notebook:
https://www.kaggle.com/asollie/intel-image-multiclass-pytorch-94-test-acc

References for Adversarial attacks:
https://pytorch.org/tutorials/beginner/fgsm_tutorial.html
https://colab.research.google.com/drive/19N9VWTukXTPUj9eukeie55XIu3HKR5TT


# What is Adversarial Attack?







# Project
For this project, I will be training a model (using Pytorch) with modified images (using adversarial attack) to see if the new model will be accurate in predicting modified test images. I took the images dataset from [Kaggle](https://www.kaggle.com/puneet6060/intel-image-classification) and the top half of the image classification code is from [Kaggle notebook](https://www.kaggle.com/asollie/intel-image-multiclass-pytorch-94-test-acc). Instead of using 20 epochs like the original notebook, I have changed it to 10 epochs such that the run time will be faster. Like the original notebook, I have trained the Pytorch model using unmodified images.

# Train Result (Using unmodified images only) 

Best validation accuracy: 0.924

![](Images/accuracy%20.PNG)

# Test Result (Using full 
