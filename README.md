# Creating an Image Classification model to counter against adversarial attacks

[My Kaggle Notebook](https://www.kaggle.com/andylow1704/image-multiclass-pytorch-with-adversarial-attack)

List of references:

[Dataset](https://www.kaggle.com/puneet6060/intel-image-classification)

[Reference Notebook](https://www.kaggle.com/asollie/intel-image-multiclass-pytorch-94-test-acc)

[FGSM method](https://pytorch.org/tutorials/beginner/fgsm_tutorial.html)

[Article on Adversarial attack](https://colab.research.google.com/drive/19N9VWTukXTPUj9eukeie55XIu3HKR5TT)


# What is Adversarial Attack?







# Project
For this project, I will be training a model (using Pytorch) with modified images (using adversarial attack) to see if the new model will be accurate in predicting modified test images. I took the images dataset from [Kaggle](https://www.kaggle.com/puneet6060/intel-image-classification) and the top half of the image classification code is from [Kaggle notebook](https://www.kaggle.com/asollie/intel-image-multiclass-pytorch-94-test-acc). Instead of using 20 epochs like the original notebook, I have changed it to 10 epochs such that the run time will be faster. Like the original notebook, I have trained the Pytorch model using unmodified images.

# Train Result (Training with full set of unmodified images) 

Best validation accuracy: 0.924

![](https://github.com/andylow1704/Adversarial_Attack/blob/main/Images/Training%20History(Before).PNG)

# Test Result (Using full set of unmodified test images) 

Test accuracy: 0.92

![](https://github.com/andylow1704/Adversarial_Attack/blob/main/Images/Report(OG).PNG)


# Implementing Fast Gradient Sign Method (FGSM) on test images 

For the test images, I will vary the epsilon(0,0.05,0.1,0.15,0.2,0.25,0.3) for the FGSM. 

![](https://github.com/andylow1704/Adversarial_Attack/blob/main/Images/accuracy%20vs%20epilson.PNG)

The fall in test accuracy(epsilon = 0) compared to the previous test accuracy is because of the implemention of requires_grad of tensor which was important for the adversarial attack. 

From the graph above, we can see that epsilon is negatively correlated with the accuracy. 

# Varying the epsilon on test images

In the real world, the epsilons may not always be constant, hence I will test my current model using 2 different types of test datasets:
1) Mixture of original test images and varying epsilon images
2) Full set of varying epsilon images 

# Test result for case 1 : 
![](https://github.com/andylow1704/Adversarial_Attack/blob/main/Images/Report(Before%2Cmix).PNG) 

# Test result for case 2 :
![](https://github.com/andylow1704/Adversarial_Attack/blob/main/Images/Report(Before%2Cfull).PNG)

# Counter against FGSM 
.
.
.
.
.
.
Similar to the previous method, I will test my new model using a mixture of varying epsilon images and original test images(case 1 ) and a full set of varying epsilon images (case 2). 

# Test result for case 1 : 
![](https://github.com/andylow1704/Adversarial_Attack/blob/main/Images/Report(After%2Cmix).PNG)

# Test result for case 2 :
![](https://github.com/andylow1704/Adversarial_Attack/blob/main/Images/Report(After%2Cfull).PNG)



# Conclusion 
![](https://github.com/andylow1704/Adversarial_Attack/blob/main/Images/Evaluation%20table.PNG)

We can conclude that training our model with modified images will help to increase the model accuracy in detecting modified images.

The accuracy of the model can be further improved if we transform eac input image to generate a number of randomly modified images. In thsi way, there will be more training samples to train our model , hence improving its accuracy. However, such method will be time consuming since there is an increased number of training samples. 






