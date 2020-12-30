# Adversarial attack on Image Classification Model 

[My Kaggle Notebook](https://www.kaggle.com/andylow1704/image-multiclass-pytorch-with-adversarial-attack)

[Dataset](https://www.kaggle.com/puneet6060/intel-image-classification)

[Reference Notebook](https://www.kaggle.com/asollie/intel-image-multiclass-pytorch-94-test-acc)

[FGSM method](https://pytorch.org/tutorials/beginner/fgsm_tutorial.html)

[Article on Adversarial attack](https://colab.research.google.com/drive/19N9VWTukXTPUj9eukeie55XIu3HKR5TT)


# What is Adversarial Attack?

Adversarial attack refers to the deliberate modification of input images such that it is perceived differently by the AI model. In layman's terms, the attacker is generating optical illusions to confuse the AI.   

There are many types of adversarial attacks but in this project, I will be focusing on deliberate image misclassification. In the case of image misclassification, the attacker only wants the output classification to be inaccurate and he/she does not care what the output is. Repeated output misclassification will result in an inaccurate image classification model.

In this project, I will using the [Fast Gradient Sign Attack(FGSM)](https://pytorch.org/tutorials/beginner/fgsm_tutorial.html) as the form of adversarial attack.

Described by Goodfollow et.al in [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572), the FGSM method is designed to attack neural networks by leveraging the way they learn - gradients. A normal neural network model works by minimizing the loss by adjusting the weights based on the backpropogated gradients. How this attack works is that it uses the gradient of the loss w.r.t the input data, then adjusts the input data to maximize the loss, casuing the classification model to be inaccurate. 

![](https://github.com/andylow1704/Adversarial_Attack/blob/main/Images/panda.PNG)
[Source:](https://arxiv.org/abs/1412.6572)

The FGSM method adjusts the input images by changing the epsilon which is part of the sign. By changing the sign, it will result in a different output. As shown above, the output changes from "panda" to "gibbon" when the epsilon changes. 

FGSM code:
![](https://github.com/andylow1704/Adversarial_Attack/blob/main/Images/FGSM.png)



# Project
For this project, I will be training a model (using Pytorch) with modified images (using adversarial attack) to see if the new model will be accurate in predicting modified test images. I took the images dataset from [Kaggle](https://www.kaggle.com/puneet6060/intel-image-classification) and the first half of the image classification code is from [Kaggle notebook](https://www.kaggle.com/asollie/intel-image-multiclass-pytorch-94-test-acc). Instead of using 20 epochs like the original notebook, I have changed it to 10 epochs such that the run time will be faster. Like the original notebook, I have trained the Pytorch model using unmodified images.

# Train Result (Training with full set of unmodified images) 

Best validation accuracy: 0.924

![](https://github.com/andylow1704/Adversarial_Attack/blob/main/Images/Training%20History(Before).PNG)

# Test Result (Using full set of unmodified test images) 

Test accuracy: 0.92

![](https://github.com/andylow1704/Adversarial_Attack/blob/main/Images/Report(OG).PNG)


# Implementing Fast Gradient Sign Method (FGSM) on test images 

For the test set, I will be using a constant epsilon to generate a full set of adversarial test examples to test the accuracy of my model. I will be using 7 sets of different epsilons( 0,0.05,0.1,0.15,0.2,0.25,0.3) to generate 7 different results. 

![](https://github.com/andylow1704/Adversarial_Attack/blob/main/Images/accuracy%20vs%20epilson.PNG)

The fall in test accuracy(epsilon = 0) compared to the original test accuracy(0.92) was due to the introduction of torch.clamp in the FGSM method. 

From the graph above, we can see that the test accuracy is negatively correlated with the epsilon. 

# Varying the epsilon on test images

In the real world, the attacker might not keep the epsilon constant and constantly feed the model with only adversarial examples. Hence, for each input in my test loaders, there will be a 50% chance of the attacker using the input to generate adversarial examples. If the adversarial examples were to be generated, the an epsilon ranging from 0 to 0.3 will be randomly chosen to modify the input. The model will take in the generated adversarial example and try to predict its label (0-5).     

For the testing of the model, I will be using 2 different scenarios to simulate the different types of adversarial attacks. In both scenarios, the attacker will randomly assign the epsilon (0-0.3) in the FGSM method to the input.

Scenario 1 : Attacker has a 50% chance of modifying the input
Scenario 2 : Attacker has a 100% chance of modifying the input.

# Test result for Scenario 1 : 
![](https://github.com/andylow1704/Adversarial_Attack/blob/main/Images/Report(Before%2Cmix).PNG) 

# Test result for Scenario 2 :
![](https://github.com/andylow1704/Adversarial_Attack/blob/main/Images/Report(Before%2Cfull).PNG)

From the results, we can see that our original model is not accurate in predicting adversarial examples.

# Countering FGSM 

One way to counter against adversarial attack is to [proactively generate adversarial examples as part of the training procedure](https://colab.research.google.com/drive/19N9VWTukXTPUj9eukeie55XIu3HKR5TT#scrollTo=KiDYu9gOF_aU).

In order to allow the image classification model to recognize adversarial examples, we have to train the model with adversarial examples. For my new model, I will be randomly assigning adversarial examples as my training images. This method is exactly the same as scenario 1. This is to train the model to accurately predict both normal images and adversarial examples. 

Similar to the previous testing method, I will test the accuracy of my new model using the same scenarios. 

# Test result for Scenario 1 : 
![](https://github.com/andylow1704/Adversarial_Attack/blob/main/Images/Report(After%2Cmix).PNG)

# Test result for Scenario 2 :
![](https://github.com/andylow1704/Adversarial_Attack/blob/main/Images/Report(After%2Cfull).PNG)


# Conclusion 
![](https://github.com/andylow1704/Adversarial_Attack/blob/main/Images/Evaluation%20table.PNG)

From the result above, we can see that the new model is much more accurate in predicting adversarial examples. 

The accuracy of the model can be further improved. For each input in the train dataloaders, we can use the input to generate a number of randomly modified inputs. Afterwards, we can then add these new inputs to our original training samples to increase the number of training samples that we have. The new model will be much more accurate since it has more training samples to train on . However, such method will be time consuming since and will take up lots of memory space in the GPU. 

In addition, you can try to train the model with full set of adversarial examples instead of using a combination of adversarial and normal images like mine. The model will be accurate in predicting adversarial images but the drawback is that it will be poor in predicting normal images.  


# Adversarial attack on Object Detection 1 

[Daedalus attack](https://github.com/NeuralSec/Daedalus-attack)

[My notebook](https://colab.research.google.com/drive/1Hq_Z2vVJ5GB5t7CASSdh8_U5mY7BnpzH?usp=sharing)

The Daedalus attack is a type of adversarial attack which focuses on breaking the [Non-Maximum Suppression](https://towardsdatascience.com/non-maximum-suppression-nms-93ce178e177c), an algorithm that is used in YOLO. A full explaination of how the Daedalus attack can be found [here](https://arxiv.org/abs/1902.02067v3).

Using the [Daedalus attack](https://github.com/NeuralSec/Daedalus-attack), adversarial attack can be applied on object detection model such as YOLO. The model used for prediction is a YOLOv3 model with yolov5 weights. 

![](https://github.com/andylow1704/Adversarial_Attack/blob/main/Images/Combined%20image.png)

From the pictures above, we can see that both images look similar. However, after drawing the bounding boxes on both images to predict the classes, we can see that the adversarial image contains several classes which do not belong to the image. 

![](https://github.com/andylow1704/Adversarial_Attack/blob/main/Images/Combined%20image2.png)

However, this attack will only work on the YOLOv3 model with the specific weights as stated above. 

# Adversarial attack on Object Detection 2 

[Shapeshifter Attack](https://github.com/shangtse/robust-physical-attack)

[My notebook](https://colab.research.google.com/drive/16EgJxJMSn4iyJhE7kfT3pLfRuSMlfyQk?usp=sharing)

The main idea of this attack is to create an adversarial patch using an image of a stop sign. The output class of the advasarial patch can be selected at the start of the code. I have added an additional line of code at the end to paste the output advarsarial patch on the designated image. 

Using the code, I have generated an adversarial stop sign patch with person as my target class. I tested the output advarsarial patch on 2 popular object detection models - [YOLOv3]( https://github.com/ultralytics/yolov3) and [YOLOv5](https://github.com/ultralytics/yolov5). Interestingly, the YOLOv3 model did not return any predictions for the advarsarial patch image. 

![](https://github.com/andylow1704/Adversarial_Attack/blob/main/Images/Adversarial%20patches.png) 

Pasting the advarsarial patch on an image will result in the following predictions.
![](https://github.com/andylow1704/Adversarial_Attack/blob/main/Images/adversarial%20outcomes.png)

The advarsarial patch could be resized or pasted in the middle of the image to create a much distorted prediction outcome. 














