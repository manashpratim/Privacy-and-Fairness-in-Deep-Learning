# Privacy-and-Fairness-in-Deep-Learning
Adversarial Attacks, Defences, and Mitigating Biases
# Overview

•	Implemented white-box attacks like Projected Gradient Descent [1], Carlini-Wagner L2 [2] (hw4_part1.py) and black-box attacks like Shadow Model Attacks [3] (hw5_part1.py)

•	Defended the models from these attacks using Augmented and Madry Defenses [4] (hw4_part2.py)

•	Mitigated biases in deep learning models using GAN like adversarial training to achieve demographic parity, equality of opportunity and equality of odds [5] (hw5_part2.py) as well as debiasing of word embeddings [6] (hw5_part3.py)

# Experiments and Results

**The experiments of White-box Attacks and Defences are done using the MNIST dataset.**

## White-box Attacks

The goal of an attack algorithm is to create adversarial samples that are as close to the original samples as possible such that humans can not detect the difference but the model gives wrong predictions on the adversarial samples. 

The Projected Gradient Descent has two modes: Clip and Projection.

### Projected Gradient Descent (Clip)

<p align="center">
  <img width="460" height="300" src="https://github.com/manashpratim/Privacy-and-Fairness-in-Deep-Learning/blob/master/pgdclip1.PNG">
</p>

Plot showing the mean distortions of the perturbed samples over 30 steps of adversarial sample generation. In layman terms, it shows how different the adversarial samples are from the original samples at the end of each epoch while the attack algorithm is trained for 30 epochs.
Success rate of 1 means that the attack was successful 100% of the time.

Mean distortions L0, L1, L2 and L_infinity refers to the mean difference between the original and adversarial samples using different L norms.

<p align="center">
  <img width="460" height="300" src="https://github.com/manashpratim/Privacy-and-Fairness-in-Deep-Learning/blob/master/pgdclip2.PNG">
</p>

The left part shows the original samples, the middle part shows the adversarial samples and the right part shows the difference between the two images. 

In the first image, the true label is 1, the model predicted 1 for the original sample but 2 for the adversarial sample.

In the second image, the true label is 0, the model predicted 0 for the original sample but 2 for the adversarial sample.

We, humans, can clearly distinguish between the original and adversarial samples. Hence, the algorithm is not very good even though the success rate of the algorithm is 100%.

### Projected Gradient Descent (Projection)

<p align="center">
  <img width="460" height="300" src="https://github.com/manashpratim/Privacy-and-Fairness-in-Deep-Learning/blob/master/pgdproject1.PNG">
</p>

This algorithm has a success rate of 93.75%

<p align="center">
  <img width="460" height="300" src="https://github.com/manashpratim/Privacy-and-Fairness-in-Deep-Learning/blob/master/pgdproject2.PNG">
</p>

We, humans, can clearly distinguish between the original and adversarial samples. Hence, the algorithm is not very good even though the success rate of the algorithm is 93.75%. It is worser than PGD (Clip).

### Carlini-Wagner L2

<p align="center">
  <img width="460" height="300" src="https://github.com/manashpratim/Privacy-and-Fairness-in-Deep-Learning/blob/master/cwl21.PNG">
</p>

The success rate of this algorithm is 100%

<p align="center">
  <img width="460" height="300" src="https://github.com/manashpratim/Privacy-and-Fairness-in-Deep-Learning/blob/master/cwl22.PNG">
</p>

We, humans, can not clearly distinguish between the original and adversarial samples. This algorithm is better than the previous two.

<p align="center">
  <img width="300" height="200" src="https://github.com/manashpratim/Privacy-and-Fairness-in-Deep-Learning/blob/master/three_comparison.PNG">
</p>

Comparison of the success rate of the three algorithms.

## Defences

I have attacked the models using the Carlini-Wagner L2.

Before the model was trained with a defence, the success rate of the attack was 100%.

<p align="center">
  <img width="300" height="200" src="https://github.com/manashpratim/Privacy-and-Fairness-in-Deep-Learning/blob/master/afterdefence.PNG">
</p>

After defence, the success rate of the attacker has decreased for both the defences. The success rate against Madry's Defence (PreMadry) is lower than Augmented Defence. Hence, Madry's Defence is better than Augmented Defence.

## Black-box Attack

**The experiments of this section are done using the CIFAR10 dataset.**

### Membership Inference Attack

A membership inference is the scenario in which an adversary seeks to determine if a given instance was used for training a given model. The adversary does not have direct access to the model, they only have query access in that they can ask the model to make predictions on chosen instances and observe the outcome. Shadow Model Attack is a black-box membership inference attack.

### Shadow Model Attack

In the shadow model attack, the attacker uses the shadow dataset S  to create a predictor for the question "was x used to train this model?". The shadow dataset mostly comes from the same distribution as the training dataset.

The process has the following steps: 

1. Repeat a number of times the process:

    1. Split S into two disjoint subsets Sin and Sout and train a shadow model g using only Sin. We will use this model to generalize          output behaviour of models on training instances vs. non-training instances.
    
    2. Synthesize two datasets Ain and Aout. The features in Ain are the ground truth label y and the g-predicted class distribution for        each instance (x; y) belongs to Sin while Aout has the same but for each instance of Sout. The target class in these is an              indicator whether the given instances are from Sin (indicated by 1) or from Sout (indicated by 0).
    
       We now have (y; ^y; 1) ~ to Ain where ^y = g(x) for (x; y) belongs to Sin and (y; ^y; 0) ~ Aout where ^y = g(x)
       for (x; y) belongs Sout.
       
2. Combine all of the produced Ain and Aout sets into (y; ^y; b) ~ A.

3. Train an attack model m : (y; ^y) -> b using A to predict training set membership b. In this experiment, I have created a family of attack models where each attack model is specialized to instances of only class.

**The Architecture of the Baseline Training model (Model being Attacked)**

Input (32x32x3) -> Flatten -> Dense (128,ReLU) -> Dense (64,ReLU) -> Dense (32,ReLU) -> Output (10,Softmax)

**The Architecture of the Baseline Shadow model**

Input (32x32x3) -> Flatten -> Dense (128,ReLU) -> Dense (64,ReLU) -> Dense (32,ReLU) -> Output (10,Softmax)

**The Architecture of the Attack model**

Input (20) -> Dense (40,ReLU) -> Output (1,Sigmoid)

### Attack Success Rate vs Overfitting of the Model being Attacked

To measure overfitting of the model being attacked, I have defined “Overfit” which is the difference in training and test accuracy of the model being attacked.

Here, I have created 4 different training models:

1. Baseline

2. L2(0.01): L2 regularization of lambda = 0.01 in the Dense (32,ReLU) layer

3. Dropout(0.5): Dropout of 50% before the Output (10,Softmax) layer

4. L2&Dropout: L2 regularization of lambda = 0.1 in the Dense (32,ReLU) layer and dropout of 50% before the Output (10,Softmax) layer

<p align="center">
  <img width="400" height="150" src="https://github.com/manashpratim/Privacy-and-Fairness-in-Deep-Learning/blob/master/plot1.PNG">
</p>

<p align="center">
  <img width="400" height="200" src="https://github.com/manashpratim/Privacy-and-Fairness-in-Deep-Learning/blob/master/plot2.PNG">
</p>

<p align="center">
  <img width="400" height="200" src="https://github.com/manashpratim/Privacy-and-Fairness-in-Deep-Learning/blob/master/plot3.PNG">
</p>

From the plots, we can see that attack success decreases with decrease in overfitting.

### Can an attack be carried out if the Shadow Model does not have the exact architecture as the Model being attacked?

Here, I have created 4 different shadow models:

1. Baseline

2. MLP: Input (32x32x3) -> Flatten -> Dense (64,ReLU) -> Dense (32,ReLU) -> Dense (16,ReLU) -> Output (10,Softmax)

3. CONV1: Input (32x32x3) -> CONV2D(32,(3x3),ReLU) -> MAXPOOL2D (2x2) -> Flatten -> Dense (64,ReLU) -> Output (10,Softmax)

4. CONV2: Input (32x32x3) -> CONV2D(16,(3x3),ReLU) -> CONV2D(32,(3x3),ReLU) -> Flatten -> Dense (64,ReLU) -> Output (10,Softmax)

I have defined “Similarity” as a subjective notion of how similar a shadow model is to the model being attacked.

Similarity of the Shadow models to the model under attack are 1, 0.85, 0.5, 0.4 respectively.

<p align="center">
  <img width="500" height="200" src="https://github.com/manashpratim/Privacy-and-Fairness-in-Deep-Learning/blob/master/plot4.PNG">
</p>

From the plots, we can see that the architecture of the shadow model does not have to be exact as the model being attack.

### Can the Shadow Model Attack be applied to a White-box Scenerio?

I define "Target model" as the model that was trained on the Training data. I have used the baseline architectures for this experiment.

In this experiment, I have used shadow model attack on four different types of white-box scenarios:

1. When the Target model is also trained on the shadow data and used to predict S_in and S_out. S_in and S_out are splits from the shadow data.

2. When the Target model is used to predict S_in and S_out without being trained on the shadow data. S_in and S_out are splits from the shadow data.

3) When the Target model is now trained on the training data again and used to predict S_in and S_out. S_in and S_out are splits from the training data.

4) Similar to 3 but Target model is not trained again on the training data. It is used to predict S_in and S_out. S_in and S_out are splits from the training data.

## 

# References
[1] Christian Szegedy, Wojciech Zaremba, Ilya Sutskever, Joan Bruna, Dumitru Erhan, Ian J. Goodfellow, and Rob Fergus. Intriguing properties of neural networks. CoRR, abs/1312.6199, 2013.

[2] Nicholas Carlini and David Wagner. Towards evaluating the robustness of neural networks. In 2017 ieee symposium on security and privacy (sp), pages 39–57. IEEE, 2017.

[3] Reza Shokri, Marco Stronati, Congzheng Song, and Vitaly Shmatikov. Membership inference attacks against machine learning models. In 2017 IEEE Symposium on Security and Privacy (SP), pages 3–18. IEEE, 2017.

[4] Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu. Towards deep learning models resistant to adversarial attacks, 2017.

[5] Brian Hu Zhang, Blake Lemoine, and Margaret Mitchell. Mitigating unwanted biases with adversarial learning. CoRR, abs/1801.07593, 2018.

[6] Tolga Bolukbasi, Kai-Wei Chang, James Y Zou, Venkatesh Saligrama, and Adam T Kalai. Man is to computer programmer as woman is to homemaker? debiasing word embeddings. In Advances in neural information processing systems, pages 4349–4357, 2016.

**Note: This project is part of my Homeworks. Current CMU students please refrain from going through the codes.**


