# Problem Description
The Fashion MNIST database is a grayscale image dataset of 10 types of clothing, such as shoes, t-shirts, dresses, and more. The mapping of all 0-9 integers to class labels is listed below  
• 0: T-shirt/top  
• 1: Trouser  
• 2: Pullover  
• 3: Dress  
• 4: Coat  
• 5: Sandal  
• 6: Shirt  
• 7: Sneaker  
• 8: Bag  
• 9: Ankle boot  
## Note 
Since the Fashion MNIST dataset is a black and white image dataset, the shape of the dataset is (dataset_length, 28,28). But to fit data into a conv2d layer, we need to make the input shape comply with its required format: (batch_size, image_width, image_depth, image_channels). Although batch_size can be decided later when training it, we will still need to tell the number of image channels here. Therefore, we can reshape the dataset into (dataset_length, 28,28,1).  

Every Fashion MNIST data point has two parts: an image of a piece of clothing and a corresponding label. We will call the images 𝑥 and the labels 𝑦. Both the training set and test set contain 𝑥 and 𝑦. Each image is 28 pixels by 28 pixels.  

As mentioned, the corresponding labels in the Fashion MNIST are numbers between 0 and 9. In this assignment, we regard the labels as one-hot-vectors, i.e. 0 in most dimensions, and 1 in a single dimension. In this case, the 𝑛-th digit will be represented as a vector which is 1 in the 𝑛 dimensions. For example, 3 would be [0,0,0,1,0,0,0,0,0,0].  

The assignment aims to build NNs for classifying images in the Fashion MNIST database, train the models on the training set and test them on the test set. Since the main object of this assignment is to understand the relationship between input, model, and output, high accuracy in model performance is not expected, instead, for each task, we need to identify how can we improve model performance with the change of different network structure.  

# Tasks 1
Build a neural network without convolutional layers to do the classification task. Then, change the model structure (i.e., number of dense layers, number of neurons in dense layers, or activation functions), to be able to improve network performance.
## Task 1 answer:
Since the model has no convolutional layers but just dense layer, there was a need for reshaping the image dataset into 1 dimension for the dense layer because dense layer is only applied to the last axis. So, an image of 28 by 28 needs to be converted into a single 784-long feature. 
The first model has only 1 dense layer and so the number of nodes equal to the number of classes which is 10. The activation function used is Softmax which is appropriate for multi-classification task. The input dimension is 28*28 which matches what the image data has been flattened to.  
The model architecture is shown as below:  
![image](https://github.com/user-attachments/assets/78ad2af8-6b81-4444-a927-7155a228e9e0)
The dense layer is input shape of 28*28 = 784 and the output shape is 10 which is the number of classes. The total parameters is 28*28*10 + 10 bias params = 7850.  

In the second model, another dense layer is added with 20 neurons with relu activation function. The reason why relu function is used is because it’s a suitable function for intermediary step. The last dense layer is same as the first model.  
The model architecture is shown as below:  
![image](https://github.com/user-attachments/assets/f0efe1aa-0277-4785-972d-80e651d6b433)

The accuracy on the testing set for the first model was 78.74% and it increased slightly to 79.75% in the second model where an extra dense layer with 20 neurons were added. So adding an extra dense layer increased the accuracy by about 1%. The reason for this could be that the network is a little bit deeper, hence more learning when an extra dense layer is added.  
Both models ran relatively fast with 1 or 2 seconds on each epoch, there are 5 epochs for each model.  

# Task 2
Build a neural network with the use of convolutional layers. Then, change: the number of convolutional layers, the number of filters, or activation function functions in convolutional layers, to be able to improve network performance.
## Task 2 Answer:
The first model has a convolutional layer with kernel of size 3 by 3. The input shape is the size of the image where width is 28 and height is 28, feature is only 1 because the image is in black and white. The activation layer is relu which is suitable for intermediary step.  

A flatten layer is added to turn everything into a long vector before the dense layer since the dense layer which is fully connected only takes 1 dimension.  

The last layer is the dense layer with 10 neurons and softmax activation function for multi-classification task.  
The model architecture is shown as below:  
![image](https://github.com/user-attachments/assets/28fd4644-7367-4494-8454-50ae64043260)
The first convolution layer has output shape of (28, 28, 28) because the image size is 28 by 28 and 28 filters specified by the code. The flatten layer has a vector of 28 * 28 * 28 = 21952 as output shape. The final fully connected (dense layer) has output shape of 10 which is the number of classes. The total parameters is the sum of 280 from the convolution layer, 0 from the flatten layer and 219530 from the dense layer.  

In the second model, another convolution layer, a maxpooling, a dense layer and 50% dropout rate are added. The extra convolution layer is still the same as the original one. 256 neurons are assigned to the extra dense layer.   
The model architecture is shown as below: 
![image](https://github.com/user-attachments/assets/35f69991-c43e-4294-9611-fb72c39d80e5)
The total parameters is 1,415,118 which is a lot more compared to the original with 219,810 parameters.  
The accuracy on the testing set for the first model was 82.85% and it increased slightly to 84.24% in the second model, a 1.39% difference. Adding an extra convolution and dense layer introduces more depth into the model which could lead to better accuracy. The dropout of 50% could potentially address overfitting issues in the first model. Notice that in the first model, the accuracy of training dataset is 84.08% compared to 82.85% in the testing dataset, this means that there was some overfitting. Max pooling can also address overfitting as well. The accuracy of the training dataset in the 2nd model is  83.20% compared to 84.24% in the testing dataset, which means that overfitting has been addressed.  
Despite the 2nd model being slightly more accurate, the running time is about 3 times more than the 1st model.  

# Task 3
Change the type of optimizer or learning rate that was applied in the previous tasks, and see how these changes can influence model performance
## Task 3 answer:
The learning rate of the 2nd model in task 2 is 0.002 which is low. As an experiment, changed the learning rate to 0.5 and the accuracy on the testing dataset increase by 1.73% from 84.24% to 85.97%. This could be because learning rate at 0.002 is not optimum. A low learning rate decreases the loss function but at a shallow rate, where at the optimum learning rate, a quick drop in the loss function is observed (Jordan J, 2018). In this case, learning rate at 0.5 is more optimum than 0.002. The running time is also similar between the 2 models. With a larger learning rate, it was expected that the model converges faster but have lower accuracy (Brownlee, J, 2019). However, a higher accuracy was observed with the higher learning rate in this case.  
As the last experiment, Adam optimizer is chosen. However the model did not run well on learning rate of 0.5 so reduced the learning rate to 0.0005. The final accuracy on testing dataset arrived at 91.83% which is the highest among all models. However, the running time is twice as high as the model using SGD. This was a surprised because according to Gupta A, et al., 2021, Adam optimizer generally performs worse than SGD on image classification tasks.  

# Ranking of all methods:
![image](https://github.com/user-attachments/assets/79d8530b-6522-46c0-8917-1c53554125dc)
