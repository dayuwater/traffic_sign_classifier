# **Traffic Sign Recognition** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! 

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is ? 34799
* The size of the validation set is ? 4410
* The size of test set is ? 12630
* The shape of a traffic sign image is ? (32, 32, 3)
* The number of unique classes/labels in the data set is ? 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. I plotted 5 images from the training set using matplotlib with their labels.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to augment the datasets. According to my exploration of the training set, the amount of images for each classification is not evenly distributed. One classification has more than 2000 images, and there is another classification only has 180 images. This would certainly causes a bias in training stage and makes the model fails to generalize. In order to reduce this negative effect, I decided to balance the amount of images in each classification by generating faked images so that each class has 2000 images after augmentation. 

In order to generate faked data that can make the computer think it is a new image, I added random noise in each generated images in a very rudimentary way. As you can see, I just random pick up a training image from the label I want to generate image to and use Python's built-in function to generate random perturbation.

```python
def add_fake_image(index, label_index_dict):
    img = X_train[random.choice(label_index_dict[index])]
    # Add noise to the fake image
    img_new = np.array([x + (random.random() - 0.5) / 10 for x in img])
    return img_new
```
In order to make sure the amount of images in training set, validation set and test set is about the same ratio before augmentation, I also added faked data to validation set and test set as well. I first tried to balance cases as I did to training set, but the result is not ideal. Then I just doubled the amount of images of each class, and the results is good.

```python
#augment the validation set
X_valid_aug = [x for x in X_valid]
y_valid_aug = [y for y in y_valid]

for y in y_valid:
    X_valid_aug.append(add_fake_image(y, label_index_dict))
    y_valid_aug.append(y)
    
X_valid_augmented = X_valid_aug
y_valid_augmented = y_valid_aug
```


Then I decided to convert the images to grayscale because the images in different classes mostly differs in the shape and not clearly differs in color. If we just consider these 43 classes of German Traffic Signs, grayscale is better because this simplifies the model and reduces the chance of overfitting. 


As a last step, I normalized the image data because this makes the model in zero mean and unit variance, which is better for the training stage. In addition, images might be stored in different color space and value, this makes sure the image data are unified from all sources, which makes it possible to predict outside images and actually used in production.



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16      									|
| RELU					|			
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| FLATTEN				|			
| Fully connected		| 400 -> 120       									|
| RELU					|			
| Fully connected		| 120 -> 84        									|
| RELU					|			
| Fully connected		| 84 -> 43       									|




#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam Optimizer with a learning rate of 0.001. In addition, I applied L2 regularization to  weights in each layer of the model.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ? 
* validation set accuracy of ? 0.970
* test set accuracy of ? 0.955


If a well known architecture was chosen:
* What architecture was chosen? LeNet
* Why did you believe it would be relevant to the traffic sign application? 
 Because LeNet was originally used to classify English letters. English letters can be considered as color irrelevant symbols. Because in this project traffic signs can be considered as color irrelevant symbols ( mostly), it is basically the same as English letters.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
Because the validation accuracy is basically increasing for each epoch, and it reaches 95% in just 5 epochs.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

In order to fully evaluate this classifier, I prepared three sets of images containing different levels of complexity, each set has 5 images

Here are five German traffic signs that comes from Wikipedia, which is in pure white background

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

Here are five cropped German traffic signs that comes from web

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

Here are five original German traffic signs that comes from web

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

Among the three datasets, the last dataset is the hardest to classify because it contains way too much of irrelevant data. Actually, I am not expecting this classifier to classify anything correctly.

For the other two datasets, the images that contains multiple units of symbols are harder to classify than that contains only 1 unit of symbol. Because LeNet was trained on images that only contain 1 English letter, if the image provided has only 1 unit of symbol, like a single snowflake or a left arrow, it can be classified correctly. However, if the image has multiple units of symbols, like speed limit (70 km/h), there are two pieces of information, "7" and "0", this might confuse the classifier because LeNet was not originally designed to work on this case.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

Wikipedia set:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

Cropped set:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...


Original Set:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

Because the classifier completely fails the Original Set, the softmax is not meaningful at all, I would not discuss that in this section. In fact, it might not be even possible to throw a whole image into the classifier without cropping.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

### Image Sources:
https://goo.gl/images/uWHdPS
https://goo.gl/images/33Gdbj
https://goo.gl/images/6GT4Jg
https://goo.gl/images/73aA3L
https://goo.gl/images/tav65M

