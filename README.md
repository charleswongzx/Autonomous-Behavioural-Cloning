#**Behavioral Cloning for Self-Driving Cars**

[//]: # (Image References)

[image1]: ./readme_images/architecture.png "Modified NVIDIA Architecture"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./readme_images/3views.png "3 Camera Views"
[image4]: ./readme_images/mirror.png "Mirrored Images"
[image5]: ./readme_images/cropped.png "Cropped Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

This project was a submission for Udacity's Self-driving Car Nanodegree Project 3. The aim was to build and train a convolutional neural network (CNN) capable of steering a car around a simulator. This involved learning from prior driving behaviour, as the title suggests. Built on Python and Keras, and implemented NVIDIA's end-to-end learning model. Trained on a NVIDIA 860M, HP Omen 15.

<iframe width="560" height="315" src="https://www.youtube.com/embed/DLJC36mshL4" frameborder="0" allowfullscreen></iframe>

## Model Architecture
The model used is an implementation of NVIDIA's end-to-end learning model (https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars). Designed in 2016, the model has proven itself in real-world scenarios, and proved more than sufficient for this application.

![alt text][image1]

Modified from NVIDIA's dev blog, link above.

Every convolutional layer is activated with a RELU function to introduce non-linearity, while dropouts of 0.5 are introduced after every fully connected layer except the final layer to prevent overfitting.

The error was finally calculated with a means-squared error function, and learning parameters decided by the AdamOptimizer.

## Dataset Preparation

[gif of driving about]

### Control
Data was collected by recording my driving on Udacity's provided Unity simulator. Steering and throttle were controlled with an Xbox 360 controller for finer, more precise input. The result can only be as good as the data, and it was imperative that my recorded driving be up to scratch.
Effort was made to avoid janky steering input. I attempted to steer as smoothly as possible, to allow for more gradual steering angle distribution.

![alt text][image3]
### Left, Right and Center
The simulator recorded 3 camera views - one from the bonnet and one from each wing mirror. The view from each wing mirror can be interpreted as another vehicle translated to the left or right. As such, we can use those images as additional sets of data, effectively tripling our dataset. An angular offset of 3 degrees was added to each view to compensate for the shift.

![alt text][image4]
### Mirroring
In order to help the model generalise better, all images were flipped horizontally, and their steering angles reversed. This provides a counter to each run of the track, alleviating any concerns of left-right bias in steering input.

### Recovery Driving
The vehicle needed help learning to recover from off-center situations. To assist with that, I recorded specific sections of myself recovering from the road shoulder and steering back to the center of the track.

### Extra Training
After 4 laps around the track (clockwise and counter-clockwise), I re-attempted all major corners, particularly the first bend and the two last ones. This is necessary as the vehicle spends a lot of time travelling in a straight line, and requires more data of corners being taken at speed.

### The More The Merrier
After collecting something on the order of 20,000 images, I decided to further add to my dataset by loading in Udacity's own. This nearly doubled my available data, a major plus point being that my additional data was proven to be reliable.

![alt text][image5]
cropped and normalised

### Cropping and Normalisation
As part of the network, images are cropped top and bottom. These portions contain relatively useless features like the vehicle's bonnet and sky. Images were then normalised using lambda to bring their values between -0.5 and 0.5.

## Strategy and Design
Overall, my strategy when building this model was to start simple. I began with a simple network consisting of only one fully connected layer, and with only one lap of driving data. To my surprise, the car made it all the way to the bridge on track 1. I put it down to sheer dumb luck, and decided I was heading in the right direction.

I quickly decided on implementing NVIDIA's end-to-end deep learning network for cars, which was fairly trivial to build in Keras. I continued to switch between data collection, training and testing, focusing on areas that were giving the vehicle trouble (e.g., sharp bends and such).

As my dataset grew, CUDA began throwing memory errors, and I employed a modified generator to serve up batches of data, circumventing the issue of having to load tens of thousands of images into RAM.

The car was doing better and better! One thing that continued to irk me was the car's throttle and braking motion. The car's movements were constantly oscillating about the target speed, and doing so very poorly. This was due to the PI controller used to modulate the vehicle's speed. 

After a little research, I settled on the design laid out in ivmech's implementation here (https://github.com/ivmech/ivPID). With some parameter tuning, velocity was constant! It was off to the races.

Adding images from all 3 cameras into the dataset was fairly trivial - unfortunately tuning the angular offset both sides was not! Too small a value and the vehicle tended to understeer off the track. Too high, and the vehicle weaved from side to side. Not to mention the long iteration time thanks to the model being trained! 

After much testing, I found that doing without the 3 camera views yielded a much better result, and decided to do away with them entirely.

I eventually settled on a value of 0.04, decided upon empirically after much testing.

In order to further bolster my dataset, I elected to add Udacity's provided data to mine, and load in all 3 camera images as outlined above. The angular offset for left-right cameras required some trial and error, but soon enough the car was masterfully carving its way around the track.

## Conclusion
