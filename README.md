# Automatic Colorization of Grayscale Fruit Images

An automatic colorization of grayscale fruit images of 30 types of different fruits, with Convolutional Neural Network (CNN) model.

Click [here](##-Colorizable-Fruits-List) for the list of colorizable fruits list.

## Problem Statement and Approach

### Problem:

Given a collection of grayscale images, many would deem that coloring them is a very challenging and implausible task for computers because not all dimensions with the color of the image are provided. However, as we delve more deeply, it is noted that local features and global configuration of the images would provide enough evidence for the colorization of the images in most cases. For instance, oceans are mostly dark blue, leaves on trees are typically green, and the sky is generally blue.

**We aim to develop a model that can colorize grayscale fruit images in a way that is plausible and understandable to an extent that human observers cannot distinguish.** More precisely, our task is to develop a model that learns a mapping function that can colorize grayscale fruit images based on the statistical information from the local features and global configuration.

> 📘 Important
> 
> The project is not to perform fine-tuning from exisiting colorization tasks nor to use other open-source well-developed CNN architecture in training. The goal is to develop machine learning (CNN specifically) models from scratch with the help of Python ML library, to learn the techniques of data processing, building models, evaluation and optimization given various experimentation results.

### Approach:

An Autoencoder-based CNN model with Python Keras.

### Tech Stacks

 - **Languages**: Python, JavaScript, HTML/CSS
 - **Libraries**: Keras, OpenCV, NumPy, Seaborn, Flask
 - **Developer Tools**: Jupyter Notebook, VS Code, Git, Heroku

## Methodology

### Dataset and Data Preprocessing

For the colorization model, merely color images will be enough because no tag or label is required for colorization. In terms of our model, we rely heavily on the [FIDS30](https://www.vicos.si/resources/fids30/) [[1]](#1) dataset, which is consisted of colorful images of fruits with clean backgrounds.

In general, the data preprocessing of our model consists of the following steps:

1. Filtering outliers (non-RGB color space and overly wide or long images)
2. Resize images to 256 * 256
3. Conversion of pre-processed images from RGB color space from CIELAB color space
4. Normalization
5. Data Augmentation (Multiplying dataset size by 10x)
   a. 30-degree Rotation Range
   b. 0.3 Zoom Range
   c. Horizontal Flip
Feature-wise Center
7. Train & test split (8:2)
8. Randomize training data
9. Separate L and AB

Note that in step 3, RGB color space in converted into LAB. It is studied that CIELAB color space is commonly adopted in the colorization models for its accountability on the luminance of images. The L channel in CIE L* A*B color space refers to the luminance (i.e. grayscale), while the A and B color axes measure the red-green and yellow-blue color respectively. As a result, not only can this color space reduce the number of output channels of the model from three to two, but also can this space links the information in the grayscale into color axes effectively. After the conversion of color space, they are further processed and fed into the model.

<img src="https://github.com/tohoilam/Colorization/assets/61353084/9387a92a-8937-48b5-ab4f-d9cdb3c1aee5" alt="CIE LAB Color Space" width="300"/>  [[2]](#2)

### Model

We trained an Autoencoder-based Convolutional Neural Network (CNN) model with Keras from scratch. By using CNN model, it is able to capture local features and global configuration to understand spatial information. The architecture of the CNN model is in an Autoencoder structure. On the encoder side, the input image, which contains details and noises, is transformed into an image abstraction of important features of the image. On the decoder side, with the help of upsampling layers and convolutional layers, the colorful image is restored from the image abstraction.

One only using convolutional and upsampling layers, two major issues are addressed: The issue of not colorizing the A B space to a larger magnitude and the problem where the model will apply color outside of the edge of the fruit object. As convolutional layers only extract features from a local perspective, we assumed that the issue may be created by the incapability to extract features with only a local perspective. In addition, as the model has to reconstruct the image back to a 256*256 dimension, we cannot simply expand the perspective by magnifying the convolutional window size, since it will greatly reduce the output kernel size. Therefore, we added dilated convolutional layers which help capture features in a wider perspective without reducing kernel size, by increasing the receptive field without adding computation cost.

<img width="1311" alt="CleanShot 2023-09-25 at 00 38 04@2x" src="https://github.com/tohoilam/Colorization/assets/61353084/c9972bf8-ffe8-4f6c-b81a-2f4a0ae957cc">

After performing hyperparameter tuning, the learning rate is set to 0.001 and decay set to 0.05 with MSE loss function and ADAM optimization function.

### Evaluation Metrics

As the loss and accuracy are predicted from the MSE between the A B space of the original and the predicted images, we can use them to train our model but not to evaluate on the performance of the trained model. Since the accuracy indicator does not give a clear and full picture of how effective the colorization performance is.

As a result, two other types of evaluation metrics are adopted.

#### A). Subjective Evaluation Metrics (Human Observation):

Human observation is often the best indicator to judge the quality of colorization in an early stage. Different graphs are plot and comparison between the original and the colorized image are done to aid us in determining the colorization performance. Including image in different color spaces and color pixels distribution with scatter graphs. 

#### B). Mathematical Evaluation Metrics:

Mathematical evaluation metrics adopted include:

1. PSNR (Peak Signal-to-Noise Ratio): Ratio of maximum power (signal) between two images
2. SSIM (Structural Similarity Index): Similarity between two images
3. UIQM (Underwater Image Colorfulness Measure): Colorfulness and sharpness of the color of a single image
4. UCIQE (Underwater Color Image Quality Evaluation): Statistical distribution of the pixels of a single image

<img width="800" alt="CleanShot 2023-09-25 at 01 11 19@2x" src="https://github.com/tohoilam/Colorization/assets/61353084/48f31b92-017f-4a62-943e-6ecd8d521898">


### Experiments Performed

1.  Experimented on different model layers and structures
2.	Experimented with different activation functions
3.	Experimented with different data augmentation combinations
4.	Experimented with customization of loss functions
5.	Experimented on involving different dataset

#### Progression Made from Different Experimentation and Optimization:

<img width="800" alt="CleanShot 2023-09-25 at 01 39 46@2x" src="https://github.com/tohoilam/Colorization/assets/61353084/0cf22cea-28e5-4fae-9502-fa65406a7215">


## Results

### Best Results:

<img width="800" alt="CleanShot 2023-09-25 at 01 41 35@2x" src="https://github.com/tohoilam/Colorization/assets/61353084/95b09c41-4ec7-4544-85e5-d2b4c6f5e1b5">

- Good result when there is a clear fruit object occupying most of the space of the image
- Can colorize multiple fruits simultaneously
- Can capture unique details such as leafs and seeds

### Worst Results (Failing Cases):

<img width="800" alt="CleanShot 2023-09-25 at 01 41 46@2x" src="https://github.com/tohoilam/Colorization/assets/61353084/bd5aa1b6-2899-4e49-9b95-ee6651cbfed2">

- Bad result when there is high background noise and when fruits are in smaller sizes
- Failing to detect fruits with more than one color where both exist in the image (example: the green and red apple)

## Applications

Application Link: [https://fruits-colorization.herokuapp.com](https://fruits-colorization.herokuapp.com)

A web application is built with JavaScript, HTML/CSS, and Flask for demo, and is deployed with Heroku.

<img width="800" alt="CleanShot 2023-09-25 at 02 14 35@2x" src="https://github.com/tohoilam/Colorization/assets/61353084/a5a51bf5-1fce-4ff2-9e43-6ca12be95796">

User can either upload any grayscale fruit image (under a [list of acceptable fruits](##Colorizable-Fruits-List)) or choose from a given list of fruit image examples.


## Colorizable Fruits List

- acerolas, apples, apricots, avocados, bananas
- blackberries, blueberries, cantaloupes, cherries, coconuts
- figs, grapefruits, grapes, guava, kiwifruit
- lemons, limes, mangos, olives, oranges
- passionfruit, peaches, pears, pineapples, plums
- pomegranates, raspberries, strawberries, tomatoes, watermelons

## References

<a id="1">[1]</a> 
M. Škrjanec, "Fruit Image Data set," Visual Cognitive Systems Laboratory, [Online]. Available: https://www.vicos.si/resources/fids30/#:~:text=The%20fruit%20image%20data%20set,contains%20about%2032%20different%20images..

<a id="2">[2]</a> 
The CIE L*A*B* Color Space (CIELAB). - researchgate, https://www.researchgate.net/figure/The-CIE-Lab-color-space-CIELAB_fig1_266458803 (accessed Sep. 25, 2023). 

