import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import skimage
from skimage import color
from skimage.color import rgb2gray

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

class Data():
  def __init__(self, data_dir, DIMENSION, RESAMPLE):
    # Labels
    self.labels = []
    self.data_dir = data_dir
    self.DIMENSION = DIMENSION
    self.RESAMPLE = RESAMPLE

    # Test Data
    self.testImages = []
    self.l_org_test = []
    self.org_size = []
    self.file_names = []
    self.grayscaleImages = []



  def loadImage(self):
    data_paths = []
    self.testImages = []

    ###################################
    ######### Load Data Paths #########
    ###################################
    print('Loading image paths...')
    print('')

    # Get data paths
    for dirname, _, filenames in os.walk(self.data_dir):
      for filename in filenames:
        if (filename == '.DS_Store'):
          continue
        
        file_path = os.path.join(dirname, filename)
        data_paths.append(file_path)
        

    data_size = len(data_paths)

    print('Data paths loaded!')
    print('     Number of data:', data_size)
    print('')

    ###################################
    ########### Load Images ###########
    ###################################
    print('Loading images from paths...')
    print('')

    # Load and resize image
    for data_path in data_paths:
      image = np.asarray(Image.open(data_path))
      
      # Expand if only 2 in dimensions
      if (image.ndim==2):
        image = np.tile(image[:,:,None],3)
      
      # remove CMYK or RGBA image representation
      if (np.asarray(image).shape[2] != 3):
        continue
      
      self.org_size.append((np.asarray(image).shape[1], np.asarray(image).shape[0]))
      self.file_names.append(data_path.split('/')[-1])
      # self.grayscaleImages.append(color.rgb2gray(image))
      self.grayscaleImages.append(np.dot(image[...,:3], [0.2989, 0.5870, 0.1140]))

      # Resize Image
      image = Image.fromarray(image).resize(self.DIMENSION, resample=self.RESAMPLE)

      self.testImages.append(np.asarray(image))

    # Convert to np array
    self.testImages = np.asarray(self.testImages)
    
    print('Images Loaded!')
    print('     Number of test data:', len(self.testImages))
         
  
  def dataProcessing(self, image):
    # Change to LAB Space
    lab_image = color.rgb2lab(image)

    # Get L Space
    l_image = lab_image[:,:,0]

    # Get AB Space
    ab_image = lab_image[:,:,1:] / 100.

    return l_image, ab_image
  
  
  def processTestData(self):
    self.l_org_test = []

    test_augmentation = ImageDataGenerator(rescale=1./255)

    l_list = []

    test_images_label = test_augmentation.flow(self.testImages, batch_size=128, shuffle=False)

    for numBatch in range(len(test_images_label)):
      for image in test_images_label[numBatch]:
        rgb_image = np.asarray(image)
        lab_image = color.rgb2lab(rgb_image)
        l_image = lab_image[:,:,0]

        l_list.append(l_image)
    
    l_list = np.asarray(l_list)
    l_list = l_list.reshape(l_list.shape[0], l_list.shape[1], l_list.shape[2], 1)
    self.l_org_test = l_list
      
    print("  Resized Test Image in L Space   :", self.l_org_test.shape)
    print('')
  
  def showImage(self, imageArray, axis='on', grayscale=False):
    if (grayscale):
      plt.imshow(imageArray, cmap='gray', interpolation='nearest')
    else:
      plt.imshow(imageArray, interpolation='nearest')

    plt.axis(axis)
    plt.show()
