import cv2
import os
from PIL import ImageOps

from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array

def apply_crop(images,left,top,right,bottom):
  return [img.crop((left, top, right, bottom)) for img in images]

def apply_auto_brightness_and_contrast(images, param):
  new_images = [automatic_brightness_and_contrast(img, param) for img in images]
  return new_images

# Automatic brightness and contrast optimization with optional histogram clipping
def automatic_brightness_and_contrast(img, clip_hist_percent):
    img = img_to_array(img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Calculate grayscale histogram
    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    hist_size = len(hist)
    
    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index -1] + float(hist[index]))
    
    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0
    
    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1
    
    # Locate right cut
    maximum_gray = hist_size -1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1
    
    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha
    

    auto_result = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return array_to_img(auto_result)

def apply_gray(images):
  return [ImageOps.grayscale(img) for img in images]

def make_data_from_folder(path):
  data = []
  for img in os.listdir(path):
      pic = cv2.imread(os.path.join(path,img))
      pic = cv2.cvtColor(pic,cv2.COLOR_BGR2RGB)
      # pic = cv2.resize(pic,(80,80))
      data.append(pic)
  return data

def augment_images(images, path_out):
  for img in images:
    # Initialising the ImageDataGenerator class.
    # We will pass in the augmentation parameters in the constructor.
    '''
      Itt lehet beállítani az augmentálás paramétereit
    '''
    datagen = ImageDataGenerator(
            # rotation_range = 15,
            # shear_rankge = 0.2,
            # zoom_range = 0.2,
            # horizontal_flip = True,
            # brightness_range = (0.5, 1.5))
    )
        
    # Converting the input sample image to an array
    x = img_to_array(img)
    # Reshaping the input image
    x = x.reshape((1, ) + x.shape) 
    
    # Generating and saving 5 augmented samples 
    # using the above defined parameters. 
    i = 0
    for batch in datagen.flow(x, batch_size = 1,
                              save_to_dir = path_out, 
                              save_prefix ='image', save_format ='jpeg'):
        '''
          Itt lehet beállítani hogy hányszor fusson képenként
        '''          
        # i += 1
        # if i > 5:
        #     break
        break

def make_data_from_folder(path):
  data = []
  for img in os.listdir(path):
      pic = cv2.imread(os.path.join(path,img))
      pic = cv2.cvtColor(pic,cv2.COLOR_BGR2RGB)
      # pic = cv2.resize(pic,(80,80))
      data.append(pic)
  return data