import os
from turtle import distance
import cv2
import numpy as np
from skimage import color, feature
import matplotlib.pyplot as plt
from skimage import transform
import pandas as pd
import joblib

import functions as f


'''

    MAIN FUNCTION THAT RETURN THE DISTANCE
    INPUT: IMAGE (numpy array)
    OUTPUT: DISTANCE (int)

'''
def distance_measurement(model, image):

    #   image_path = path_image
    #   image = cv2.imread(image_path)
    img = color.rgb2gray(image)

    #predict with sliding window
    indices, patches, koords = zip(*sliding_window(model, img))
    #plot result
    # vis_points(img, koords)

    #tuples into list
    koords = list(koords)

    new_koords = []
    #delete empty units
    for k in koords:
        if k != ():
            new_koords.append(k)

    # vis_points(img, koords)

    #delete duplicate units
    #   new_koords = pd.unique(pd.Series(koords))

    #separete points
    a_koords = [new_koords[1]]
    b_koords = []

    for k in new_koords[2:len(new_koords)]:
        l = a_koords[0]
        o1 = int((k[0][0] + k[1][0]) / 2), int((k[0][1] + k[1][1]) / 2)
        o2 = int((l[0][0] + l[1][0]) / 2), int((l[0][1] + l[1][1]) / 2)

        # print(np.linalg.norm(np.array(o2) - np.array(o1)))
        if np.linalg.norm(np.array(o2) - np.array(o1)) < 50:
            a_koords.append(k)
        else:
            b_koords.append(k)

    distances = calc_distances(a_koords, b_koords)
    result = False
    if np.mean(distances) < 200:
        result = True

    return np.mean(distances)
#   return result


'''

    CALCULATE TWO EACH COORDINATE'S DISTANCE
    INPUT: TWO LIST OF COORDINATES
    OUTPUT: LIST OF DISTANCES

'''
def calc_distances(a_koords, b_koords):
  distances = []
  for a, b in zip(a_koords, b_koords):
    o1 = int((a[0][0] + a[1][0]) / 2), int((a[0][1] + a[1][1]) / 2)
    o2 = int((b[0][0] + b[1][0]) / 2), int((b[0][1] + b[1][1]) / 2)

    distances.append(np.linalg.norm(np.array(o2) - np.array(o1)))

  return distances



'''

    SLIDE OVER THE IMAGE AND CLASSIFIE THE PATCHES
    INPUT: ML MODEL, IMAGE
    OUTPUT: COORDINATES

'''
def sliding_window(model, img, istep=5, jstep=5, scale=1.0):
    
    patch_size = (100, 100)
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')

    Ni, Nj = (int(scale * s) for s in patch_size)
    for i in range(0, img.shape[0] - Ni, istep):
        for j in range(0, img.shape[1] - Ni, jstep):

            koords = ()

            patch = img[i:i + Ni, j:j + Nj]
            if scale != 1:
                patch = transform.resize(patch, patch_size)
            
            #predikt??ljuk hogy ez egy seeger fog-e, hog reprezent??ci?? alapj??n
            label = 0
            if (i > 50 and i < 150) and ((j < 100 and j > 25) or (j > 225 and j < 300)): 
              label = model.predict(feature.hog(patch).reshape(1, -1))

            #ha az akkor rajzoljunk n??gyzetet k??r?? ??s adjuk vissza az ??rt??keket
            if label == 1:
              koords = ((j, i), (j + Nj, i + Ni))
              ax.add_patch(plt.Rectangle((j, i), Nj, Ni, edgecolor='red',
                                alpha=0.2, lw=2, facecolor='none'))
            yield (i, j, koords)


def startup(path_model):
    # load model
    filename = path_model
    model = joblib.load(filename)

    return model

def main():

    path_model = "/Users/bormilan/Documents/ko??d/plm/model_knn.sav"
    path_image = "/Users/bormilan/Documents/plm_kepek_detect/test_crop/2022-04-29 11_24_34-632004.jpg"
    
    model = startup(path_model)

    path_images = '/Users/bormilan/Documents/plm_kepek_detect/test4'
    test_images = f.make_data_from_folder(path_images)
    distances = []
    for img in test_images:
        distances.append(distance_measurement(model, img))

    # print(distances)
    print(len(distances))
    num = 0
    for dist in distances:
        if dist > 200:
            num += 1

    print(num)
    # measure the distance
    # distance = distance_measurement(path_model, path_image)
    # print(distance)

main()