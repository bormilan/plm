import cv2
import numpy as np
from skimage import feature
from skimage import transform
import joblib
from scipy.spatial.distance import cityblock
import time

class Distance_measure:

    model = None
    treshold = None

    def __init__(self, path_model, treshold):
        # load model
        filename = path_model
        self.model = joblib.load(filename)
        self.treshold = treshold

    '''

        MAIN FUNCTION THAT RETURN THE DISTANCE
        INPUT: IMAGE (numpy array)
        OUTPUT: DISTANCE (int)

    '''
    def distance_measurement(self, image):

        #   image_path = path_image
        #   image = cv2.imread(image_path)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img = img[200:440, 300:700]

        # cv2.imwrite("photos/"+str(datetime.datetime.now()).replace(".","-")+'.jpg', img)

        #predict with sliding window
        start_time = time.time()
        indices, patches, koords = zip(*sliding_window(self.model, img))
        #plot result
        # vis_points(img, koords)
        print("--- %s seconds ---" % (time.time() - start_time))
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

        #calculate the distance
        distances = calc_distances(a_koords, b_koords)

        #if its under the treshold, its good
        result = "Hibas felhelyezes"
        if np.mean(distances) < self.treshold:
            result = "Helyes felhelyezes"

        # return np.mean(distances)
        return result


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

    distances.append(cityblock(o1, o2))

  return distances



'''

    SLIDE OVER THE IMAGE AND CLASSIFIE THE PATCHES
    INPUT: ML MODEL, IMAGE
    OUTPUT: COORDINATES

'''
def sliding_window(model, img, istep=8, jstep=10, scale=1.0):
    
    patch_size = (100, 100)
    # fig, ax = plt.subplots()
    # ax.imshow(img, cmap='gray')

    Ni, Nj = (int(scale * s) for s in patch_size)
    for i in range(0, img.shape[0] - Ni, istep):
        for j in range(0, img.shape[1] - Ni, jstep):

            koords = ()

            patch = img[i:i + Ni, j:j + Nj]
            if scale != 1:
                patch = transform.resize(patch, patch_size)
            
            #prediktáljuk hogy ez egy seeger fog-e, hog reprezentáció alapján
            label = 0
            if (i > 50 and i < 150) and ((j < 100 and j > 25) or (j > 225 and j < 300)): 
              label = model.predict(feature.hog(patch).reshape(1, -1))

            #ha az akkor rajzoljunk négyzetet köré és adjuk vissza az értékeket
            if label == 1:
              koords = ((j, i), (j + Nj, i + Ni))
            #   ax.add_patch(plt.Rectangle((j, i), Nj, Ni, edgecolor='red',
            #                     alpha=0.2, lw=2, facecolor='none'))
            yield (i, j, koords)