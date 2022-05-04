from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
import cv2
from PIL import Image
import os.path, sys
import datetime
from os import listdir 

import functions as f

'''

EZ A SCRIPT KÉPEK ELŐFELDOLGOZÁSÁT VÉGZI
INPUT: KÉPEKET TARTALMAZÓ GYÖKÉRMAPPA
OUTPUT: FELDOLGOZOTT KÉPEKET TARTALMAZÓ GYÖKÉRMAPPA

FELDOLGOZÁSI LÉPÉSEK:
    - KÉPEK LEVÁGÁSA
    - KÉPEK SZÁMÁNAK NÖVELÉSE AUGMENTÁLÁSSAL
    - KÉPEK KONTRASZTJÁNAK ÉS FÉNYEREJÉNEK AUTOMATIKUS BEÁLLÍTÁSA
    - KÉPEK KONVERTÁLÁSA FEKETE-FEHÉR FORMÁTUMRA

'''

#Adathalmaz gyökérmappájának path-ja
path = '/Users/bormilan/Documents/plm_kepek'

temp = "test"
#Célmappa root
target_path = f'/Users/bormilan/Documents/plm_kepek/{temp}'
os.mkdir(target_path)

# képek beolvasása

dirs = listdir(path)

images_root = []
folder_num = 0
for dir in dirs:
    if dir != ".DS_Store" and dir != "test" and dir != "test2" and dir != "test3" and dir != "test4" and dir != "test_dl":
        images_root.append([Image.open(os.path.join(path, dir, img)) for img in listdir(os.path.join(path, dir)) if img != ".DS_Store"])
        #a belső mappák létrehozása
        os.mkdir(os.path.join(target_path, str(folder_num)))
        folder_num += 1

#mindegyik mappán végigfut
for i, set in enumerate(images_root):
  # képek croppolása
  # 400, 200, 700, 500 - sima 
  # 520, 280, 620, 380 - a gyűrű két foga
  # 400, 200, 800, 600 - új
  set = f.apply_crop(set, 250, 350, 650, 750)
  # képek auto contrast és brightness-elése
  set = f.apply_auto_brightness_and_contrast(set, 10)
  # képek grayscale-ezése
  # set = f.apply_gray(set)
  # képek augmentálása és exportálása
  f.augment_images(set, os.path.join(path, temp, str(i), ))