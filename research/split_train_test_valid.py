import os
import shutil
import random

# Adatok megfelelő mappaszerkezetbe rendezése
def split_dataset(path, validSize = 0.2, testSize = 0.1):
    print("Current directory", os.getcwd())
    os.chdir(path)    
    if os.path.isdir("train/") is False:
        folders = os.listdir('./')
        os.mkdir('train')
        os.mkdir('valid')
        os.mkdir('test')

        for i in folders: 
            if i != ".DS_Store":                  
                shutil.move(f'{i}', 'train')
                os.mkdir(f'valid/{i}')
                os.mkdir(f'test/{i}')

                valid_samples = random.sample(os.listdir(f'train/{i}'), int(len(os.listdir(f'train/{i}'))*validSize))
                for j in valid_samples:
                    shutil.move(f'train/{i}/{j}', f'valid/{i}')

                test_samples = random.sample(os.listdir(f'train/{i}'), int(len(os.listdir(f'train/{i}'))*testSize))
                for k in test_samples:
                    shutil.move(f'train/{i}/{k}', f'test/{i}')
    # os.chdir("/content")    
    print("Valid, test, train sets created.")

split_dataset("/Users/bormilan/Documents/plm_kepek_dl")