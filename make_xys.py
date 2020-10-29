import numpy as np
from skimage import io
from skimage.transform import rotate, resize
import os
import cv2
import pandas as pd
import argparse
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--labels_folder', type=str, default='../Labels/',
                    help='destination path for the labels folder')
parser.add_argument('--patch_size', type=int, default=32,
                    help='dimensions of the patch size you wish to use')
parser.add_argument('--step', type=int, default=19,
                    help='step that will be used to extract the patches along the x and y direction')

args = parser.parse_args()

train_areas = ['abudhabi', 'beihai', 'aguasclaras', 'beirut', 'bercy', 'bordeaux', 'cupertino',

                 'hongkong', 'mumbai', 'nantes', 'rennes', 'saclay_e', 'pisa', 'rennes']

step=args.step
patch_s=args.patch_size

def shuffle(vector):
  vector = np.asarray(vector)
  p=np.random.permutation(len(vector))
  vector=vector[p]
  return vector


def sliding_window_train(i_city, labeled_areas, label, window_size, step):
    city=[]
    fpatches_labels=[]

    x=0
    while (x!=label.shape[0]):
     y=0
     while(y!=label.shape[1]):

               if (not y+window_size > label.shape[1]) and (not x+window_size > label.shape[0]):
                line=np.array([x,y, labeled_areas.index(i_city), 0]) # (x,y) are the saved coordinates, 
                                                                     # labeled_areas.index(i_city)... are the image ids, e.g according to train_areas,
                                                                           #the indice for abudhabi in the list is 0, for beihai it is 1, for beirut is 3, etc..
                                                                     # the fourth element which has been set as 0, represents the transformadion index,
                                                                           #which in this case indicates that no data augmentation will be performed for the
                                                                           #specific patch 
                city.append(line)

                ##############CONDITIONS####################################
                new_patch_label = label[x:x + window_size, y:y + window_size]
                ff=np.where(new_patch_label==2)
                perc = ff[0].shape[0]/float(window_size*window_size)
                if ff[0].shape[0]==0:
                       stride=window_size
                else:
                       stride=step
                if perc>=0.05: #if percentage of change exceeds a threshold, perform data augmentation on this patch
                               #Below, 1, 2, 3 are transformation indexes that will be used by the custom dataloader
                               #to perform various rotations
                       line=np.array([x,y, labeled_areas.index(i_city), 1])
                       city.append(line)
                       line=np.array([x,y, labeled_areas.index(i_city), 2])
                       city.append(line)
                       line=np.array([x,y, labeled_areas.index(i_city), 3])
                       city.append(line)
                 ###############CONDITIONS####################################

               if y + window_size == label.shape[1]:
                  break

               if y + window_size > label.shape[1]:
                y = label.shape[1] - window_size
               else:
                y = y+stride

     if x + window_size == label.shape[0]:
        break

     if x + window_size > label.shape[0]:
       x = label.shape[0] - window_size
     else:
      x = x+stride

    return np.asarray(city)


cities=[]
for i_city in train_areas:
 path=args.labels_folder+'{}/cm/{}-cm.tif'.format(i_city, i_city)
 im_name = os.path.basename(path)
 print('icity', i_city)
 train_gt = io.imread(path)
 xy_city =  sliding_window_train(i_city, train_areas, train_gt, patch_s, step)
 cities.append(xy_city)

#from all training (x,y) locations, divide 4/5 for training and 1/5 for validation
final_cities = np.concatenate(cities, axis=0)
size_len = len(final_cities)
portion=int(size_len/5)
final_cities=shuffle(final_cities)
final_cities_train = final_cities[:4*portion]
final_cities_val = final_cities[4*portion:]


save_folder = './xys/' #where to save the models and training progress
if os.path.exists(save_folder):
    shutil.rmtree(save_folder)
os.mkdir(save_folder)

##save train to csv file
df = pd.DataFrame({'X': list(final_cities_train[:,0]),
                   'Y': list(final_cities_train[:,1]),
                   'image_ID': list(final_cities_train[:,2]),
                   'transform_ID': list(final_cities_train[:,3]),
                   })
df.to_csv(save_folder + 'myxys_train.csv', index=False, columns=["X", "Y", "image_ID", "transform_ID"])


df = pd.DataFrame({'X': list(final_cities_val[:,0]),
                   'Y': list(final_cities_val[:,1]),
                   'image_ID': list(final_cities_val[:,2]),
                   'transform_ID': list(final_cities_val[:,3]),
                   })
df.to_csv(save_folder + 'myxys_val.csv', index=False, columns=["X", "Y", "image_ID", "transform_ID"])
