import glob
import cv2
from skimage import io
import numpy as np
import os
from tqdm import tqdm
import torch
import tools
import argparse
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--images_folder', type=str, default='../images_nonlabeled/',
                    help='destination path for the images folder')
parser.add_argument('--labels_folder', type=str, default='../Labels/',
                    help='destination path for the labels folder')
parser.add_argument('--xys', type=str, default='./xys/',
                    help='destination path for the csv files')
parser.add_argument('--saved_model', type=str, default='./models/model_3.pt',
                    help='destination path for the trained model')
parser.add_argument('--patch_size', type=int, default=32,
                    help='dimensions of the patch size you wish to use')
parser.add_argument('--step', type=int, default=32,
                    help='step you wish to use for the extraction of patches for inference')
parser.add_argument('--nb_dates', type=list, default=[1,2],
                    help='number of dates you wish to use')
parser.add_argument('--model_type', type=str, default='lstm',
                    help='simple or lstm')

args = parser.parse_args()


def sliding_window(IMAGE, patch_size, step):
    prediction = np.zeros((IMAGE.shape[3], IMAGE.shape[4], 2))
    x=0
    while (x!=IMAGE.shape[0]):
     y=0
     while(y!=IMAGE.shape[1]):

               if (not y+patch_size > IMAGE.shape[4]) and (not x+patch_size > IMAGE.shape[3]):
                patch = IMAGE[:, :, :, x:x + patch_size, y:y + patch_size]
                patch = tools.to_cuda(torch.from_numpy(patch).float())
                output = model(patch)
                output = output.cpu().data.numpy().squeeze()
                output = np.transpose(output, (1,2,0))
                for i in range(0, patch_size):
                    for j in range(0, patch_size):
                        prediction[x+i, y+j] += (output[i,j,:])

                stride=step

               if y + patch_size == IMAGE.shape[4]:
                  break

               if y + patch_size > IMAGE.shape[4]:
                y = IMAGE.shape[4] - patch_size
               else:
                y = y+stride

     if x + patch_size == IMAGE.shape[3]:
        break

     if x + patch_size > IMAGE.shape[3]:
       x = IMAGE.shape[3] - patch_size
     else:
      x = x+stride

    final_pred = np.zeros((IMAGE.shape[3], IMAGE.shape[4]))
    print('ok')
    for i in range(0, final_pred.shape[0]):
        for j in range(0, final_pred.shape[1]):
            final_pred[i,j] = np.argmax(prediction[i,j])

    final_pred[final_pred==1]=2
    final_pred[final_pred==0]=1

    return final_pred


patch_size = args.patch_size
step = args.step

networks_folder_path = './networks/'
import sys
sys.path.insert(0, networks_folder_path)

model_type = args.model_type #choose network type ('simple' or 'lstm')
                      #'simple' refers to a simple U-Net while 'lstm' refers to a U-Net involving LSTM blocks
if model_type == 'simple':
    import network
    model=tools.to_cuda(network.U_Net(4,2,nb_dates))
elif model_type=='lstm':
    import networkL
    model=tools.to_cuda(networkL.U_Net(4,2,patch_size))
else:
 print('invalid on_network_argument')


test_areas = ['brasilia', 'milano', 'norcia', 'chongqing', 'dubai', 'lasvegas', 'montpellier', 'rio', 'saclay_w', 'valencia']
nb_dates = args.nb_dates
patch_size = args.patch_size
step = args.step

save_folder = 'PREDICTIONS'
if os.path.exists(save_folder):
    shutil.rmtree(save_folder)
os.mkdir(save_folder)

model.load_state_dict(torch.load(args.saved_model)) #ena apo to 5D
model = model.eval()

FOLDER = args.images_folder

for id in test_areas:
   print('test_area', id)
   
   imgs = []
   for nd in nb_dates:
       img = np.load(FOLDER + id + '/' + id + '_{}.npy'.format(str(nd)))
       img = np.concatenate((img[:,:,1:4], np.reshape(img[:,:,7], (img.shape[0],img.shape[1],1))), 2)
       img = np.transpose(img, (2,0,1))
       imgs.append(img)
   imgs = np.asarray(imgs)
   imgs = np.reshape(imgs, (imgs.shape[0], 1, imgs.shape[1], imgs.shape[2], imgs.shape[3]))

   pred = sliding_window(imgs, patch_size, step)
   cv2.imwrite('./' + save_folder + '/' + id + '.tif', pred)
