import glob
import cv2
from skimage import io
import numpy as np
import os
from tqdm import tqdm
import torch
import network
import tools

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

test_areas = ['brasilia', 'milano', 'norcia', 'chongqing', 'dubai', 'lasvegas', 'montpellier', 'rio', 'saclay_w', 'valencia']
test_areas=['brasilia']
nb_dates = [1,2,3,4,5]
patch_size = 32
step = 16
model=network.U_Net(4,2,nb_dates)
BATCH_SIZE=1
save_dir = 'PREDICTIONS'
os.mkdir(save_dir)
model.load_state_dict(torch.load('./saved_models/model_22.pt')) #ena apo to 5D
model=tools.to_cuda(model)
model = model.eval()

FOLDER = './IMGS_PREPROCESSED/'

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
   cv2.imwrite('./' + save_dir + '/' + id + '.tif', pred)
