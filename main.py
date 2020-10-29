import os
import glob
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchnet as tnt
from skimage import io
import tools
import custom
from torch.utils.data import DataLoader
import argparse
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--images_folder', type=str, default='../images_labeled/',
                    help='destination path for the images folder')
parser.add_argument('--labels_folder', type=str, default='../Labels/',
                    help='destination path for the labels folder')
parser.add_argument('--xys', type=str, default='./xys/',
                    help='destination path for the csv files')
parser.add_argument('--patch_size', type=int, default=32,
                    help='dimensions of the patch size you wish to use')
parser.add_argument('--nb_dates', type=list, default=[1,5],
                    help='number of dates you wish to use')
parser.add_argument('--model_type', type=str, default='lstm',
                    help='simple or lstm')

args = parser.parse_args()


train_areas = ['abudhabi', 'beihai', 'aguasclaras', 'beirut', 'bercy', 'bordeaux', 'cupertino',

                 'hongkong', 'mumbai', 'nantes', 'rennes', 'saclay_e', 'pisa', 'rennes']

csv_file_train = args.xys + 'myxys_train.csv'
csv_file_val = args.xys + 'myxys_val.csv'
img_folder = args.images_folder #folder with preprocessed images according to preprocess.py
lbl_folder = args.labels_folder #folder with OSCD dataset's labels
patch_size=args.patch_size
nb_dates = args.nb_dates #specify the number of dates you want to use, e.g put [1,2,3,4,5] if you want to use all five dates
                 #or [1,2,5] to use just three of them

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

change_dataset_train =  custom.MyDataset(csv_file_train, train_areas, img_folder, lbl_folder, nb_dates, patch_size)
change_dataset_val =  custom.MyDataset(csv_file_val, train_areas, img_folder, lbl_folder, nb_dates, patch_size)
mydataset_val = DataLoader(change_dataset_val, batch_size=32)


#images_train, labels_train, images_val, labels_val = tools.make_data(size_len, portion, change_dataset)
base_lr=0.0001
optimizer = optim.Adam(model.parameters(), lr=base_lr)
weight_tensor=torch.FloatTensor(2)
weight_tensor[0]= 0.20
weight_tensor[1]= 0.80
criterion=tools.to_cuda(nn.CrossEntropyLoss(tools.to_cuda(weight_tensor)))
confusion_matrix = tnt.meter.ConfusionMeter(2, normalized=True)
epochs=60

save_folder = 'models' #where to save the models and training progress
if os.path.exists(save_folder):
    shutil.rmtree(save_folder)
os.mkdir(save_folder)

ff=open('./' + save_folder + '/progress.txt','w')
iter_=0
for epoch in range(1,epochs+1):
    mydataset = DataLoader(change_dataset_train, batch_size=32, shuffle=True)
    model.train()
    train_losses = []
    confusion_matrix.reset()

    for i, batch, in enumerate(tqdm(mydataset)):
        img_batch, lbl_batch = batch
        img_batch, lbl_batch = tools.to_cuda(img_batch.permute(1,0,2,3,4)), tools.to_cuda(lbl_batch)

        optimizer.zero_grad()
        output=model(img_batch.float())
        output_conf, target_conf = tools.conf_m(output, lbl_batch)
        confusion_matrix.add(output_conf, target_conf)

        loss=criterion(output, lbl_batch.long())
        train_losses.append(loss.item())
        loss.backward()
        optimizer.step()

        _, preds = output.data.max(1)
        if iter_ % 100 == 0:
         pred = np.argmax(output.data.cpu().numpy()[0], axis=0)
         gt = lbl_batch.data.cpu().numpy()[0]
         print('Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {}'.format(
                      epoch, epochs, i, len(mydataset),100.*i/len(mydataset), loss.item(), tools.accuracy(pred, gt)))

        iter_ += 1
        del(img_batch, lbl_batch, loss)

    train_acc=(np.trace(confusion_matrix.conf)/float(np.ndarray.sum(confusion_matrix.conf))) *100

    print('TRAIN_LOSS: ', '%.3f' % np.mean(train_losses), 'TRAIN_ACC: ', '%.3f' % train_acc)
    confusion_matrix.reset()

    ##VALIDATION
    with torch.no_grad():
        model.eval()

        val_losses = []

        for i, batch, in enumerate(tqdm(mydataset_val)):
            img_batch, lbl_batch = batch
            img_batch, lbl_batch = tools.to_cuda(img_batch.permute(1,0,2,3,4)), tools.to_cuda(lbl_batch)

            output=model(img_batch.float())
            loss=criterion(output, lbl_batch.long())
            val_losses.append(loss.item())
            output_conf, target_conf = tools.conf_m(output, lbl_batch)
            confusion_matrix.add(output_conf, target_conf)

        print(confusion_matrix.conf)
        test_acc=(np.trace(confusion_matrix.conf)/float(np.ndarray.sum(confusion_matrix.conf)))*100
        change_acc=confusion_matrix.conf[1,1]/float(confusion_matrix.conf[1,0]+confusion_matrix.conf[1,1])*100
        non_ch=confusion_matrix.conf[0,0]/float(confusion_matrix.conf[0,0]+confusion_matrix.conf[0,1])*100
        print('VAL_LOSS: ', '%.3f' % np.mean(val_losses), 'VAL_ACC:  ', '%.3f' % test_acc, 'Non_ch_Acc: ', '%.3f' % non_ch, 'Change_Accuracy: ', '%.3f' % change_acc)
        confusion_matrix.reset()



    tools.write_results(ff, save_folder, epoch, train_acc, test_acc, change_acc, non_ch, np.mean(train_losses), np.mean(val_losses))
    if epoch%5==0: #save model every 5 epochs
       torch.save(model.state_dict(), './' + save_folder + '/model_{}.pt'.format(epoch))
