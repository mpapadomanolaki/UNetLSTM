# UNetLSTM
Code of the following manuscript:

'Detecting Urban Changes With Recurrent Neural Networks From Multitemporal Sentinel-2 Data'

https://arxiv.org/abs/1910.07778

# Steps
1. Preprocessing with preprocess.py
   
   Create a folder (e.g 'Images') of the raw data with the following structure:

   / Images / city / imgs_i / (13 tif 2D images of sentinel channels)

   where i=[1,2,3,4,5] 

   and city = ['abudhabi', 'aguasclaras', 'beihai', 'beirut', 'bercy', 'bordeaux', 'brasilia', 'chongqing',
        'cupertino', 'dubai', 'hongkong', 'lasvegas', 'milano', 'montpellier', 'mumbai', 'nantes',
        'norcia', 'paris', 'pisa', 'rennes', 'rio', 'saclay_e', 'saclay_w', 'valencia']

   For example, if you have 5 dates for each city, each folder should look like this (e.g for abudhabi): 
    
   ```ruby 
   mariapap@pikolo2:~/DATA/Images/abudhabi$ ls
   imgs_1  imgs_2  imgs_3  imgs_4  imgs_5

   ```
   where imgs_i are the subfolders with the 13 available channels for every city. 
   Use preprocess.py to preprocess these images.
   
   In the end, each preprocessed city folder should look like this:
   ```ruby
   mariapap@pikolo2:~/DATA/Preprocessed_Images/abudhabi$ ls
   abudhabi_1.npy  abudhabi_2.npy  abudhabi_3.npy  abudhabi_4.npy  abudhabi_5.npy
   ```
   
2. Create csv files with (x,y) locations for patch extraction during the training and validation process using make_xys.py
   Here you need to specify the folder with the OSCD dataset's Labels.
   The csv files will be saved in a folder named 'xys'.
   Also, inside the make_xys.py script there is a list containing the names of the training cities.
   
   ```ruby
   train_areas = ['abudhabi', 'beihai', 'aguasclaras', 'beirut', 'bercy', 'bordeaux', 'cupertino',
      
                 'hongkong', 'mumbai', 'nantes', 'rennes', 'saclay_e', 'pisa', 'rennes']
   ```
   
3. Start the training process with main.py

   Note that 'train_areas' list should be defined in the same sequence as in make_xys.py script

4. Make predictions on the OSCD dataset's testing images with inference.py

   Comments are included in the scripts for further instructions.

If you find this work useful, please consider citing: 

M.Papadomanolaki, Sagar Verma, M. Vakalopoulou, S. Gupta, K., 'Detecting Urban Changes With Recurrent Neural Networks From Multitemporal Sentinel-2 Data', IGARSS 2019, Yokohama, Japan
