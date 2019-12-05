# UNetLSTM
Code of the following manuscript:

'Detecting Urban Changes With Recurrent Neural Networks From Multitemporal Sentinel-2 Data'

https://arxiv.org/abs/1910.07778

# Steps
# 1. Preprocessing with preprocess.py
Create a folder (e.g 'images') of the raw data with the following structure:

/ images / city / imgs_i / (13 tif 2D images of sentinel channels)

where i=[1,2,3,4,5] 

and city = ['abudhabi', 'aguasclaras', 'beihai', 'beirut', 'bercy', 'bordeaux', 'brasilia', 'chongqing',
        'cupertino', 'dubai', 'hongkong', 'lasvegas', 'milano', 'montpellier', 'mumbai', 'nantes',
        'norcia', 'paris', 'pisa', 'rennes', 'rio', 'saclay_e', 'saclay_w', 'valencia']

Use preprocess.py to preprocess the images of the OSCD dataset.

# 2. Create csv file with (x,y) locations for patch extraction during the training process using make_xys.py
Here you need to specify the folder with the OSCD dataset's Labels.

Note that 'train_areas' list should be defined in the same way both in make_xys.py and main.py

# 3. Start the training process with main.py

# 4. Make predictions on the OSCD dataset's testing images with inference.py

Comments are included in the scripts for further instructions.

If you find this work useful, please consider citing: M.Papadomanolaki, Sagar Verma, M. Vakalopoulou, S. Gupta, K., 'Detecting Urban Changes With Recurrent Neural Networks From Multitemporal Sentinel-2 Data', IGARSS 2019, Yokohama, Japan
