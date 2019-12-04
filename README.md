# UNetLSTM


# Steps
# 1. Preprocessing with preprocess.py
Create a folder (e.g 'images') of the raw data with the following structure:

/images/city/imgs_i/(13 tif 2D images of sentinel channels)

where i=[1,2,3,4,5] 

IMG_FOLDER = './images/' #folder of the form ./images/abudhabi/imgs_1/..(13 tif 2D images of sentinel channels)..
                                 #           ./images/abudhabi/imgs_2/..(13 tif 2D images of sentinel channels)..
                                 #           ....
                                 #           ./images/abudhabi/imgs_n/..(13 tif 2D images of sentinel channels)..
                                 #           where n = number of dates

Use preprocess.py to preprocess the images of the OSCD dataset.



2. Use make_xys.py to create csv file with (x,y) locations for patch extraction during the training process.
3. Use main.py to start the training process.
4. Use inference.py to make predictions on the testing images.

Comments are included in the scripts for further instructions.

If you find this work useful, please consider citing: M.Papadomanolaki, Sagar Verma, M. Vakalopoulou, S. Gupta, K., 'DETECTING URBAN CHANGES WITH RECURRENT NEURAL NETWORKS FROMMULTITEMPORAL SENTINEL-2 DATA'
