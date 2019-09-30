from imgaug import augmenters as iaa
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import json
import glob 
from PIL import Image
import scipy.misc
import random

def GetID(ID_file_path,filename):
    IDlist=[]
    with open(ID_file_path+filename, "r") as f:
      for line in f:
        IDlist.append(line.strip())
    # with open(ID_file_path+filename, 'r') as f:
    #      IDlist = json.loads(f.read())

    return IDlist
#----------------------------------------------------------------------------
FOLDER = r"I:\Krongrath\Transfer\dataset\\"

# train_no_threat_list = GetID(FOLDER,'train_no_threat.txt')
# train_threat_list = GetID(FOLDER,'train_threat.txt')
# print(ID_list)

#----------------------------------------------------------------------------
def GetName(Folder,ID_file_path,filename):
    list_of_ID = GetID(ID_file_path,filename)
    image_angle = np.array([1,3,5,7,9,11,13,15])
    image_list = []
    for ID in list_of_ID:
         for i in np.nditer(image_angle):
             image = Folder+'ID_{}_angle_{}.jpg'.format(ID,i) 
             image_list.append(image)
    return image_list

#----------------------------------------------------------------------------
def GetName2(Folder,ID_file_path,list_of_ID):
    image_angle = np.array([1,3,5,7,9,11,13,15])
    image_list = []
    for ID in list_of_ID:
         for i in np.nditer(image_angle):
             image = Folder+'ID_{}_angle_{}.jpg'.format(ID,i) 
             image_list.append(image)
    return image_list
#----------------------------------------------------------------------------
#For train no threat
# TRAIN_NO_THREAT_FOLDER = r"I:\Krongrath\Transfer\dataset\imageFile\train_no_threat\\"
# ID_list = GetID(FOLDER,'train_no_threat.txt')
# image_train_no_threat = GetName(TRAIN_NO_THREAT_FOLDER, FOLDER, 'train_no_threat.txt')

#Save the array of all images into .npy file
#images_train_no_threat = np.array([np.array(Image.open(fname)) for fname in image_train_no_threat])
#np.save('images_train_no_threat_array',images_train_no_threat)
#load 4D array of images (shape = (1576,660,512,3))
#images = np.load('./images_train_no_threat_array.npy')

#----------------------------------------------------------------------------
#For train threat (only random 270 samples will be augmented)
TRAIN_THREAT_FOLDER = r"I:\Krongrath\Transfer\dataset\imageFile\train_threat\\"
ID_list = GetID(FOLDER,'train_threat.txt')
image_train_threat = GetName2(TRAIN_THREAT_FOLDER, FOLDER, ID_list)

num_to_select = 270                         # set the number to select here.
ID_list_270 = random.sample(ID_list, num_to_select)
image_train_threat_270 = GetName2(TRAIN_THREAT_FOLDER, FOLDER, ID_list_270)

#write all ID in train threat that were augmented
#with open(r'.\dataset\train_threat_augment.txt', 'w') as outfile:
#    json.dump(ID_list_270, outfile)

#Save the array of all images into .npy file
#images_train_threat = np.array([np.array(Image.open(fname)) for fname in image_train_threat])
#np.save('images_train_threat_array',images_train_threat)
#print('image_train_threat:',images_train_threat.shape)

#images_train_threat_270 = np.array([np.array(Image.open(fname)) for fname in image_train_threat_270])
#np.save('images_train_threat_270_array',images_train_threat_270)
#print('image_train_threat_270:',images_train_threat_270.shape)

#load 4D array of images 
images = np.load('./images_train_threat_array.npy')
images_270 = np.load('./images_train_threat_270_array.npy')
print(images_270.shape)

#----------------------------------------------------------------------------
#test the augmentation methods 
# flipper = iaa.Fliplr(1.0) # always horizontally flip each input image
# images_flip = flipper.augment_image(images[5])

# translater = iaa.Affine(translate_px={"x": 30})
# images_translate = translater.augment_image(images[7]) 

# sharpener = iaa.Sharpen(alpha=0.3,lightness=0.3)
# images_sharp = sharpener.augment_images(images[10])

# pre_post_translate=np.append(images[7],images_translate).reshape(2,512,660,3)
#----------------------------------------------------------------------------
#apply the sequential agumentation methods
# sometimes = lambda aug: iaa.Sometimes(0.5,aug)

# seq = iaa.Sequential(
#   [    
#        sometimes(iaa.Affine(
#         translate_px={"y": (-50,50)}
#        )),

#        sometimes(iaa.Sharpen(
#         alpha=(0.3,0.5), lightness=(0.3,0.5)
#        ))
#   ], random_order=True
# )

seq_translate_x = iaa.Sequential([
     iaa.Affine(
      translate_px={"x": (20,40)})
  ])

seq_translate_y_up = iaa.Sequential([ #USE THIS
     iaa.Affine(
      translate_px={"y": (-100,-20)})
  ])

seq_translate_y_down = iaa.Sequential([ #USE THIS
     iaa.Affine(
        translate_px={"y": (20,100)})
    ])

seq_translate_up_down = iaa.Sequential([
     iaa.Affine(
      translate_px={"y": (-100,100)})
  ])

seq_translate_xy = iaa.Sequential([
     iaa.Affine(
      translate_px={"x": (20,40), "y": (-100,100)})
  ])

seq_sharpen = iaa.Sequential([ #USE THIS
   iaa.Sharpen(
        alpha=(0.3,0.5))
   ])

seq_lightness = iaa.Sequential([ #USE THIS
     iaa.Sharpen(
            alpha=(0.3,0.5), lightness=(0.3,0.8))
     ])

seq_translate_sharpen = iaa.Sequential([ #USE THIS 
     iaa.Affine(
      translate_px={"y": (-100,100)}),

     iaa.Sharpen(
        alpha=(0.3,0.5), lightness=(0.3,0.8))
  ])

#For train no threat (augment = 765 files)
#images_translate_y_up = seq_translate_y_up.augment_images(images)
#images_translate_y_down = seq_translate_y_down.augment_images(images)
#images_sharp = seq_sharpen.augment_images(images)
#images_lightness = seq_lightness.augment_images(images)
#images_translate_sharp = seq_translate_sharpen.augment_images(images)

#For train threat(augment=290 files)
images_translate_y_up_down = seq_translate_up_down.augment_images(images_270)
print('image_translate_y_up_down.shape=',images_translate_y_up_down.shape)
print('Finish augmentation')

#----------------------------------------------------------------------------
#save images to .jpg
def save_image_aug(FOLDER,type_of_image,ID_list,images):
    count = 0
    image_angle = np.array([1,3,5,7,9,11,13,15])
    for ID in ID_list:
        for j in image_angle:
             scipy.misc.imsave(FOLDER + '{}_ID_{}_angle_{}.jpg'.format(type_of_image,ID,j), images[count])
             count += 1


#AUGMENT_FOLDER = r'I:\Krongrath\Transfer\dataset\imageFile\train_no_threat_augment\\'
AUGMENT_FOLDER = r'I:\Krongrath\Transfer\dataset\imageFile\train\train_threat_augment\\'

#save all images from different types of augmentations together
#save_image_aug(AUGMENT_FOLDER,'original', ID_list,images)
#save_image_aug(AUGMENT_FOLDER,'translate_y_up',ID_list,images_translate_y_up)
#save_image_aug(AUGMENT_FOLDER,'translate_y_down',ID_list,images_translate_y_down)
#save_image_aug(AUGMENT_FOLDER,'sharp',ID_list,images_sharp)
#save_image_aug(AUGMENT_FOLDER,'lightness',ID_list,images_lightness)
#save_image_aug(AUGMENT_FOLDER,'translate_sharp',ID_list,images_translate_sharp)

#save_image_aug(FOLDER=AUGMENT_FOLDER,type_of_image='original',ID_list=ID_list,images=images)
#print('finish saving original')
save_image_aug(AUGMENT_FOLDER,'translate_y_up_down',ID_list_270,images_translate_y_up_down)
#----------------------------------------------------------------------------
# create the ID list to use in extract_feature.py
def create_ID_list(type_of_image,ID_list):
    all_ID = []
    for type in type_of_image:
        for ID in ID_list:
             single_ID = '{}_ID_{}'.format(type,ID)
             all_ID.append(single_ID)

    return all_ID

#type_of_image = ['original','translate_y_up','translate_y_down','sharp','lightness','translate_sharp']
#all_ID = creat_ID_list(type_of_image,ID_list)
# print(all_ID)
#----------------------------------------------------------------------------
def show_images(images, cols = 1, titles = None):
    """Display a list of images in a single figure with matplotlib.
    
    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.
    
    cols (Default = 1): Number of columns in figure (number of rows is 
                        set to np.ceil(n_images/float(cols))).
    
    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()

# show_images(pre_post_translate)
  