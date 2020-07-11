from PIL import Image
import os

data_path = 'data/BSDS500/data/images'

train_dir = ['test', 'train']
val_dir = ['val']

for dir in train_dir:

    img_dir = os.path.join(data_path,dir)
    for img in os.listdir(img_dir):

        im = Image.open(os.path.join(img_dir,img))
        half_w, half_h = im.size[0]/2, im.size[1]/2
        left = half_w-90
        top = half_h-90
        right = half_w+90
        bottom = half_h+90
        im_crop = im.crop((left,top,right,bottom))