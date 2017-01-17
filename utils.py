"""
Some codes from https://github.com/Newmu/dcgan_code
"""
import math
import json
import pprint
from PIL import Image
import scipy.misc
import numpy as np
import math
import pdb
from skimage.transform import resize

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])


def create_mask(images):
    mask = [images >-1.][0]*1.
    return mask

def get_image(input_,sr_label_,ang_label_,image_size,is_crop=True):
    LF_wid = input_.shape[1]
    LF_hei = input_.shape[0]
    randx = np.random.randint(LF_wid - image_size)
    randy = np.random.randint(LF_hei - image_size)
    inputs = transform(input_,randx,randy,image_size)
    sr_gt = transform(sr_label_,randx,randy,image_size)
    ang_gt = transform(np.expand_dims(ang_label_,axis=-1),randx,randy,image_size)
    return np.concatenate((inputs,sr_gt,ang_gt),axis=2)

def get_image_test(input_):
    inputs = input_/255.0
    return inputs
"""
def get_image_test(input_,sr_label_,ang_label_):
    inputs = input_/255.0
    sr_gt = sr_label_
    ang_gt = np.expand_dims(ang_label_,axis=-1)
    return np.concatenate((inputs,sr_gt,ang_gt),axis=2)
"""
def imread(path):
	#image = Image.open(path)
	#image = image.convert('YCbCr')
	image = scipy.misc.imread(path).astype(np.float)
	return image

def rgb2ycbcr(image):
	Y = (0.257*image[:,:,0])+(0.504*image[:,:,1])+(0.098*image[:,:,2]) + 16.0
	return Y


def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def merge_images(images, size):
    return inverse_transform(images)

def inverse_normalize(images):
    batchnum  = images.shape[0]
    inv_ = np.zeros((images.shape[0],images.shape[1],images.shape[2],3)).astype(float)
    for batch in range(batchnum):
        y = images[batch,:,:,0]
        x = images[batch,:,:,1]
        z = np.ones((images.shape[1],images.shape[2])).astype(float)
        is_zero = (x == -1).astype(int)
        norm = np.sqrt(np.power(x,2)+np.power(y,2)+1.)
        yy = y/norm
        xx = x/norm
        zz = z/norm

        inv = np.dstack((yy,xx,zz))
        inv = (inv*2.0)+1.
        inv[is_zero ==1]= 0.0
        inv_[batch,:,:,:] = inv      
    return inv_




def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if images.shape[-1] == 3:
        img = np.zeros((h * size[0], w * size[1], 3))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx / size[1]
            img[j*h:j*h+h, i*w:i*w+w, :] = image
    else:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx / size[1]
            img[j*h:j*h+h, i*w:i*w+w] = np.squeeze(image)

        
    return img

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))


def center_crop(x, crop_h, crop_w=None, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w],
                               [resize_w, resize_w])
def transform_normal(image, npx, randx,randy,is_crop=True):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = random_crop(image, npx,randx,randy)
        #cropped_image = center_crop(image, npx)
    else:
        cropped_image = image
    mean = 1.0
    std = 0.05
    cropped_image = cropped_image * np.random.normal(mean,std)
    max_val = np.max(cropped_image)
    cropped_image = cropped_image /max_val
    #scipy.misc.imshow(cropped_image)
    #print('cropped image dim:',cropped_image.shape)
    #print('x:%d y:%d' % (randx,randy))
    return np.array(cropped_image)*2. -1.

def random_crop(x,randx,randy,npx):
    #npx =64
    return x[randy:randy+npx, randx:randx+npx,:]


def transform(image, randx,randy,image_size,is_crop=True):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = random_crop(image,randx,randy,image_size)
    else:
        cropped_image = image
    return np.array(cropped_image)/255.0


def inverse_transform(images):
    return (images+1.)/2.

def make_gif(images, fname, duration=2, true_image=False):
  import moviepy.editor as mpy

  def make_frame(t):
    try:
      x = images[int(len(images)/duration*t)]
    except:
      x = images[-1]

    if true_image:
      return x.astype(np.uint8)
    else:
      return ((x+1)/2*255).astype(np.uint8)

  clip = mpy.VideoClip(make_frame, duration=duration)
  clip.write_gif(fname, fps = len(images) / duration)
