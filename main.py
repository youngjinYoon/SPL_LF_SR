import os
import time
from model import LFSR
import tensorflow as tf
from utils import pp,get_image_test 
from scipy.ndimage import gaussian_filter
import scipy.misc
from load_data import * 
import scipy.io as sio
import numpy as np
import math
import time
import pdb

flags =tf.app.flags
flags.DEFINE_integer("epochs", 1000000,"Epoch to train")
flags.DEFINE_float("learning_rate",0.000001,"learning_rate for training")
flags.DEFINE_integer("image_wid",32,"cropping size")
flags.DEFINE_integer("image_hei",32,"cropping size")
flags.DEFINE_string("dataset","vertical","the name of training name")
flags.DEFINE_string("checkpoint_dir","checkpoint","the name to save the training network")
flags.DEFINE_string("output","output","The directory name to testset output image ")
flags.DEFINE_integer("batch_size",16,"batch_size")
flags.DEFINE_boolean("is_train",False,"True for training,False for testing")
flags.DEFINE_boolean("is_finetune",True,"True for training,False for testing")
flags.DEFINE_float("gpu",0.5,"GPU fraction per process")
FLAGS = flags.FLAGS

def main(_):
	pp.pprint(flags.FLAGS.__flags)
	date = time.strftime('%d%m')
	if not os.path.exists(FLAGS.checkpoint_dir):
        	os.makedirs(FLAGS.checkpoint_dir)
	if not os.path.exists(FLAGS.output):
        	os.makedirs(FLAGS.output)
    	if not os.path.exists(os.path.join('./logs',date)):
		os.makedirs(os.path.join('./logs',date))
    	gpu_config = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu)
    	with tf.Session(config=tf.ConfigProto(gpu_options=gpu_config)) as sess:
		if FLAGS.is_train:
        		SR = LFSR(sess,date, image_wid=FLAGS.image_wid,image_hei =FLAGS.image_hei ,batch_size=FLAGS.batch_size,\
	        	dataset_name=FLAGS.dataset, checkpoint_dir=FLAGS.checkpoint_dir)
			SR.train(FLAGS)
		else:
        		SR = LFSR(sess,date, image_wid=None ,image_hei = None ,batch_size=FLAGS.batch_size,\
	        	dataset_name=FLAGS.dataset, checkpoint_dir=FLAGS.checkpoint_dir)
			if SR.loadnet(FLAGS.checkpoint_dir,'finetune'):
				print('Load pretrained network \n')
			else:
				print('Fail to Load network \n')
	
			# Load validation data	
			#[val_ver_input,val_hor_input,val_views_input] = load_testdata()
			valdata = sio.loadmat('testmat/LF_real_test_vertical_finetune_ycbcr_input.mat')
			val_ver_input = valdata['low_Input']
			val_batch_idxs_views = val_ver_input.shape[-1]/FLAGS.batch_size


			


			######## For SR depth estimation ##########
			"""
			valdata = sio.loadmat('depth_vertical_finetune_ycbcr_input.mat')
			val_ver_input = valdata['LF_input']
			valdata = sio.loadmat('depth_horizontal_finetune_ycbcr_input.mat')
			val_hor_input = valdata['LF_input']
			valdata = sio.loadmat('depth_4views_finetune_ycbcr_input.mat')
			val_views_input = valdata['LF_input']
			valdata = sio.loadmat('depth_sr_finetune_ycbcr_input.mat')
			val_sr_input = valdata['LF_input']
			#val_batch_idxs_views = val_sr_input.shape[-1]/FLAGS.batch_size
			#val_batch_idxs_views = val_ver_input.shape[-1]/FLAGS.batch_size
			#val_batch_idxs_views = val_hor_input.shape[-1]/FLAGS.batch_size
			val_batch_idxs_views = val_views_input.shape[-1]/FLAGS.batch_size
			"""
			#############################################
			for val_idx in xrange(0,val_batch_idxs_views):		
				view =0
				if view ==0:
					batch_files = range(val_ver_input.shape[-1])[val_idx*FLAGS.batch_size:(val_idx+1)*FLAGS.batch_size]
					batches =[get_image_test(val_ver_input[0,batch]) for batch in batch_files]
					batches = np.array(batches).astype(np.float32)
					
					final_sr_output =np.zeros((batches.shape[1],batches.shape[2],3)).astype(np.float32)
					final_output =np.zeros((batches.shape[1],batches.shape[2],3)).astype(np.float32)
					for ch in range(3):
						input1 = batches[:,:,:,ch]
						input1 = np.expand_dims(input1,axis=-1)
						input2 = batches[:,:,:,ch+3]
						input2 = np.expand_dims(input2,axis=-1)
						
        					start_time = time.time()
						sr_output,output = SR.sess.run([SR.output_spa1,SR.output_ver],feed_dict={SR.train_input1:input1,SR.train_input2:input2})
						print('Processing time: %.6f \n' %(time.time()-start_time))	
						sr_output = np.squeeze(sr_output[0])
						final_sr_output[:,:,ch] = sr_output	
						output = np.squeeze(output[0])
						final_output[:,:,ch] = output	
					sio.savemat(os.path.join('test/2011/','ver_predict_%04d.mat' %val_idx),{'Predict':final_output})

				elif view==1:
					batch_files = range(val_hor_input.shape[-1])[val_idx*FLAGS.batch_size:(val_idx+1)*FLAGS.batch_size]	
					batches =[get_image_test(val_hor_input[0,batch]) for batch in batch_files]
					batches = np.array(batches).astype(np.float32)
					final_output =np.zeros((batches.shape[1],batches.shape[2],3)).astype(np.float32)
					for ch in range(3):
						input1 = batches[:,:,:,ch]
						input1 = np.expand_dims(input1,axis=-1)
						input2 = batches[:,:,:,ch+3]
						input2 = np.expand_dims(input2,axis=-1)
						output = SR.sess.run([SR.output_hor],feed_dict={SR.train_input1:input1,SR.train_input2:input2})
						output = np.squeeze(output[0])
						final_output[:,:,ch] = output	
					sio.savemat(os.path.join('depth','hor_predict_%04d.mat' %val_idx),{'Predict':final_output})


				elif view==2:
					batch_files = range(val_views_input.shape[-1])[val_idx*FLAGS.batch_size:(val_idx+1)*FLAGS.batch_size]	
					batches =[get_image_test(val_views_input[0,batch]) for batch in batch_files]
					batches = np.array(batches).astype(np.float32)
					final_output =np.zeros((batches.shape[1],batches.shape[2],3)).astype(np.float32)
					for ch in range(3):
						input1 = batches[:,:,:,ch]
						input1 = np.expand_dims(input1,axis=-1)
						input2 = batches[:,:,:,ch+3]
						input2 = np.expand_dims(input2,axis=-1)
						input3 = batches[:,:,:,ch+6]
						input3 = np.expand_dims(input3,axis=-1)
						input4 = batches[:,:,:,ch+9]
						input4 = np.expand_dims(input4,axis=-1)
						output = SR.sess.run([SR.output_views],feed_dict={SR.train_input1:input1,SR.train_input2:input2,SR.train_input3:input3,SR.train_input4:input4})
						output = np.squeeze(output[0])
						final_output[:,:,ch] = output	
					sio.savemat(os.path.join('depth','4views_predict_%04d.mat' %val_idx),{'Predict':final_output})

				else:
					batch_files = range(val_sr_input.shape[-1])[val_idx*FLAGS.batch_size:(val_idx+1)*FLAGS.batch_size]	
					batches =[get_image_test(val_sr_input[0,batch]) for batch in batch_files]
					batches = np.array(batches).astype(np.float32)
					final_output =np.zeros((batches.shape[1],batches.shape[2],3)).astype(np.float32)
					for ch in range(3):
						input1 = batches[:,:,:,ch]
						input1 = np.expand_dims(input1,axis=-1)
						output = SR.sess.run([SR.output_spa1],feed_dict={SR.train_input1:input1})
						output = np.squeeze(output[0])
						final_output[:,:,ch] = output	
					sio.savemat(os.path.join('depth','sr_predict_%04d.mat' %val_idx),{'Predict':final_output})


				
def ssim_exact(img1, img2, sd=1.5, C1=0.01**2, C2=0.03**2):

    mu1 = gaussian_filter(img1, sd)
    mu2 = gaussian_filter(img2, sd)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = gaussian_filter(img1 * img1, sd) - mu1_sq
    sigma2_sq = gaussian_filter(img2 * img2, sd) - mu2_sq
    sigma12 = gaussian_filter(img1 * img2, sd) - mu1_mu2

    ssim_num = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))

    ssim_den = ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    ssim_map = ssim_num / ssim_den
    return np.mean(ssim_map)

def psnr(img1,img2):
	mse = np.mean((img1-img2)**2)
	if mse ==0:
		return 100
	PIXEL_MAX = 255.0
	return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

if __name__=='__main__':
	tf.app.run()
