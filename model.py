import os
import time
from glob import glob
import numpy as np
from numpy import inf
import tensorflow as tf
import pdb
import scipy.io as sio
from load_data import * 
from utils import *
from ops import *
class LFSR(object):
	def __init__(self,sess,date,image_wid=64,image_hei=64,batch_size=32,dataset_name='default',checkpoint_dir=None):
		self.sess = sess
		self.batch_size = batch_size
		self.dataset_name = dataset_name
		self.checkpoint_dir = checkpoint_dir
		self.image_wid = image_wid
		self.image_hei = image_hei
		self.LF_wid = 552
		self.LF_hei = 383
		self.count = 0
		self.date = date
		self.build_model()
	def build_model(self):
		
		self.train_input1 = tf.placeholder(tf.float32,[self.batch_size,self.image_hei,self.image_wid,1],name ='train_input1')
		self.train_input2 = tf.placeholder(tf.float32,[self.batch_size,self.image_hei,self.image_wid,1],name ='train_input2')
		self.train_input3 = tf.placeholder(tf.float32,[self.batch_size,self.image_hei,self.image_wid,1],name ='train_input3')
		self.train_input4 = tf.placeholder(tf.float32,[self.batch_size,self.image_hei,self.image_wid,1],name ='train_input4')
		#self.train_input_2views = tf.placeholder(tf.float32,[self.batch_size,self.image_hei,self.image_wid,2],name ='train_input_2view')
		#self.train_input_4views = tf.placeholder(tf.float32,[self.batch_size,self.image_hei,self.image_wid,4],name ='train_input_4views')
		self.train_spa_gt1 = tf.placeholder(tf.float32,[self.batch_size,self.image_hei,self.image_wid,1],name ='train_spa_gt1')
		self.train_spa_gt2 = tf.placeholder(tf.float32,[self.batch_size,self.image_hei,self.image_wid,1],name ='train_spa_gt2')
		self.train_spa_gt3 = tf.placeholder(tf.float32,[self.batch_size,self.image_hei,self.image_wid,1],name ='train_spa_gt3')
		self.train_spa_gt4 = tf.placeholder(tf.float32,[self.batch_size,self.image_hei,self.image_wid,1],name ='train_spa_gt4')
		self.train_ang_gt = tf.placeholder(tf.float32,[self.batch_size,self.image_hei,self.image_wid,1],name ='train_ang_gt')
		self.output_spa1 = self.spa_net(self.train_input1)
		self.output_spa2 = self.spa_net(self.train_input2,reuse=True)
		self.output_spa3 = self.spa_net(self.train_input3,reuse=True)
		self.output_spa4 = self.spa_net(self.train_input4,reuse=True)

		self.ver_output = self.vertical_net(tf.concat(3,[self.output_spa1,self.output_spa2]))
		self.hor_output = self.horizontal_net(tf.concat(3,[self.output_spa1,self.output_spa2]))
		self.views_output = self.views_net(tf.concat(3,[self.output_spa1,self.output_spa2,self.output_spa3,self.output_spa4]))
		self.output_ver = self.shared_net(self.ver_output)
		self.output_hor = self.shared_net(self.hor_output,reuse=True )
		self.output_views = self.shared_net(self.views_output,reuse=True)
		
		self.loss_spa1 = tf.reduce_mean(tf.square(tf.sub(self.output_spa1,self.train_spa_gt1)))# MSE
		self.loss_spa2 = tf.reduce_mean(tf.square(tf.sub(self.output_spa2,self.train_spa_gt2)))# MSE
		self.loss_spa3 = tf.reduce_mean(tf.square(tf.sub(self.output_spa3,self.train_spa_gt3)))# MSE
		self.loss_spa4 = tf.reduce_mean(tf.square(tf.sub(self.output_spa4,self.train_spa_gt4)))# MSE
		self.loss_ver = tf.reduce_mean(tf.square(tf.sub(self.output_ver,self.train_ang_gt)))# MSE
		self.loss_hor = tf.reduce_mean(tf.square(tf.sub(self.output_hor,self.train_ang_gt)))# MSE
		self.loss_views = tf.reduce_mean(tf.square(tf.sub(self.output_views,self.train_ang_gt)))# MSE

		self.loss_fine_ver = self.loss_spa1 + self.loss_spa2+self.loss_ver
		self.loss_fine_hor = self.loss_spa1 + self.loss_spa2+self.loss_hor
		self.loss_fine_views = self.loss_spa1 + self.loss_spa2 + self.loss_spa3 + self.loss_spa4 +self.loss_views
		self.saver = tf.train.Saver(max_to_keep=1)

	def train(self,config):
		global_step1 = tf.Variable(0,name='global_step_train1',trainable=False)	
		global_step2 = tf.Variable(0,name='global_step_train2',trainable=False)	
		global_step3 = tf.Variable(0,name='global_step_train3',trainable=False)	

		train_optim_ver = tf.train.AdamOptimizer(config.learning_rate).minimize(self.loss_fine_ver,global_step=global_step1)
		train_optim_hor = tf.train.AdamOptimizer(config.learning_rate).minimize(self.loss_fine_hor,global_step=global_step2)
		train_optim_views = tf.train.AdamOptimizer(config.learning_rate).minimize(self.loss_fine_views,global_step=global_step3)

		t_vars = tf.trainable_variables()
	        self.var_list1 = [var for var in t_vars if 'first_two_' in var.name]
	        self.var_list2 = [var for var in t_vars if 'last' in var.name]
	        self.var_list3 = [var for var in t_vars if 'ver' in var.name]
	        self.var_list4 = [var for var in t_vars if 'hor' in var.name]
	        self.var_list5 = [var for var in t_vars if 'views' in var.name]
	        self.var_list6 = [var for var in t_vars if 'shared_' in var.name]
	        self.var_list7 = [var for var in t_vars if 'shread_' in var.name]

		tf.initialize_all_variables().run()

		if config.is_finetune: 
			# Initialize Spanet and load pretrained network
			tmp = self.var_list1 + self.var_list2
			self.saver = tf.train.Saver(var_list=tmp,max_to_keep=1)
			tf.initialize_variables(tmp).run()#load trained network
			if self.loadnet(self.checkpoint_dir,'spaSR'):  #Load Spatial SR network
				print('Load pretrained spatial network')
			else:
				print(' Load Fail!!')

			tmp = self.var_list3 + self.var_list4+self.var_list5 + self.var_list6 +self.var_list7
			self.saver = tf.train.Saver(var_list=tmp,max_to_keep=1)
			tf.initialize_variables(tmp).run()

			if self.loadnet(self.checkpoint_dir,'allviews'):  #Load Spatial SR network
				print('Load pretrained angular network')
			else:
				print(' Load Fail!!')
			self.saver = tf.train.Saver(max_to_keep=1)

		else:
			self.saver = tf.train.Saver(max_to_keep=1)
			if self.loadnet(self.checkpoint_dir,'finetune'):  #Load Spatial SR network
				print('Load pretrained angular network')
			else:
				print(' Load Fail!!')

		
		train_ver_input,train_hor_input,train_views_input,train_ver_sr_gt,train_hor_sr_gt,train_views_sr_gt,train_ver_ang_gt,train_hor_ang_gt,train_views_ang_gt = load_traindata()	
		[val_ver_input,val_hor_input,val_views_input,val_ver_sr_gt,val_hor_sr_gt,val_views_sr_gt,val_ver_ang_gt,val_hor_ang_gt,val_views_ang_gt] = load_valdata()	
		
		batch_idxs_views = train_views_input.shape[-1]/self.batch_size
		val_batch_idxs_views = val_views_input.shape[-1]/self.batch_size
		for epoch in xrange(config.epochs):
			rand_idx_ver = np.random.permutation(range(train_ver_input.shape[-1]))
			rand_idx_hor = np.random.permutation(range(train_hor_input.shape[-1]))
			rand_idx_views = np.random.permutation(range(train_views_input.shape[-1]))
			val_rand_idx_ver = np.random.permutation(range(val_ver_input.shape[-1]))
			val_rand_idx_hor = np.random.permutation(range(val_hor_input.shape[-1]))
			val_rand_idx_views = np.random.permutation(range(val_views_input.shape[-1]))

			train_spa_MSE = 0.0
			train_ang_MSE =0.0
			train_total_MSE=0.0 
			val_spa_MSE = 0.0
			val_ang_MSE =0.0
			val_total_MSE=0.0 

			for idx in xrange(0,batch_idxs_views):
				if epoch ==0:
					f_train_epoch = open(os.path.join("logs",self.date,'train_epoch.log'),'w')
					f_val = open(os.path.join("logs",self.date,'val.log'),'w')
				else:
					f_train_epoch = open(os.path.join("logs",self.date,'train_epoch.log'),'aw')
					f_val = open(os.path.join("logs",self.date,'val.log'),'aw')
				randview =np.random.permutation(range(3))
				for view in randview:
					if view ==0:
						batch_files = rand_idx_ver[idx*config.batch_size:(idx+1)*config.batch_size]	
						batches =[get_image(train_ver_input[0,batch],train_ver_sr_gt[0,batch],train_ver_ang_gt[0,batch],self.image_wid) for batch in batch_files]
						batches = np.array(batches).astype(np.float32)
						input1 = batches[:,:,:,0]
						input1 = np.expand_dims(input1,axis=-1)
						input2 = batches[:,:,:,1]
						input2 = np.expand_dims(input2,axis=-1)
						spa_gt1 = batches[:,:,:,2]
						spa_gt1 = np.expand_dims(spa_gt1,axis=-1)
						spa_gt2 = batches[:,:,:,3]
						spa_gt2 = np.expand_dims(spa_gt2,axis=-1)
						ang_gt = batches[:,:,:,4]
						ang_gt = np.expand_dims(ang_gt,axis=-1)
						
						_,total_MSE,spa1_MSE,spa2_MSE,ang_MSE = self.sess.run([train_optim_ver,self.loss_fine_ver,self.loss_spa1,self.loss_spa2,self.loss_ver],feed_dict={self.train_input1:input1,self.train_input2:input2,self.train_spa_gt1:spa_gt1,self.train_spa_gt2:spa_gt2,self.train_ang_gt:ang_gt})
						self.count +=1
						train_ang_MSE +=ang_MSE
						train_total_MSE +=total_MSE
						train_spa_MSE = (spa1_MSE + spa2_MSE)/2. + train_spa_MSE
					elif view==1:
						batch_files = rand_idx_hor[idx*config.batch_size:(idx+1)*config.batch_size]	
						batches =[get_image(train_hor_input[0,batch],train_hor_sr_gt[0,batch],train_hor_ang_gt[0,batch],self.image_wid) for batch in batch_files]
						batches = np.array(batches).astype(np.float32)
						
						input1 = batches[:,:,:,0]
						input1 = np.expand_dims(input1,axis=-1)
						input2 = batches[:,:,:,1]
						input2 = np.expand_dims(input2,axis=-1)
						spa_gt1 = batches[:,:,:,2]
						spa_gt1 = np.expand_dims(spa_gt1,axis=-1)
						spa_gt2 = batches[:,:,:,3]
						spa_gt2 = np.expand_dims(spa_gt2,axis=-1)
						ang_gt = batches[:,:,:,-1]
						ang_gt = np.expand_dims(ang_gt,axis=-1)
						_,total_MSE,spa1_MSE,spa2_MSE,ang_MSE = self.sess.run([train_optim_hor,self.loss_fine_hor,self.loss_spa1,self.loss_spa2,self.loss_hor],feed_dict={self.train_input1:input1,self.train_input2:input2,self.train_spa_gt1:spa_gt1,self.train_spa_gt2:spa_gt2,self.train_ang_gt:ang_gt})


						self.count +=1
						train_ang_MSE +=ang_MSE
						train_total_MSE +=total_MSE
						train_spa_MSE = (spa1_MSE + spa2_MSE)/2. + train_spa_MSE

					else:
						batch_files = rand_idx_views[idx*config.batch_size:(idx+1)*config.batch_size]	
						batches =[get_image(train_views_input[0,batch],train_views_sr_gt[0,batch],train_views_ang_gt[0,batch],self.image_wid) for batch in batch_files]
						batches = np.array(batches).astype(np.float32)
						input1 = batches[:,:,:,0]
						input1 = np.expand_dims(input1,axis=-1)
						input2 = batches[:,:,:,1]
						input2 = np.expand_dims(input2,axis=-1)
						input3 = batches[:,:,:,2]
						input3 = np.expand_dims(input3,axis=-1)
						input4 = batches[:,:,:,3]
						input4 = np.expand_dims(input4,axis=-1)
						spa_gt1 = batches[:,:,:,4]
						spa_gt1 = np.expand_dims(spa_gt1,axis=-1)
						spa_gt2 = batches[:,:,:,5]
						spa_gt2 = np.expand_dims(spa_gt2,axis=-1)
						spa_gt3 = batches[:,:,:,6]
						spa_gt3 = np.expand_dims(spa_gt3,axis=-1)
						spa_gt4 = batches[:,:,:,7]
						spa_gt4 = np.expand_dims(spa_gt4,axis=-1)

						ang_gt = batches[:,:,:,-1]
						ang_gt = np.expand_dims(ang_gt,axis=-1)
						_,total_MSE,spa1_MSE,spa2_MSE,spa3_MSE,spa4_MSE,ang_MSE = self.sess.run([train_optim_views,self.loss_fine_views,self.loss_spa1,self.loss_spa2,self.loss_spa3,self.loss_spa4,self.loss_views],feed_dict={self.train_input1:input1,self.train_input2:input2,self.train_input3:input3,self.train_input4:input4,self.train_spa_gt1:spa_gt1,self.train_spa_gt2:spa_gt2,self.train_spa_gt3:spa_gt3,self.train_spa_gt4:spa_gt4,self.train_ang_gt:ang_gt})


						self.count +=1
						train_ang_MSE +=ang_MSE
						train_spa_MSE =(spa1_MSE +spa2_MSE +spa3_MSE + spa4_MSE)/4. + train_spa_MSE
						train_total_MSE +=total_MSE



			print('Epoch train[%2d] total MSE: %.4f spa MSE: %.4f ang MSE: %.4f \n' %(epoch,train_total_MSE/(3*batch_idxs_views),train_spa_MSE/(3*batch_idxs_views),train_ang_MSE/(3*batch_idxs_views)))
			
			#Validation
			for val_idx in xrange(0,val_batch_idxs_views):		
				
				randview =np.random.permutation(range(3))
				for view in randview:
					if view ==0:
						batch_files = val_rand_idx_ver[val_idx*config.batch_size:(val_idx+1)*config.batch_size]	
						batches =[get_image(val_ver_input[0,batch],val_ver_sr_gt[0,batch],val_ver_ang_gt[0,batch],self.image_wid) for batch in batch_files]
						batches = np.array(batches).astype(np.float32)
						input1 = batches[:,:,:,0]
						input1 = np.expand_dims(input1,axis=-1)
						input2 = batches[:,:,:,1]
						input2 = np.expand_dims(input2,axis=-1)
						spa_gt1 = batches[:,:,:,2]
						spa_gt1 = np.expand_dims(spa_gt1,axis=-1)
						spa_gt2 = batches[:,:,:,3]
						spa_gt2 = np.expand_dims(spa_gt2,axis=-1)
						ang_gt = batches[:,:,:,4]
						ang_gt = np.expand_dims(ang_gt,axis=-1)
						total_MSE,spa1_MSE,spa2_MSE,ang_MSE = self.sess.run([self.loss_fine_ver,self.loss_spa1,self.loss_spa2,self.loss_ver],feed_dict={self.train_input1:input1,self.train_input2:input2,self.train_spa_gt1:spa_gt1,self.train_spa_gt2:spa_gt2,self.train_ang_gt:ang_gt})
						
						val_ang_MSE +=ang_MSE
						val_total_MSE +=total_MSE
						val_spa_MSE = spa1_MSE + spa2_MSE + train_spa_MSE

					elif view==1:
						batch_files = val_rand_idx_hor[val_idx*config.batch_size:(val_idx+1)*config.batch_size]	
						batches =[get_image(val_hor_input[0,batch],val_hor_sr_gt[0,batch],val_hor_ang_gt[0,batch],self.image_wid) for batch in batch_files]
						batches = np.array(batches).astype(np.float32)
						input1 = batches[:,:,:,0]
						input1 = np.expand_dims(input1,axis=-1)
						input2 = batches[:,:,:,1]
						input2 = np.expand_dims(input2,axis=-1)
						spa_gt1 = batches[:,:,:,2]
						spa_gt1 = np.expand_dims(spa_gt1,axis=-1)
						spa_gt2 = batches[:,:,:,3]
						spa_gt2 = np.expand_dims(spa_gt2,axis=-1)
						ang_gt = batches[:,:,:,-1]
						ang_gt = np.expand_dims(ang_gt,axis=-1)
						total_MSE,spa1_MSE,spa2_MSE,ang_MSE = self.sess.run([self.loss_fine_hor,self.loss_spa1,self.loss_spa2,self.loss_hor],feed_dict={self.train_input1:input1,self.train_input2:input2,self.train_spa_gt1:spa_gt1,self.train_spa_gt2:spa_gt2,self.train_ang_gt:ang_gt})
						val_ang_MSE +=ang_MSE
						val_total_MSE +=total_MSE
						val_spa_MSE = spa1_MSE + spa2_MSE + train_spa_MSE


					else:
						batch_files = val_rand_idx_views[val_idx*config.batch_size:(val_idx+1)*config.batch_size]	
						batches =[get_image(val_views_input[0,batch],val_views_sr_gt[0,batch],val_views_ang_gt[0,batch],self.image_wid) for batch in batch_files]
						batches = np.array(batches).astype(np.float32)
						input1 = batches[:,:,:,0]
						input1 = np.expand_dims(input1,axis=-1)
						input2 = batches[:,:,:,1]
						input2 = np.expand_dims(input2,axis=-1)
						input3 = batches[:,:,:,2]
						input3 = np.expand_dims(input3,axis=-1)
						input4 = batches[:,:,:,3]
						input4 = np.expand_dims(input4,axis=-1)
						spa_gt1 = batches[:,:,:,4]
						spa_gt1 = np.expand_dims(spa_gt1,axis=-1)
						spa_gt2 = batches[:,:,:,5]
						spa_gt2 = np.expand_dims(spa_gt2,axis=-1)
						spa_gt3 = batches[:,:,:,6]
						spa_gt3 = np.expand_dims(spa_gt3,axis=-1)
						spa_gt4 = batches[:,:,:,7]
						spa_gt4 = np.expand_dims(spa_gt4,axis=-1)

						ang_gt = batches[:,:,:,-1]
						ang_gt = np.expand_dims(ang_gt,axis=-1)
						total_MSE,spa1_MSE,spa2_MSE,spa3_MSE,spa4_MSE,ang_MSE = self.sess.run([self.loss_fine_views,self.loss_spa1,self.loss_spa2,self.loss_spa3,self.loss_spa4,self.loss_views],feed_dict={self.train_input1:input1,self.train_input2:input2,self.train_input3:input3,self.train_input4:input4,self.train_spa_gt1:spa_gt1,self.train_spa_gt2:spa_gt2,self.train_spa_gt3:spa_gt3,self.train_spa_gt4:spa_gt4,self.train_ang_gt:ang_gt})

						val_ang_MSE +=ang_MSE
						val_spa_MSE =spa1_MSE +spa2_MSE +spa3_MSE + spa4_MSE + train_spa_MSE
						val_total_MSE +=total_MSE


			print('Epoch val[%2d] total MSE: %.4f spa MSE: %.4f ang MSE: %.4f \n' %(epoch,val_total_MSE/(3*val_batch_idxs_views),val_spa_MSE/(3*val_batch_idxs_views),val_ang_MSE/(3*val_batch_idxs_views)))
			if np.mod(epoch,100) ==0:
				f_train_epoch.write('epoch %06d mean_total_MSE %.6f  mean_spa_MSE %.6f mean_ang_MSE %.6f\n' %(epoch,train_total_MSE/(3*batch_idxs_views),train_spa_MSE/(3*batch_idxs_views),train_ang_MSE/(3*batch_idxs_views)))
				f_train_epoch.close()
				f_val.write('epoch %06d mean_total_MSE %.6f  mean_spa_MSE %.6f mean_ang_MSE %.6f\n' %(epoch,val_total_MSE/(3*batch_idxs_views),val_spa_MSE/(3*batch_idxs_views),val_ang_MSE/(3*batch_idxs_views)))
				f_val.close()	
	                    	self.save(config.checkpoint_dir,0)

	def spa_net(self,input_,reuse=None):
		with tf.variable_scope(tf.get_variable_scope(),reuse=reuse):
			h1 = conv2d(input_,64,k_h=9,k_w=9,d_h=1,d_w=1,padding='SAME',name='first_two_1')
			h1 = tf.nn.relu(h1)
			h2 = conv2d(h1,32,k_h=1,k_w=1,d_h=1,d_w=1,padding='SAME',name='first_two_2')
			h2 = tf.nn.relu(h2)
			h3 = conv2d(h2,1,k_h=5,k_w=5,d_h=1,d_w=1,padding='SAME',name='last')
			return h3
	
	def vertical_net(self,ver_input):
		h1 = tf.nn.relu(conv2d(ver_input,64,k_h=9,k_w=9,d_h=1,d_w=1,padding='SAME',name='ver' ))
		return h1

	def horizontal_net(self,hor_input):
		h1 = tf.nn.relu(conv2d(hor_input,64,k_h=9,k_w=9,d_h=1,d_w=1,padding='SAME',name='hor' ))
		return h1
	def views_net(self,views_input):
		h1 = tf.nn.relu(conv2d(views_input,64,k_h=9,k_w=9,d_h=1,d_w=1,padding='SAME',name='views' ))
		return h1

	
	def shared_net(self,input_,reuse=None):
		with tf.variable_scope(tf.get_variable_scope(),reuse=reuse):
			h1 = tf.nn.relu(conv2d(input_,32,k_h=5,k_w=5,d_h=1,d_w=1,padding='SAME',name='shared_1' ))
			h2 = conv2d(h1,1,k_h=5,k_w=5,d_h=1,d_w=1,padding='SAME',name='shared_2')
			return h2

	def loadnet(self,checkpoint_dir,model_dir):
		#model_dir = '%s' %(self.dataset_name)
		checkpoint_dir = os.path.join(checkpoint_dir,model_dir)
		ckpt = 	tf.train.get_checkpoint_state(checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
			self.saver.restore(self.sess,os.path.join(checkpoint_dir,ckpt_name))	
			return True
		else:
			return False
	def save(self, checkpoint_dir, step):
        	model_name = "LFSR.model"
	        model_dir = "%s" % (self.dataset_name)
	        #model_dir = "%s_%s" % (self.dataset_name, self.batch_size)
        	checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

	        if not os.path.exists(checkpoint_dir):
        	    os.makedirs(checkpoint_dir)

	        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),global_step=step)

