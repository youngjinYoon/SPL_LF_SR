import os
import numpy as np
import scipy.io as sio
import pdb
def load_traindata():
	# Load training data
	print('Load training data \n')
	################## Vertical ###################
	traindata = sio.loadmat('/research2/SPL/real/fine-tune/traindata/LF_real_train_vertical_finetune_input.mat')
	train_input_vertical = traindata['LF_input']
	
	traindata = sio.loadmat('/research2/SPL/real/fine-tune/traindata/LF_real_train_vertical_finetune_SR_gt.mat')
	train_vertical_sr_gt = traindata['LF_SR']
	
	traindata = sio.loadmat('/research2/SPL/real/fine-tune/traindata/LF_real_train_vertical_finetune_ANG_gt.mat')
	train_vertical_ang_gt = traindata['LF_ANG']
	################## Horizontal ###################
	traindata = sio.loadmat('/research2/SPL/real/fine-tune/traindata/LF_real_train_horizontal_finetune_input.mat')
	train_input_horizontal = traindata['LF_input']
	
	traindata = sio.loadmat('/research2/SPL/real/fine-tune/traindata/LF_real_train_horizontal_finetune_SR_gt.mat')
	train_horizontal_sr_gt = traindata['LF_SR']
	
	traindata = sio.loadmat('/research2/SPL/real/fine-tune/traindata/LF_real_train_horizontal_finetune_ANG_gt.mat')
	train_horizontal_ang_gt = traindata['LF_ANG']

	################## 4 views ###################

	traindata = sio.loadmat('/research2/SPL/real/fine-tune/traindata/LF_real_train_finetune_4views_input_1.mat')
	train_input_4views1 = traindata['LF_input']
	traindata = sio.loadmat('/research2/SPL/real/fine-tune/traindata/LF_real_train_finetune_4views_input_2.mat')
	train_input_4views2 = traindata['LF_input']
	train_input_4views= np.concatenate([train_input_4views1,train_input_4views2],axis=-1)
	
	traindata = sio.loadmat('/research2/SPL/real/fine-tune/traindata/LF_real_train_finetune_4views_SR_gt1.mat')
	train_gt_4views1 = traindata['LF_SR']
	traindata = sio.loadmat('/research2/SPL/real/fine-tune/traindata/LF_real_train_finetune_4views_SR_gt2.mat')
	train_gt_4views2 = traindata['LF_SR']
	train_4views_sr_gt= np.concatenate([train_gt_4views1,train_gt_4views2],axis=-1)
	
	traindata = sio.loadmat('/research2/SPL/real/fine-tune/traindata/LF_real_train_finetune_4views_ANG_gt1.mat')
	train_gt_4views1 = traindata['LF_ANG']
	traindata = sio.loadmat('/research2/SPL/real/fine-tune/traindata/LF_real_train_finetune_4views_ANG_gt2.mat')
	train_gt_4views2 = traindata['LF_ANG']
	train_4views_ang_gt= np.concatenate([train_gt_4views1,train_gt_4views2],axis=-1)


	return train_input_vertical,train_input_horizontal,train_input_4views,train_vertical_sr_gt,train_horizontal_sr_gt,train_4views_sr_gt,train_vertical_ang_gt,train_horizontal_ang_gt,train_4views_ang_gt

def load_valdata():
	valdata = sio.loadmat('/research2/SPL/real/fine-tune/traindata/LF_real_val_vertical_finetune_input.mat')
	val_input_vertical = valdata['LF_input']
	valdata = sio.loadmat('/research2/SPL/real/fine-tune/traindata/LF_real_val_vertical_finetune_SR_gt.mat')
	val_vertical_sr_gt = valdata['LF_SR']
	valdata = sio.loadmat('/research2/SPL/real/fine-tune/traindata/LF_real_val_vertical_finetune_ANG_gt.mat')
	val_vertical_ang_gt = valdata['LF_ANG']

	valdata = sio.loadmat('/research2/SPL/real/fine-tune/traindata/LF_real_val_horizontal_finetune_input.mat')
	val_input_horizontal = valdata['LF_input']
	valdata = sio.loadmat('/research2/SPL/real/fine-tune/traindata/LF_real_val_horizontal_finetune_SR_gt.mat')
	val_horizontal_sr_gt = valdata['LF_SR']
	valdata = sio.loadmat('/research2/SPL/real/fine-tune/traindata/LF_real_val_horizontal_finetune_ANG_gt.mat')
	val_horizontal_ang_gt = valdata['LF_ANG']

	valdata = sio.loadmat('/research2/SPL/real/fine-tune/traindata/LF_real_val_4views_finetune_input_1.mat')
	val_input_4views1 = valdata['LF_input']
	valdata = sio.loadmat('/research2/SPL/real/fine-tune/traindata/LF_real_val_4views_finetune_input_2.mat')
	val_input_4views2 = valdata['LF_input']
	valdata = sio.loadmat('/research2/SPL/real/fine-tune/traindata/LF_real_val_4views_finetune_input_3.mat')
	val_input_4views3 = valdata['LF_input']
	val_input_4views = np.concatenate([val_input_4views1,val_input_4views2,val_input_4views3],axis=-1)

	valdata = sio.loadmat('/research2/SPL/real/fine-tune/traindata/LF_real_val_4views_finetune_SR_gt1.mat')
	val_4views_sr_gt1 = valdata['LF_SR']
	valdata = sio.loadmat('/research2/SPL/real/fine-tune/traindata/LF_real_val_4views_finetune_SR_gt2.mat')
	val_4views_sr_gt2 = valdata['LF_SR']
	valdata = sio.loadmat('/research2/SPL/real/fine-tune/traindata/LF_real_val_4views_finetune_SR_gt3.mat')
	val_4views_sr_gt3 = valdata['LF_SR']
	val_4views_sr_gt = np.concatenate([val_4views_sr_gt1,val_4views_sr_gt2,val_4views_sr_gt3],axis=-1)

	valdata = sio.loadmat('/research2/SPL/real/fine-tune/traindata/LF_real_val_4views_finetune_ANG_gt1.mat')
	val_4views_ang_gt1 = valdata['LF_ANG']
	valdata = sio.loadmat('/research2/SPL/real/fine-tune/traindata/LF_real_val_4views_finetune_ANG_gt2.mat')
	val_4views_ang_gt2 = valdata['LF_ANG']
	valdata = sio.loadmat('/research2/SPL/real/fine-tune/traindata/LF_real_val_4views_finetune_ANG_gt3.mat')
	val_4views_ang_gt3 = valdata['LF_ANG']
	val_4views_ang_gt = np.concatenate([val_4views_ang_gt1,val_4views_ang_gt2,val_4views_ang_gt3],axis=-1)

	return val_input_vertical,val_input_horizontal,val_input_4views,val_vertical_sr_gt,val_horizontal_sr_gt,val_4views_sr_gt,val_vertical_ang_gt,val_horizontal_ang_gt,val_4views_ang_gt

def load_testdata():
	testdata = sio.loadmat('/research2/SPL/real/fine-tune/traindata/LF_real_test_vertical_finetune_ycbcr_input.mat')
	test_input_vertical = testdata['low_Input']

	testdata = sio.loadmat('/research2/SPL/real/fine-tune/traindata/LF_real_val_horizontal_finetune_ycbcr_input.mat')
	test_input_horizontal = testdata['LF_input']

	testdata = sio.loadmat('/research2/SPL/real/fine-tune/traindata/LF_real_test_4views_finetune_ycbcr_input.mat')
	test_input_4views = testdata['low_Input']

	return test_input_vertical,test_input_horizontal,test_input_4views

