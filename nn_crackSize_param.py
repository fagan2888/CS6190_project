import pickle
import pandas as pd
import os, glob
import matplotlib.pyplot as plt
from scipy import optimize
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split, KFold

from keras.models import Sequential
from keras.layers import Dense, Activation

def get_data(main_dir, crack_size, param, BC, run, size):
	
	os.chdir(main_dir)
	os.chdir('data')
	data = pickle.load(open(crack_size+'_'+param+'_'+BC+'_'+run+'_'+size+'_clean.pickle', 'rb'))
	print(crack_size+'_'+param+'_'+BC+'_'+run+'_'+size+'_clean.pickle')
	
	if crack_size == '0-25':
		df_new = data.rename(index={2.25:2.0, 4.25:4.0, 6.25:6.0, 8.25:8.0, 10.25:10.0, 12.25:12.0, 14.25:14.0, \
								16.25:16.0, 18.25:18.0, 20.25:20.0, 22.25:22.0, 24.25:24.0, 26.25:26.0})
	elif crack_size == '0-45':
		df_new = data.rename(index={2.05:2.0, 4.05:4.0, 6.05:6.0, 8.05:8.0, 10.05:10.0, 12.05:12.0, 14.05:14.0, \
								16.05:16.0, 18.05:18.0, 20.05:20.0, 22.05:22.0, 24.05:24.0, 26.05:26.0})
	else:
		df_new = data

	return df_new
	
def adjust_fit_points(crack_size, fit_points):
	
	if crack_size == '0-25':
		fit_points = [x + 0.25 for x in fit_points]
	elif crack_size == '0-45':
		fit_points = [x + 0.05 for x in fit_points]
		
	return fit_points
	
def select_case(BC, crack_size, find_parameter):
	
	# get parameters to use in running specific case
	# INPUT:
	# BC: type of boundary condition 'Free' or 'Submodeling'
	# crack_size: size of crack '0-25','0-45','1-0','3-0'
	# find_parameter: RVE parameter to find 'd1' or 'd2'
	
	# OUTPUT:
	# fixed_parameter: RVE parameter fixed during case
	# runs: microstructure instantiation numbers
	# sizes: fixed parameter sizes corresponding to runs
	
	if (BC=='Free') and (crack_size=='0-25') and (find_parameter=='d1'):
		fixed_parameter = 'd2'
		
		runs =  [1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
		sizes = [14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14]
		
	elif (BC=='Free') and (crack_size=='0-45') and (find_parameter=='d1'):
		fixed_parameter = 'd2'
		
		runs =  [1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
		sizes = [14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14]
		
	elif (BC=='Free') and (crack_size=='1-0') and (find_parameter=='d1'):
		fixed_parameter = 'd2'
		
		runs =  [1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
		sizes = [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16]
		
	elif (BC=='Free') and (crack_size=='3-0') and (find_parameter=='d1'):
		fixed_parameter = 'd2'
		
		runs =  [1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
		sizes = [14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14]
		
	elif (BC=='Free') and (crack_size=='0-25') and (find_parameter=='d2'):
		fixed_parameter = 'd1'
		
		runs =  [1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
		sizes = [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12]
		
	elif (BC=='Free') and (crack_size=='0-45') and (find_parameter=='d2'):
		fixed_parameter = 'd1'
		
		runs =  [1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
		sizes = [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12]
		
	elif (BC=='Free') and (crack_size=='1-0') and (find_parameter=='d2'):
		fixed_parameter = 'd1'
		
		runs =  [1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
		sizes = [14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14]
		
	elif (BC=='Free') and (crack_size=='3-0') and (find_parameter=='d2'):
		fixed_parameter = 'd1'
		
		runs =  [1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
		sizes = [14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14]
	
	elif (BC=='Submodeling') and (crack_size=='0-25') and (find_parameter=='d1'):
		fixed_parameter = 'd2'
		
		runs =  [1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
		sizes = [14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14]
		
	elif (BC=='Submodeling') and (crack_size=='0-45') and (find_parameter=='d1'):
		fixed_parameter = 'd2'
		
		runs =  [1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
		sizes = [14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14]
		
	elif (BC=='Submodeling') and (crack_size=='1-0') and (find_parameter=='d1'):
		fixed_parameter = 'd2'
		
		runs =  [1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
		sizes = [14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14]
		
	elif (BC=='Submodeling') and (crack_size=='3-0') and (find_parameter=='d1'):
		fixed_parameter = 'd2'
		
		runs =  [1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
		sizes = [14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14]
	
	elif (BC=='Submodeling') and (crack_size=='0-25') and (find_parameter=='d2'):
		fixed_parameter = 'd1'
		
		runs =  [1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
		sizes = [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12]
		
	elif (BC=='Submodeling') and (crack_size=='0-45') and (find_parameter=='d2'):
		fixed_parameter = 'd1'
		
		runs =  [1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
		sizes = [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12]
		
	elif (BC=='Submodeling') and (crack_size=='1-0') and (find_parameter=='d2'):
		fixed_parameter = 'd1'
		
		runs =  [1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
		sizes = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
		
	elif (BC=='Submodeling') and (crack_size=='3-0') and (find_parameter=='d2'):
		fixed_parameter = 'd1'
		
		runs =  [1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
		sizes = [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12]
	
	return [fixed_parameter, runs, sizes]
	
def train_NN(num_nodes, activation, X_train, X_val, y_train, y_val, learning_rate, sigma_sq, batch_size):
	
	concat_data = np.concatenate((X_train,y_train), axis=1)
	
	N = batch_size
	D_in = X_train.shape[1]
	D_out = y_train.shape[1]
	
	y_out = tf.compat.v1.placeholder(tf.float32, shape=[D_out, N])
	X_in = tf.compat.v1.placeholder(tf.float32, shape=[D_in, N])
	
	# initialize posterior mean and variance
	zeros_initializer = tf.zeros_initializer()
	mu_1 = tf.Variable(zeros_initializer(shape=[num_nodes,D_in]), name='mu1')
	mu_2 = tf.Variable(zeros_initializer(shape=[num_nodes,num_nodes]), name='mu2')
	mu_3 = tf.Variable(zeros_initializer(shape=[D_out,num_nodes]), name='mu3')
	ones_initializer = tf.ones_initializer()
	rho_1 = tf.Variable(ones_initializer(shape=[num_nodes,D_in]), name='rho1')
	rho_2 = tf.Variable(ones_initializer(shape=[num_nodes,num_nodes]), name='rho2')
	rho_3 = tf.Variable(ones_initializer(shape=[D_out,num_nodes]), name='rho3')
	
	# initialize weights
	#norm_initializer = tf.initializers.random_normal(mean=0,stddev=1)
	#eps_1 = tf.Variable(norm_initializer(shape=[num_nodes,D_in]),name='eps1')
	#eps_2 = tf.Variable(norm_initializer(shape=[num_nodes,num_nodes]),name='eps2')
	#eps_3 = tf.Variable(norm_initializer(shape=[D_out,num_nodes]),name='eps3')
	eps_1 = tf.random.normal(shape=[num_nodes,D_in],mean=0,stddev=1)
	eps_2 = tf.random.normal(shape=[num_nodes,num_nodes],mean=0,stddev=1)
	eps_3 = tf.random.normal(shape=[D_out,num_nodes],mean=0,stddev=1)
	w_1 = mu_1 + tf.math.multiply(eps_1,tf.math.log(1+tf.math.exp(rho_1)))
	w_2 = mu_2 + tf.math.multiply(eps_2,tf.math.log(1+tf.math.exp(rho_2)))
	w_3 = mu_3 + tf.math.multiply(eps_3,tf.math.log(1+tf.math.exp(rho_3)))
	
	if activation == 'RELU':
		h_1 = tf.nn.relu(tf.matmul(w_1, X_in))
		h_2 = tf.nn.relu(tf.matmul(w_2, h_1))
		y_pred = tf.matmul(w_3, h_2)
	elif activation == 'tanh':
		h_1 = tf.nn.tanh(tf.matmul(w_1, X_in))
		h_2 = tf.nn.tanh(tf.matmul(w_2, h_1))
		y_pred = tf.matmul(w_3, h_2)
	else:
		raise ValueError('model not defined for activation given')
	
	loss_term1 = (1./2.)*(tf.reduce_sum(tf.math.log(1 + tf.math.exp(rho_1))**2, [0,1]) + tf.reduce_sum(mu_1**2, [0,1]))
	loss_term2 = (1./2.)*(tf.reduce_sum(tf.math.log(1 + tf.math.exp(rho_2))**2, [0,1]) + tf.reduce_sum(mu_2**2, [0,1]))
	loss_term3 = (1./2.)*(tf.reduce_sum(tf.math.log(1 + tf.math.exp(rho_3))**2, [0,1]) + tf.reduce_sum(mu_3**2, [0,1]))
	loss_term4 = -(1./2.)*(tf.reduce_sum(tf.math.log(tf.math.log(1 + tf.math.exp(rho_1))*2*np.pi*tf.math.exp(tf.constant(1.0))), [0,1]))
	loss_term5 = -(1./2.)*(tf.reduce_sum(tf.math.log(tf.math.log(1 + tf.math.exp(rho_2))*2*np.pi*tf.math.exp(tf.constant(1.0))), [0,1]))
	loss_term6 = -(1./2.)*(tf.reduce_sum(tf.math.log(tf.math.log(1 + tf.math.exp(rho_3))*2*np.pi*tf.math.exp(tf.constant(1.0))), [0,1]))
	sum_square_error = (1./2.)*(tf.reduce_sum((y_pred - y_out)**2/sigma_sq, [0,1]))
	loss = loss_term1 + loss_term2 + loss_term3 + loss_term4 + loss_term5 + loss_term6 + sum_square_error

	train_step = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=loss, var_list=[mu_1,mu_2,mu_3,rho_1,rho_2,rho_3])
	
	sess = tf.compat.v1.Session()
	sess.run(tf.compat.v1.global_variables_initializer())
	
	loss_plot_train = []
	SSE_plot_train = []
	loss_plot_val = []
	SSE_plot_val = []
	for i in range(5000):
		
		np.random.shuffle(concat_data)
		X_train_batch = concat_data[0:batch_size,0:D_in]
		y_train_batch = concat_data[0:batch_size,0:D_out]
				
		sess.run(train_step, feed_dict={X_in:X_train_batch.T, y_out:y_train_batch.T})
		
		mu = [sess.run(mu_1), sess.run(mu_2), sess.run(mu_3)]
		rho = [sess.run(rho_1), sess.run(rho_2), sess.run(rho_3)]
		w = [sess.run(w_1), sess.run(w_2), sess.run(w_3)]
		
		train_loss = sess.run(loss, feed_dict={X_in:X_train_batch.T, y_out:y_train_batch.T})
		train_SSE = sess.run(sum_square_error, feed_dict={X_in:X_train_batch.T, y_out:y_train_batch.T})
		
		loss_plot_train.append(train_loss)
		SSE_plot_train.append(train_SSE)
		
		'''
		val_loss, val_SSE = get_test_loss_SSE(num_nodes, activation, y_val, X_val, w, mu, rho, sigma_sq)
		
		loss_plot_val.append(val_loss)
		SSE_plot_val.append(val_SSE)
		print(i, train_loss, train_SSE, val_loss, val_SSE)
		'''
		print(i, train_loss, train_SSE)
	'''
	plt.figure()
	plt.plot(loss_plot_train, label='Train loss')
	plt.plot(loss_plot_val, label='Validation loss')
	plt.plot(SSE_plot_train, label='Train SSE')
	plt.plot(SSE_plot_val, label='Validation SSE')
	plt.legend()
	plt.show()
	
	plt.figure()
	plt.plot(loss_plot_train, label='Training loss')
	plt.legend()
	plt.show()
	'''
	#mu = [sess.run(mu_1), sess.run(mu_2), sess.run(mu_3)]
	#rho = [sess.run(rho_1), sess.run(rho_2), sess.run(rho_3)]
	#w = [sess.run(w_1), sess.run(w_2), sess.run(w_3)]
	
	return(train_loss, train_SSE, mu, rho, w)
	
def plot_results_raw(w, X, y, activation, X_vol, y_vol):
	
	N = len(y)
	D_in = X.shape[1]
	D_out = y.shape[1]
	
	y_out = tf.compat.v1.placeholder(tf.float32, shape=[D_out, N])
	X_in = tf.compat.v1.placeholder(tf.float32, shape=[D_in, N])
	
	w_1 = w[0]
	w_2 = w[1]
	w_3 = w[2]
	
	if activation == 'RELU':
		h_1 = tf.nn.relu(tf.matmul(w_1, X_in))
		h_2 = tf.nn.relu(tf.matmul(w_2, h_1))
		y_pred = tf.matmul(w_3, h_2)
	elif activation == 'tanh':
		h_1 = tf.nn.tanh(tf.matmul(w_1, X_in))
		h_2 = tf.nn.tanh(tf.matmul(w_2, h_1))
		y_pred = tf.matmul(w_3, h_2)
	else:
		raise ValueError('model not defined for activation given')
		
	sess = tf.compat.v1.Session()
	sess.run(tf.compat.v1.global_variables_initializer())
	
	for i in range(1):
		plt.figure()
		plt.plot(X_vol, X[i], '.', label='training data')
		plt.errorbar(x=y_vol, y=y[i], fmt='.', yerr=y[i]*0.01, label='actual data (1% errorbar)')
		plt.plot(y_vol, np.transpose(sess.run(y_pred, feed_dict={X_in:X.T, y_out:y.T}))[i], '.', label='predicted data')
		plt.legend()
		plt.show()
		
def plot_results_scatter(mu, rho, X, y, activation, X_vol, y_vol):
	
	N = len(y)
	D_in = X.shape[1]
	D_out = y.shape[1]
	
	y_out = tf.compat.v1.placeholder(tf.float32, shape=[D_out, N])
	X_in = tf.compat.v1.placeholder(tf.float32, shape=[D_in, N])
	
	mu_1 = mu[0]
	mu_2 = mu[1]
	mu_3 = mu[2]
	
	rho_1 = rho[0]
	rho_2 = rho[1]
	rho_3 = rho[2]
	
	eps_1 = tf.random.normal(shape=[num_nodes,D_in],mean=0,stddev=1)
	eps_2 = tf.random.normal(shape=[num_nodes,num_nodes],mean=0,stddev=1)
	eps_3 = tf.random.normal(shape=[D_out,num_nodes],mean=0,stddev=1)
	w_1 = mu_1 + tf.math.multiply(eps_1,tf.math.log(1+tf.math.exp(rho_1)))
	w_2 = mu_2 + tf.math.multiply(eps_2,tf.math.log(1+tf.math.exp(rho_2)))
	w_3 = mu_3 + tf.math.multiply(eps_3,tf.math.log(1+tf.math.exp(rho_3)))
	
	if activation == 'RELU':
		h_1 = tf.nn.relu(tf.matmul(w_1, X_in))
		h_2 = tf.nn.relu(tf.matmul(w_2, h_1))
		y_pred = tf.matmul(w_3, h_2)
	elif activation == 'tanh':
		h_1 = tf.nn.tanh(tf.matmul(w_1, X_in))
		h_2 = tf.nn.tanh(tf.matmul(w_2, h_1))
		y_pred = tf.matmul(w_3, h_2)
	else:
		raise ValueError('model not defined for activation given')
		
	sess = tf.compat.v1.Session()
	sess.run(tf.compat.v1.global_variables_initializer())
	
	for i in range(1):
		plt.figure()
		plt.plot(X_vol, X[i], '.', label='training data')
		plt.errorbar(x=y_vol, y=y[i], fmt='.', yerr=y[i]*0.01, label='actual data (1% errorbar)')
		for j in range(25):
			plt.plot(y_vol, np.transpose(sess.run(y_pred, feed_dict={X_in:X.T, y_out:y.T}))[i], 'g.')
		plt.plot(y_vol, np.transpose(sess.run(y_pred, feed_dict={X_in:X.T, y_out:y.T}))[i], 'g.', label='predicted data')
		plt.legend()
		plt.show()

	
def get_test_loss_SSE(num_nodes, activation, y_test, X_test, w, mu, rho, sigma_sq):
	
	w_1 = w[0]
	w_2 = w[1]
	w_3 = w[2]
	
	mu_1 = mu[0]
	mu_2 = mu[1]
	mu_3 = mu[2]
	
	rho_1 = rho[0]
	rho_2 = rho[1]
	rho_3 = rho[2]
	
	N_test = len(y_test)
	D_in_test = X_test.shape[1]
	D_out_test = y_test.shape[1]
	
	y_out_test = tf.compat.v1.placeholder(tf.float32, shape=[D_out_test, N_test])
	X_in_test = tf.compat.v1.placeholder(tf.float32, shape=[D_in_test, N_test])
	
	if activation == 'RELU':
		h_1_test = tf.nn.relu(tf.matmul(w_1, X_in_test))
		h_2_test = tf.nn.relu(tf.matmul(w_2, h_1_test))
		y_pred_test = tf.matmul(w_3, h_2_test)
	elif activation == 'tanh':
		h_1_test = tf.nn.tanh(tf.matmul(w_1, X_in_test))
		h_2_test = tf.nn.tanh(tf.matmul(w_2, h_1_test))
		y_pred_test = tf.matmul(w_3, h_2_test)
	else:
		raise ValueError('model not defined for activation given')
		
	loss_test1 = (1./2.)*(tf.reduce_sum(tf.math.log(1 + tf.math.exp(rho_1))**2, [0,1]) + tf.reduce_sum(mu_1**2, [0,1]))
	loss_test2 = (1./2.)*(tf.reduce_sum(tf.math.log(1 + tf.math.exp(rho_2))**2, [0,1]) + tf.reduce_sum(mu_2**2, [0,1]))
	loss_test3 = (1./2.)*(tf.reduce_sum(tf.math.log(1 + tf.math.exp(rho_3))**2, [0,1]) + tf.reduce_sum(mu_3**2, [0,1]))
	loss_test4 = -(1./2.)*(tf.reduce_sum(tf.math.log(tf.math.log(1 + tf.math.exp(rho_1))*2*np.pi*tf.math.exp(tf.constant(1.0))), [0,1]))
	loss_test5 = -(1./2.)*(tf.reduce_sum(tf.math.log(tf.math.log(1 + tf.math.exp(rho_2))*2*np.pi*tf.math.exp(tf.constant(1.0))), [0,1]))
	loss_test6 = -(1./2.)*(tf.reduce_sum(tf.math.log(tf.math.log(1 + tf.math.exp(rho_3))*2*np.pi*tf.math.exp(tf.constant(1.0))), [0,1]))
	sum_square_error_test = (1./2.)*(tf.reduce_sum((y_pred_test - y_out_test)**2/sigma_sq, [0,1]))
	loss_test = loss_test1 + loss_test2 + loss_test3 + loss_test4 + loss_test5 + loss_test6 + sum_square_error_test
	
	sess = tf.compat.v1.Session()
	sess.run(tf.compat.v1.global_variables_initializer())
	
	test_loss = sess.run(loss_test, feed_dict={X_in_test:X_test.T, y_out_test:y_test.T})
	test_SSE = sess.run(sum_square_error_test, feed_dict={X_in_test:X_test.T, y_out_test:y_test.T})
	
	return(test_loss, test_SSE)
	
def get_percent_error(percent_error, w, X, y, activation):
	
	N = len(y)
	D_in = X.shape[1]
	D_out = y.shape[1]
	
	y_out = tf.compat.v1.placeholder(tf.float32, shape=[D_out, N])
	X_in = tf.compat.v1.placeholder(tf.float32, shape=[D_in, N])
	
	w_1 = w[0]
	w_2 = w[1]
	w_3 = w[2]
	
	if activation == 'RELU':
		h_1 = tf.nn.relu(tf.matmul(w_1, X_in))
		h_2 = tf.nn.relu(tf.matmul(w_2, h_1))
		y_pred = tf.matmul(w_3, h_2)
	elif activation == 'tanh':
		h_1 = tf.nn.tanh(tf.matmul(w_1, X_in))
		h_2 = tf.nn.tanh(tf.matmul(w_2, h_1))
		y_pred = tf.matmul(w_3, h_2)
	else:
		raise ValueError('model not defined for activation given')
		
	sess = tf.compat.v1.Session()
	sess.run(tf.compat.v1.global_variables_initializer())
	
	pred = sess.run(y_pred, feed_dict={X_in:X.T, y_out:y.T})
	for i in range(N):
		for j in range(len(y[i])):
			percent_error.append(abs((pred[j,i]-y[i,j])/y[i,j])*100)
			
	return(percent_error)
	
def plot_histogram(percent_error, crack_size, param, BC):
	
	plt.figure()
	plt.hist(percent_error, bins=25, density=True)
	plt.title('a/g='+crack_size+', '+param+', '+BC)
	plt.show()

def compare_baseline(activation, y_test, X_test, w):
	
	w_1 = w[0]
	w_2 = w[1]
	w_3 = w[2]
	
	N_test = len(y_test)
	D_in_test = X_test.shape[1]
	D_out_test = y_test.shape[1]
	
	y_out_test = tf.compat.v1.placeholder(tf.float32, shape=[D_out_test, N_test])
	X_in_test = tf.compat.v1.placeholder(tf.float32, shape=[D_in_test, N_test])
	
	if activation == 'RELU':
		h_1_test = tf.nn.relu(tf.matmul(w_1, X_in_test))
		h_2_test = tf.nn.relu(tf.matmul(w_2, h_1_test))
		y_pred_test = tf.matmul(w_2, h_2_test)
	elif activation == 'tanh':
		h_1_test = tf.nn.tanh(tf.matmul(w_1, X_in_test))
		h_2_test = tf.nn.tanh(tf.matmul(w_2, h_1_test))
		y_pred_test = tf.matmul(w_3, h_2_test)
	else:
		raise ValueError('model not defined for activation given')
		
	baseline = tf.math.reduce_mean(tf.math.square(tf.math.subtract(y_out_test[-1,:], X_in_test[-1,:])),axis=[0])
	model_perf = tf.math.reduce_mean(tf.math.square(tf.math.subtract(y_out_test[-1,:], y_pred_test[-1,:])),axis=[0])
	
	test1 = tf.math.subtract(y_out_test[-1,:], y_pred_test[-1,:])
	test2 = tf.math.subtract(y_out_test[-1,:], X_in_test[-1,:])
	
	sess = tf.compat.v1.Session()
	sess.run(tf.compat.v1.global_variables_initializer())
	
	baseline_test = sess.run(baseline, feed_dict={X_in_test:X_test.T, y_out_test:y_test.T})
	model_perf_test = sess.run(model_perf, feed_dict={X_in_test:X_test.T, y_out_test:y_test.T})
	
	return(baseline_test, model_perf_test)
	
def test_keras(X_train, X_test, y_train, y_test, X_vol, y_vol):
	
	N = len(y_train)
	D_in = X_train.shape[1]
	D_out = y_train.shape[1]
	
	model = Sequential([
		Dense(10, input_shape=(D_in,)),
		Activation('relu'),
		Dense(10),
		Activation('relu'),
		Dense(4),
		Activation('linear'),
	])
	
	model.compile(optimizer='adam',loss='mse')
	model.fit(x=X_train, y=y_train, epochs=100)
	y_pred = model.predict(X_test)
	
	print(y_pred)
	
	for i in range(len(y_pred)):
		plt.figure()
		plt.plot(X_vol, X_test[i], '.', label='train')
		plt.plot(y_vol, y_pred[i], '.', label='predict')
		plt.plot(y_vol, y_test[i], '.', label='actual')
		plt.legend()
		plt.show()


if __name__ == '__main__':
	
	main_dir = os.getcwd()
	
	f = open('crackSize_param_crossVal.txt', 'a')
	
	X_vol = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]
	y_vol = [18.0, 20.0, 22.0, 24.0, 26.0]
	#clean_data(X_vol)
	
	num_nodes = 10
	activation = 'RELU'
	learning_rate = 1.0e-3
	sigma_sq = 1
	batch_size = 32
	BC = 'Submodeling'
	
	f.write('Batch size: '+str(batch_size)+'\n')
	f.write('Sigma squared: '+str(sigma_sq)+'\n')
	f.write('Layer nodes: '+str(num_nodes)+'\n')
	f.write('Learning rate: '+str(learning_rate)+'\n')
	
	run_list = [12, 2, 18, 9, 6, 21, 20, 3, 7, 5, 4, 1, 8, 10, 19, 15, 11, 22, 17, 16, 13, 14]
	#np.random.shuffle(run_list)
	print(run_list)
	train_1 = run_list[0:4]
	train_2 = run_list[4:8]
	train_3 = run_list[8:12]
	train_4 = run_list[12:16]
	test = run_list[16:]
	print(train_1, train_2, train_3, train_4, test)
	
	for crack_size in ['0-25','0-45','1-0','3-0']:
		for param in ['d1','d2']:
			
			f.write('Crack size: '+crack_size+'\n')
			f.write('RVE parameter: '+param+'\n')
			
			data_1 = pd.DataFrame()
			data_2 = pd.DataFrame()
			data_3 = pd.DataFrame()
			data_4 = pd.DataFrame()
			data_test = pd.DataFrame()
				
			fixed_parameter, runs, sizes = select_case(BC, crack_size, param)
				
			percent_error = []
			
			for i in range(len(sizes)):
				run = str(runs[i])
				size = str(sizes[i])
				
				if (int(run) in train_1):
					data_temp = get_data(main_dir, crack_size, param, BC, run, size)
					data_1 = pd.concat(objs=[data_1, data_temp], axis=1)
				elif (int(run) in train_2):
					data_temp = get_data(main_dir, crack_size, param, BC, run, size)
					data_2 = pd.concat(objs=[data_2, data_temp], axis=1)
				elif (int(run) in train_3):
					data_temp = get_data(main_dir, crack_size, param, BC, run, size)
					data_3 = pd.concat(objs=[data_3, data_temp], axis=1)
				elif (int(run) in train_4):
					data_temp = get_data(main_dir, crack_size, param, BC, run, size)
					data_4 = pd.concat(objs=[data_4, data_temp], axis=1)
				elif (int(run) in test):
					data_temp = get_data(main_dir, crack_size, param, BC, run, size)
					data_test = pd.concat(objs=[data_test, data_temp], axis=1)
				else:
					raise ValueError('Class missing')
			
			val_base = []		
			val_model = []		
			test_base = []		
			test_model = []		
			# cross validation 1
			print('Cross validation 1')
			f.write('Cross validation 1'+'\n')
			data_train = pd.concat(objs=[data_2, data_3, data_4], axis=1)
			data_val = data_1
			data_test = data_test
				
			X_train = np.asarray(data_train.loc[X_vol]).T
			y_train = np.asarray(data_train.loc[y_vol]).T
			X_val = np.asarray(data_val.loc[X_vol]).T
			y_val = np.asarray(data_val.loc[y_vol]).T
			X_test = np.asarray(data_test.loc[X_vol]).T
			y_test = np.asarray(data_test.loc[y_vol]).T
			
			train_loss, train_SSE, mu, rho, w = train_NN(num_nodes, activation, X_train, X_val, y_train, y_val, learning_rate, sigma_sq, batch_size)
			#plot_results_raw(w, X_train, y_train, activation, X_vol, y_vol)
			#plot_results_raw(w, X_test, y_test, activation, X_vol, y_vol)
			#plot_results_scatter(mu, rho, X_test, y_test, activation, X_vol, y_vol)
			
			baseline_val, model_perf_val = compare_baseline(activation, y_val, X_val, w)
			print('Baseline (validation):')
			print(baseline_val)
			val_base.append(baseline_val)
			print('Model performance (validation):')
			print(model_perf_val)
			val_model.append(model_perf_val)
			f.write('Baseline (validation): '+str(baseline_val)+'\n')
			f.write('Model performance (validation): '+str(model_perf_val)+'\n')
			
			baseline_test, model_perf_test = compare_baseline(activation, y_test, X_test, w)
			print('Baseline: (test)')
			print(baseline_test)
			test_base.append(baseline_test)
			print('Model performance: (test)')
			print(model_perf_test)
			test_model.append(model_perf_test)
			f.write('Baseline (test): '+str(baseline_test)+'\n')
			f.write('Model performance (test): '+str(model_perf_test)+'\n')
					
			# cross validation 2
			print('Cross validation 2')
			f.write('Cross validation 2'+'\n')
			data_train = pd.concat(objs=[data_1, data_3, data_4], axis=1)
			data_val = data_2
			data_test = data_test
				
			X_train = np.asarray(data_train.loc[X_vol]).T
			y_train = np.asarray(data_train.loc[y_vol]).T
			X_val = np.asarray(data_val.loc[X_vol]).T
			y_val = np.asarray(data_val.loc[y_vol]).T
			X_test = np.asarray(data_test.loc[X_vol]).T
			y_test = np.asarray(data_test.loc[y_vol]).T
			
			train_loss, train_SSE, mu, rho, w = train_NN(num_nodes, activation, X_train, X_val, y_train, y_val, learning_rate, sigma_sq, batch_size)
			#plot_results_raw(w, X_train, y_train, activation, X_vol, y_vol)
			#plot_results_raw(w, X_test, y_test, activation, X_vol, y_vol)
			#plot_results_scatter(mu, rho, X_test, y_test, activation, X_vol, y_vol)
			
			baseline_val, model_perf_val = compare_baseline(activation, y_val, X_val, w)
			print('Baseline (validation):')
			print(baseline_val)
			val_base.append(baseline_val)
			print('Model performance (validation):')
			print(model_perf_val)
			val_model.append(model_perf_val)
			f.write('Baseline (validation): '+str(baseline_val)+'\n')
			f.write('Model performance (validation): '+str(model_perf_val)+'\n')
			
			baseline_test, model_perf_test = compare_baseline(activation, y_test, X_test, w)
			print('Baseline: (test)')
			print(baseline_test)
			test_base.append(baseline_test)
			print('Model performance: (test)')
			print(model_perf_test)
			test_model.append(model_perf_test)
			f.write('Baseline (test): '+str(baseline_test)+'\n')
			f.write('Model performance (test): '+str(model_perf_test)+'\n')
					
			# cross validation 3
			print('Cross validation 3')
			f.write('Cross validation 3'+'\n')
			data_train = pd.concat(objs=[data_1, data_2, data_4], axis=1)
			data_val = data_3
			data_test = data_test
				
			X_train = np.asarray(data_train.loc[X_vol]).T
			y_train = np.asarray(data_train.loc[y_vol]).T
			X_val = np.asarray(data_val.loc[X_vol]).T
			y_val = np.asarray(data_val.loc[y_vol]).T
			X_test = np.asarray(data_test.loc[X_vol]).T
			y_test = np.asarray(data_test.loc[y_vol]).T
			
			train_loss, train_SSE, mu, rho, w = train_NN(num_nodes, activation, X_train, X_val, y_train, y_val, learning_rate, sigma_sq, batch_size)
			#plot_results_raw(w, X_train, y_train, activation, X_vol, y_vol)
			#plot_results_raw(w, X_test, y_test, activation, X_vol, y_vol)
			#plot_results_scatter(mu, rho, X_test, y_test, activation, X_vol, y_vol)
			
			baseline_val, model_perf_val = compare_baseline(activation, y_val, X_val, w)
			print('Baseline (validation):')
			print(baseline_val)
			val_base.append(baseline_val)
			print('Model performance (validation):')
			print(model_perf_val)
			val_model.append(model_perf_val)
			f.write('Baseline (validation): '+str(baseline_val)+'\n')
			f.write('Model performance (validation): '+str(model_perf_val)+'\n')
			
			baseline_test, model_perf_test = compare_baseline(activation, y_test, X_test, w)
			print('Baseline: (test)')
			print(baseline_test)
			test_base.append(baseline_test)
			print('Model performance: (test)')
			print(model_perf_test)
			test_model.append(model_perf_test)
			f.write('Baseline (test): '+str(baseline_test)+'\n')
			f.write('Model performance (test): '+str(model_perf_test)+'\n')
					
			# cross validation 4
			print('Cross validation 4')
			f.write('Cross validation 4'+'\n')
			data_train = pd.concat(objs=[data_1, data_2, data_3], axis=1)
			data_val = data_3
			data_test = data_test
				
			X_train = np.asarray(data_train.loc[X_vol]).T
			y_train = np.asarray(data_train.loc[y_vol]).T
			X_val = np.asarray(data_val.loc[X_vol]).T
			y_val = np.asarray(data_val.loc[y_vol]).T
			X_test = np.asarray(data_test.loc[X_vol]).T
			y_test = np.asarray(data_test.loc[y_vol]).T
			
			train_loss, train_SSE, mu, rho, w = train_NN(num_nodes, activation, X_train, X_val, y_train, y_val, learning_rate, sigma_sq, batch_size)
			#plot_results_raw(w, X_train, y_train, activation, X_vol, y_vol)
			#plot_results_raw(w, X_test, y_test, activation, X_vol, y_vol)
			#plot_results_scatter(mu, rho, X_test, y_test, activation, X_vol, y_vol)
			
			baseline_val, model_perf_val = compare_baseline(activation, y_val, X_val, w)
			print('Baseline (validation):')
			print(baseline_val)
			val_base.append(baseline_val)
			print('Model performance (validation):')
			print(model_perf_val)
			val_model.append(model_perf_val)
			f.write('Baseline (validation): '+str(baseline_val)+'\n')
			f.write('Model performance (validation): '+str(model_perf_val)+'\n')
			
			baseline_test, model_perf_test = compare_baseline(activation, y_test, X_test, w)
			print('Baseline: (test)')
			print(baseline_test)
			test_base.append(baseline_test)
			print('Model performance: (test)')
			print(model_perf_test)
			test_model.append(model_perf_test)
			f.write('Baseline (test): '+str(baseline_test)+'\n')
			f.write('Model performance (test): '+str(model_perf_test)+'\n')
			
			# average
			print('Baseline (validation, average):')
			print(np.mean(val_base))
			print('Model performance (validation, average):')
			print(np.mean(val_model))
			f.write('Baseline (validation, average): '+str(np.mean(val_base))+'\n')
			f.write('Model performance (validation, average): '+str(np.mean(val_model))+'\n')
			
			print('Baseline: (test, average)')
			print(np.mean(test_base))
			print('Model performance: (test, average)')
			print(np.mean(test_model))
			f.write('Baseline (test, average): '+str(np.mean(test_base))+'\n')
			f.write('Model performance (test, average): '+str(np.mean(test_model))+'\n')
			
			#percent_error = get_percent_error(percent_error, w, X_test, y_test, activation)
		
		#plot_histogram(percent_error, crack_size, param, BC)
		
	f.close()
					
					
					
					
			
