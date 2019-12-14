import pickle
import pandas as pd
import os, glob
import matplotlib.pyplot as plt
from scipy import optimize
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

def get_data(main_dir, crack_size, param, BC, run, size):
	
	os.chdir(main_dir)
	os.chdir('data')
	data = pickle.load(open(crack_size+'_'+param+'_'+BC+'_'+run+'_'+size+'_clean.pickle', 'rb'))
	print(crack_size+'_'+param+'_'+BC+'_'+run+'_'+size+'_clean.pickle')

	return data
	
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
	
def train_NN(num_nodes, activation, y_train, X_train, learning_rate, sigma_sq):
	
	N = len(y_train)
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
	w_1 = mu_1 + tf.math.multiply(eps_1,rho_1)
	w_2 = mu_2 + tf.math.multiply(eps_2,rho_2)
	w_3 = mu_3 + tf.math.multiply(eps_3,rho_3)
	
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
	'''
	pred_raw = tf.math.greater(tf.math.sigmoid(y_out),0.5)
	pred = tf.dtypes.cast(pred_raw, dtype=tf.int32)
	diff = tf.math.abs(tf.math.subtract(tf.transpose(pred),tf.dtypes.cast(y_train, dtype=tf.int32)))
	acc = 1 - tf.reduce_sum(diff, [0,1])/N
	'''
	train_step = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=loss, var_list=[mu_1,mu_2,mu_3,rho_1,rho_2,rho_3])
	
	sess = tf.compat.v1.Session()
	sess.run(tf.compat.v1.global_variables_initializer())
	
	loss_plot_train = []
	SSE_plot_train = []
	loss_plot_test = []
	SSE_plot_test = []
	for i in range(1000):
		sess.run(train_step, feed_dict={X_in:X_train.T, y_out:y_train.T})
		
		temp_pred1 = sess.run(tf.math.reduce_max(y_pred, axis=[0,1]), feed_dict={X_in:X_train.T, y_out:y_train.T})
		temp_pred2 = sess.run(tf.math.reduce_min(y_pred, axis=[0,1]), feed_dict={X_in:X_train.T, y_out:y_train.T})
		temp_pred3 = sess.run(tf.math.reduce_mean(y_pred, axis=[0,1]), feed_dict={X_in:X_train.T, y_out:y_train.T})
		print(temp_pred1)
		print(temp_pred2)
		print(temp_pred3)
		input('...')
		
		mu = [sess.run(mu_1), sess.run(mu_2), sess.run(mu_3)]
		rho = [sess.run(rho_1), sess.run(rho_2), sess.run(rho_3)]
		w = [sess.run(w_1), sess.run(w_2), sess.run(w_3)]
		
		train_loss = sess.run(loss, feed_dict={X_in:X_train.T, y_out:y_train.T})
		train_SSE = sess.run(sum_square_error, feed_dict={X_in:X_train.T, y_out:y_train.T})
		
		'''
		test_loss, test_SSE = get_test_loss_SSE(num_nodes, activation, y_test, X_test, w, mu, rho, sigma_sq)
		
		loss_plot_train.append(train_loss)
		SSE_plot_train.append(train_SSE)
		loss_plot_test.append(test_loss)
		SSE_plot_test.append(test_SSE)
		print(i, train_loss, train_SSE, test_loss, test_SSE)
		'''
		print(i, train_loss, train_SSE)
	
	'''	
	plt.figure()
	plt.plot(loss_plot_train, label='Train loss')
	plt.plot(loss_plot_test, label='Test loss')
	plt.plot(SSE_plot_train, label='Train SSE')
	plt.plot(SSE_plot_test, label='Test SSE')
	plt.legend()
	plt.show()

	plt.figure()
	plt.plot(SSE_plot_train, label='Train SSE')
	plt.plot(SSE_plot_test, label='Test SSE')
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
	w_1 = mu_1 + tf.math.multiply(eps_1,rho_1)
	w_2 = mu_2 + tf.math.multiply(eps_2,rho_2)
	w_3 = mu_3 + tf.math.multiply(eps_3,rho_3)
	
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


if __name__ == '__main__':
	
	crack_size = '0-25'
	param = 'd1'
	BC = 'Free'
	
	main_dir = os.getcwd()
	
	num_nodes = 10
	activation = 'RELU'
	learning_rate = 1.0e-1
	train_SSE = 5
	sigma_sq = 1
	
	data = pd.DataFrame()
	
	for crack_size in ['0-25','0-45','1-0','3-0']:
		for param in ['d1','d2']:
			for BC in ['Free','Submodeling']:
				
				#data = pd.DataFrame()
				
				fixed_parameter, runs, sizes = select_case(BC, crack_size, param)
				X_vol = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]
				y_vol = [18.0, 20.0, 22.0, 24.0, 26.0]
				if (param=='d1'):
					X_vol = adjust_fit_points(crack_size, X_vol)
					y_vol = adjust_fit_points(crack_size, y_vol)
					
				percent_error = []
				
				for i in range(len(sizes)):
					run = str(runs[i])
					size = str(sizes[i])
					
					data1 = get_data(main_dir, crack_size, param, BC, run, size)
					data = pd.concat(objs=[data, data1], axis=1)
					
					print(crack_size)
					print(data1)
					
				'''
				data_X = np.asarray(data.loc[X_vol])
				data_y = np.asarray(data.loc[y_vol])
				X_train, X_test, y_train, y_test = train_test_split(data_X.T, data_y.T, test_size=0.2, random_state=42)
					
				#while (train_SSE>1):
				train_loss, train_SSE, mu, rho, w = train_NN(num_nodes, activation, y_train, X_train, learning_rate, sigma_sq)
				plot_results_raw(w, X_train, y_train, activation, X_vol, y_vol)
				plot_results_raw(w, X_test, y_test, activation, X_vol, y_vol)
				plot_results_scatter(mu, rho, X_test, y_test, activation, X_vol, y_vol)
				
				baseline_test, model_perf_test = compare_baseline(activation, y_test, X_test, w)
				print('Baseline:')
				print(baseline_test)
				print('Model performance:')
				print(model_perf_test)
				
				percent_error = get_percent_error(percent_error, w, X_test, y_test, activation)
				train_SSE = 5
				
				plot_histogram(percent_error, crack_size, param, BC)
				'''
	print(data)
					
					
					
					
			
