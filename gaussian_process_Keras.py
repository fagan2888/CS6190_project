import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.model_selection import train_test_split
from prep_data import clean_data
import pandas as pd

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

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
	
def adjust_fit_points(crack_size, fit_points):
	
	if crack_size == '0-25':
		fit_points = [x + 0.25 for x in fit_points]
	elif crack_size == '0-45':
		fit_points = [x + 0.05 for x in fit_points]
		
	return fit_points
	
def test_keras(X_train, X_test, y_train, y_test, X_vol, y_vol):
	
	N = len(y_train)
	D_in = X_train.shape[1]
	D_out = y_train.shape[1]
	
	model = Sequential([
		Dense(10, input_shape=(D_in,)),
		Activation('relu'),
		Dense(10),
		Activation('relu'),
		Dense(D_out),
		Activation('linear'),
	])
	
	model.compile(optimizer='adam',loss='mse')
	model.fit(x=X_train, y=y_train, epochs=100)
	y_pred = model.predict(X_test)
	
	print(y_pred)
	
	for i in range(5):
		plt.figure()
		plt.plot(X_vol, X_test[i], '.', label='train')
		plt.plot(y_vol, y_pred[i], '.', label='predict')
		plt.plot(y_vol, y_test[i], '.', label='actual')
		plt.legend()
		plt.show()
		
	baseline = np.array(y_test)
	baseline[:,0] = X_test[:,-1]
	baseline[:,1] = X_test[:,-1]
	baseline[:,2] = X_test[:,-1]
	baseline[:,3] = X_test[:,-1]
	baseline[:,4] = X_test[:,-1]
	print(baseline)
	print(y_test.shape)
	
	print('Test MSE')
	print(np.mean((y_test-y_pred)**2))
	print('Baseline MSE')
	print(np.mean((y_test-baseline)**2))
	input('Enter to continue...')

if __name__ == '__main__':
	
	main_dir = os.getcwd()
	
	#f = open('temp.txt', 'a')
	
	X_vol = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]
	y_vol = [18.0, 20.0, 22.0, 24.0, 26.0]
	#clean_data(X_vol)
	
	run_list = [12, 2, 18, 9, 6, 21, 20, 3, 7, 5, 4, 1, 8, 10, 19, 15, 11, 22, 17, 16, 13, 14]
	#np.random.shuffle(run_list)
	print(run_list)
	train_1 = run_list[0:4]
	train_2 = run_list[4:8]
	train_3 = run_list[8:12]
	train_4 = run_list[12:16]
	test = run_list[16:]
	print(train_1, train_2, train_3, train_4, test)
	
	BC = 'Submodeling'
	
	for param in ['d1','d2']:
		
		#f.write('RVE parameter: '+param+'\n')
		
		for crack_size in ['0-25','0-45','1-0','3-0']:
			
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
		#f.write('Cross validation 1'+'\n')
		data_train = pd.concat(objs=[data_2, data_3, data_4], axis=1)
		data_val = data_1
		data_test = data_test
			
		X_train = np.asarray(data_train.loc[X_vol]).T
		y_train = np.asarray(data_train.loc[y_vol]).T
		X_val = np.asarray(data_val.loc[X_vol]).T
		y_val = np.asarray(data_val.loc[y_vol]).T
		X_test = np.asarray(data_test.loc[X_vol]).T
		y_test = np.asarray(data_test.loc[y_vol]).T
		
		D_in = X_train.shape[1]
		D_out = y_train.shape[1]
		batch_size = 512
		
		concat_data = np.concatenate((X_train,y_train), axis=1)
		np.random.shuffle(concat_data)
		X_train_batch = concat_data[0:batch_size,0:D_in]
		y_train_batch = concat_data[0:batch_size,0:D_out]
		
		kernel = RBF(10, (1e-2, 1e2))
		gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
		gp.fit(X_train_batch, y_train_batch)
		
		y_pred, sigma = gp.predict(X_test, return_std=True)
		
		plt.figure()
		plt.xlabel(r'Volume $d_1$')
		plt.ylabel('Normalized J-integral')
		plt.plot(X_vol, X_test[1], 'r.', label='Input data')
		plt.errorbar(x=y_vol, y=y_test[1], fmt='g.', yerr=y_test[1]*0.1, label='Actual data (5% errorbar)')
		plt.plot(y_vol, y_pred[1], 'b.', label='Predicted data')
		plt.legend()
		plt.show()
		
		baseline = np.array(y_test)
		baseline[:,0] = X_test[:,-1]
		baseline[:,1] = X_test[:,-1]
		baseline[:,2] = X_test[:,-1]
		baseline[:,3] = X_test[:,-1]
		baseline[:,4] = X_test[:,-1]
		print(baseline)
		print(y_test.shape)
		
		print('Test MSE')
		print(np.mean((y_test-y_pred)**2))
		print('Baseline MSE')
		print(np.mean((y_test-baseline)**2))
		
		test_keras(X_train, X_test, y_train, y_test, X_vol, y_vol)
		
