import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_batch():
	
	files = ['param_crossVal_batch8.txt','param_crossVal_batch16.txt','param_crossVal_batch32.txt','param_crossVal_batch64.txt','param_crossVal_batch128.txt','param_crossVal_batch256.txt']
	
	batch_label = [8,16,32,64,128,256]
	
	batch = []
	val_MSE = []
	test_MSE = []
	
	mean_data_val = {8:[], 16:[], 32:[], 64:[], 128:[], 256:[]}
	mean_data_test = {8:[], 16:[], 32:[], 64:[], 128:[], 256:[]}
	
	batch_size = 0
	for file_i in files:
		f = open(file_i, 'r')
		data = f.readlines()
		for line in data:
			if 'Batch size' in line:
				batch_size = float(line.split(' ')[-1].split('\n')[0])
			elif 'Model performance (validation)' in line:
				batch.append(batch_size)
				val_MSE.append(float(line.split(' ')[-1].split('\n')[0]))
				mean_data_val[batch_size].append(float(line.split(' ')[-1].split('\n')[0]))
			elif 'Model performance (test)' in line:
				test_MSE.append(float(line.split(' ')[-1].split('\n')[0]))
				mean_data_test[batch_size].append(float(line.split(' ')[-1].split('\n')[0]))
				
	means_val = [np.mean(mean_data_val[8]), np.mean(mean_data_val[16]), np.mean(mean_data_val[32]), np.mean(mean_data_val[64]), np.mean(mean_data_val[128]), np.mean(mean_data_val[256])]
	means_test = [np.mean(mean_data_test[8]), np.mean(mean_data_test[16]), np.mean(mean_data_test[32]), np.mean(mean_data_test[64]), np.mean(mean_data_test[128]), np.mean(mean_data_test[256])]

	plt.figure()
	plt.scatter(batch, val_MSE, label='Validation')
	plt.scatter(batch, test_MSE, label='Test')
	plt.plot(batch_label,means_val,'x-',label='Validation mean')
	plt.plot(batch_label,means_test,'x-',label='Test mean')
	plt.xlabel('Batch size')
	plt.ylabel('MSE')
	plt.legend()
	plt.xscale('log')
	plt.xticks(batch_label, batch_label)
	plt.show()
	
def plot_sigmaSq():
	
	files = ['param_crossVal_sigmaSq0.02.txt','param_crossVal_sigmaSq0.04.txt','param_crossVal_sigmaSq1.0.txt','param_crossVal_sigmaSq0.2.txt','param_crossVal_sigmaSq5.0.txt','param_crossVal_sigmaSq25.0.txt','param_crossVal_sigmaSq50.0.txt']
	
	sigma_sq = []
	val_MSE = []
	test_MSE = []
	
	mean_data_val = {0.02:[], 0.04:[], 0.2:[], 1.0:[], 5.0:[], 25.0:[], 50.0:[]}
	mean_data_test = {0.02:[], 0.04:[], 0.2:[], 1.0:[], 5.0:[], 25.0:[], 50.0:[]}
	
	sigma_label = [0.02, 0.04, 0.2, 1.0, 5.0, 25.0, 50.0]
	
	for file_i in files:
		f = open(file_i, 'r')
		data = f.readlines()
		for line in data:
			if 'Sigma squared' in line:
				sigma_sq_val = float(line.split(' ')[-1].split('\n')[0])
			elif 'Model performance (validation)' in line:
				sigma_sq.append(sigma_sq_val)
				val_MSE.append(float(line.split(' ')[-1].split('\n')[0]))
				mean_data_val[sigma_sq_val].append(float(line.split(' ')[-1].split('\n')[0]))
			elif 'Model performance (test)' in line:
				test_MSE.append(float(line.split(' ')[-1].split('\n')[0]))
				mean_data_test[sigma_sq_val].append(float(line.split(' ')[-1].split('\n')[0]))
				
	means_val = [np.mean(mean_data_val[0.02]), np.mean(mean_data_val[0.04]), np.mean(mean_data_val[0.2]), np.mean(mean_data_val[1.0]), np.mean(mean_data_val[5.0]), np.mean(mean_data_val[25.0]), np.mean(mean_data_val[50.0])]
	means_test = [np.mean(mean_data_test[0.02]), np.mean(mean_data_test[0.04]), np.mean(mean_data_test[0.2]), np.mean(mean_data_test[1.0]), np.mean(mean_data_test[5.0]), np.mean(mean_data_test[25.0]), np.mean(mean_data_test[50.0])]

	plt.figure()
	plt.scatter(sigma_sq, val_MSE, label='Validation')
	plt.scatter(sigma_sq, test_MSE, label='Test')
	plt.plot(sigma_label,means_val,'x-',label='Validation mean')
	plt.plot(sigma_label,means_test,'x-',label='Test mean')
	plt.xlabel(r'$\sigma^2$')
	plt.ylabel('MSE')
	plt.legend()
	plt.xscale('log')
	plt.xticks(sigma_label, sigma_label)
	plt.show()
	
def plot_numNodes():
	
	files = ['param_crossVal_numNodes5.txt','param_crossVal_numNodes10.txt','param_crossVal_numNodes25.txt','param_crossVal_numNodes50.txt']
	
	num_nodes = []
	val_MSE = []
	test_MSE = []
	
	mean_data_val = {5:[], 10:[], 25:[], 50:[]}
	mean_data_test = {5:[], 10:[], 25:[], 50:[]}
	
	numNodes_label = [5,10,25,50]
	
	for file_i in files:
		f = open(file_i, 'r')
		data = f.readlines()
		for line in data:
			if 'Layer nodes' in line:
				numNodes_val = float(line.split(' ')[-1].split('\n')[0])
			elif 'Model performance (validation)' in line:
				num_nodes.append(numNodes_val)
				val_MSE.append(float(line.split(' ')[-1].split('\n')[0]))
				mean_data_val[numNodes_val].append(float(line.split(' ')[-1].split('\n')[0]))
			elif 'Model performance (test)' in line:
				test_MSE.append(float(line.split(' ')[-1].split('\n')[0]))
				mean_data_test[numNodes_val].append(float(line.split(' ')[-1].split('\n')[0]))
				
	means_val = [np.mean(mean_data_val[5]), np.mean(mean_data_val[10]), np.mean(mean_data_val[25]), np.mean(mean_data_val[50])]
	means_test = [np.mean(mean_data_test[5]), np.mean(mean_data_test[10]), np.mean(mean_data_test[25]), np.mean(mean_data_test[50])]

	plt.figure()
	plt.scatter(num_nodes, val_MSE, label='Validation')
	plt.scatter(num_nodes, test_MSE, label='Test')
	plt.plot(numNodes_label,means_val,'x-',label='Validation mean')
	plt.plot(numNodes_label,means_test,'x-',label='Test mean')
	plt.xlabel('Number of nodes per layer')
	plt.ylabel('MSE')
	plt.legend()
	plt.xticks(numNodes_label, numNodes_label)
	plt.show()
	
def plot_results():
	
	files = ['param_crossVal_final.txt']
	
	val_base = []
	val_MSE = []
	test_base = []
	test_MSE = []
	
	for file_i in files:
		f = open(file_i, 'r')
		data = f.readlines()
		for line in data:
			if 'Baseline (validation)' in line:
				val_base.append(float(line.split(' ')[-1].split('\n')[0]))
			elif 'Model performance (validation)' in line:
				val_MSE.append(float(line.split(' ')[-1].split('\n')[0]))
			elif 'Baseline (test)' in line:
				test_base.append(float(line.split(' ')[-1].split('\n')[0]))
			elif 'Model performance (test)' in line:
				test_MSE.append(float(line.split(' ')[-1].split('\n')[0]))

	plt.figure()
	plt.plot([1,2,3,4], val_base[0:4], 'ro', label=r'Validation baseline')
	plt.plot([5,6,7,8], val_base[4:8], 'ro')
	plt.plot([1,2,3,4], val_MSE[0:4], 'rx', label=r'Validation MSE')
	plt.plot([5,6,7,8], val_MSE[4:8], 'rx')
	plt.plot([1,2,3,4], test_base[0:4], 'bo', label=r'Test baseline')
	plt.plot([5,6,7,8], test_base[4:8], 'bo')
	plt.plot([1,2,3,4], test_MSE[0:4], 'bx', label=r'Test MSE')
	plt.plot([5,6,7,8], test_MSE[4:8], 'bx')
	plt.xlabel('Cross-validation number')
	plt.ylabel('MSE')
	plt.legend()
	plt.ylim((-0.1,1))
	plt.show()

if __name__ == '__main__':
	
	#plot_batch()
	#plot_sigmaSq()
	#plot_numNodes()
	plot_results()
