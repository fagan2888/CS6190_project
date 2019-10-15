import pickle
import pandas as pd
import os, glob
import matplotlib.pyplot as plt
from scipy import optimize
import numpy as np

def get_data(crack_size, param, BC, run, size):
	
	os.chdir('data')
	data = pickle.load(open(crack_size+'_'+param+'_'+BC+'_'+run+'_'+size+'.pickle', 'rb'))

	return data
	
def exp_function_1(x,a,b,c):
	return a+b*np.exp(-1.4*x)
	
def exp_function_2(x,a,b,c):
	return a+b*np.exp(-0.9*x)
	
def exp_function_3(x,a,b,c):
	return a+b*np.exp(-0.4*x)
	
def fit_data_1(x_fit, y_fit):
	
	try:
		bounds = ([-np.inf,np.inf,-np.inf],[-np.inf,np.inf,0])
		params, params_covariance = optimize.curve_fit(exp_function_1, x_fit, y_fit, p0=(y_fit[-1],1.,-1.))
	except:
		params = 'failed'
		plt.figure()
		plt.scatter(x_fit, y_fit)
		plt.show()
		raise ValueError('Fit failed to converge')
		
	return params
	
def fit_data_2(x_fit, y_fit):
	
	try:
		bounds = ([-np.inf,np.inf,-np.inf],[-np.inf,np.inf,0])
		params, params_covariance = optimize.curve_fit(exp_function_2, x_fit, y_fit, p0=(y_fit[-1],1.,-1.))
	except:
		params = 'failed'
		plt.figure()
		plt.scatter(x_fit, y_fit)
		plt.show()
		raise ValueError('Fit failed to converge')
		
	return params
	
def fit_data_3(x_fit, y_fit):
	
	try:
		bounds = ([-np.inf,np.inf,-np.inf],[-np.inf,np.inf,0])
		params, params_covariance = optimize.curve_fit(exp_function_3, x_fit, y_fit, p0=(y_fit[-1],1.,-1.))
	except:
		params = 'failed'
		plt.figure()
		plt.scatter(x_fit, y_fit)
		plt.show()
		raise ValueError('Fit failed to converge')
		
	return params
	
def fit_data_c(x_fit, y_fit, c):
	
	try:
		eps = abs(c)*0.00001
		bounds = ([-np.inf,np.inf,c-eps],[-np.inf,np.inf,c+eps])
		params, params_covariance = optimize.curve_fit(exp_function, x_fit, y_fit, p0=(y_fit[-1],1.,-1.))
		#print(abs(c-params[2])/c)
		if (params[2]>c+eps) or (params[2]<c-eps):
			print('Limits: '+str([c-eps,c+eps]))
			print('C:'+str(params[2]))
			raise ValueError('Parameter out of bounds')
	except:
		params = 'failed'
		plt.figure()
		plt.scatter(x_fit, y_fit)
		plt.show()
		raise ValueError('Fit failed to converge')
		
	return params
	
def adjust_fit_points(crack_size, fit_points):
	
	if crack_size == '0-25':
		fit_points = [x + 0.25 for x in fit_points]
	elif crack_size == '0-45':
		fit_points = [x + 0.05 for x in fit_points]
		
	return fit_points
	
	
if __name__ == '__main__':
	
	'''
	crack_size = '0-45'
	param = 'd1'
	BC = 'Submodeling'
	run = '4'
	size = '14'
	'''
	
	c = []
	
	os.chdir('data')
	
	for file_i in glob.glob('*.pickle'):
		
		print(file_i)
		
		crack_size = file_i.split('_')[0]
		param = file_i.split('_')[1]
	
		fit_points = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]
		fit_points = adjust_fit_points(crack_size, fit_points)
		
		#df = get_data(crack_size, param, BC, run, size)
		df = pickle.load(open(file_i, 'rb'))
		
		for i in range(0,len(df.columns)):
			
			if (i == 5) and ((file_i.split('_')[3] == '12') or (file_i.split('_')[3] == '13')):
				
				angle = df.columns[i]
				
				x_all = df.index.values.tolist()
				y_all = df[angle][df.index.values].tolist()
				
				x_fit = fit_points
				y_fit = df[angle][x_fit].tolist()
				
				# scale data
				y_all = [x*1.e6 for x in y_all]
				y_fit = [x*1.e6 for x in y_fit]
				
				params_1 = fit_data_1(x_fit, y_fit)
				x = np.linspace(0, 30, 500)
				
				params_2 = fit_data_2(x_fit, y_fit)
				x = np.linspace(0, 30, 500)
				
				params_3 = fit_data_3(x_fit, y_fit)
				x = np.linspace(0, 30, 500)
				
				print('[a,b,c]='+str(params_1))
				print('[a,b,c]='+str(params_2))
				print('[a,b,c]='+str(params_3))
				
				plt.figure()
				plt.ylim(min(y_all)-abs(min(y_all)*0.01),max(y_all)+abs(max(y_all)*0.01))
				plt.errorbar(x_all,y_all,label='All data (1% errorbars)',yerr=np.asarray(y_all)*0.01,fmt='.')
				plt.scatter(x_fit,y_fit,label='Data used in fit',color='orange')
				plt.plot(x,exp_function_1(x,params_1[0],params_1[1],params_1[2]),'r-',label='Curve fit (c=-1.4)')
				plt.plot(x,exp_function_2(x,params_2[0],params_2[1],params_2[2]),'r-.',label='Curve fit (c=-0.9)')
				plt.plot(x,exp_function_3(x,params_3[0],params_3[1],params_3[2]),'r--',label='Curve fit (c=-0.4)')
				plt.legend()
				plt.xlabel(param+' [grains]')
				plt.ylabel(r'J-integral [$10^{-6}\ N \cdot \mu m / \mu m^2 $]')
				plt.show()
				

