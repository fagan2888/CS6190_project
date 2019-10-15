import pickle
import pandas as pd
import matplotlib.pyplot as plt
import os,glob
import numpy as np
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error

def get_data(crack_size, param, BC, run, size):
	
	os.chdir('data')
	data = pickle.load(open(crack_size+'_'+param+'_'+BC+'_'+run+'_'+size+'.pickle', 'rb'))

	return data
	
def adjust_fit_points(crack_size, fit_points, param):
	
	if (crack_size == '0-25') and (param == 'd1'):
		fit_points = [x + 0.25 for x in fit_points]
	elif (crack_size == '0-45') and (param == 'd1'):
		fit_points = [x + 0.05 for x in fit_points]
		
	return fit_points
	
def check_x_vals(x_vals, x_all):
	
	# check that x_vals referenced exist in data
	# INPUT
	# x_vals: x values to reference 
	# df: dataset
	
	# OUTPUT:
	# x_checked: x values to reference that exist in data
	
	x_checked = []
	for x in x_vals:
		if x in x_all:
			x_checked.append(x)
			
	return x_checked
	
	
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
		run = file_i.split('_')[3]
	
		fit_points = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0]
		fit_points = adjust_fit_points(crack_size, fit_points, param)
		
		#df = get_data(crack_size, param, BC, run, size)
		df = pickle.load(open(file_i, 'rb'))
		
		for i in range(0,len(df.columns)):
				
			if (i==5) and (run != '12') and (run != '15'):
				angle = df.columns[i]
				
				x_all = df.index.values.tolist()
				y_all = df[angle][df.index.values].tolist()
				
				x_fit = fit_points
				x_fit = check_x_vals(x_fit, x_all)
				y_fit = df[angle][x_fit].tolist()
				
				# scale data
				y_all = [x*1.e6 for x in y_all]
				y_fit = [x*1.e6 for x in y_fit]
				
				num_train = 6
				maxlag=3
				
				y_train, y_test = y_fit[0:num_train], y_fit[num_train:len(y_fit)]
				x_train, x_test = x_fit[0:num_train], x_fit[num_train:len(y_fit)]
				predictions = []
				model = AR(y_train)
				model_fit = model.fit(ic='aic', maxlag=maxlag)
				#model_fit = model.fit(maxlag=maxlag)
				output_1 = model_fit.predict(len(y_train), len(y_train)+len(y_test)-1)
				x_test_1 = x_test
				
				num_train = 5
				maxlag=3
				
				y_train, y_test = y_fit[0:num_train], y_fit[num_train:len(y_fit)]
				x_train, x_test = x_fit[0:num_train], x_fit[num_train:len(y_fit)]
				predictions = []
				model = AR(y_train)
				model_fit = model.fit(ic='aic', maxlag=maxlag)
				#model_fit = model.fit(maxlag=maxlag)
				output_2 = model_fit.predict(len(y_train), len(y_train)+len(y_test)-1)
				x_test_2 = x_test
				
				num_train = 4
				maxlag=3
				
				y_train, y_test = y_fit[0:num_train], y_fit[num_train:len(y_fit)]
				x_train, x_test = x_fit[0:num_train], x_fit[num_train:len(y_fit)]
				predictions = []
				model = AR(y_train)
				model_fit = model.fit(ic='aic', maxlag=maxlag)
				#model_fit = model.fit(maxlag=maxlag)
				output_3 = model_fit.predict(len(y_train), len(y_train)+len(y_test)-1)
				x_test_3 = x_test
				
				# plot
				yerr = []
				for i in y_all:
					yerr.append(i*0.01)
				plt.errorbar(x_all, y_all, yerr=yerr, label='Actual (1% errorbar)', color='black', fmt='.')
				plt.scatter(x_fit[:4],y_fit[:4], color='orange', label='Training')
				plt.scatter(x_fit[7:],y_fit[7:], color='violet', label='Testing')
				plt.scatter(x_fit[6],y_fit[6], color='green', label='Start of testing (6 points)')
				plt.scatter(x_fit[5],y_fit[5], color='blue', label='Start of testing (5 points)')
				plt.scatter(x_fit[4],y_fit[4], color='red', label='Start of testing (4 points)')
				plt.plot(x_test_1, output_1, 'g-', label='Predicted (6 points)')
				plt.plot(x_test_2, output_2, 'b--', label='Predicted (5 points)')
				plt.plot(x_test_3, output_3, 'r-.', label='Predicted (4 points)')
				plt.xlabel(param+' [grains]')
				plt.ylabel(r'J-integral [$10^{-6}\ N \cdot \mu m / \mu m^2 $]')
				plt.legend(loc='lower right')
				plt.show()
