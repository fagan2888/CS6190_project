import pickle
import pandas as pd
import os, glob
import matplotlib.pyplot as plt
import numpy as np

def get_data(main_dir, crack_size, param, BC, run, size):
	
	os.chdir(main_dir)
	os.chdir('data')
	data = pickle.load(open(crack_size+'_'+param+'_'+BC+'_'+run+'_'+size+'.pickle', 'rb'))

	return data
	
def adjust_fit_points(crack_size, fit_points):
	
	if crack_size == '0-25':
		fit_points = [x + 0.25 for x in fit_points]
	elif crack_size == '0-45':
		fit_points = [x + 0.05 for x in fit_points]
		
	return fit_points
	
def linear_interpolation(x, x_data, y_data):
	
	found_val = False
	for i in range(len(x_data)-1):
		if (found_val==False):
			if (x>=x_data[i]) and (x<=x_data[i+1]):
				x1 = x_data[i]
				x2 = x_data[i+1]
				y1 = y_data[i]
				y2 = y_data[i+1]
				found_val = True
	
	if (found_val==True):
		y = (((y2-y1)/(x2-x1))*(x-x1)) + y1
	else:
		print(x_data)
		print(x)
		y = y_data[-1]
		
	return(y)
	
def normalize_data(y_all, y_fit, x_fit, X_vol):
	
	y_vol_data = []
	max_delta = 0
	for i in range(len(x_fit)-1):
		if (x_fit[i] in X_vol) and (x_fit[i+1] in X_vol):
		
			delta = abs(y_fit[i] - y_fit[i+1])
			if (delta>max_delta):
				max_delta = delta
				
			y_vol_data.append(y_fit[i])
			y_vol_data.append(y_fit[i+1])
			
	min_val = min(y_vol_data) - max_delta
	max_val = max(y_vol_data) + max_delta
	
	y_fit_norm = (y_fit-min_val)/(max_val - min_val)
	y_all_norm = (y_all-min_val)/(max_val - min_val)
	
	return(min_val, max_val, y_fit_norm, y_all_norm)
	
def clean_data(X_vol_temp):

	main_dir = os.getcwd()
	
	os.chdir('data')
	
	for file_i in glob.glob('*.pickle'):
		if ('clean' not in file_i):
		
			print(str(file_i))
			
			crack_size = file_i.split('_')[0]
			rve_param = file_i.split('_')[1]
			
			if (rve_param=='d1'):
				X_vol = adjust_fit_points(crack_size, X_vol_temp)
			else:
				X_vol = X_vol_temp
		
			fit_points = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0]
			if rve_param=='d1':
				fit_points = adjust_fit_points(crack_size, fit_points)
			
			#df = get_data(crack_size, param, BC, run, size)
			df = pickle.load(open(file_i, 'rb'))
			df = df.multiply(10**6)
			
			df_new = pd.DataFrame(index=fit_points)
			df_min_max = pd.DataFrame()
			
			for i in range(0,len(df.columns)):
					
				angle = df.columns[i]
				
				x_all = df.index.values.tolist()
				y_all = df[angle][df.index.values].tolist()
				
				x_fit = fit_points
				y_fit = []
				
				for x in fit_points:
					if x not in x_all:
						y = linear_interpolation(x, x_all, y_all)
						y_fit.append(y)
					else:
						y = df[angle][x]
						y_fit.append(y)
				
				min_val, max_val, y_fit_norm, y_all_norm = normalize_data(y_all, y_fit, x_fit, X_vol)
				
				df_new[angle] = y_fit_norm
				df_min_max.loc['min',angle] = min_val
				df_min_max.loc['max',angle] = max_val
				
				'''
				if i%57==0:
					plt.figure()
					plt.plot(x_all,y_all_norm,'o')
					plt.plot(x_fit,y_fit_norm,':x')
					plt.show()
				'''
			
			df_new = pd.concat(objs=[df_new, df_min_max], axis=0)	
				
			file_new = file_i.split('.')[0]+'_clean.pickle'
			pickle.dump(df_new, open(file_new, 'wb'))
		

if __name__ == '__main__':
	
	fit_points = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0]
	clean_data(fit_points)


			
