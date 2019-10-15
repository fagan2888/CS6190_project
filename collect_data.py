import pickle
import os, glob
import pandas as pd

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
		
		runs =  [1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 12, 13, 14, 14, 15, 16, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22]
		sizes = [14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 16, 14, 14, 16, 14, 14, 14, 14, 16, 14, 16, 14, 16, 14, 16, 14, 16]
		
	elif (BC=='Free') and (crack_size=='0-45') and (find_parameter=='d1'):
		fixed_parameter = 'd2'
		
		runs =  [1,  2,  3,  4,  4,  5,  6,  7,  8,  9,  10, 11, 11, 12, 12, 13, 14, 14, 15, 16, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22]
		sizes = [14, 14, 14, 14, 16, 14, 14, 14, 14, 14, 14, 14, 16, 14, 16, 14, 14, 16, 14, 14, 14, 14, 16, 14, 16, 14, 16, 14, 16, 14]
		
	elif (BC=='Free') and (crack_size=='1-0') and (find_parameter=='d1'):
		fixed_parameter = 'd2'
		
		runs =  [1,  2,  3,  4,  5,  6,  7,  7,  8,  8,  9,  10, 10, 11, 11, 11, 12, 12, 13, 14, 15, 15, 16, 16, 16, 17, 17, 18, 19, 19, 20, 20, 21, 21, 21, 21, 21, 22, 22, 22, 22, 22, 22]
		sizes = [16, 16, 16, 16, 16, 16, 16, 18, 16, 18, 16, 16, 18, 16, 18, 20, 16, 18, 16, 16, 16, 18, 16, 18, 20, 16, 18, 16, 16, 18, 16, 18, 16, 18, 20, 21, 22, 16, 18, 19, 20, 21, 22]
		
	elif (BC=='Free') and (crack_size=='3-0') and (find_parameter=='d1'):
		fixed_parameter = 'd2'
		
		runs =  [1,  1,  2,  2,  3,  4,  4,  5,  5,  6,  7,  7,  8,  9,  9,  10, 11, 11, 12, 13, 14, 15, 16, 16, 17, 18, 18, 19, 20, 20, 21, 21, 21, 21, 22, 22, 22, 22, 22, 22]
		sizes = [14, 16, 14, 16, 14, 14, 16, 14, 16, 14, 14, 16, 14, 14, 16, 14, 14, 16, 14, 14, 14, 14, 14, 16, 14, 14, 16, 14, 14, 16, 14, 18, 20, 22, 14, 18, 19, 20, 21, 22]
		
	elif (BC=='Free') and (crack_size=='0-25') and (find_parameter=='d2'):
		fixed_parameter = 'd1'
		
		runs =  [1,  1,  1,  1,  2,  3,  4,  5,  5,  5,  6,  6,  6,  7,  8,  9,  10, 11, 11, 12, 12, 12, 12, 13, 14, 14, 15, 16, 17, 18, 18, 19, 19, 20, 20, 20, 21, 21, 22, 22]
		sizes = [12, 14, 15, 16, 12, 12, 12, 12, 14, 15, 12, 14, 15, 12, 12, 12, 12, 12, 14, 12, 14, 15, 16, 12, 12, 14, 12, 12, 12, 12, 14, 12, 14, 12, 14, 15, 12, 14, 12, 14]
		
	elif (BC=='Free') and (crack_size=='0-45') and (find_parameter=='d2'):
		fixed_parameter = 'd1'
		
		runs =  [1,  1,  2,  3,  4,  5,  5,  6,  6,  7,  8,  9,  10, 11, 11, 12, 12, 12, 13, 14, 14, 15, 15, 16, 16, 17, 18, 18, 18, 19, 19, 20, 21, 21, 21, 22]
		sizes = [12, 14, 12, 12, 12, 12, 14, 12, 14, 12, 12, 12, 12, 12, 14, 12, 14, 15, 12, 12, 14, 12, 14, 12, 14, 12, 12, 14, 15, 12, 14, 12, 12, 14, 15, 12]
		
	elif (BC=='Free') and (crack_size=='1-0') and (find_parameter=='d2'):
		fixed_parameter = 'd1'
		
		runs =  [1,  1,  1,  1,  1,  1,  2,  2,  2,  3,  3,  4,  4,  5,  5,  5,  6,  7,  7,  8,  9,  9,  10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 18, 18, 19, 19, 20, 20, 21, 21, 21, 21, 21, 21, 21, 22, 22, 22, 22]
		sizes = [14, 16, 17, 18, 19, 20, 14, 15, 16, 14, 16, 14, 16, 14, 15, 16, 14, 14, 16, 14, 14, 16, 14, 14, 16, 14, 16, 14, 16, 14, 16, 14, 16, 14, 16, 14, 14, 16, 14, 16, 14, 16, 14, 16, 18, 19, 20, 21, 22, 14, 16, 18, 20]
		
	elif (BC=='Free') and (crack_size=='3-0') and (find_parameter=='d2'):
		fixed_parameter = 'd1'
		
		runs =  [1,  1,  1,  2,  3,  4,  5,  6,  7,  8,  8,  9,  9,  10, 10, 11, 12, 13, 13, 13, 14, 14, 15, 15, 16, 17, 17, 18, 18, 18, 18, 18, 19, 20, 20, 20, 21, 21, 21, 21, 21, 22, 22, 22, 22, 22]
		sizes = [14, 16, 18, 14, 14, 14, 14, 14, 14, 14, 16, 14, 16, 14, 16, 14, 14, 14, 16, 18, 14, 16, 14, 16, 14, 14, 16, 12, 14, 15, 16, 18, 14, 12, 14, 16, 14, 18, 20, 21, 22, 14, 16, 18, 20, 22]
	
	elif (BC=='Submodeling') and (crack_size=='0-25') and (find_parameter=='d1'):
		fixed_parameter = 'd2'
		
		runs =  [1,  1,  1,  2,  2,  3,  3,  3,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 12, 13, 14, 15, 15, 16, 17, 18, 18, 19, 19, 20, 20, 21, 22]
		sizes = [12, 14, 16, 14, 16, 10, 12, 14, 16, 14, 14, 14, 14, 14, 14, 14, 14, 14, 16, 14, 14, 14, 16, 14, 14, 14, 16, 14, 16, 14, 16, 14, 14]
		
	elif (BC=='Submodeling') and (crack_size=='0-45') and (find_parameter=='d1'):
		fixed_parameter = 'd2'
		
		runs =  [1,  2,  3,  4,  4,  5,  6,  7,  8,  9,  10, 11, 12, 12, 13, 14, 15, 16, 17, 18, 19, 19, 20, 20, 21, 21, 22]
		sizes = [14, 14, 14, 14, 16, 14, 14, 14, 14, 14, 14, 14, 14, 16, 14, 14, 14, 14, 14, 14, 14, 16, 14, 16, 14, 16, 14]
		
	elif (BC=='Submodeling') and (crack_size=='1-0') and (find_parameter=='d1'):
		fixed_parameter = 'd2'
		
		runs =  [1,  1,  1,  1,  1,  2,  2,  2,  2,  3,  3,  3,  3,  4,  4,  4,  4,  5,  6,  7,  8,  9,  10, 11, 12, 12, 12, 12, 12, 12, 13, 14, 14, 15, 16, 16, 17, 18, 19, 19, 20, 21, 21, 21, 21, 22, 22, 22, 22]
		sizes = [8,  10, 12, 14, 16, 10, 12, 14, 16, 8,  10, 12, 14, 12, 14, 16, 18, 14, 14, 14, 14, 14, 14, 14, 8,  10, 12, 14, 16, 18, 14, 14, 16, 14, 14, 16, 14, 14, 14, 16, 14, 14, 16, 18, 20, 14, 16, 18, 20]
		
	elif (BC=='Submodeling') and (crack_size=='3-0') and (find_parameter=='d1'):
		fixed_parameter = 'd2'
		
		runs =  [1,  1,  1,  2,  2,  2,  2,  3,  3,  3,  3,  4,  4,  4,  5,  6,  7,  7,  7,  7,  8,  9,  9,  10, 10, 11, 12, 12, 13, 14, 15, 15, 16, 16, 16, 16, 17, 18, 18, 18, 18, 19, 20, 21, 21, 22, 22]
		sizes = [10, 12, 14, 10, 12, 14, 16, 10, 12, 14, 16, 10, 12, 14, 14, 14, 12, 14, 15, 16, 14, 14, 16, 14, 16, 14, 14, 16, 14, 14, 14, 16, 12, 14, 15, 16, 14, 12, 14, 15, 16, 14, 14, 14, 18, 14, 18]
	
	elif (BC=='Submodeling') and (crack_size=='0-25') and (find_parameter=='d2'):
		fixed_parameter = 'd1'
		
		runs =  [1,  1,  1,  1,  2,  2,  3,  4,  5,  6,  6,  6,  7,  8,  9,  9,  10, 10, 11, 11, 12, 12, 12, 13, 14, 15, 16, 17, 18, 18, 18, 19, 19, 20, 20, 21, 21, 22]
		sizes = [8,  10, 12, 14, 10, 12, 12, 12, 12, 10, 12, 14, 12, 12, 10, 12, 12, 16, 12, 16, 10, 12, 14, 12, 12, 12, 12, 12, 10, 12, 14, 12, 14, 12, 14, 12, 14, 12]
		
	elif (BC=='Submodeling') and (crack_size=='0-45') and (find_parameter=='d2'):
		fixed_parameter = 'd1'
		
		runs =  [1,  2,  3,  4,  5,  6,  6,  7,  8,  9,  10, 11, 12, 12, 13, 14, 14, 15, 15, 16, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22]
		sizes = [12, 12, 12, 12, 12, 12, 14, 12, 12, 12, 12, 12, 12, 14, 12, 12, 14, 12, 14, 12, 12, 12, 14, 12, 14, 12, 14, 12, 14, 12]
		
	elif (BC=='Submodeling') and (crack_size=='1-0') and (find_parameter=='d2'):
		fixed_parameter = 'd1'
		
		runs =  [1,  1,  1,  2,  2,  2,  3,  3,  3,  4,  5,  6,  7,  7,  7,  7,  8,  9,  9,  9,  10, 11, 11, 12, 13, 14, 14, 14, 15, 16, 16, 16, 17, 18, 18, 19, 20, 21, 21, 21, 22, 22, 22]
		sizes = [10, 12, 14, 10, 12, 14, 10, 12, 16, 10, 10, 10, 10, 12, 14, 16, 10, 10, 12, 16, 10, 10, 12, 10, 10, 10, 12, 16, 10, 10, 12, 16, 10, 10, 16, 10, 10, 10, 12, 14, 10, 12, 14]
		
	elif (BC=='Submodeling') and (crack_size=='3-0') and (find_parameter=='d2'):
		fixed_parameter = 'd1'
		
		runs =  [1,  1,  1,  1,  2,  2,  2,  2,  3,  3,  3,  3,  4,  4,  4,  4,  5,  6,  7,  7,  7,  7,  8,  9,  9,  9,  10, 11, 12, 13, 13, 13, 13, 14, 15, 15, 15, 16, 17, 18, 18, 18, 18, 19, 20, 20, 21, 21, 22, 22, 22]
		sizes = [10, 12, 14, 16, 10, 12, 14, 16, 10, 12, 14, 16, 10, 12, 14, 16, 12, 12, 12, 14, 16, 18, 12, 12, 14, 16, 12, 12, 12, 12, 14, 16, 18, 12, 12, 14, 16, 12, 12, 12, 14, 16, 18, 12, 12, 14, 12, 16, 12, 16, 18]
	
	return [fixed_parameter, runs, sizes]

def get_sheetname(BC, crack_size, find_parameter):
	
	# get sheetname to get data from
	# INPUT:
	# BC: type of boundary condition 'Free' or 'Submodeling'
	# crack_size: size of crack '0-25','0-45','1-0','3-0'
	# find_parameter: RVE parameter to find 'd1' or 'd2'
	
	# OUTPUT:
	# sheetname: name of excel sheet to get J-integral data from
	
	if (BC=='Free') and (crack_size=='0-25') and (find_parameter=='d1'):
		sheetname = 'd2='+size+'.0 d1=26.25 raw data'
	elif (BC=='Free') and (crack_size=='0-45') and (find_parameter=='d1'):
		sheetname = 'd2='+size+'.0 d1=26.05 raw data'
	elif (BC=='Free') and (crack_size=='1-0') and (find_parameter=='d1'):
		sheetname = 'd2='+size+'.0 d1=26.0 raw data'
	elif (BC=='Free') and (crack_size=='3-0') and (find_parameter=='d1'):
		sheetname = 'd2='+size+'.0 d1=24.0 raw data'
	elif (BC=='Free') and (crack_size=='0-25') and (find_parameter=='d2'):
		sheetname = 'd2=26.0 d1='+size+'.25 raw data'
	elif (BC=='Free') and (crack_size=='0-45') and (find_parameter=='d2'):
		sheetname = 'd2=26.0 d1='+size+'.05 raw data'
	elif (BC=='Free') and (crack_size=='1-0') and (find_parameter=='d2'):
		sheetname = 'd2=26.0 d1='+size+'.0 raw data'
	elif (BC=='Free') and (crack_size=='3-0') and (find_parameter=='d2'):
		sheetname = 'd2=26.0 d1='+size+'.0 raw data'
	elif (BC=='Submodeling') and (crack_size=='0-25') and (find_parameter=='d1'):
		sheetname = 'd2='+size+'.0 d1=26.25 raw data'
	elif (BC=='Submodeling') and (crack_size=='0-45') and (find_parameter=='d1'):
		sheetname = 'd2='+size+'.0 d1=26.05 raw data'
	elif (BC=='Submodeling') and (crack_size=='1-0') and (find_parameter=='d1'):
		sheetname = 'd2='+size+'.0 d1=26.0 raw data'
	elif (BC=='Submodeling') and (crack_size=='3-0') and (find_parameter=='d1'):
		sheetname = 'd2='+size+'.0 d1=24.0 raw data'
	elif (BC=='Submodeling') and (crack_size=='0-25') and (find_parameter=='d2'):
		sheetname = 'd2=26.0 d1='+size+'.25 raw data'
	elif (BC=='Submodeling') and (crack_size=='0-45') and (find_parameter=='d2'):
		sheetname = 'd2=26.0 d1='+size+'.05 raw data'
	elif (BC=='Submodeling') and (crack_size=='1-0') and (find_parameter=='d2'):
		sheetname = 'd2=26.0 d1='+size+'.0 raw data'
	elif (BC=='Submodeling') and (crack_size=='3-0') and (find_parameter=='d2'):
		sheetname = 'd2=26.0 d1='+size+'.0 raw data'
	
	return sheetname


if __name__ == '__main__':
	for crack_size in ['0-25','0-45','1-0','3-0']:
			for find_parameter in ['d1','d2']:
				for BC in ['Submodeling','Free']:

					print('case='+BC+'_a/g='+crack_size+'_'+find_parameter)
					
					fixed_parameter, runs, sizes = select_case(BC, crack_size, find_parameter)
					
					for set_i in range(len(runs)):
						run = str(runs[set_i])
						size = str(sizes[set_i])
						
						sheetname = get_sheetname(BC, crack_size, find_parameter)

						print('run '+run)
						
						try:
							path = 'I:\\RVE Results\\'+BC+'\\ag'+crack_size+'\\Run '+run+'\\Find '+find_parameter+'\\'+fixed_parameter+'='+size+'\\data'
							os.chdir(path)
						except:
							path = 'I:\\RVE Results\\'+BC+'\\ag'+crack_size+'\\Run'+run+'\\Find '+find_parameter+'\\'+fixed_parameter+'='+size+'\\data'
							os.chdir(path)

						file_name = glob.glob('point*.xlsx')
						
						df = pd.read_excel(file_name[0], sheet_name=sheetname, index_col=0)
						df = df.sort_index(axis=0,ascending=True)
						df = df.dropna()
						
						os.chdir('C:\\Users\\Karen\\Desktop\\CS 6190\\Project\\Code\\data')
						
						pickle.dump( df, open( crack_size+'_'+find_parameter+'_'+BC+'_'+run+'_'+size+'.pickle', 'wb' ) )
