import pandas as pd
import pickle
from joblib import load

#reads and prepares datasets
def getdata(dataset):
	
	#read an prepare heart failure data set
	if dataset == 'heart':
		
		#read data set from csv
		data = pd.read_csv('dataset/heart_failure_clinical_records_dataset.csv')
		
		#extract feature values and names
		array = data.values
		names = list(data.columns.values)
				
		#split into prediction features and class
		X = array[:,0:12]
		y = array[:,12]
		
		encoded_data = data
		
	#read an prepare heart failure data set without time
	elif dataset == 'heartWT':
		
		#read data set from csv
		data = pd.read_csv('dataset/heart_failure_clinical_records_dataset.csv')
		
		#extract feature values and names
		array = data.values
		names = list(data.columns.values)
		
		#remove temporal variable name
		names.remove('time')
		
		#split into prediction features and class
		X = array[:,0:11]
		y = array[:,12]
		
		encoded_data = data
		
	#read an prepare bug report data set
	elif dataset == 'bugreport':
		
		#read data set from csv
		data = pd.read_csv('dataset/BRCTP.csv')
		
		#drop first column
		data = data.drop(data.columns[0], axis=1)
	
		#create X and y df
		Xdf = data.drop(['OOSLA'], axis=1)
		ydf = data[['OOSLA']]
		
		#extract feature values and names
		array = data.values
		names = list(data.columns.values)
				
		#split into prediction features and class
		X = array[:,:-1]
		y = array[:,-1]	
		
		#tests
		#print(Xdf.shape)
		#print(ydf.shape)
		#print(ydf.value_counts())
		#print(data.drop(data.columns[-1], axis=1).dtypes)	

		encoded_data = data
	
	#read an prepare java data set
	elif dataset == 'java':
		
		#read data set from csv
		dfX = pd.read_csv('dataset/data_Java.csv')
		dfy = pd.read_csv('dataset/label_Java.csv')
		data = pd.concat([dfX, dfy] ,axis=1)
		
		#sample part of dataset
		data = data.sample(frac=0.75, random_state = 12345678).reset_index(drop=True)

		#extract feature values and names
		array = data.values
		names = list(data.columns.values)
				
		#split into prediction features and class
		X = array[:,:-1]
		y = array[:,-1]	
		
		#tests
		#print(dfX.shape)
		#print(dfy.shape)
		#print(dfy.value_counts())
		#print(data.drop(data.columns[-1], axis=1).dtypes)	

		encoded_data = data
		
	#read an prepare php data set
	elif dataset == 'php':
		
		#read data set from csv
		dfX = pd.read_csv('dataset/data_PHP.csv')
		dfy = pd.read_csv('dataset/label_PHP.csv')
		data = pd.concat([dfX, dfy] ,axis=1)
		
		#sample part of dataset
		data = data.sample(frac=0.4, random_state = 12345678).reset_index(drop=True)

		#extract feature values and names
		array = data.values
		names = list(data.columns.values)
				
		#split into prediction features and class
		X = array[:,:-1]
		y = array[:,-1]	
		
		#tests
		#print(dfX.shape)
		#print(dfy.shape)
		#print(dfy.value_counts())
		#print(data.drop(data.columns[-1], axis=1).dtypes)	

		encoded_data = data
		
	#read an prepare python data set
	elif dataset == 'python':
		
		#read data set from csv
		dfX = pd.read_csv('dataset/data_Python.csv')
		dfy = pd.read_csv('dataset/label_Python.csv')
		data = pd.concat([dfX, dfy] ,axis=1)
		
		#sample part of dataset
		data = data.sample(frac=0.4, random_state = 12345678).reset_index(drop=True)

		#extract feature values and names
		array = data.values
		names = list(data.columns.values)
				
		#split into prediction features and class
		X = array[:,:-1]
		y = array[:,-1]	
		
		#tests
		#print(dfX.shape)
		#print(dfy.shape)
		#print(dfy.value_counts())
		#print(data.drop(data.columns[-1], axis=1).dtypes)	

		encoded_data = data
		
	#read an prepare ruby data set
	elif dataset == 'ruby':
		
		#read data set from csv
		dfX = pd.read_csv('dataset/data_Ruby.csv')
		dfy = pd.read_csv('dataset/label_Ruby.csv')
		data = pd.concat([dfX, dfy] ,axis=1)
		
		#sample part of dataset
		data = data.sample(frac=0.5, random_state = 12345678).reset_index(drop=True)
		
		#extract feature values and names
		array = data.values
		names = list(data.columns.values)
				
		#split into prediction features and class
		X = array[:,:-1]
		y = array[:,-1]	
		
		#tests
		#print(dfX.shape)
		#print(dfy.shape)
		#print(dfy.value_counts())
		#print(data.drop(data.columns[-1], axis=1).dtypes)	

		encoded_data = data
		
	#read an prepare spam email data set
	elif dataset == 'spam':
		
		#read data set from csv
		data_base = pd.read_csv('dataset/spacm_email.csv')
		
		#sample part of dataset
		data_sample = data_base.sample(frac=0.03, random_state = 12345678).reset_index(drop=True)
	
		#remove index column
		data_sample = data_sample.drop('id', axis = 1)
			
		#extract feature values and names
		array = data_sample.values
		names = list(data_sample.columns.values)
				
		#split into prediction features and class
		X = array[:,:-1]
		y = array[:,-1]	
		
		#tests
		#print(data_sample.drop(data_sample.columns[-1], axis=1).dtypes)	
		
		encoded_data = data_sample
		
	#read an prepare water quality data set
	elif dataset == 'water':
		
		#read data set from csv
		data_base = pd.read_csv('dataset/water_potability.csv')
		
		#remove lines with empty cells
		data_nona = data_base.dropna().reset_index(drop=True)

		#extract feature values and names
		array = data_nona.values
		names = list(data_nona.columns.values)
				
		#split into prediction features and class
		X = array[:,:-1]
		y = array[:,-1]	
		
		#tests
		#label = data_nona.columns[-1]
		#features = data_nona.columns[:-1]
		#test_x, test_y = data_nona[features], data_nona[label]
		#print(test_x.shape, test_y.shape)
		#print(data_nona.drop(data_nona.columns[-1], axis=1).dtypes)	

		encoded_data = data_nona
		
	#read an prepare hotel bookings data set
	elif dataset == 'hotel':
		
		#read data set from csv
		data_base = pd.read_csv('dataset/hotel_bookings.csv')
		
		#sample part of dataset
		data_base = data_base.sample(frac=0.17, random_state=12345678).reset_index(drop=True)

		#drop company column (majority null)
		data_base = data_base.drop(['company'], axis = 1)
		
		#set 4 NaN in children to 0
		data_base['children'] = data_base['children'].fillna(0)
	
		#replace hotel names with 0 and 1
		data_base['hotel'] = data_base['hotel'].map({'Resort Hotel':0, 'City Hotel':1})	
		
		#replace arival months with numbers
		data_base['arrival_date_month'] = data_base['arrival_date_month'].map({'January':1, 'February': 2, 'March':3, 'April':4, 'May':5, 'June':6, 'July':7,'August':8, 'September':9, 'October':10, 'November':11, 'December':12})	
																		
		#new column for deposit (refundable as no deposit)
		data_base['deposit'] = data_base['deposit_type'].map({'No Deposit':0, 'Refundable':0, 'Non Refund':1})
		#drop deposit_type
		data_base = data_base.drop(['deposit_type'], axis = 1)
		
		#function to check if family
		def is_family(row):
			if ((row['adults'] > 0) & (row['children'] > 0) | (row['adults'] > 0) & (row['babies'] > 0)):
				family = 1
			else:
				family = 0
			return family
		#apply function to create new family column	
		data_base["family"] = data_base.apply(is_family, axis = 1)
		#create new column for customer amount
		data_base["customers"] = data_base["adults"] + data_base["children"] + data_base["babies"]		
		#remove now redundant columns
		data_base = data_base.drop(columns = ['adults', 'babies', 'children'])
		
		#column for nights stayed
		data_base["nights"] = data_base["stays_in_weekend_nights"] + data_base["stays_in_week_nights"]
		
		#remove not relevant and duplicate info
		data_base = data_base.drop(columns = ['reservation_status_date', 'arrival_date_week_number', 'reservation_status'])
		
		#drop country with over 300 classes
		data_base = data_base.drop(['country'], axis = 1)
		
		#drop agent (many NaN, many classes)
		data_base = data_base.drop(['agent'], axis = 1)
		
		#one hot encode multi class features
		dummy_df = pd.get_dummies(data = data_base, columns = ['meal', 'market_segment', 'distribution_channel', 'reserved_room_type', 'assigned_room_type', 'customer_type'])
		#rename dummys with / white space and - in names
		dummy_df.rename(columns={'market_segment_Offline TA/TO' : 'market_segment_Offline_TA_TO', 'market_segment_Online TA' : 'market_segment_Online_TA', 'distribution_channel_TA/TO' : 'distribution_channel_TA_TO', 'customer_type_Transient-Party' : 'customer_type_Transient_Party'}, inplace=True)
			
		#split X and y
		dfy = dummy_df["is_canceled"]
		dfX = dummy_df.drop(["is_canceled"], axis=1)
		
		data = pd.concat([dfX, dfy] ,axis=1)

		#extract feature values and names
		array = data.values
		names = list(data.columns.values)
				
		#split into prediction features and class
		X = array[:,:-1]
		y = array[:,-1]	
		
		#tests
		#print(dfX.shape)
		#print(dfy.shape)
		#print(dfy.value_counts())
		#print(data.drop(data.columns[-1], axis=1).dtypes)	

		encoded_data = data

	#read an prepare job change data set
	elif dataset == 'job':
		
		#read data set from csv
		data_base = pd.read_csv('dataset/job_change.csv')

		#remove enrollee_id (all unique classes)
		data_base = data_base.drop(["enrollee_id"], axis=1)	
		
		#remove city (123 unique classes to much to one hot encode)
		data_base = data_base.drop(["city"], axis=1)	

		#replace relevant_experience with 0 and 1
		data_base['relevent_experience'] = data_base['relevent_experience'].map({'No relevent experience':0, 'Has relevent experience':1})	
		
		#replace experience >20 with 21 and <1 with 0 and NaN with 0
		data_base['experience'] = data_base['experience'].fillna(0)
		data_base['experience'] = data_base['experience'].replace({'<1':0, '>20':21})
		
		#replace last_new_job >4 with 5 and never with 0 and NaN with 0
		data_base['last_new_job'] = data_base['last_new_job'].fillna(0)
		data_base['last_new_job'] = data_base['last_new_job'].replace({'>4':5, 'never':0})		
		
		#rename company size
		data_base['company_size'] = data_base['company_size'].replace({'<10':'1_9', '10/49':'10_49', '50-99':'50_99', '100-500':'100_499', '500-999':'500_999', '1000-4999':'1000_4999', '5000-9999':'5000_9999', '10000+':'10000'})
	
		#set NaN to unknown
		data_base['gender'] = data_base['gender'].fillna('unknown')
		data_base['enrolled_university'] = data_base['enrolled_university'].fillna('unknown')
		data_base['education_level'] = data_base['education_level'].fillna('unknown')
		data_base['major_discipline'] = data_base['major_discipline'].fillna('unknown')
		data_base['company_size'] = data_base['company_size'].fillna('unknown')
		data_base['company_type'] = data_base['company_type'].fillna('unknown')
		
		#one hot encode for categorical features
		dummy_df = pd.get_dummies(data = data_base, columns = ['gender', 'enrolled_university', 'education_level', 'major_discipline', 'company_size', 'company_type'])
		
		#rename encoded columns
		dummy_df.rename(columns={'enrolled_university_Full time course' : 'enrolled_university_Full_time_course',
								 'education_level_High School' : 'education_level_High_School',
								 'education_level_Primary School' : 'education_level_Primary_School',
								 'major_discipline_Business Degree' : 'major_discipline_Business_Degree',
								 'major_discipline_No Major' : 'major_discipline_No_Major',
								 'company_type_Pvt Ltd' : 'company_type_Pvt_Ltd',
								 'company_type_Funded Startup' : 'company_type_Funded_Startup',
								 'company_type_Public Sector' : 'company_type_Public_Sector',
								 'company_type_Early Stage Startup' : 'company_type_Early_Stage_Startup',
								 'enrolled_university_Part time course' : 'enrolled_university_Part_time_course'}, inplace=True)
		
		#set type for experience and last_new_job
		dummy_df = dummy_df.astype({'experience': 'int64', 'last_new_job': 'int64'})
		
		#print(dummy_df.columns)	
		#print(dummy_df['experience'].value_counts(dropna=False))
		#print(data_base['experience'].nunique())
			
		#split X and y
		dfy = dummy_df["target"]
		dfX = dummy_df.drop(["target"], axis=1)
				
		data = pd.concat([dfX, dfy] ,axis=1)

		#extract feature values and names
		array = data.values
		names = list(data.columns.values)
				
		#split into prediction features and class
		X = array[:,:-1]
		y = array[:,-1]	
		
		#tests
		#print(dfX.shape)
		#print(dfy.shape)
		#print(dfy.value_counts())
		#print(data.drop(data.columns[-1], axis=1).dtypes)	

		encoded_data = data
		
	#read an prepare bank marketing data set
	elif dataset == 'bank':
		
		#read data set from csv
		data_base = pd.read_csv('dataset/bank.csv', sep = ';')
		
		#sample part of dataset
		data_base = data_base.sample(frac=0.5, random_state=12345678).reset_index(drop=True)
		
		#set unknown marital to single
		data_base['marital'] = data_base['marital'].replace({'unknown':'single'})
		
		#map in contact column cellular:0 telephone:1
		data_base['contact'] = data_base['contact'].map({'cellular':0, 'telephone':1})

		#replace months with numbers
		data_base['month'] = data_base['month'].map({'mar':3, 'apr':4, 'may':5, 'jun':6, 'jul':7,'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12})	

		#replace day of week with numbers
		data_base['day_of_week'] = data_base['day_of_week'].map({'mon':1, 'tue':2, 'wed':3, 'thu':4, 'fri':5})	
			
		#map employee number
		data_base['employee'] = data_base['nr.employed'].map({5228.1:1, 5099.1:2, 5191.0:3, 5195.8:4, 5076.2:5, 5017.5:6, 4991.6:7, 5008.7:8, 4963.6:9, 5023.5:10, 5176.3:10})
		data_base = data_base.drop(["nr.employed"], axis=1)	

		#replace yes no in target
		data_base['y'] = data_base['y'].map({'yes':1, 'no':0})	
						
		#one hot encode for categorical features
		dummy_df = pd.get_dummies(data = data_base, columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'poutcome', 'employee'])
		
		#rename columns
		dummy_df.rename(columns={'job_blue-collar' : 'job_blue_collar',
								 'job_self-employed' : 'job_self_employed',
								 'education_university.degree' : 'education_university_degree',
								 'education_high.school' : 'education_high_school',
								 'education_basic.9y' : 'education_basic_9y',
								 'education_basic.4y' : 'education_basic_4y',
								 'education_basic.6y' : 'education_basic_6y',
								 'education_professional.course' : 'education_professional_course',
								 'emp.var.rate' : 'emp_var_rate',
								 'cons.price.idx' : 'cons_price_idx',
								 'cons.conf.idx' : 'cons_conf_idx',
								 'job_admin.' : 'job_admin'}, inplace=True)

		#print(dummy_df.columns)		
		#print(data_base['y'].value_counts(dropna=False))
		#print(data_base['y'].nunique())
		
		#split X and y
		dfy = dummy_df["y"]
		dfX = dummy_df.drop(["y"], axis=1)
				
		data = pd.concat([dfX, dfy] ,axis=1)

		#extract feature values and names
		array = data.values
		names = list(data.columns.values)
				
		#split into prediction features and class
		X = array[:,:-1]
		y = array[:,-1]	
		
		#tests
		#print(dfX.shape)
		#print(dfy.shape)
		#print(dfy.value_counts())
		#print(data.drop(data.columns[-1], axis=1).dtypes)	

		encoded_data = data
		
	return X, y, names, encoded_data.drop(encoded_data.columns[-1], axis=1)

#read X and y from testset 
def loadTestSet(dataset, modeltype, number):
	
	#read data set from csv
	Xdata = pd.read_csv('blackboxes/testdata/'+dataset+modeltype+'Xtest'+number+'.csv', sep=',')
	ydata = pd.read_csv('blackboxes/testdata/'+dataset+modeltype+'ytest'+number+'.csv', sep=',')

	return Xdata, ydata
		
#load model from .pickle/.joblib file
def loadpickle(path, nametype):
			
	#check if .pickle or .joblib
	if path[-6:] == 'pickle':
		#load model form file
		model = pickle.load(open(path, 'rb'))
	elif path[-6:] == 'joblib':
		model = load(path)
	
	return model

#returns dict with relevant features
def get_relevant(dataset, XData):

	if dataset == "heart":
		relevant = {
				   'age' : 'I',
				   'anaemia': 'D',
				   'creatinine_phosphokinase': 'I',
				   'diabetes': 'D',
				   'ejection_fraction': 'I',
				   'high_blood_pressure': 'D',
				   'platelets': 'I',
				   'serum_creatinine': 'C',
				   'serum_sodium': 'I',
				   'sex': 'D',
				   'smoking': 'D',
				   'time': 'I'
		}
	elif dataset == "heartWT":
		relevant = {
				   'age' : 'I',
				   'anaemia': 'D',
				   'creatinine_phosphokinase': 'I',
				   'diabetes': 'D',
				   'ejection_fraction': 'I',
				   'high_blood_pressure': 'D',
				   'platelets': 'I',
				   'serum_creatinine': 'C',
				   'serum_sodium': 'I',
				   'sex': 'D',
				   'smoking': 'D',
		}
	else:
		
		#creat dict with relevant features and types for mmd
		relevant = {}
		for column in XData:
			if XData[column].dtype == 'int64':
				uniques = XData[column].nunique()
				if uniques > 2:
					relevant[column] = 'I'
				elif uniques == 2:
					relevant[column] = 'D'
			elif XData[column].dtype == 'float64':
				relevant[column] = 'C'

	return relevant
