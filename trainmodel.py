from sklearn.ensemble import RandomForestClassifier
from modelevaluation import evaluate_model
from sklearn.model_selection import train_test_split
from sklearn import tree, svm
import numpy as np
import graphviz

#set seed for randomness
def set_seed(seed):
	np.random.RandomState(seed)
	np.random.seed(seed)
	
#train blackboxes from given dataset
def trainmodel(data, target, modeltype, datasetname, names, encoded_X):
	
	#set random seed
	seed = 12345678
	set_seed(seed)
	
	#store trained models
	models = []
	
	#if rfc
	if modeltype == 'rfc':	
		#calculate accuracy, precision, recall, specificity, f1 and mcc
		evaluate_model(data, target, modeltype)
	
	#train blackbox model and write test sets
	for i in range(5):
		
		#split into training and test sets
		if datasetname in ['heart', 'heartWT']:	
			X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=1000*i+1, stratify=target)		
		else:
			X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=1000*i+1, stratify=target)

		#if rfc
		if modeltype == 'rfc':
	
			#create rf classifier
			model = RandomForestClassifier(random_state=seed)
		
		#if xgb
		elif modeltype == 'svm':
			
			#create xgb classifier
			model = svm.SVC(random_state=seed)
			
		#fit data
		model.fit(X_train, y_train)
			
		#add trained model to list
		models.append(model)
	
		#if heartfailure dataset
		if datasetname == 'heart':
		
			#save test set data
			with open('blackboxes/testdata/'+datasetname+modeltype+'Xtest' + str(i+1) + '.csv', 'w') as FOUT:
				np.savetxt(FOUT, [names[:-1]], fmt='%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s')
				np.savetxt(FOUT, X_test, fmt='%d,%d,%d,%d,%d,%d,%d,%f,%d,%d,%d,%d')
			with open('blackboxes/testdata/'+datasetname+modeltype+'ytest' + str(i+1) + '.csv', 'w') as FOUT:
				np.savetxt(FOUT, [names[-1]], fmt='%s')
				np.savetxt(FOUT, y_test, fmt='%d')
		
		#if heartfailure dataset without time	
		elif datasetname == 'heartWT':
			
			#save test set data
			with open('blackboxes/testdata/'+datasetname+modeltype+'Xtest' + str(i+1) + '.csv', 'w') as FOUT:
				np.savetxt(FOUT, [names[:-1]], fmt='%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s')
				np.savetxt(FOUT, X_test, fmt='%d,%d,%d,%d,%d,%d,%d,%f,%d,%d,%d')
			with open('blackboxes/testdata/'+datasetname+modeltype+'ytest' + str(i+1) + '.csv', 'w') as FOUT:
				np.savetxt(FOUT, [names[-1]], fmt='%s')
				np.savetxt(FOUT, y_test, fmt='%d')
				
		#for other datasets
		else:
			#build format strings
			format_mapping = {
				'int64': '%d',
				'uint8': '%d',
				'float64': '%f'
			}
			formats = [format_mapping[dtype.name] for dtype in encoded_X.dtypes]
			format_string_features = ','.join(formats)
			format_string_names = ','.join(['%s'] * len(encoded_X.columns))

			#save test set data
			with open('blackboxes/testdata/'+datasetname+modeltype+'Xtest' + str(i+1) + '.csv', 'w') as FOUT:
				np.savetxt(FOUT, [names[:-1]], fmt=format_string_names)
				np.savetxt(FOUT, X_test, fmt=format_string_features)
			with open('blackboxes/testdata/'+datasetname+modeltype+'ytest' + str(i+1) + '.csv', 'w') as FOUT:
				np.savetxt(FOUT, [names[-1]], fmt='%s')
				np.savetxt(FOUT, y_test, fmt='%d')
			
	return models

#trains decision tree for misprediction rule set
def train_tree(inputs, blackbox, modeltype, learner, coverage, depth, runs, rseed, model_number):

	#extract X and y from dataframe
	X = inputs.iloc[:,:-2].to_numpy()
	y = inputs[inputs.columns[-1]].to_numpy()
	
	#extract names
	names = list(inputs.columns)
	del names[-2]
		
	#train tree on inputs
	model = tree.DecisionTreeClassifier(max_depth=depth, random_state = rseed) #min_samples_leaf=round(len(X)/20)
	model.fit(X,y)
	
	#save trees as pngs
	dot_data = tree.export_graphviz(model, feature_names = names[0:-1])
	graph = graphviz.Source(dot_data)
	graph.render(filename = "tree/" + str(blackbox) + "_" + str(modeltype) + "_" + str(learner) + "_" + str(coverage) + "_" + str(depth) + "_" + model_number + "_" + str(runs))
		
	return model, names
