from sklearn.model_selection import cross_validate, RepeatedStratifiedKFold
from sklearn.metrics import recall_score, make_scorer, matthews_corrcoef
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

#set seed for randomness
def set_seed(seed):
	np.random.RandomState(seed)
	np.random.seed(seed)
		
#calculate accuracy, precision, recall, specificity, f1, and mcc scores of a given model
def evaluate_model(X, y, modeltype):
	
	#set random seed
	seed = 12345678
	set_seed(seed)
	
	#create models
	if modeltype == 'rfc':
		model = RandomForestClassifier(random_state=seed)
	elif modeltype == 'svm':
		model = svm.SVC(random_state=seed)
	
	#custom scorer for TNs(specificity) and MCC(Matthews correlation coefficient)
	specificity = make_scorer(recall_score, pos_label=0)
	matthews = make_scorer(matthews_corrcoef)
	
	#define which metrics to calculate
	scoring = {'acc': 'accuracy',
			   'prec': 'precision',
			   'rec': 'recall',
			   'spec': specificity,
			   'f1': 'f1',
			   'mcc': matthews}
	
	#n_repeats different 80/20 train/test splits
	cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=seed)

	#calculate accuracy, precision, recall, specificity, f1, and mcc scores
	scores = cross_validate(model, X, y, scoring=scoring, cv=cv, n_jobs=-1)

	#calculate and print mean and std
	acc = [scores['test_acc'].mean(), scores['test_acc'].std()]
	prec = [scores['test_prec'].mean(), scores['test_prec'].std()]
	rec = [scores['test_rec'].mean(), scores['test_rec'].std()]
	spec = [scores['test_spec'].mean(), scores['test_spec'].std()]
	f1 = [scores['test_f1'].mean(), scores['test_f1'].std()]
	mcc = [scores['test_mcc'].mean(), scores['test_mcc'].std()]
	print('Performance metrics:')
	print('Accuracy mean:   ', format(acc[0], '.4f'), 'Accuracy std:   ', format(acc[1], '.4f'),
		  '\nPrecision mean:  ', format(prec[0], '.4f'), 'Precision std:  ', format(prec[1], '.4f'),
		  '\nRecall mean:     ', format(rec[0], '.4f'), 'Recall std:     ', format(rec[1], '.4f'),
		  '\nSpecificity mean:', format(spec[0], '.4f'), 'Specificity std:', format(spec[1], '.4f'),
		  '\nF1 mean:         ', format(f1[0], '.4f'), 'F1 std:         ', format(f1[1], '.4f'),
		  '\nMCC mean:        ', format(mcc[0], '.4f'), 'MCC std:        ', format(mcc[1], '.4f'))
				
#determin for which inputs the model mispredicts
def test_mispredict(XData, yData, model):
	
	inputs = XData.values
	groundt = yData.values.ravel()
	
	#predict label for test inputs
	predictions = model.predict(inputs)
	
	#check for mispredictions
	mispredictions = np.where(groundt == predictions, 0, 1)
	
	#convert to pandas dataframe
	df_mispredictions = pd.DataFrame({'misprediction': mispredictions})

	return df_mispredictions


