import getopt
import sys
import numpy as np
import pandas as pd
from joblib import dump, load
from typing import Dict, Tuple
from dataload import getdata, loadTestSet, loadpickle, get_relevant
from modelevaluation import test_mispredict
from trainmodel import trainmodel, train_tree
from sklearn.ensemble import RandomForestClassifier
from skopt import gp_minimize
from mmd.diagnoser import discover
from islearn.learner import InvariantLearner
from isla.language import ISLaUnparser, Formula, DerivationTree
from fuzzingbook.Parser import EarleyParser
from fuzzingbook.Grammars import Grammar, unreachable_nonterminals, reachable_nonterminals
from grammars.heartfailure import HEARTFAILURE as grammarHF
from grammars.heartfailureWT import HEARTFAILUREWT as grammarHFWT
from grammars.bugreport import BUGREPORT as grammarBR
from grammars.java import JAVA as grammarJAVA
from grammars.php import PHP as grammarPHP
from grammars.python import PYTHON as grammarPYTHON
from grammars.ruby import RUBY as grammarRUBY
from grammars.spam import SPAM as grammarSPAM
from grammars.water import WATER as grammarWATER
from grammars.hotel import HOTEL as grammarHOTEL
from grammars.job import JOB as grammarJOB
from grammars.bank import BANK as grammarBANK
from ruleset import mmd_ruleset_string, islearn_ruleset_string, RuleSet
import warnings
import time

#set seed for randomness
def set_seed(seed):
	np.random.RandomState(seed)
	np.random.seed(seed)
	
#handle inputs
def main(argv):
	arg_help = ("\nmisprediction.py -b <blackbox> -m <model> -l <learner> -r <rules> -c <coverage> -s <seed> -n <model_number>\n"
				"blackbox: heart(heart failure data set), heartWT(heart failure data set without time), bugreport, java, php, python, ruby, bank, hotel, job, spam, water \n"
				"model: rfc, svm \n"
				"learner: 0, >=1 (use or don't use reduced inputs - if >= 1 defines number of most influential features used) \n"
				"rules: induction(mmd with rule induction), islearn, tree, bayesian \n"
				"coverage: float misprediction coverage percantage of ruleset \n"
				"seed: random seed for execution (int with len == 8) \n"
				"model_number: which of the 10 trained models should be used (int 1-5) \n\n"
				"Or misprediction.py -d <set> -m <model> to train 10 blackboxes and save as .joblib file \n"
				"set: heart(heart failure data set), heartWT(heart failure data set without time), bugreport, java, php, python, ruby, bank, hotel, job, spam, water \n"
				"model: rfc, svm \n"
				)
				
	arg_data = ""
	arg_model = ""
	arg_blackbox = ""
	arg_learner = ""
	arg_rules = ""
	arg_coverage = ""
	arg_seed = ""
	arg_number = ""
	
	try:
		opts, args = getopt.getopt(argv[1:], "hb:d:m:l:r:c:s:n:", ["help", "blackbox=", "data=", "model=", "learner=", "rules=", "coverage=", "seed=", "number="])
	except:
		print(arg_help)
		sys.exit(2)
	
	for opt, arg in opts:
		if opt in ("-h", "--help"):
			print(arg_help)
			sys.exit(2)
		elif opt in ("-d", "--dataset"):
			if arg in ("heart", "heartWT", "bugreport", "java", "php", "python", "ruby", "bank", "hotel", "job", "spam", "water"):
				arg_data = arg
			else:
				print('Dataset has to be in: "heart", "heartWT", "bugreport", "java", "php", "python", "ruby", "bank", "hotel", "job", "spam", "water"')
				sys.exit(2)
		elif opt in ("-m", "--model"):
			if arg in ("rfc", "svm"):
				arg_model = arg
			else:
				print('Model has to be in: "rfc", "svm"')
				sys.exit(2)
		elif opt in ("-b", "--blackbox"):
			if arg in ("heart", "heartWT", "bugreport", "java", "php", "python", "ruby", "bank", "hotel", "job", "spam", "water"):
				arg_blackbox = arg			
			else:
				print('Blackbox has to be in: "heart", "heartWT", "bugreport", "java", "php", "python", "ruby", "bank", "hotel", "job", "spam", "water"')
				sys.exit(2)
		elif opt in ("-l", "--learner"):
			if arg.isdigit():
				if int(arg) >= 0:
					arg_learner = arg			
				else:
					print('Learner has to be in: "0", >="1"')
					sys.exit(2)
			else:
				print('Learner has to be in: "0", >="1"')
				sys.exit(2)
		elif opt in ("-r", "--rules"):
			if arg in ("induction", "islearn", "tree", "bayesian"):
				arg_rules = arg			
			else:
				print('Rules has to be in: "induction", "islearn", "tree", "bayesian"')
				sys.exit(2)
		elif opt in ("-c", "--coverage"):
			try:		
				if float(arg) > 0:
					if float(arg) <=1:
						arg_coverage = arg
					else:
						print('Coverage has to be float x 0<x<=1')
						sys.exit(2)	
				else:
					print('Coverage has to be float x 0<x<=1')
					sys.exit(2)		
			except ValueError:
				print('Coverage has to be float x 0<x<=1')
				sys.exit(2)
		elif opt in ("-s", "--seed"):
			if arg.isdigit():
				if len(arg) == 8:
					arg_seed = arg			
				else:
					print('Seed has to be int with len(seed) == 8')
					sys.exit(2)
			else:
				print('Seed has to be int with len(seed) == 8')
				sys.exit(2)
		elif opt in ("-n", "--number"):
			if arg.isdigit():
				if int(arg) > 0 and int(arg) < 6:
					arg_number = arg			
				else:
					print('Model number has to be int 1-5')
					sys.exit(2)
			else:
				print('Model number has to be int 1-5')
				sys.exit(2)
								
	if  ((arg_blackbox == '' and arg_learner == '' and arg_rules == '' and arg_coverage == '' and arg_seed == '' and arg_number == '' and (arg_data != '' and arg_model != '')) or
		(arg_blackbox != '' and arg_model != '' and arg_learner != '' and arg_rules != '' and arg_coverage != '' and arg_seed != '' and arg_number != '' and (arg_data == ''))):		
		return arg_blackbox, arg_data, arg_model, arg_learner, arg_rules, arg_coverage, arg_seed, arg_number
	else:
		print(arg_help)
		sys.exit(2)
	
if __name__ == "__main__":
	arg_blackbox, arg_data, arg_model, arg_learner, arg_rules, arg_coverage, arg_seed, arg_number = main(sys.argv)

#bayesian optimization objective function
def objective(params):

	feature1_idx = params[0]
	feature2_idx = params[1]
	x = params[2]
	y = params[3]
	
	#calculate weighted f score for each option
	option1 = inputs[(inputs.iloc[:, feature1_idx] > x) & (inputs.iloc[:, feature2_idx] > y)]
	option2 = inputs[(inputs.iloc[:, feature1_idx] > x) & (inputs.iloc[:, feature2_idx] <= y)]
	option3 = inputs[(inputs.iloc[:, feature1_idx] <= x) & (inputs.iloc[:, feature2_idx] > y)]
	option4 = inputs[(inputs.iloc[:, feature1_idx] <= x) & (inputs.iloc[:, feature2_idx] <= y)]
	option5 = inputs[(inputs.iloc[:, feature1_idx] <= x)]
	option6 = inputs[(inputs.iloc[:, feature1_idx] > x)]
	option7 = inputs[(inputs.iloc[:, feature2_idx] <= y)]
	option8 = inputs[(inputs.iloc[:, feature2_idx] > y)]
	
	f_scores = []
	
	for option in [option1, option2, option3, option4, option5, option6, option7, option8]:
		if len(option) == 0:
			f_scores.append(0)
		else:
			true_positives = sum(option['misprediction'] == 1)
			false_positives = sum(option['misprediction'] == 0)
			false_negatives = sum(inputs['misprediction'] == 1) - true_positives
			
			if true_positives + false_positives == 0:
				precision = 0
			else:
				precision = true_positives / (true_positives + false_positives)
				
			if true_positives + false_negatives == 0:
				recall = 0
			else:
				recall = true_positives / (true_positives + false_negatives)
				
			if precision + recall == 0:
				f_score = 0
			else:
				f_score = ((1+(0.5 ** 2)) * precision * recall) / (((0.5 ** 2) * precision) + recall)
				
			f_scores.append(f_score)

	#return best f score
	return -max(f_scores)
	
#pandas print options for better debugging
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

#build misprediction explanation	
if arg_blackbox != "":
	
	#set random seed to input
	seed = int(arg_seed)
	set_seed(seed)
	
	#convert inputs
	arg_learner = int(arg_learner)
	arg_coverage = float(arg_coverage)
	
	#load blackbox from .joblib/.pickle
	print('Loading model from .joblib...')
	model = loadpickle("blackboxes/"+ arg_blackbox + arg_model + arg_number +".joblib", arg_blackbox)
	
	#read dataset from .csv
	XData, yData = loadTestSet(arg_blackbox, arg_model, arg_number)

	print('Check which test inputs are mispredicted by the model...')
	#test_mispredict returns pandas dataframe column containing label for mispredictions
	mispredictions = test_mispredict(XData, yData, model)
	
	#when only most influential features should be used
	if arg_learner > 0:
		
		#start CPU time of extracting most influential features
		start_time_learn = time.process_time()
		
		#print all inputs used to determin influential features
		#print('Print all inputs used to determin influential features...')  
		#print(XData)
			
		#learn most influential input features 
		print('Learn most influential input features...')
		
		#train random forest classifier on misprediction data.
		print('Train random forest classifier on misprediction data...')
		mispredict_model = RandomForestClassifier(random_state=seed)
				
		#use feature importance to explain model
		print('Use feature importance to explain model...')
		mispredict_model.fit(XData, mispredictions.values.ravel())		
		importances = mispredict_model.feature_importances_
		sort_result = importances.argsort()
		influential = [XData.columns[i] for i in sort_result[-arg_learner:]]
		
		#stop CPU time of extracting most influential features
		end_time_learn = time.process_time()	
			
		#print most influential features
		print('Most influential features:')
		print(influential)
		
	#if all input features are used
	if arg_learner == 0:
		
		#build data frames		
		df = pd.concat([XData, yData, mispredictions], axis=1)

		#mmd
		if arg_rules == "induction":
			
			#define relevant features
			relevant = get_relevant(arg_blackbox, XData)
			
			#start CPU time for mmd
			start_time_mmd = time.process_time()
			
			#run mmd for no disjunctions
			print('Run MMD to create misprediction explanation with rule induction(MMD) without disjunctions...')			
			result = discover(df, ('misprediction', True), relevant_attributes=relevant, coverage=arg_coverage, allow_disjunctions=False)
			
			#stop CPU time for mmd
			end_time_mmd = time.process_time()
			
			result_df = result.dataframe()
			
			#select rule sets that reached the coverage
			found = result_df[result_df['recall'] >= arg_coverage].copy()
			
			#choose best rule with weighted f1 score
			#if no rule set reached coverage
			if found.empty:
				result_df['f1_score'] = np.where((result_df['precision'] + result_df['recall']) == 0, 0, (1+(0.5 ** 2)) * (result_df['precision'] * result_df['recall']) / ((0.5 ** 2) * result_df['precision'] + result_df['recall']))
				best = result_df.loc[result_df['f1_score'].idxmax()]

			else:
				found['f1_score'] = np.where((found['precision'] + found['recall']) == 0, 0, (1+(0.5 ** 2)) * (found['precision'] * found['recall']) / ((0.5 ** 2) * found['precision'] + found['recall']))
				best = found.loc[found['f1_score'].idxmax()]
			
			#result.print()
			#print(result.dataframe().to_string())
			#print(result_df)
			#print(found)

			#extact and rewrite best ruleset
			print('Best ruleset with all features and rule induction(MMD) without disjunctions:\n')
			mmd_best = best.values
			mmd_output_string = mmd_ruleset_string(mmd_best[0])

			#calculate specificity
			ruleset = RuleSet()
			ruleset.set_ruleset(mmd_output_string)
			specificity = ruleset.eval_ruleset_spec(df)
			
			#print rule set and metrics
			print(mmd_output_string)
			print("\nSpecificity: " + str(round(specificity, 3)) + " Precision: " + str(round(mmd_best[1], 3)) + " Recall: " + str(round(mmd_best[2], 3))) 
			#print CPU time used
			print(f"CPU time used without disjunctions: {end_time_mmd - start_time_mmd} seconds")
			
			#start CPU time for mmd with disjunctions
			start_time_mmd_dis = time.process_time()
			
			#run mmd with disjunctions
			print('Run MMD to create misprediction explanation with rule induction(MMD) with disjunctions...')
			result = discover(df, ('misprediction', True), relevant_attributes=relevant, coverage=arg_coverage, allow_disjunctions=True)
			
			#end CPU time for mmd with disjunctions
			end_time_mmd_dis = time.process_time()			
			
			result_df = result.dataframe()
			
			#select rule sets that reached the coverage
			found = result_df[result_df['recall'] >= arg_coverage].copy()
			
			#choose best rule with weighted f1 score
			#if no rule set reached coverage
			if found.empty:
				result_df['f1_score'] = np.where((result_df['precision'] + result_df['recall']) == 0, 0, (1+(0.5 ** 2)) * (result_df['precision'] * result_df['recall']) / ((0.5 ** 2) * result_df['precision'] + result_df['recall']))
				best = result_df.loc[result_df['f1_score'].idxmax()]

			else:
				found['f1_score'] = np.where((found['precision'] + found['recall']) == 0, 0, (1+(0.5 ** 2)) * (found['precision'] * found['recall']) / ((0.5 ** 2) * found['precision'] + found['recall']))
				best = found.loc[found['f1_score'].idxmax()]
			
			#result.print()
			#print(result_df)
			#print(found)
			#print(result.dataframe().to_string())
		
			#extact and rewrite best ruleset
			print('Best ruleset with all features and rule induction(MMD) with disjunctions:\n')
			mmd_best = best.values
			mmd_output_string = mmd_ruleset_string(mmd_best[0])

			#calculate specificity
			ruleset = RuleSet()
			ruleset.set_ruleset(mmd_output_string)
			specificity = ruleset.eval_ruleset_spec(df)
			
			#print rule set and metrics
			print(mmd_output_string)
			print("\nSpecificity: " + str(round(specificity, 3)) + " Precision: " + str(round(mmd_best[1], 3)) + " Recall: " + str(round(mmd_best[2], 3)))
			#print CPU time used
			print(f"CPU time used with disjunctions: {end_time_mmd_dis - start_time_mmd_dis} seconds") 

		#islearn
		elif arg_rules == "islearn":
			
			#start CPU time for islearn
			start_time_islearn = time.process_time()
			
			#set the correct grammar for islearn
			if arg_blackbox == "heart":
				islagrammar = grammarHF
			elif arg_blackbox == "heartWT":
				islagrammar = grammarHFWT
			elif arg_blackbox == "bugreport":
				islagrammar = grammarBR
			elif arg_blackbox == "java":
				islagrammar = grammarJAVA
			elif arg_blackbox == "php":
				islagrammar = grammarPHP
			elif arg_blackbox == "python":
				islagrammar = grammarPYTHON
			elif arg_blackbox == "ruby":
				islagrammar = grammarRUBY
			elif arg_blackbox == "spam":
				islagrammar = grammarSPAM
			elif arg_blackbox == "water":
				islagrammar = grammarWATER
			elif arg_blackbox == "hotel":
				islagrammar = grammarHOTEL
			elif arg_blackbox == "job":
				islagrammar = grammarJOB
			elif arg_blackbox == "bank":
				islagrammar = grammarBANK
		
			print('Preprocess data for islearn...')				
			#preprocess data because islearn cant handle floats and negative values
			normalized = df.copy()
			for col in df.columns:
				if df[col].min() != df[col].max():
					normalized[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
			rounded = normalized.round(8)
			multiplied = (rounded * 100000000).round()
			to_int = multiplied.astype(int)

			#split df into positive and negative for misprediction and remove target information
			positive = to_int.loc[to_int['misprediction'] == 100000000].iloc[:,:-2]
			negative = to_int.loc[to_int['misprediction'] == 0].iloc[:,:-2]

			print('Construct string inputs...')			
			#build inputs from dataframe for positive and negative examples
			buildstringinputsPositive = positive.to_string(header=False, index=False, index_names=False).split('\n')
			buildstringinputsNegative = negative.to_string(header=False, index=False, index_names=False).split('\n')	
			POSITIVE_INPUTS = [' '.join(ele.split()) for ele in buildstringinputsPositive]
			NEGATIVE_INPUTS = [' '.join(ele.split()) for ele in buildstringinputsNegative]	
			positive_input_list = [DerivationTree.from_parse_tree(next(EarleyParser(islagrammar).parse(inp))) for inp in POSITIVE_INPUTS]
			negative_input_list = [DerivationTree.from_parse_tree(next(EarleyParser(islagrammar).parse(inp))) for inp in NEGATIVE_INPUTS]
						
			#find unrelevant nonterminals in grammar
			all_nonterminals = set()
			relevant_nonterminals = set()
			all_nonterminals.update(reachable_nonterminals(islagrammar, '<start>'))
			relevant_nonterminals.update(islagrammar['<start>'][0].split())
			unrelevant_nonterminals = all_nonterminals - relevant_nonterminals
			unrelevant_nonterminals.remove('<start>')
			print('Unrelevant nonterminals in grammar...')
			print(unrelevant_nonterminals)
					
			print('Run IsLearn to create misprediction explanation...')
			#run islearn with pos and neg misprediction examples
			result: Dict[Formula, Tuple[float, float]] = InvariantLearner(
				islagrammar,
				positive_examples = positive_input_list,
				negative_examples = negative_input_list,
				pattern_file = "used_patterns.toml",
				min_recall = arg_coverage,
				min_specificity = 0.0,
				max_disjunction_size = 1,
				max_conjunction_size = 2,
				generate_new_learning_samples = False,
				do_generate_more_inputs = False,
				filter_inputs_for_learning_by_kpaths = False,
				exclude_nonterminals = unrelevant_nonterminals
			).learn_invariants()

			#end CPU time for islearn
			end_time_islearn = time.process_time()
						
			#print islearn results
			#rulesets_it = map(
				#lambda p: f"{p[1]}: " + ISLaUnparser(p[0]).unparse(),
				#{f: p for f, p in result.items() if p[0] > .0}.items())
			#rulesets = list(rulesets_it)
			#print(rulesets[0])
			
			#extact and rewrite best ruleset
			islearn_best = list(next(iter(result.items())))
			islearn_output_string = islearn_ruleset_string(ISLaUnparser(islearn_best[0]).unparse(), df)
			
			#calculate precision of full ruleset
			ruleset = RuleSet()
			ruleset.set_ruleset(islearn_output_string)
			precision = ruleset.eval_ruleset_prec(df)
			
			#print rule set and metrics
			print('Best ruleset with all features and islearn without disjunctions:\n')
			print(islearn_output_string)
			print("\nSpecificity: " + str(round(islearn_best[1][0], 3)) + " Precision: " + str(round(precision, 3)) + " Recall: " + str(round(islearn_best[1][1], 3)))
			#print CPU time used
			print(f"CPU time used without disjunctions: {end_time_islearn - start_time_islearn} seconds") 
			
		#decision trees
		elif arg_rules == "tree":
			
			#start CPU time for decision trees
			start_time_tree = time.process_time()	
					
			current_coverage = 0.0
			uncovered = df
			ruleset = RuleSet()
			runs = 0
						
			#check for max decimal places for rounding
			max_decimal_places = 0
			for col in uncovered.columns:
				if uncovered[col].dtype == float:
					max_decimal_places = max(max_decimal_places, (uncovered[col].astype(str).str.split('.').str[1].str.len().max()))

			print('Create misprediction explanation by learning decision trees...')
			#do until desired coverage of mispredictions is reached
			while current_coverage < arg_coverage:
				
				#print(uncovered)
				
				runs += 1

				#train decision tree on uncovered inputs
				model, names = train_tree(uncovered, arg_blackbox, arg_model, arg_learner, arg_coverage, 2, runs, seed, arg_number)
				
				#extract rules from decision tree
				ruleset.extract_rules(model, names, max_decimal_places+1)
				
				#add best extracted rule to ruleset
				ruleset.add_best()
				
				#calculate coverage with ruleset build so far
				current_coverage = ruleset.check_coverage()
				
				#if not enough coverage...
				if current_coverage < arg_coverage:	
					#split off inputs that are still uncovered
					uncovered = ruleset.get_uncovered(uncovered)
				
				#print current results
				#print("\n" + str(runs) + ". Run:")
				#print("Rule set:\n\n" + ruleset.get_ruleset())		
				#calculate precision of ruleset
				#precision = ruleset.eval_ruleset_prec(df)
				#specificity precision of ruleset
				#specificity = ruleset.eval_ruleset_spec(df)
				#print("\nSpecificity: " + str(round(specificity, 3)) + " Precision: " + str(round(precision, 3)) + " Recall: " + str(round(current_coverage, 3)))
				
				#result for decision trees without disjunctions
				if runs == 1:
					#start CPU time for decision trees without disjunctions output
					start_time_tree_wo = time.process_time()
					
					#print rule set and metrics
					print('Best ruleset with all features and decision trees without disjunctions:\n')
					print(ruleset.get_ruleset())
					#calculate precision of full ruleset
					precision = ruleset.eval_ruleset_prec(df)
					#specificity precision of full ruleset
					specificity = ruleset.eval_ruleset_spec(df)
					print("\nSpecificity: " + str(round(specificity, 3)) + " Precision: " + str(round(precision, 3)) + " Recall: " + str(round(current_coverage, 3)))
					#print CPU time used
					print(f"CPU time used without disjunctions: {start_time_tree_wo - start_time_tree} seconds\n") 
					
					#end CPU time for decision trees without disjunctions output
					end_time_tree_wo = time.process_time()	
				
			#end CPU time for decision trees with disjunctions
			end_time_tree = time.process_time()
			
			#print rule set and metrics
			print('Best ruleset with all features and decision trees with disjunctions:\n')
			print(ruleset.get_ruleset())
			#calculate precision of full ruleset
			precision = ruleset.eval_ruleset_prec(df)
			#specificity precision of full ruleset
			specificity = ruleset.eval_ruleset_spec(df)
			print("\nSpecificity: " + str(round(specificity, 3)) + " Precision: " + str(round(precision, 3)) + " Recall: " + str(round(current_coverage, 3)))
			#print CPU time used
			print(f"CPU time used with disjunctions: {end_time_tree - start_time_tree - (end_time_tree_wo - start_time_tree_wo)} seconds") 	
										
		#bayesian optimization
		elif arg_rules == "bayesian":
			
			#start CPU time for bayesian
			start_time_bayes = time.process_time()
		
			current_coverage = 0.0
			ruleset = RuleSet()
			runs = 0
						
			#prepare inputs
			uncovered = df.astype(float).round(2)
	
			#set bounds for bayesian optimization
			bounds = [
				(0, len(uncovered.columns) - 3),
				(0, len(uncovered.columns) - 3),
				(0.0, 1.0),
				(0.0, 1.0),
			]	
				
			print('Create misprediction explanation by utilizing bayesian optimization...')
			#do until desired coverage of mispredictions is reached
			while current_coverage < arg_coverage:
				
				#normalize inputs
				inputs = uncovered.copy()
				for col in uncovered.columns:
					if uncovered[col].min() != uncovered[col].max():
						inputs[col] = (uncovered[col] - uncovered[col].min()) / (uncovered[col].max() - uncovered[col].min())
						
				runs += 1

				#run bayesian optimization(filter warnings)
				with warnings.catch_warnings():
					warnings.filterwarnings("ignore", category=UserWarning, message="The objective has been evaluated at this point before.")
					result = gp_minimize(objective, bounds, n_calls=40, n_initial_points=25, acq_optimizer="lbfgs", n_points=250, random_state=seed)
				
				#based on the results find out which operators have to be used
				best_params = result.x
				feature1_idx = best_params[0]
				feature2_idx = best_params[1]
				x_norm = best_params[2]
				y_norm = best_params[3]
				
				#recreate original value range for x and y
				x_min = uncovered.iloc[:, feature1_idx].min()
				x_max = uncovered.iloc[:, feature1_idx].max()
				x = x_norm * (x_max - x_min) + x_min
				
				y_min = uncovered.iloc[:, feature2_idx].min()
				y_max = uncovered.iloc[:, feature2_idx].max()
				y = y_norm * (y_max - y_min) + y_min
							
				#calculate weighted f score for each option
				option1 = uncovered[(uncovered.iloc[:, feature1_idx] > x) & (uncovered.iloc[:, feature2_idx] > y)]
				option2 = uncovered[(uncovered.iloc[:, feature1_idx] > x) & (uncovered.iloc[:, feature2_idx] <= y)]
				option3 = uncovered[(uncovered.iloc[:, feature1_idx] <= x) & (uncovered.iloc[:, feature2_idx] > y)]
				option4 = uncovered[(uncovered.iloc[:, feature1_idx] <= x) & (uncovered.iloc[:, feature2_idx] <= y)]
				option5 = inputs[(inputs.iloc[:, feature1_idx] <= x)]
				option6 = inputs[(inputs.iloc[:, feature1_idx] > x)]
				option7 = inputs[(inputs.iloc[:, feature2_idx] <= y)]
				option8 = inputs[(inputs.iloc[:, feature2_idx] > y)]
				
				f_scores = []
				
				for option in [option1, option2, option3, option4, option5, option6, option7, option8]:
					if len(option) == 0:
						f_scores.append(0)
					else:
						true_positives = sum(option['misprediction'] == 1)
						false_positives = sum(option['misprediction'] == 0)
						false_negatives = sum(uncovered['misprediction'] == 1) - true_positives
						
						if true_positives + false_positives == 0:
							precision_score = 0
						else:
							precision_score = true_positives / (true_positives + false_positives)
							
						if true_positives + false_negatives == 0:
							recall_score = 0
						else:
							recall_score = true_positives / (true_positives + false_negatives)
							
						if precision_score + recall_score == 0:
							f_score = 0
						else:
							f_score = ((1+(0.5 ** 2)) * precision_score * recall_score) / (((0.5 ** 2) * precision_score) + recall_score)
												
						f_scores.append(f_score)	
		
				#build best rule string
				option_num = f_scores.index(max(f_scores))
				string_options = [
					f"{uncovered.columns[feature1_idx]} > {x:.2f} and {uncovered.columns[feature2_idx]} > {y:.2f}",
					f"{uncovered.columns[feature1_idx]} > {x:.2f} and {uncovered.columns[feature2_idx]} <= {y:.2f}",
					f"{uncovered.columns[feature1_idx]} <= {x:.2f} and {uncovered.columns[feature2_idx]} > {y:.2f}",
					f"{uncovered.columns[feature1_idx]} <= {x:.2f} and {uncovered.columns[feature2_idx]} <= {y:.2f}",
					f"{uncovered.columns[feature1_idx]} <= {x:.2f}",
					f"{uncovered.columns[feature1_idx]} > {x:.2f}",
					f"{uncovered.columns[feature2_idx]} <= {y:.2f}",
					f"{uncovered.columns[feature2_idx]} > {y:.2f}"
				]
				
				best_rule = string_options[option_num]				
		
				#add best extracted rule to ruleset
				if runs > 1:
					ruleset.add_ruleset(" or ")
				ruleset.add_ruleset(best_rule)
		
				#calculate coverage with ruleset build so far
				current_coverage = ruleset.check_coverage_df(df)
								
				#if not enough coverage...
				if current_coverage < arg_coverage:	
					#split off inputs that are still uncovered
					uncovered = ruleset.get_uncovered(uncovered)
					
				#print current results
				#print("\n" + str(runs) + ". Run:")
				#print("Rule set:\n\n" + ruleset.get_ruleset())
				#calculate precision of ruleset
				#precision = ruleset.eval_ruleset_prec(df)
				#specificity precision of ruleset
				#specificity = ruleset.eval_ruleset_spec(df)
				#print("\nSpecificity: " + str(round(specificity, 3)) + " Precision: " + str(round(precision, 3)) + " Recall: " + str(round(current_coverage, 3)))

				#result for bayesian without disjunctions
				if runs == 1:
					#start CPU time for bayesian without disjunctions output
					start_time_bayes_wo = time.process_time()
					
					#print rule set and metrics
					print('Best ruleset with all features and bayesian optimization without disjunctions:\n')
					print(ruleset.get_ruleset())
					#calculate precision of full ruleset
					precision = ruleset.eval_ruleset_prec(df)
					#specificity precision of full ruleset
					specificity = ruleset.eval_ruleset_spec(df)
					print("\nSpecificity: " + str(round(specificity, 3)) + " Precision: " + str(round(precision, 3)) + " Recall: " + str(round(current_coverage, 3)))
					#print CPU time used
					print(f"CPU time used without disjunctions: {start_time_bayes_wo - start_time_bayes} seconds\n") 
					
					#end CPU time for bayesian without disjunctions output
					end_time_bayes_wo = time.process_time()	

				if (time.process_time() - start_time_bayes) > 7200:
					print('Result after max time reached:')
					break

			#end CPU time for bayesian
			end_time_bayes = time.process_time()

			#print rule set and metrics
			print('Best ruleset with all features and bayesian optimization with disjunctions:\n')
			print(ruleset.get_ruleset())
			#calculate precision of full ruleset
			precision = ruleset.eval_ruleset_prec(df)
			#specificity precision of full ruleset
			specificity = ruleset.eval_ruleset_spec(df)
			print("\nSpecificity: " + str(round(specificity, 3)) + " Precision: " + str(round(precision, 3)) + " Recall: " + str(round(current_coverage, 3)))
			#print CPU time used
			print(f"CPU time used with disjunctions: {end_time_bayes - start_time_bayes - (end_time_bayes_wo - start_time_bayes_wo)} seconds") 
							
	#if only the most influential input features are used
	elif arg_learner > 0:
			
		#build new input dataframe with relevant features
		print('Build new input dataframe with relevant features...')
		df = XData.loc[:, influential]
		dff = pd.concat([df, yData, mispredictions], axis=1)
		
		#print('Dataframe used for rule set generation:')
		#print(dff.to_string())

		#mmd
		if arg_rules == "induction":
		
			#create dict with relevant features for mmd
			relevant = get_relevant(arg_blackbox, XData)
			relevantf = {i: j for i, j in relevant.items() if i in influential}
					
			#start CPU time for mmd without disjunctions
			start_time_mmd = time.process_time()					
					
			#run mmd for no disjunctions
			print('Run MMD to create misprediction explanation with rule induction(MMD) without disjunctions...')
			resultf = discover(dff, ('misprediction', True), relevant_attributes=relevantf, coverage=arg_coverage, allow_disjunctions=False)
			
			#end CPU time for mmd without disjunctions
			end_time_mmd = time.process_time()
			
			result_df = resultf.dataframe()
			
			#select rule sets that reached the coverage
			found = result_df[result_df['recall'] >= arg_coverage].copy()
			
			#choose best rule with weighted f1 score
			#if no rule set reached coverage
			if found.empty:
				result_df['f1_score'] = np.where((result_df['precision'] + result_df['recall']) == 0, 0, (1+(0.5 ** 2)) * (result_df['precision'] * result_df['recall']) / ((0.5 ** 2) * result_df['precision'] + result_df['recall']))
				best = result_df.loc[result_df['f1_score'].idxmax()]

			else:
				found['f1_score'] = np.where((found['precision'] + found['recall']) == 0, 0, (1+(0.5 ** 2)) * (found['precision'] * found['recall']) / ((0.5 ** 2) * found['precision'] + found['recall']))
				best = found.loc[found['f1_score'].idxmax()]
			
			#resultf.print()
			#print(result_df)
			#print(found)
			#print(resultf.dataframe().to_string())
		
			#extact and rewrite best ruleset
			print('Best ruleset with most influential features and rule induction(MMD) without disjunctions:\n')
			mmd_best = best.values
			mmd_output_string = mmd_ruleset_string(mmd_best[0])

			#calculate specificity
			ruleset = RuleSet()
			ruleset.set_ruleset(mmd_output_string)
			specificity = ruleset.eval_ruleset_spec(dff)
			
			#print rule set and metrics
			print(mmd_output_string)
			print("\nSpecificity: " + str(round(specificity, 3)) + " Precision: " + str(round(mmd_best[1], 3)) + " Recall: " + str(round(mmd_best[2], 3))) 
			#print CPU time used
			print(f"CPU time used extracting features: {end_time_learn - start_time_learn} seconds")
			print(f"CPU time used overall without disjunctions: {(end_time_mmd - start_time_mmd) + (end_time_learn - start_time_learn)} seconds") 
			
			#start CPU time for mmd with disjunctions
			start_time_mmd_dis = time.process_time()

			#run mmd with disjunctions
			print('Run MMD to create misprediction explanation with rule induction(MMD) with disjunctions...')
			resultf = discover(dff, ('misprediction', True), relevant_attributes=relevantf, coverage=arg_coverage, allow_disjunctions=True)
			
			#end CPU time for mmd with disjunctions
			end_time_mmd_dis = time.process_time()
			
			result_df = resultf.dataframe()
			
			#select rule sets that reached the coverage
			found = result_df[result_df['recall'] >= arg_coverage].copy()
			
			#choose best rule with weighted f1 score
			#if no rule set reached coverage
			if found.empty:
				result_df['f1_score'] = np.where((result_df['precision'] + result_df['recall']) == 0, 0, (1+(0.5 ** 2)) * (result_df['precision'] * result_df['recall']) / ((0.5 ** 2) * result_df['precision'] + result_df['recall']))
				best = result_df.loc[result_df['f1_score'].idxmax()]

			else:
				found['f1_score'] = np.where((found['precision'] + found['recall']) == 0, 0, (1+(0.5 ** 2)) * (found['precision'] * found['recall']) / ((0.5 ** 2) * found['precision'] + found['recall']))
				best = found.loc[found['f1_score'].idxmax()]
			
			#resultf.print()
			#print(result_df)
			#print(found)
			#print(resultf.dataframe().to_string())
		
			#extact and rewrite best ruleset
			print('Best ruleset with most influential features and rule induction(MMD) with disjunctions:\n')
			mmd_best = best.values
			mmd_output_string = mmd_ruleset_string(mmd_best[0])

			#calculate specificity
			ruleset = RuleSet()
			ruleset.set_ruleset(mmd_output_string)
			specificity = ruleset.eval_ruleset_spec(dff)
			
			#print rule set and metrics
			print(mmd_output_string)
			print("\nSpecificity: " + str(round(specificity, 3)) + " Precision: " + str(round(mmd_best[1], 3)) + " Recall: " + str(round(mmd_best[2], 3))) 		
			#print CPU time used
			print(f"CPU time used extracting features: {end_time_learn - start_time_learn} seconds") 
			print(f"CPU time used overall with disjunctions: {(end_time_mmd_dis - start_time_mmd_dis) + (end_time_learn - start_time_learn)} seconds") 
	
		#islearn
		elif arg_rules == "islearn":

			#start CPU time for islearn preparation
			start_time_islearn_prep = time.process_time()

			#set the correct grammar for islearn
			if arg_blackbox == "heart":
				islagrammar = grammarHF
			elif arg_blackbox == "heartWT":
				islagrammar = grammarHFWT
			elif arg_blackbox == "bugreport":
				islagrammar = grammarBR
			elif arg_blackbox == "java":
				islagrammar = grammarJAVA
			elif arg_blackbox == "php":
				islagrammar = grammarPHP
			elif arg_blackbox == "python":
				islagrammar = grammarPYTHON
			elif arg_blackbox == "ruby":
				islagrammar = grammarRUBY
			elif arg_blackbox == "spam":
				islagrammar = grammarSPAM
			elif arg_blackbox == "water":
				islagrammar = grammarWATER
			elif arg_blackbox == "hotel":
				islagrammar = grammarHOTEL
			elif arg_blackbox == "job":
				islagrammar = grammarJOB
			elif arg_blackbox == "bank":
				islagrammar = grammarBANK

			print('Preprocess data for islearn...')
			#preprocess data because islearn cant handle floats and negative values
			normalized = dff.copy()
			for col in df.columns:
				if df[col].min() != df[col].max():
					normalized[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
			rounded = normalized.round(8)
			multiplied = (rounded * 100000000).round()
			to_int = multiplied.astype(int)

			#split df into positive and negative for misprediction and remove target information
			positive = to_int.loc[to_int['misprediction'] == 100000000].iloc[:,:-2]
			negative = to_int.loc[to_int['misprediction'] == 0].iloc[:,:-2]

			print('Construct string inputs...')	
			#build inputs from dataframe for positive and negative examples
			buildstringinputsPositive = positive.to_string(header=False, index=False, index_names=False).split('\n')
			buildstringinputsNegative = negative.to_string(header=False, index=False, index_names=False).split('\n')	
			POSITIVE_INPUTS = [' '.join(ele.split()) for ele in buildstringinputsPositive]
			NEGATIVE_INPUTS = [' '.join(ele.split()) for ele in buildstringinputsNegative]	

			#change grammar to match inputs			
			column_names = list(dff.columns.values)
			start = islagrammar['<start>'][0].split()
			print(dff.columns.values)
			print(influential)
			#adjust start
			newstart = ''												
			for x in column_names:
				for y in influential:
					if x == y:
						newstart += '<' + x + '> '
						
			newstart = newstart[:-1]
			islagrammar['<start>'] = [newstart]
			
			#remove unreachable
			unneeded = unreachable_nonterminals(islagrammar, '<start>')
			for x in unneeded:
				del islagrammar[x]
	
			print('New grammar...')
			print(islagrammar)

			#build DerivationTree inputs for islearn
			positive_input_list = [DerivationTree.from_parse_tree(next(EarleyParser(islagrammar).parse(inp))) for inp in POSITIVE_INPUTS]
			negative_input_list = [DerivationTree.from_parse_tree(next(EarleyParser(islagrammar).parse(inp))) for inp in NEGATIVE_INPUTS]
			
			#find unrelevant nonterminals in grammar
			all_nonterminals = set()
			relevant_nonterminals = set()
			all_nonterminals.update(reachable_nonterminals(islagrammar, '<start>'))
			relevant_nonterminals.update(islagrammar['<start>'][0].split())
			unrelevant_nonterminals = all_nonterminals - relevant_nonterminals
			unrelevant_nonterminals.remove('<start>')
			print('Unrelevant nonterminals in grammar...')
			print(unrelevant_nonterminals)
			
			#end CPU time for islearn preparation
			end_time_islearn_prep = time.process_time()
			
			#start CPU time for islearn without disjunctions
			start_time_islearn = time.process_time()

			print('Run IsLearn to create misprediction explanation without disjunctions...')
			#run islearn with pos and neg misprediction examples
			result: Dict[Formula, Tuple[float, float]] = InvariantLearner(
				islagrammar,
				positive_examples = positive_input_list,
				negative_examples = negative_input_list,
				pattern_file = "used_patterns.toml",
				min_recall = arg_coverage,
				min_specificity = 0.0,
				max_disjunction_size = 1,
				max_conjunction_size = 2,
				generate_new_learning_samples = False,
				do_generate_more_inputs = False,
				filter_inputs_for_learning_by_kpaths = False,
				exclude_nonterminals = unrelevant_nonterminals
			).learn_invariants()

			#end CPU time for islearn without disjunctions
			end_time_islearn = time.process_time()
									
			#print islearn results
			#rulesets_it = map(
				#lambda p: f"{p[1]}: " + ISLaUnparser(p[0]).unparse(),
				#{f: p for f, p in result.items() if p[0] > .0}.items())
			#rulesets = list(rulesets_it)
			#print(rulesets[0])
			
			#extact and rewrite best ruleset
			islearn_best = list(next(iter(result.items())))
			islearn_output_string = islearn_ruleset_string(ISLaUnparser(islearn_best[0]).unparse(), dff)
			
			#calculate precision of full ruleset
			ruleset = RuleSet()
			ruleset.set_ruleset(islearn_output_string)
			precision = ruleset.eval_ruleset_prec(dff)
			
			#print rule set and metrics
			print('Best ruleset with most influential features and Islearn without disjunctions:\n')
			print(islearn_output_string)
			print("\nSpecificity: " + str(round(islearn_best[1][0], 3)) + " Precision: " + str(round(precision, 3)) + " Recall: " + str(round(islearn_best[1][1], 3)))
			#print CPU time used
			print(f"CPU time used extracting features: {end_time_learn - start_time_learn} seconds") 
			print(f"CPU time used overall without disjunctions: {(end_time_islearn- start_time_islearn) + (end_time_learn - start_time_learn) + (end_time_islearn_prep - start_time_islearn_prep)} seconds") 	
			
			#ISLearn with disjunctions too expensive
			"""
			#start CPU time for islearn with disjunctions
			start_time_islearn_dis = time.process_time()
			
			print('Run IsLearn to create misprediction explanation with disjunctions...')
			#run islearn with pos and neg misprediction examples
			result2: Dict[Formula, Tuple[float, float]] = InvariantLearner(
				islagrammar,
				positive_examples = positive_input_list,
				negative_examples = negative_input_list,
				pattern_file = "used_patterns.toml",
				min_recall = arg_coverage,
				min_specificity = 0.0,
				max_disjunction_size = 2,
				max_conjunction_size = 2,
				reduce_inputs_for_learning = False,
				generate_new_learning_samples = False,
				do_generate_more_inputs = False,
				filter_inputs_for_learning_by_kpaths = False,
				exclude_nonterminals = unrelevant_nonterminals
			).learn_invariants()			

			#end CPU time for islearn with disjunctions
			end_time_islearn_dis = time.process_time()

			#extact and rewrite best ruleset
			islearn_best2 = list(next(iter(result2.items())))
			islearn_output_string2 = islearn_ruleset_string(ISLaUnparser(islearn_best2[0]).unparse(), dff)
			
			#calculate precision of full ruleset
			ruleset2 = RuleSet()
			ruleset2.set_ruleset(islearn_output_string2)
			precision2 = ruleset2.eval_ruleset_prec(dff)
			
			#print rule set and metrics
			print('Best ruleset with most influential features and Islearn with disjunctions:\n')
			print(islearn_output_string2)
			print("\nSpecificity: " + str(round(islearn_best2[1][0], 3)) + " Precision: " + str(round(precision2, 3)) + " Recall: " + str(round(islearn_best2[1][1], 3)))
			#print CPU time used
			print(f"CPU time used extracting features: {end_time_learn - start_time_learn} seconds") 
			print(f"CPU time used with disjunctions: {(end_time_islearn_dis - start_time_islearn_dis) + (end_time_islearn_prep - start_time_islearn_prep)} seconds") 
			print(f"CPU time used overall with disjunctions: {(end_time_islearn_dis - start_time_islearn_dis) + (end_time_learn - start_time_learn) + (end_time_islearn_prep - start_time_islearn_prep)} seconds") 			
			"""		
			
		#decision trees
		elif arg_rules == "tree":

			#start CPU time for trees
			start_time_tree = time.process_time()
			
			current_coverage = 0.0
			uncovered = dff
			ruleset = RuleSet()
			runs = 0

			#check for max decimal places for rounding
			max_decimal_places = 0
			for col in uncovered.columns:
				if uncovered[col].dtype == float:
					max_decimal_places = int(max(max_decimal_places, (uncovered[col].astype(str).str.split('.').str[1].str.len().max())))

			print('Create misprediction explanation by learning decision trees...')
			#do until desired coverage of mispredictions is reached
			while current_coverage < arg_coverage:
				
				#print(uncovered)
				
				runs += 1
				
				#train decision tree on uncovered inputs
				model, names = train_tree(uncovered, arg_blackbox, arg_model, arg_learner, arg_coverage, 2, runs, seed, arg_number)
				
				#extract rules from decision tree
				ruleset.extract_rules(model, names, max_decimal_places+1)
				
				#add best extracted rule to ruleset
				ruleset.add_best()
				
				#calculate coverage with ruleset build so far
				current_coverage = ruleset.check_coverage()
				
				#if not enough coverage...
				if current_coverage < arg_coverage:	
					#split off inputs that are still uncovered
					uncovered = ruleset.get_uncovered(uncovered)
				
				#print current results
				#print("\n" + str(runs) + ". Run:")
				#print("Rule set:\n\n" + ruleset.get_ruleset())
				#calculate precision of ruleset
				#precision = ruleset.eval_ruleset_prec(dff)
				#calculate specificity of ruleset
				#specificity = ruleset.eval_ruleset_spec(dff)
				#print("\nSpecificity: " + str(round(specificity, 3)) + " Precision: " + str(round(precision, 3)) + " Recall: " + str(round(current_coverage, 3)))	
				
				#result for decision trees without disjunctions
				if runs == 1:
					#start CPU time for decision trees without disjunctions output
					start_time_tree_wo = time.process_time()
					
					#print rule set and metrics
					print('Best ruleset with most influential features and decision trees without disjunctions:\n')
					print(ruleset.get_ruleset())
					#calculate precision of full ruleset
					precision = ruleset.eval_ruleset_prec(dff)
					#specificity precision of full ruleset
					specificity = ruleset.eval_ruleset_spec(dff)
					print("\nSpecificity: " + str(round(specificity, 3)) + " Precision: " + str(round(precision, 3)) + " Recall: " + str(round(current_coverage, 3)))
					#print CPU time used
					print(f"CPU time used extracting features: {end_time_learn - start_time_learn} seconds") 
					print(f"CPU time used overall without disjunctions: {(start_time_tree_wo - start_time_tree) + (end_time_learn - start_time_learn)} seconds\n") 
					
					#end CPU time for decision trees without disjunctions output
					end_time_tree_wo = time.process_time()
			
			#end CPU time for trees
			end_time_tree = time.process_time()
								
			#print rule set and metrics
			print('Best ruleset with most influential features and decision trees with disjunctions:\n')
			print(ruleset.get_ruleset())
			#calculate precision of full ruleset
			precision = ruleset.eval_ruleset_prec(dff)
			#specificity precision of full ruleset
			specificity = ruleset.eval_ruleset_spec(dff)
			print("\nSpecificity: " + str(round(specificity, 3)) + " Precision: " + str(round(precision, 3)) + " Recall: " + str(round(current_coverage, 3)))
			#print CPU time used
			print(f"CPU time used extracting features: {end_time_learn - start_time_learn} seconds") 
			print(f"CPU time used overall with disjunctions: {(end_time_tree- start_time_tree) + (end_time_learn - start_time_learn) - (end_time_tree_wo - start_time_tree_wo)} seconds") 	
	
		#bayesian optimization
		elif arg_rules == "bayesian":

			#start CPU time for bayesian
			start_time_bayes = time.process_time()

			current_coverage = 0.0
			ruleset = RuleSet()
			runs = 0
			
			#prepare inputs
			uncovered = dff.astype(float).round(2)
	
			#set bounds for bayesian optimization
			bounds = [
				(0, len(uncovered.columns) - 3),
				(0, len(uncovered.columns) - 3),
				(0.0, 1.0),
				(0.0, 1.0)	
			]
					
			print('Create misprediction explanation utilizing bayesian optimization...')
			#do until desired coverage of mispredictions is reached
			while current_coverage < arg_coverage:
				
				#print(uncovered)
				
				#normalize inputs
				inputs = uncovered.copy()
				for col in uncovered.columns:
					if uncovered[col].min() != uncovered[col].max():
						inputs[col] = (uncovered[col] - uncovered[col].min()) / (uncovered[col].max() - uncovered[col].min())
				
				runs += 1

				#run bayesian optimization(filter warnings)
				with warnings.catch_warnings():
					warnings.filterwarnings("ignore", category=UserWarning, message="The objective has been evaluated at this point before.")
					result = gp_minimize(objective, bounds, n_calls=40, n_initial_points=25, acq_optimizer="lbfgs", n_points=250, random_state=seed)
				
				#based on the results find out which operators have to be used
				best_params = result.x
				feature1_idx = best_params[0]
				feature2_idx = best_params[1]
				x_norm = best_params[2]
				y_norm = best_params[3]
				
				#recreate original value range for x and y
				x_min = uncovered.iloc[:, feature1_idx].min()
				x_max = uncovered.iloc[:, feature1_idx].max()
				x = x_norm * (x_max - x_min) + x_min
				
				y_min = uncovered.iloc[:, feature2_idx].min()
				y_max = uncovered.iloc[:, feature2_idx].max()
				y = y_norm * (y_max - y_min) + y_min
							
				#calculate weighted f score for each option
				option1 = uncovered[(uncovered.iloc[:, feature1_idx] > x) & (uncovered.iloc[:, feature2_idx] > y)]
				option2 = uncovered[(uncovered.iloc[:, feature1_idx] > x) & (uncovered.iloc[:, feature2_idx] <= y)]
				option3 = uncovered[(uncovered.iloc[:, feature1_idx] <= x) & (uncovered.iloc[:, feature2_idx] > y)]
				option4 = uncovered[(uncovered.iloc[:, feature1_idx] <= x) & (uncovered.iloc[:, feature2_idx] <= y)]
				option5 = inputs[(inputs.iloc[:, feature1_idx] <= x)]
				option6 = inputs[(inputs.iloc[:, feature1_idx] > x)]
				option7 = inputs[(inputs.iloc[:, feature2_idx] <= y)]
				option8 = inputs[(inputs.iloc[:, feature2_idx] > y)]
				
				f_scores = []
				
				for option in [option1, option2, option3, option4, option5, option6, option7, option8]:
					if len(option) == 0:
						f_scores.append(0)
					else:
						true_positives = sum(option['misprediction'] == 1)
						false_positives = sum(option['misprediction'] == 0)
						false_negatives = sum(uncovered['misprediction'] == 1) - true_positives
						
						if true_positives + false_positives == 0:
							precision_score = 0
						else:
							precision_score = true_positives / (true_positives + false_positives)
							
						if true_positives + false_negatives == 0:
							recall_score = 0
						else:
							recall_score = true_positives / (true_positives + false_negatives)
							
						if precision_score + recall_score == 0:
							f_score = 0
						else:
							f_score = ((1+(0.5 ** 2)) * precision_score * recall_score) / (((0.5 ** 2) * precision_score) + recall_score)
												
						f_scores.append(f_score)	
								
				#build best rule string
				option_num = f_scores.index(max(f_scores))
				string_options = [
					f"{uncovered.columns[feature1_idx]} > {x:.2f} and {uncovered.columns[feature2_idx]} > {y:.2f}",
					f"{uncovered.columns[feature1_idx]} > {x:.2f} and {uncovered.columns[feature2_idx]} <= {y:.2f}",
					f"{uncovered.columns[feature1_idx]} <= {x:.2f} and {uncovered.columns[feature2_idx]} > {y:.2f}",
					f"{uncovered.columns[feature1_idx]} <= {x:.2f} and {uncovered.columns[feature2_idx]} <= {y:.2f}",
					f"{uncovered.columns[feature1_idx]} <= {x:.2f}",
					f"{uncovered.columns[feature1_idx]} > {x:.2f}",
					f"{uncovered.columns[feature2_idx]} <= {y:.2f}",
					f"{uncovered.columns[feature2_idx]} > {y:.2f}"
				]
				
				best_rule = string_options[option_num]				
		
				#add best extracted rule to ruleset
				if runs > 1:
					ruleset.add_ruleset(" or ")
				ruleset.add_ruleset(best_rule)
		
				#calculate coverage with ruleset build so far
				current_coverage = ruleset.check_coverage_df(dff)
								
				#if not enough coverage...
				if current_coverage < arg_coverage:	
					#split off inputs that are still uncovered
					uncovered = ruleset.get_uncovered(uncovered)
					
				#print current results
				#print("\n" + str(runs) + ". Run:")
				#print("Rule set:\n\n" + ruleset.get_ruleset())
				#calculate precision of ruleset
				#precision = ruleset.eval_ruleset_prec(dff)
				#specificity precision of ruleset
				#specificity = ruleset.eval_ruleset_spec(dff)
				#print("\nSpecificity: " + str(round(specificity, 3)) + " Precision: " + str(round(precision, 3)) + " Recall: " + str(round(current_coverage, 3)))
				
				#result for bayesian without disjunctions
				if runs == 1:
					#start CPU time for bayesian without disjunctions output
					start_time_bayes_wo = time.process_time()
					
					#print rule set and metrics
					print('Best ruleset with most influential features and bayesian optimization without disjunctions:\n')
					print(ruleset.get_ruleset())
					#calculate precision of full ruleset
					precision = ruleset.eval_ruleset_prec(dff)
					#specificity precision of full ruleset
					specificity = ruleset.eval_ruleset_spec(dff)
					print("\nSpecificity: " + str(round(specificity, 3)) + " Precision: " + str(round(precision, 3)) + " Recall: " + str(round(current_coverage, 3)))
					#print CPU time used
					print(f"CPU time used extracting features: {end_time_learn - start_time_learn} seconds") 
					print(f"CPU time used overall without disjunctions: {(start_time_bayes_wo - start_time_bayes) + (end_time_learn - start_time_learn)} seconds\n") 
					
					#end CPU time for bayesian without disjunctions output
					end_time_bayes_wo = time.process_time()	
			
				if (time.process_time() - start_time_bayes) > 7200:
					print('Result after max time reached:')
					break
			
			#end CPU time for bayesian
			end_time_bayes = time.process_time()

			#print rule set and metrics
			print('Best ruleset with most influential features and bayesian optimization with disjunctions:\n')
			print(ruleset.get_ruleset())
			#calculate precision of full ruleset
			precision = ruleset.eval_ruleset_prec(dff)
			#specificity precision of full ruleset
			specificity = ruleset.eval_ruleset_spec(dff)
			print("\nSpecificity: " + str(round(specificity, 3)) + " Precision: " + str(round(precision, 3)) + " Recall: " + str(round(current_coverage, 3)))
			#print CPU time used
			print(f"CPU time used extracting features: {end_time_learn - start_time_learn} seconds") 
			print(f"CPU time used overall with disjunctions: {(end_time_bayes - start_time_bayes) + (end_time_learn - start_time_learn) -(end_time_bayes_wo - start_time_bayes_wo)} seconds") 
		
#train model from dataset and save blackbox to .joblib file
elif arg_blackbox == "":
					
	#get database for the blackbox (data_arg: 'heart'(heart failure dataset), 'heartWT'(heart failure dataset without time))
	print('Fetching and preparing database...')
	X, y, names, encoded_X = getdata(arg_data)

	#train on default model parameters and evaluate
	#arg_data: 'heart'(heart failure dataset) , 'heartWT'(heart failure dataset without time)  
	#arg_model: 'rfc'(random forest classifier), 'xgb' (gradient boosting trees) 	
	print('Training and evaluating models...')
	models = trainmodel(X, y, arg_model, arg_data, names, encoded_X)	
							
	#save models as .joblib file
	print('Saving models as .joblib file...')
	for i in range(len(models)):
		dump(models[i], "blackboxes/"+arg_data+arg_model+str(i+1)+".joblib")


