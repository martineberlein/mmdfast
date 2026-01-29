import sys
import os
import pandas as pd
import csv
import subprocess
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from threading import Lock, Condition
import re
import time
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams

#help prints
def usage():
	print(f"Usage: {sys.argv[0]} -m <mode>")
	print("  -m mode  specify evaluation task (create, evaluate, extract, combine, graph)")
	print("  -h, --help       print help")

args = sys.argv[1:]
mode = None

#read arguments
while args:
	arg = args.pop(0)
	if arg in ["-h", "--help"]:
		usage()
		sys.exit(0)
	elif arg == "-m":
		if not args:
			print(f"Error: {arg} requires mode name")
			usage()
			sys.exit(1)
		mode = args.pop(0)
	else:
		print(f"Error: wrong argument {arg}")
		usage()
		sys.exit(1)
		
#check if -m was read
if mode is None:
	print("Error: missing required argument -m <mode>")
	usage()
	sys.exit(1)

#check if mode argument in options
if mode not in ['create', 'evaluate', 'extract', 'combine', 'graph']:
	print("Error: not an correct mode argument")
	usage()
	sys.exit(1)	 

#how many times all experiments should be run
runs = 5

#create checklist file for all evaluation runs
if mode == 'create':
	
	#create checklist for a number of runs of all experiments
	for i in range(runs):
	
		#path for checklist file
		checklist_path = f"evaluation/checklist{i+1}.csv"
		
		#define column names
		columns = ['settings', 'done']
		
		#define settings column
		settings = []
		blackbox_list = ['heartWT', 'bugreport', 'java', 'python', 'php', 'ruby', 'bank', 'hotel', 'job', 'spam', 'water']
		learner_list = ['2', '3', '4', '6']
		method_list = ['induction', 'islearn', 'tree', 'bayesian']
		recall_list = ['0.3', '0.4', '0.5', '0.6', '0.7', '0.8']
		seed_list = [['12345678', '1'], ['23456789', '2'], ['34567890', '3'], ['45678901', '4'], ['56789012', '5']]
		
		"""
		for blackbox in ['heartWT']:
			for method in ['tree']:
				for recall in recall_list:
					for seed in seed_list:
						settings_string = f"-b {blackbox} -m rfc -l 0 -r {method} -c {recall} -s {seed[0]} -n {seed[1]}"
						settings.append(settings_string)		
		"""
		
		for blackbox in blackbox_list:
			for method in method_list:
				for recall in recall_list:
					for seed in seed_list:
						settings_string = f"-b {blackbox} -m rfc -l 0 -r {method} -c {recall} -s {seed[0]} -n {seed[1]}"
						settings.append(settings_string)
						
		for blackbox in blackbox_list:
			for method in method_list:
				for recall in recall_list:
					for seed in seed_list:
						settings_string = f"-b {blackbox} -m rfc -l 5 -r {method} -c {recall} -s {seed[0]} -n {seed[1]}"
						settings.append(settings_string)
						
		for blackbox in blackbox_list:
			for method in method_list:
				for learner in learner_list:
					for seed in seed_list:
						settings_string = f"-b {blackbox} -m rfc -l {learner} -r {method} -c 0.6 -s {seed[0]} -n {seed[1]}"
						settings.append(settings_string)
		
		#remove combination of islearn and spam dataset
		trimmed_settings = [x for x in settings if not ("-r islearn" in x and "-b spam" in x and "-l 0" in x)]
		
		#sort bayesian executions to the end of the list
		sorted_settings = sorted(trimmed_settings, key=lambda x: 'bayesian' in x.lower())
		
		#write .csv checklist
		with open(checklist_path, 'w') as checklist:
			
			writer = csv.writer(checklist)
			
			#write columns
			writer.writerow(columns)
			
			#write settings
			for item in sorted_settings:
				writer.writerow([item, 0])
		
		print(f"Checklist {i+1} created.")
	
#run all evaluations not already done (checklist)	
elif mode == 'evaluate':
	
	#do for a number of runs of all experiments
	for r in range(runs):
			
		#path for checklist file
		checklist_path = f"evaluation/checklist{r+1}.csv"
		#max executions
		max_parallel_executions = 70
		#lock, condition and counter to make sure bayes executions run solo
		bayes_lock = Lock()
		execution_counter = 0
		execution_condition = Condition()
		
		#run until all experiments in checklist finished correctly
		while True:
		
			#start timing
			start_time = time.time()
		
			#find all parameters not already executed from checklist
			parameters = []
			with open(checklist_path, mode='r') as checklist:
				reader = csv.reader(checklist)
				next(reader)
				for row in reader:
					if row[1] == '0':
						parameters.append(row[0])
			
			#break out if while loop when all experiments done
			if not parameters:
				print(f"Safety check: All experiments for checklist{r+1} done.")
				break
			
			#if not empty
			if parameters:
				
				#create queue
				parameter_queue = Queue()
				for parameter in parameters:
					parameter_queue.put(parameter)
					
				#create lock for .csv usage
				csv_lock = Lock()
				
				#define worker function
				def worker():
					
					global execution_counter
					
					#check if still parameters in queue
					while not parameter_queue.empty():
						
						#get next parameter set
						parameter = parameter_queue.get()
						
						#sleep when currently a bayesian execution is running
						wait_print=0
						while bayes_lock.locked():
							if wait_print == 0:
								wait_print = 1
								print(f"Execution with {parameter} waiting while other bayesian execution is running...")
							time.sleep(2)
						
						#check if bayesian parameter
						if 'bayesian' in parameter:
							bayes_lock.acquire()
							
							#wait for all executions to finish
							with execution_condition:
								print(f"Bayesian execution with {parameter} waiting until all other executions are finished...")
								while execution_counter > 0:
									execution_condition.wait()
									
							print(f"Bayesian execution with {parameter} continues...")
							
						#increase execution counter
						with execution_condition:
							execution_counter += 1
						
						parameter_list = parameter.split()
						print(f"Executing with parameters '{parameter}'.")
						
						#write output to files
						output_filename = f'output/{r+1}__misprediction_{parameter}.txt'
						output_filename = output_filename.replace('-b', '').replace('-m', '').replace('-l', '').replace('-r', '').replace('-c', '').replace('-s', '').replace('-n', '').replace(' ', '_')
						with open(output_filename, 'w') as output_file:
						
							#execute misprediction.py with parameters
							#result = subprocess.run(['nice', '-n', '19', 'python3', 'misprediction.py', *parameter_list], stdout=output_file)
							result = subprocess.run(['python3', 'misprediction.py', *parameter_list], stdout=output_file)
							
						#when executed successfully update .csv checklist
						if result.returncode == 0:
							
							#get lock
							with csv_lock:
								
								#get all rows
								rows = []
								with open(checklist_path, mode='r') as checklist:
									reader = csv.reader(checklist)
									rows = list(reader)
									
								#change row of used parameters
								for row in rows:
									if row[0] == parameter:
										row[1] = '1'
										break
								
								#change checklist
								with open(checklist_path, mode='w', newline='') as checklist:
									writer = csv.writer(checklist)
									writer.writerows(rows)
								
								print(f"Execution with parameters '{parameter}' finished successfully.")
									
						else:
								
							print(f"Execution with parameters '{parameter}' did not finish successfully.")
						
						#reduce execution counter and notify
						with execution_condition:
							execution_counter -= 1
							execution_condition.notify_all()
						
						#release lock if needed
						if 'bayesian' in parameter:
							bayes_lock.release()
						
				#create pool of worker threads
				with ThreadPoolExecutor(max_workers=max_parallel_executions) as executor:
					for _ in range(max_parallel_executions):
						executor.submit(worker)											

				end_time = time.time()
				print(f"Full evaluation with {max_parallel_executions} parallel executions took {end_time-start_time} seconds.")

#extract results from output files	
elif mode == 'extract':
		
	#folder of ouput files	
	folder_path = 'output'
	
	#write into results.csv
	with open('evaluation/results.csv', mode='w') as results:
		
		writer = csv.writer(results)
		#column names
		columns = ['run','blackbox', 'model', 'learner', 'rules', 'coverage', 'seed', 'model_number', 'specificity_without', 'precision_without', 'recall_without', 'time_without',
				   'and_without', 'or_without', 'predicates_without', 'specificity_with', 'precision_with', 'recall_with', 'time_with', 'and_with', 'or_with', 'predicates_with', 'time_influential']
				   
		#write column names
		writer.writerow(columns)
		
		#for every output file in folder
		for filename in sorted(os.listdir(folder_path)):
			if filename.endswith('.txt'):
				with open(os.path.join(folder_path, filename), 'r') as output:
					
					print(filename)
					#read output
					text = output.read()
					
					#extract arguments from filename
					arguments = filename.replace('.txt', '').replace('misprediction__', '').split('__')
					
					#find all important values
					specificity = re.findall(r'Specificity:\s*([\d.]+)', text)
					recall = re.findall(r'Recall:\s*([\d.]+)', text)
					precision = re.findall(r'Precision:\s*([\d.]+)', text)
					time_without = re.search(r'without disjunctions:\s*([\d.]+)', text).group(1)
					time_with = re.findall(r'with disjunctions:\s*([\d.]+)', text)
					#check because islearn output does not have disjunctions
					if len(time_with) == 0:
						
						#test with fixed error
						time_with_err = re.findall(r'with disjuncitons:\s*([\d.]+)', text)
						if len(time_with_err) == 0:
							time_with = -1
						else:
							time_with = time_with_err[0]
							
					else:
						time_with = time_with[0]
					time_inf = re.findall(r'CPU time used extracting features:\s*([\d.]+)', text)
					#check because time for influential feature extraction not there when using all features
					if len(time_inf) == 0:
						time_inf = -1
					else:
						time_inf = time_inf[0]
					
					#extract number of predicates for ruleset without disjunctions
					ruleset_without = re.search(r'without disjunctions:\n\n(.*?)\n\n', text, re.DOTALL)
					and_without = ruleset_without.group(1).count(' and ')
					or_without = ruleset_without.group(1).count(' or\n')
					predicates_without = and_without + or_without + 1

					#extract number of predicates for ruleset with disjunctions
					ruleset_with = re.search(r'with disjunctions:\n\n(.*?)\n\n', text, re.DOTALL)
					#check because islearn output does not have disjunctions
					if ruleset_with is None:
						and_with = -1
						or_with = -1
						predicates_with = "-1"
						specificity_dis = "-1"
						precision_dis = "-1"
						recall_dis = "-1"
					else:
						and_with = ruleset_with.group(1).count('and')
						or_with = ruleset_with.group(1).count('or')
						predicates_with = and_with + or_with + 1
						specificity_dis = specificity[1]
						precision_dis = precision[1]
						recall_dis = recall[1]
						
					#print(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6])
					#print(specificity[0], precision[0], recall[0], str(round(float(time_without), 3)))
					#print(str(and_without), str(or_without), str(predicates_without), specificity_dis, precision_dis, recall_dis)
					#print(str(round(float(time_with), 3)), str(and_with), str(or_with), str(predicates_with), str(round(float(time_inf), 3)))
					
					#write results row
					writer.writerow([arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], specificity[0], precision[0], recall[0], str(round(float(time_without), 3)),
									 str(and_without), str(or_without), str(predicates_without), specificity_dis, precision_dis, recall_dis, str(round(float(time_with), 3)), str(and_with), str(or_with), str(predicates_with), str(round(float(time_inf), 3))])
				   
#combine extracted results and calculate min/max/median/std deviation
elif mode == 'combine':
	
	#read extracted results
	results = pd.read_csv('evaluation/results.csv')

	#group together data of different runs
	group_results = results.groupby(['blackbox', 'model', 'learner', 'rules', 'coverage', 'seed', 'model_number'])
	
	#new data frame with mean of computation times
	combined_results = group_results.agg({'time_without': 'mean', 'time_with': 'mean'}).reset_index()
	
	#round mean to 3 decimal places
	combined_results['time_without'] = combined_results['time_without'].round(3)
	combined_results['time_with'] = combined_results['time_with'].round(3)
	
	#add other columns to new data frame
	first_rows = group_results.first()
	for column in first_rows.columns:
		if column not in combined_results.columns and column != 'run': 
			combined_results[column] = first_rows[column].values
	
	#add weighted f-score to every row
	combined_results['fscore_without'] = (1+(0.5 ** 2)) * (combined_results['precision_without'] * combined_results['recall_without']) / ((0.5 ** 2) * combined_results['precision_without'] + combined_results['recall_without'])
	combined_results['fscore_with'] = (1+(0.5 ** 2)) * (combined_results['precision_with'] * combined_results['recall_with']) / ((0.5 ** 2) * combined_results['precision_with'] + combined_results['recall_with'])
	combined_results['fscore_without'] = combined_results['fscore_without'].round(3)
	combined_results['fscore_with'] = combined_results['fscore_with'].round(3)
	
	#save to .csv
	combined_results.to_csv('evaluation/combined_time_results.csv', index=False)
	
	#remove unneeded columns 
	combined_results = combined_results.drop(columns=['seed', 'model_number', 'model'])
	
	#group together data of different models
	group_combined_results = combined_results.groupby(['blackbox', 'learner', 'rules', 'coverage'])

	#new data frame with columns for min/max/median/std of combined rows
	final_results = group_combined_results.agg({col: ['min', 'max', 'median', 'std'] for col in combined_results.columns if col not in ['blackbox', 'learner', 'rules', 'coverage']}).reset_index()

	#column names
	final_results.columns = ['_'.join(col).strip() for col in final_results.columns.values]
	
	#round new columns to 3 decimal places
	for col in final_results.columns:
		if col.endswith('_min') or col.endswith('_max') or col.endswith('_median') or col.endswith('_std'):
			final_results[col] = final_results[col].round(3)

	#save to .csv
	final_results.to_csv('evaluation/final_combined_results.csv', index=False)
	
	#calculate average f-scores
	group_average_fscore = final_results.groupby(['blackbox_', 'learner_', 'rules_'])
	average_fscore = group_average_fscore.agg({'fscore_with_median': 'mean'}).reset_index()
	average_fscore['fscore_with_median'] = average_fscore['fscore_with_median'].round(3)
	average_fscore.to_csv('evaluation/final_average_fscore_coverage.csv', index=False)
	
	#calculate average specificity
	group_average_specificity = final_results.groupby(['blackbox_', 'learner_', 'rules_'])
	average_specificity = group_average_specificity.agg({'specificity_with_median': 'mean'}).reset_index()
	average_specificity['specificity_with_median'] = average_specificity['specificity_with_median'].round(3)
	average_specificity.to_csv('evaluation/final_average_specificity_coverage.csv', index=False)
	
	#calculate average rule length
	group_average_length = final_results.groupby(['blackbox_', 'learner_', 'rules_'])
	average_length = group_average_length.agg({'predicates_with_median': 'mean'}).reset_index()
	average_length['predicates_with_median'] = average_length['predicates_with_median'].round(3)
	average_length.to_csv('evaluation/final_average_length_coverage.csv', index=False)
	
	#calculate average time
	group_average_time = final_results.groupby(['blackbox_', 'learner_', 'rules_'])
	average_time = group_average_time.agg({'time_with_median': 'mean'}).reset_index()
	average_time['time_with_median'] = average_time['time_with_median'].round(3)
	average_time.to_csv('evaluation/final_average_time_coverage.csv', index=False)

#combine extracted results and calculate min/max/median/std deviation
elif mode == 'graph':
	
	#read results
	results = pd.read_csv('evaluation/combined_time_results.csv')
	#set figure size
	width, height = 3840, 1080
	dpi = 200
	rcParams['figure.figsize'] = width/dpi, height/dpi
	#change font size
	sns.set_context("notebook", font_scale=1.1)
	
	
	#-------------------------------------------------------------
	#plot for computational demand with all features
	
	#plot for mmd
	#remove unused rows
	mmd_comp_plot_data = results.loc[results['learner'] == 0]
	mmd_comp_plot_data = mmd_comp_plot_data.loc[mmd_comp_plot_data['rules'] == 'induction']
	#sort rows correctly
	mmd_comp_plot_data = mmd_comp_plot_data.sort_values(by=['coverage', 'blackbox']).reset_index()
	#define color palette
	palette = sns.light_palette('green', reverse=False)
	#create box plot
	ax = sns.boxplot(x='blackbox', y='time_with', hue='coverage', data=mmd_comp_plot_data, width=0.8, dodge=True, palette=palette, medianprops={'color' : 'blue'})
	#change labels
	ax.set_xlabel('black boxes')
	ax.set_ylabel('CPU time in seconds')
	ax.legend_.set_title('desired coverage')
	#save figure
	fig = ax.get_figure()
	fig.savefig('evaluation/graph/time_mmd.png', bbox_inches='tight')
	#clear figure
	plt.clf()
	
	
	#plot for decision trees
	#remove unused rows
	mmd_comp_plot_data = results.loc[results['learner'] == 0]
	mmd_comp_plot_data = mmd_comp_plot_data.loc[mmd_comp_plot_data['rules'] == 'tree']
	#sort rows correctly
	mmd_comp_plot_data = mmd_comp_plot_data.sort_values(by=['coverage', 'blackbox']).reset_index()
	#define color palette
	palette = sns.light_palette('green', reverse=False)
	#create box plot
	ax = sns.boxplot(x='blackbox', y='time_with', hue='coverage', data=mmd_comp_plot_data, width=0.8, dodge=True, palette=palette, medianprops={'color' : 'blue'})
	#change labels
	ax.set_xlabel('black boxes')
	ax.set_ylabel('CPU time in seconds')
	ax.legend_.set_title('desired coverage')
	#save figure
	fig = ax.get_figure()
	fig.savefig('evaluation/graph/time_tree.png', bbox_inches='tight')
	#clear figure
	plt.clf()
	
	#plot for bayesian
	#remove unused rows
	mmd_comp_plot_data = results.loc[results['learner'] == 0]
	mmd_comp_plot_data = mmd_comp_plot_data.loc[mmd_comp_plot_data['rules'] == 'bayesian']
	#sort rows correctly
	mmd_comp_plot_data = mmd_comp_plot_data.sort_values(by=['coverage', 'blackbox']).reset_index()
	#define color palette
	palette = sns.light_palette('green', reverse=False)
	#create box plot
	ax = sns.boxplot(x='blackbox', y='time_with', hue='coverage', data=mmd_comp_plot_data, width=0.8, dodge=True, palette=palette, medianprops={'color' : 'blue'})
	#change labels
	ax.set_xlabel('black boxes')
	ax.set_ylabel('CPU time in seconds')
	ax.legend_.set_title('desired coverage')
	#save figure
	fig = ax.get_figure()
	fig.savefig('evaluation/graph/time_bayesian.png', bbox_inches='tight')
	#clear figure
	plt.clf()
	
	#-------------------------------------------------------------
	#plot for computational demand with 5 important features
	
	#plot for mmd
	#remove unused rows
	mmd_comp_plot_data = results.loc[results['learner'] == 5]
	mmd_comp_plot_data = mmd_comp_plot_data.loc[mmd_comp_plot_data['rules'] == 'induction']
	#sort rows correctly
	mmd_comp_plot_data = mmd_comp_plot_data.sort_values(by=['coverage', 'blackbox']).reset_index()
	#define color palette
	palette = sns.light_palette('green', reverse=False)
	#create box plot
	ax = sns.boxplot(x='blackbox', y='time_with', hue='coverage', data=mmd_comp_plot_data, width=0.8, dodge=True, palette=palette, medianprops={'color' : 'blue'})
	#change labels
	ax.set_xlabel('black boxes')
	ax.set_ylabel('CPU time in seconds')
	ax.legend_.set_title('desired coverage')
	#save figure
	fig = ax.get_figure()
	fig.savefig('evaluation/graph/time_important_mmd.png', bbox_inches='tight')
	#clear figure
	plt.clf()
	
	#plot for tree
	#remove unused rows
	mmd_comp_plot_data = results.loc[results['learner'] == 5]
	mmd_comp_plot_data = mmd_comp_plot_data.loc[mmd_comp_plot_data['rules'] == 'tree']
	#sort rows correctly
	mmd_comp_plot_data = mmd_comp_plot_data.sort_values(by=['coverage', 'blackbox']).reset_index()
	#define color palette
	palette = sns.light_palette('green', reverse=False)
	#create box plot
	ax = sns.boxplot(x='blackbox', y='time_with', hue='coverage', data=mmd_comp_plot_data, width=0.8, dodge=True, palette=palette, medianprops={'color' : 'blue'})
	#change labels
	ax.set_xlabel('black boxes')
	ax.set_ylabel('CPU time in seconds')
	ax.legend_.set_title('desired coverage')
	#save figure
	fig = ax.get_figure()
	fig.savefig('evaluation/graph/time_important_tree.png', bbox_inches='tight')
	#clear figure
	plt.clf()
	
	#plot for bayesian
	#remove unused rows
	mmd_comp_plot_data = results.loc[results['learner'] == 5]
	mmd_comp_plot_data = mmd_comp_plot_data.loc[mmd_comp_plot_data['rules'] == 'bayesian']
	#sort rows correctly
	mmd_comp_plot_data = mmd_comp_plot_data.sort_values(by=['coverage', 'blackbox']).reset_index()
	#define color palette
	palette = sns.light_palette('green', reverse=False)
	#create box plot
	ax = sns.boxplot(x='blackbox', y='time_with', hue='coverage', data=mmd_comp_plot_data, width=0.8, dodge=True, palette=palette, medianprops={'color' : 'blue'})
	#change labels
	ax.set_xlabel('black boxes')
	ax.set_ylabel('CPU time in seconds')
	ax.legend_.set_title('desired coverage')
	#set y axis start
	ax.set_ylim(bottom = -50)
	#save figure
	fig = ax.get_figure()
	fig.savefig('evaluation/graph/time_important_bayesian.png', bbox_inches='tight')
	#clear figure
	plt.clf()
	
	#-------------------------------------------------------------
	#plot for computational demand with different amount of important features and recall 0.6
	
	#plot for mmd
	#remove unused rows
	mmd_comp_plot_data = results.loc[(results['learner'] > 0) & (results['learner'] != 5)]
	mmd_comp_plot_data = mmd_comp_plot_data.loc[mmd_comp_plot_data['rules'] == 'induction']
	#sort rows correctly
	mmd_comp_plot_data = mmd_comp_plot_data.sort_values(by=['learner', 'blackbox']).reset_index()
	#define color palette
	palette = sns.light_palette('green', reverse=False)
	#create box plot
	ax = sns.boxplot(x='blackbox', y='time_with', hue='learner', data=mmd_comp_plot_data, width=0.8, dodge=True, palette=palette, medianprops={'color' : 'blue'})
	#change labels
	ax.set_xlabel('black boxes')
	ax.set_ylabel('CPU time in seconds')
	ax.legend_.set_title('number of used features')
	#move legend
	sns.move_legend(ax, "upper left")
	#save figure
	fig = ax.get_figure()
	fig.savefig('evaluation/graph/time_recall_mmd.png', bbox_inches='tight')
	#clear figure
	plt.clf()
	
	#plot for tree
	#remove unused rows
	mmd_comp_plot_data = results.loc[(results['learner'] > 0) & (results['learner'] != 5)]
	mmd_comp_plot_data = mmd_comp_plot_data.loc[mmd_comp_plot_data['rules'] == 'tree']
	#sort rows correctly
	mmd_comp_plot_data = mmd_comp_plot_data.sort_values(by=['learner', 'blackbox']).reset_index()
	#define color palette
	palette = sns.light_palette('green', reverse=False)
	#create box plot
	ax = sns.boxplot(x='blackbox', y='time_with', hue='learner', data=mmd_comp_plot_data, width=0.8, dodge=True, palette=palette, medianprops={'color' : 'blue'})
	#change labels
	ax.set_xlabel('black boxes')
	ax.set_ylabel('CPU time in seconds')
	ax.legend_.set_title('number of used features')
	#save figure
	fig = ax.get_figure()
	fig.savefig('evaluation/graph/time_recall_tree.png', bbox_inches='tight')
	#clear figure
	plt.clf()
	
	#plot for bayesian
	#remove unused rows
	mmd_comp_plot_data = results.loc[(results['learner'] > 0) & (results['learner'] != 5)]
	mmd_comp_plot_data = mmd_comp_plot_data.loc[mmd_comp_plot_data['rules'] == 'bayesian']
	#sort rows correctly
	mmd_comp_plot_data = mmd_comp_plot_data.sort_values(by=['learner', 'blackbox']).reset_index()
	#define color palette
	palette = sns.light_palette('green', reverse=False)
	#create box plot
	ax = sns.boxplot(x='blackbox', y='time_with', hue='learner', data=mmd_comp_plot_data, width=0.8, dodge=True, palette=palette, medianprops={'color' : 'blue'})
	#change labels
	ax.set_xlabel('black boxes')
	ax.set_ylabel('CPU time in seconds')
	ax.legend_.set_title('number of used features')
	#save figure
	fig = ax.get_figure()
	fig.savefig('evaluation/graph/time_recall_bayesian.png', bbox_inches='tight')
	#clear figure
	plt.clf()
	#plt.show()
	
	#-------------------------------------------------------------
	#plot for length with all features
	
	#plot for mmd
	#remove unused rows
	mmd_comp_plot_data = results.loc[results['learner'] == 0]
	mmd_comp_plot_data = mmd_comp_plot_data.loc[mmd_comp_plot_data['rules'] == 'induction']
	#sort rows correctly
	mmd_comp_plot_data = mmd_comp_plot_data.sort_values(by=['coverage', 'blackbox']).reset_index()
	#define color palette
	palette = sns.light_palette('green', reverse=False)
	#create box plot
	ax = sns.boxplot(x='blackbox', y='predicates_with', hue='coverage', data=mmd_comp_plot_data, width=0.8, dodge=True, palette=palette, medianprops={'color' : 'blue'})
	#change labels
	ax.set_xlabel('black boxes')
	ax.set_ylabel('Number of predicates in explanation')
	ax.legend_.set_title('desired coverage')
	#set y axis start
	ax.set_ylim(bottom = -0.5)
	#save figure
	fig = ax.get_figure()
	fig.savefig('evaluation/graph/length_mmd.png', bbox_inches='tight')
	#clear figure
	plt.clf()
	
	#plot for trees
	#remove unused rows
	mmd_comp_plot_data = results.loc[results['learner'] == 0]
	mmd_comp_plot_data = mmd_comp_plot_data.loc[mmd_comp_plot_data['rules'] == 'tree']
	#sort rows correctly
	mmd_comp_plot_data = mmd_comp_plot_data.sort_values(by=['coverage', 'blackbox']).reset_index()
	#define color palette
	palette = sns.light_palette('green', reverse=False)
	#create box plot
	ax = sns.boxplot(x='blackbox', y='predicates_with', hue='coverage', data=mmd_comp_plot_data, width=0.8, dodge=True, palette=palette, medianprops={'color' : 'blue'})
	#change labels
	ax.set_xlabel('black boxes')
	ax.set_ylabel('Number of predicates in explanation')
	ax.legend_.set_title('desired coverage')
	#set y axis start
	ax.set_ylim(bottom = -0.5)
	#set ticks
	ax.set_yticks(range(0,20,2))
	#move legend
	sns.move_legend(ax, "upper left", bbox_to_anchor=(0.84, 1))
	#save figure
	fig = ax.get_figure()
	fig.savefig('evaluation/graph/length_trees.png', bbox_inches='tight')
	#clear figure
	plt.clf()
	
	#plot for bayesian
	#remove unused rows
	mmd_comp_plot_data = results.loc[results['learner'] == 0]
	mmd_comp_plot_data = mmd_comp_plot_data.loc[mmd_comp_plot_data['rules'] == 'bayesian']
	#sort rows correctly
	mmd_comp_plot_data = mmd_comp_plot_data.sort_values(by=['coverage', 'blackbox']).reset_index()
	#define color palette
	palette = sns.light_palette('green', reverse=False)
	#create box plot
	ax = sns.boxplot(x='blackbox', y='predicates_with', hue='coverage', data=mmd_comp_plot_data, width=0.8, dodge=True, palette=palette, medianprops={'color' : 'blue'})
	#change labels
	ax.set_xlabel('black boxes')
	ax.set_ylabel('Number of predicates in explanation')
	ax.legend_.set_title('desired coverage')
	#set y axis start
	ax.set_ylim(bottom = -0.5)
	#save figure
	fig = ax.get_figure()
	fig.savefig('evaluation/graph/length_bayesian.png', bbox_inches='tight')
	#clear figure
	plt.clf()
	
	#-------------------------------------------------------------
	#plot for length with 5 important features
	
	#plot for mmd
	#remove unused rows
	mmd_comp_plot_data = results.loc[results['learner'] == 5]
	mmd_comp_plot_data = mmd_comp_plot_data.loc[mmd_comp_plot_data['rules'] == 'induction']
	#sort rows correctly
	mmd_comp_plot_data = mmd_comp_plot_data.sort_values(by=['coverage', 'blackbox']).reset_index()
	#define color palette
	palette = sns.light_palette('green', reverse=False)
	#create box plot
	ax = sns.boxplot(x='blackbox', y='predicates_with', hue='coverage', data=mmd_comp_plot_data, width=0.8, dodge=True, palette=palette, medianprops={'color' : 'blue'})
	#change labels
	ax.set_xlabel('black boxes')
	ax.set_ylabel('Number of predicates in explanation')
	#set y axis start
	ax.set_ylim(bottom = -0.5)
	#set legend font size
	legend = plt.legend(title = 'desired coverage')
	current_title = legend.get_title().get_fontsize()
	current_text = legend.get_texts()[0].get_fontsize()
	legend.get_title().set_fontsize(current_title - 2)
	for text in legend.get_texts():
		text.set_fontsize(current_text - 2)
	#save figure
	fig = ax.get_figure()
	fig.savefig('evaluation/graph/length_important_mmd.png', bbox_inches='tight')
	#clear figure
	plt.clf()
	
	#plot for trees
	#remove unused rows
	mmd_comp_plot_data = results.loc[results['learner'] == 5]
	mmd_comp_plot_data = mmd_comp_plot_data.loc[mmd_comp_plot_data['rules'] == 'tree']
	#sort rows correctly
	mmd_comp_plot_data = mmd_comp_plot_data.sort_values(by=['coverage', 'blackbox']).reset_index()
	#define color palette
	palette = sns.light_palette('green', reverse=False)
	#create box plot
	ax = sns.boxplot(x='blackbox', y='predicates_with', hue='coverage', data=mmd_comp_plot_data, width=0.8, dodge=True, palette=palette, medianprops={'color' : 'blue'})
	#change labels
	ax.set_xlabel('black boxes')
	ax.set_ylabel('Number of predicates in explanation')
	ax.legend_.set_title('desired coverage')
	#set y axis start
	ax.set_ylim(bottom = -0.5)
	#move legend
	sns.move_legend(ax, "upper left", bbox_to_anchor=(0.84, 1))
	#save figure
	fig = ax.get_figure()
	fig.savefig('evaluation/graph/length_important_trees.png', bbox_inches='tight')
	#clear figure
	plt.clf()
	
	#plot for bayesian
	#remove unused rows
	mmd_comp_plot_data = results.loc[results['learner'] == 5]
	mmd_comp_plot_data = mmd_comp_plot_data.loc[mmd_comp_plot_data['rules'] == 'bayesian']
	#sort rows correctly
	mmd_comp_plot_data = mmd_comp_plot_data.sort_values(by=['coverage', 'blackbox']).reset_index()
	#define color palette
	palette = sns.light_palette('green', reverse=False)
	#create box plot
	ax = sns.boxplot(x='blackbox', y='predicates_with', hue='coverage', data=mmd_comp_plot_data, width=0.8, dodge=True, palette=palette, medianprops={'color' : 'blue'})
	#change labels
	ax.set_xlabel('black boxes')
	ax.set_ylabel('Number of predicates in explanation')
	ax.legend_.set_title('desired coverage')
	#set y axis start
	ax.set_ylim(bottom = -0.5)
	#save figure
	fig = ax.get_figure()
	fig.savefig('evaluation/graph/length_important_bayesian.png', bbox_inches='tight')
	#clear figure
	plt.clf()
	
	#-------------------------------------------------------------
	#plot for length with different amount of important features and recall 0.6
	
	#plot for mmd
	#remove unused rows
	mmd_comp_plot_data = results.loc[(results['learner'] > 0) & (results['learner'] != 5)]
	mmd_comp_plot_data = mmd_comp_plot_data.loc[mmd_comp_plot_data['rules'] == 'induction']
	#sort rows correctly
	mmd_comp_plot_data = mmd_comp_plot_data.sort_values(by=['learner', 'blackbox']).reset_index()
	#define color palette
	palette = sns.light_palette('green', reverse=False)
	#create box plot
	ax = sns.boxplot(x='blackbox', y='predicates_with', hue='learner', data=mmd_comp_plot_data, width=0.8, dodge=True, palette=palette, medianprops={'color' : 'blue'})
	#change labels
	ax.set_xlabel('black boxes')
	ax.set_ylabel('Number of predicates in explanation')
	ax.legend_.set_title('number of used features')
	#set y axis start
	ax.set_ylim(bottom = -0.5)
	#save figure
	fig = ax.get_figure()
	fig.savefig('evaluation/graph/length_recall_mmd.png', bbox_inches='tight')
	#clear figure
	plt.clf()
	
	#plot for trees
	#remove unused rows
	mmd_comp_plot_data = results.loc[(results['learner'] > 0) & (results['learner'] != 5)]
	mmd_comp_plot_data = mmd_comp_plot_data.loc[mmd_comp_plot_data['rules'] == 'tree']
	#sort rows correctly
	mmd_comp_plot_data = mmd_comp_plot_data.sort_values(by=['learner', 'blackbox']).reset_index()
	#define color palette
	palette = sns.light_palette('green', reverse=False)
	#create box plot
	ax = sns.boxplot(x='blackbox', y='predicates_with', hue='learner', data=mmd_comp_plot_data, width=0.8, dodge=True, palette=palette, medianprops={'color' : 'blue'})
	#change labels
	ax.set_xlabel('black boxes')
	ax.set_ylabel('Number of predicates in explanation')
	ax.legend_.set_title('number of used features')
	#set y axis start
	ax.set_ylim(bottom = -0.5)
	#save figure
	fig = ax.get_figure()
	fig.savefig('evaluation/graph/length_recall_trees.png', bbox_inches='tight')
	#clear figure
	plt.clf()
	
	#plot for bayesian
	#remove unused rows
	mmd_comp_plot_data = results.loc[(results['learner'] > 0) & (results['learner'] != 5)]
	mmd_comp_plot_data = mmd_comp_plot_data.loc[mmd_comp_plot_data['rules'] == 'bayesian']
	#sort rows correctly
	mmd_comp_plot_data = mmd_comp_plot_data.sort_values(by=['learner', 'blackbox']).reset_index()
	#define color palette
	palette = sns.light_palette('green', reverse=False)
	#create box plot
	ax = sns.boxplot(x='blackbox', y='predicates_with', hue='learner', data=mmd_comp_plot_data, width=0.8, dodge=True, palette=palette, medianprops={'color' : 'blue'})
	#change labels
	ax.set_xlabel('black boxes')
	ax.set_ylabel('Number of predicates in explanation')
	ax.legend_.set_title('number of used features')
	#set y axis start
	ax.set_ylim(bottom = -0.5)
	#save figure
	fig = ax.get_figure()
	fig.savefig('evaluation/graph/length_recall_bayesian.png', bbox_inches='tight')
	#clear figure
	plt.clf()
	
	#-------------------------------------------------------------
	#plot for performance with all features (fscore)
	
	#plot for mmd
	#remove unused rows
	mmd_comp_plot_data = results.loc[results['learner'] == 0]
	mmd_comp_plot_data = mmd_comp_plot_data.loc[mmd_comp_plot_data['rules'] == 'induction']
	#sort rows correctly
	mmd_comp_plot_data = mmd_comp_plot_data.sort_values(by=['coverage', 'blackbox']).reset_index()
	#define color palette
	palette = sns.light_palette('green', reverse=False)
	#create box plot
	ax = sns.boxplot(x='blackbox', y='fscore_with', hue='coverage', data=mmd_comp_plot_data, width=0.8, dodge=True, palette=palette, medianprops={'color' : 'blue'})
	#change labels
	ax.set_xlabel('black boxes')
	ax.set_ylabel('weighted F-score')
	ax.legend_.set_title('desired coverage')
	#set y axis scale
	ax.set_ylim(-0.05,1.05)
	#save figure
	fig = ax.get_figure()
	fig.savefig('evaluation/graph/performance_fscore_mmd.png', bbox_inches='tight')
	#clear figure
	plt.clf()

	#plot for trees
	#remove unused rows
	mmd_comp_plot_data = results.loc[results['learner'] == 0]
	mmd_comp_plot_data = mmd_comp_plot_data.loc[mmd_comp_plot_data['rules'] == 'tree']
	#sort rows correctly
	mmd_comp_plot_data = mmd_comp_plot_data.sort_values(by=['coverage', 'blackbox']).reset_index()
	#define color palette
	palette = sns.light_palette('green', reverse=False)
	#create box plot
	ax = sns.boxplot(x='blackbox', y='fscore_with', hue='coverage', data=mmd_comp_plot_data, width=0.8, dodge=True, palette=palette, medianprops={'color' : 'blue'})
	#change labels
	ax.set_xlabel('black boxes')
	ax.set_ylabel('weighted F-score')
	ax.legend_.set_title('desired coverage')
	#set y axis scale
	ax.set_ylim(-0.05,1.05)
	#save figure
	fig = ax.get_figure()
	fig.savefig('evaluation/graph/performance_fscore_tree.png', bbox_inches='tight')
	#clear figure
	plt.clf()
	
	#plot for bayesian
	#remove unused rows
	mmd_comp_plot_data = results.loc[results['learner'] == 0]
	mmd_comp_plot_data = mmd_comp_plot_data.loc[mmd_comp_plot_data['rules'] == 'bayesian']
	#sort rows correctly
	mmd_comp_plot_data = mmd_comp_plot_data.sort_values(by=['coverage', 'blackbox']).reset_index()
	#define color palette
	palette = sns.light_palette('green', reverse=False)
	#create box plot
	ax = sns.boxplot(x='blackbox', y='fscore_with', hue='coverage', data=mmd_comp_plot_data, width=0.8, dodge=True, palette=palette, medianprops={'color' : 'blue'})
	#change labels
	ax.set_xlabel('black boxes')
	ax.set_ylabel('weighted F-score')
	ax.legend_.set_title('desired coverage')
	#set y axis scale
	ax.set_ylim(-0.05,1.05)
	#save figure
	fig = ax.get_figure()
	fig.savefig('evaluation/graph/performance_fscore_bayesian.png', bbox_inches='tight')
	#clear figure
	plt.clf()
	
	#-------------------------------------------------------------
	#plot for performance with all features (specificity)
	
	#plot for mmd
	#remove unused rows
	mmd_comp_plot_data = results.loc[results['learner'] == 0]
	mmd_comp_plot_data = mmd_comp_plot_data.loc[mmd_comp_plot_data['rules'] == 'induction']
	#sort rows correctly
	mmd_comp_plot_data = mmd_comp_plot_data.sort_values(by=['coverage', 'blackbox']).reset_index()
	#define color palette
	palette = sns.light_palette('green', reverse=False)
	#create box plot
	ax = sns.boxplot(x='blackbox', y='specificity_with', hue='coverage', data=mmd_comp_plot_data, width=0.8, dodge=True, palette=palette, medianprops={'color' : 'blue'})
	#change labels
	ax.set_xlabel('black boxes')
	ax.set_ylabel('specificity')
	ax.legend_.set_title('desired coverage')
	#set y axis scale
	ax.set_ylim(-0.05,1.05)
	#move legend
	sns.move_legend(ax, "lower left", bbox_to_anchor=(0.6, 0))
	#save figure
	fig = ax.get_figure()
	fig.savefig('evaluation/graph/performance_specificity_mmd.png', bbox_inches='tight')
	#clear figure
	plt.clf()

	#plot for trees
	#remove unused rows
	mmd_comp_plot_data = results.loc[results['learner'] == 0]
	mmd_comp_plot_data = mmd_comp_plot_data.loc[mmd_comp_plot_data['rules'] == 'tree']
	#sort rows correctly
	mmd_comp_plot_data = mmd_comp_plot_data.sort_values(by=['coverage', 'blackbox']).reset_index()
	#define color palette
	palette = sns.light_palette('green', reverse=False)
	#create box plot
	ax = sns.boxplot(x='blackbox', y='specificity_with', hue='coverage', data=mmd_comp_plot_data, width=0.8, dodge=True, palette=palette, medianprops={'color' : 'blue'})
	#change labels
	ax.set_xlabel('black boxes')
	ax.set_ylabel('specificity')
	ax.legend_.set_title('desired coverage')
	#set y axis scale
	ax.set_ylim(-0.05,1.05)
	#move legend
	sns.move_legend(ax, "lower left")
	#save figure
	fig = ax.get_figure()
	fig.savefig('evaluation/graph/performance_specificity_tree.png', bbox_inches='tight')
	#clear figure
	plt.clf()
	
	#plot for bayesian
	#remove unused rows
	mmd_comp_plot_data = results.loc[results['learner'] == 0]
	mmd_comp_plot_data = mmd_comp_plot_data.loc[mmd_comp_plot_data['rules'] == 'bayesian']
	#sort rows correctly
	mmd_comp_plot_data = mmd_comp_plot_data.sort_values(by=['coverage', 'blackbox']).reset_index()
	#define color palette
	palette = sns.light_palette('green', reverse=False)
	#create box plot
	ax = sns.boxplot(x='blackbox', y='specificity_with', hue='coverage', data=mmd_comp_plot_data, width=0.8, dodge=True, palette=palette, medianprops={'color' : 'blue'})
	#change labels
	ax.set_xlabel('black boxes')
	ax.set_ylabel('specificity')
	#set y axis scale
	ax.set_ylim(-0.05,1.05)
	#set legend font size
	legend = plt.legend(title = 'desired coverage')
	current_title = legend.get_title().get_fontsize()
	current_text = legend.get_texts()[0].get_fontsize()
	legend.get_title().set_fontsize(current_title - 2)
	for text in legend.get_texts():
		text.set_fontsize(current_text - 2)
	#move legend
	sns.move_legend(ax, "lower left")
	#save figure
	fig = ax.get_figure()
	fig.savefig('evaluation/graph/performance_specificity_bayesian.png', bbox_inches='tight')
	#clear figure
	plt.clf()

	#-------------------------------------------------------------
	#plot for performance with all 5 important features (fscore)
	
	#plot for mmd
	#remove unused rows
	mmd_comp_plot_data = results.loc[results['learner'] == 5]
	mmd_comp_plot_data = mmd_comp_plot_data.loc[mmd_comp_plot_data['rules'] == 'induction']
	#sort rows correctly
	mmd_comp_plot_data = mmd_comp_plot_data.sort_values(by=['coverage', 'blackbox']).reset_index()
	#define color palette
	palette = sns.light_palette('green', reverse=False)
	#create box plot
	ax = sns.boxplot(x='blackbox', y='fscore_with', hue='coverage', data=mmd_comp_plot_data, width=0.8, dodge=True, palette=palette, medianprops={'color' : 'blue'})
	#change labels
	ax.set_xlabel('black boxes')
	ax.set_ylabel('weighted F-score')
	ax.legend_.set_title('desired coverage')
	#set y axis scale
	ax.set_ylim(-0.05,1.05)
	#save figure
	fig = ax.get_figure()
	fig.savefig('evaluation/graph/performance_important_fscore_mmd.png', bbox_inches='tight')
	#clear figure
	plt.clf()

	#plot for trees
	#remove unused rows
	mmd_comp_plot_data = results.loc[results['learner'] == 5]
	mmd_comp_plot_data = mmd_comp_plot_data.loc[mmd_comp_plot_data['rules'] == 'tree']
	#sort rows correctly
	mmd_comp_plot_data = mmd_comp_plot_data.sort_values(by=['coverage', 'blackbox']).reset_index()
	#define color palette
	palette = sns.light_palette('green', reverse=False)
	#create box plot
	ax = sns.boxplot(x='blackbox', y='fscore_with', hue='coverage', data=mmd_comp_plot_data, width=0.8, dodge=True, palette=palette, medianprops={'color' : 'blue'})
	#change labels
	ax.set_xlabel('black boxes')
	ax.set_ylabel('weighted F-score')
	ax.legend_.set_title('desired coverage')
	#set y axis scale
	ax.set_ylim(-0.05,1.05)
	#save figure
	fig = ax.get_figure()
	fig.savefig('evaluation/graph/performance_important_fscore_tree.png', bbox_inches='tight')
	#clear figure
	plt.clf()
	
	#plot for bayesian
	#remove unused rows
	mmd_comp_plot_data = results.loc[results['learner'] == 5]
	mmd_comp_plot_data = mmd_comp_plot_data.loc[mmd_comp_plot_data['rules'] == 'bayesian']
	#sort rows correctly
	mmd_comp_plot_data = mmd_comp_plot_data.sort_values(by=['coverage', 'blackbox']).reset_index()
	#define color palette
	palette = sns.light_palette('green', reverse=False)
	#create box plot
	ax = sns.boxplot(x='blackbox', y='fscore_with', hue='coverage', data=mmd_comp_plot_data, width=0.8, dodge=True, palette=palette, medianprops={'color' : 'blue'})
	#change labels
	ax.set_xlabel('black boxes')
	ax.set_ylabel('weighted F-score')
	ax.legend_.set_title('desired coverage')
	#set y axis scale
	ax.set_ylim(-0.05,1.05)
	#save figure
	fig = ax.get_figure()
	fig.savefig('evaluation/graph/performance_important_fscore_bayesian.png', bbox_inches='tight')
	#clear figure
	plt.clf()
	
	#-------------------------------------------------------------
	#plot for performance with 5 important features (specificity)
	
	#plot for mmd
	#remove unused rows
	mmd_comp_plot_data = results.loc[results['learner'] == 5]
	mmd_comp_plot_data = mmd_comp_plot_data.loc[mmd_comp_plot_data['rules'] == 'induction']
	#sort rows correctly
	mmd_comp_plot_data = mmd_comp_plot_data.sort_values(by=['coverage', 'blackbox']).reset_index()
	#define color palette
	palette = sns.light_palette('green', reverse=False)
	#create box plot
	ax = sns.boxplot(x='blackbox', y='specificity_with', hue='coverage', data=mmd_comp_plot_data, width=0.8, dodge=True, palette=palette, medianprops={'color' : 'blue'})
	#change labels
	ax.set_xlabel('black boxes')
	ax.set_ylabel('specificity')
	#set y axis scale
	ax.set_ylim(-0.05,1.05)
	#move legend
	sns.move_legend(ax, "lower left")
	#set legend font size
	legend = plt.legend(title = 'desired coverage')
	current_title = legend.get_title().get_fontsize()
	current_text = legend.get_texts()[0].get_fontsize()
	legend.get_title().set_fontsize(current_title - 2)
	for text in legend.get_texts():
		text.set_fontsize(current_text - 2)
	#save figure
	fig = ax.get_figure()
	fig.savefig('evaluation/graph/performance_important_specificity_mmd.png', bbox_inches='tight')
	#clear figure
	plt.clf()

	#plot for trees
	#remove unused rows
	mmd_comp_plot_data = results.loc[results['learner'] == 5]
	mmd_comp_plot_data = mmd_comp_plot_data.loc[mmd_comp_plot_data['rules'] == 'tree']
	#sort rows correctly
	mmd_comp_plot_data = mmd_comp_plot_data.sort_values(by=['coverage', 'blackbox']).reset_index()
	#define color palette
	palette = sns.light_palette('green', reverse=False)
	#create box plot
	ax = sns.boxplot(x='blackbox', y='specificity_with', hue='coverage', data=mmd_comp_plot_data, width=0.8, dodge=True, palette=palette, medianprops={'color' : 'blue'})
	#change labels
	ax.set_xlabel('black boxes')
	ax.set_ylabel('specificity')
	ax.legend_.set_title('desired coverage')
	#set y axis scale
	ax.set_ylim(-0.05,1.05)
	#move legend
	sns.move_legend(ax, "lower left")
	#save figure
	fig = ax.get_figure()
	fig.savefig('evaluation/graph/performance_important_specificity_tree.png', bbox_inches='tight')
	#clear figure
	plt.clf()
	
	#plot for bayesian
	#remove unused rows
	mmd_comp_plot_data = results.loc[results['learner'] == 5]
	mmd_comp_plot_data = mmd_comp_plot_data.loc[mmd_comp_plot_data['rules'] == 'bayesian']
	#sort rows correctly
	mmd_comp_plot_data = mmd_comp_plot_data.sort_values(by=['coverage', 'blackbox']).reset_index()
	#define color palette
	palette = sns.light_palette('green', reverse=False)
	#create box plot
	ax = sns.boxplot(x='blackbox', y='specificity_with', hue='coverage', data=mmd_comp_plot_data, width=0.8, dodge=True, palette=palette, medianprops={'color' : 'blue'})
	#change labels
	ax.set_xlabel('black boxes')
	ax.set_ylabel('specificity')
	ax.legend_.set_title('desired coverage')
	#set y axis scale
	ax.set_ylim(-0.05,1.05)
	#move legend
	sns.move_legend(ax, "lower left")
	#save figure
	fig = ax.get_figure()
	fig.savefig('evaluation/graph/performance_important_specificity_bayesian.png', bbox_inches='tight')
	#clear figure
	plt.clf()
	
	#-------------------------------------------------------------
	#plot for performance with different number of important features and recall 0.6 (fscore)
	
	#plot for mmd
	#remove unused rows
	mmd_comp_plot_data = results.loc[(results['learner'] > 0) & (results['learner'] != 5)]
	mmd_comp_plot_data = mmd_comp_plot_data.loc[mmd_comp_plot_data['rules'] == 'induction']
	#sort rows correctly
	mmd_comp_plot_data = mmd_comp_plot_data.sort_values(by=['learner', 'blackbox']).reset_index()
	#define color palette
	palette = sns.light_palette('green', reverse=False)
	#create box plot
	ax = sns.boxplot(x='blackbox', y='fscore_with', hue='learner', data=mmd_comp_plot_data, width=0.8, dodge=True, palette=palette, medianprops={'color' : 'blue'})
	#change labels
	ax.set_xlabel('black boxes')
	ax.set_ylabel('weighted F-score')
	ax.legend_.set_title('number of used features')
	#set y axis scale
	ax.set_ylim(-0.05,1.05)
	#save figure
	fig = ax.get_figure()
	fig.savefig('evaluation/graph/performance_recall_fscore_mmd.png', bbox_inches='tight')
	#clear figure
	plt.clf()

	#plot for trees
	#remove unused rows
	mmd_comp_plot_data = results.loc[(results['learner'] > 0) & (results['learner'] != 5)]
	mmd_comp_plot_data = mmd_comp_plot_data.loc[mmd_comp_plot_data['rules'] == 'tree']
	#sort rows correctly
	mmd_comp_plot_data = mmd_comp_plot_data.sort_values(by=['learner', 'blackbox']).reset_index()
	#define color palette
	palette = sns.light_palette('green', reverse=False)
	#create box plot
	ax = sns.boxplot(x='blackbox', y='fscore_with', hue='learner', data=mmd_comp_plot_data, width=0.8, dodge=True, palette=palette, medianprops={'color' : 'blue'})
	#change labels
	ax.set_xlabel('black boxes')
	ax.set_ylabel('weighted F-score')
	ax.legend_.set_title('number of used features')
	#set y axis scale
	ax.set_ylim(-0.05,1.05)
	#save figure
	fig = ax.get_figure()
	fig.savefig('evaluation/graph/performance_recall_fscore_tree.png', bbox_inches='tight')
	#clear figure
	plt.clf()
	
	#plot for bayesian
	#remove unused rows
	mmd_comp_plot_data = results.loc[(results['learner'] > 0) & (results['learner'] != 5)]
	mmd_comp_plot_data = mmd_comp_plot_data.loc[mmd_comp_plot_data['rules'] == 'bayesian']
	#sort rows correctly
	mmd_comp_plot_data = mmd_comp_plot_data.sort_values(by=['learner', 'blackbox']).reset_index()
	#define color palette
	palette = sns.light_palette('green', reverse=False)
	#create box plot
	ax = sns.boxplot(x='blackbox', y='fscore_with', hue='learner', data=mmd_comp_plot_data, width=0.8, dodge=True, palette=palette, medianprops={'color' : 'blue'})
	#change labels
	ax.set_xlabel('black boxes')
	ax.set_ylabel('weighted F-score')
	ax.legend_.set_title('number of used features')
	#set y axis scale
	ax.set_ylim(-0.05,1.05)
	#save figure
	fig = ax.get_figure()
	fig.savefig('evaluation/graph/performance_recall_fscore_bayesian.png', bbox_inches='tight')
	#clear figure
	plt.clf()
	
	#-------------------------------------------------------------
	#plot for performance with different number of important features and recall 0.6 (specificity)
	
	#plot for mmd
	#remove unused rows
	mmd_comp_plot_data = results.loc[(results['learner'] > 0) & (results['learner'] != 5)]
	mmd_comp_plot_data = mmd_comp_plot_data.loc[mmd_comp_plot_data['rules'] == 'induction']
	#sort rows correctly
	mmd_comp_plot_data = mmd_comp_plot_data.sort_values(by=['learner', 'blackbox']).reset_index()
	#define color palette
	palette = sns.light_palette('green', reverse=False)
	#create box plot
	ax = sns.boxplot(x='blackbox', y='specificity_with', hue='learner', data=mmd_comp_plot_data, width=0.8, dodge=True, palette=palette, medianprops={'color' : 'blue'})
	#change labels
	ax.set_xlabel('black boxes')
	ax.set_ylabel('specificity')
	ax.legend_.set_title('number of used features')
	#set y axis scale
	ax.set_ylim(-0.05,1.05)
	#move legend
	sns.move_legend(ax, "lower left")
	#save figure
	fig = ax.get_figure()
	fig.savefig('evaluation/graph/performance_recall_specificity_mmd.png', bbox_inches='tight')
	#clear figure
	plt.clf()

	#plot for trees
	#remove unused rows
	mmd_comp_plot_data = results.loc[(results['learner'] > 0) & (results['learner'] != 5)]
	mmd_comp_plot_data = mmd_comp_plot_data.loc[mmd_comp_plot_data['rules'] == 'tree']
	#sort rows correctly
	mmd_comp_plot_data = mmd_comp_plot_data.sort_values(by=['learner', 'blackbox']).reset_index()
	#define color palette
	palette = sns.light_palette('green', reverse=False)
	#create box plot
	ax = sns.boxplot(x='blackbox', y='specificity_with', hue='learner', data=mmd_comp_plot_data, width=0.8, dodge=True, palette=palette, medianprops={'color' : 'blue'})
	#change labels
	ax.set_xlabel('black boxes')
	ax.set_ylabel('specificity')
	ax.legend_.set_title('number of used features')
	#set y axis scale
	ax.set_ylim(-0.05,1.05)
	#move legend
	sns.move_legend(ax, "lower left")
	#save figure
	fig = ax.get_figure()
	fig.savefig('evaluation/graph/performance_recall_specificity_tree.png', bbox_inches='tight')
	#clear figure
	plt.clf()
	
	#plot for bayesian
	#remove unused rows
	mmd_comp_plot_data = results.loc[(results['learner'] > 0) & (results['learner'] != 5)]
	mmd_comp_plot_data = mmd_comp_plot_data.loc[mmd_comp_plot_data['rules'] == 'bayesian']
	#sort rows correctly
	mmd_comp_plot_data = mmd_comp_plot_data.sort_values(by=['learner', 'blackbox']).reset_index()
	#define color palette
	palette = sns.light_palette('green', reverse=False)
	#create box plot
	ax = sns.boxplot(x='blackbox', y='specificity_with', hue='learner', data=mmd_comp_plot_data, width=0.8, dodge=True, palette=palette, medianprops={'color' : 'blue'})
	#change labels
	ax.set_xlabel('black boxes')
	ax.set_ylabel('specificity')
	ax.legend_.set_title('number of used features')
	#set y axis scale
	ax.set_ylim(-0.05,1.05)
	#move legend
	sns.move_legend(ax, "lower left")
	#save figure
	fig = ax.get_figure()
	fig.savefig('evaluation/graph/performance_recall_specificity_bayesian.png', bbox_inches='tight')
	#clear figure
	plt.clf()

	
	#-------------------------------------------------------------
	#plot for single rule performance with all features (fscore and specificity)
	
	#plot for mmd/tree/bayesian (fscore)
	#remove unused rows
	mmd_comp_plot_data = results.loc[results['learner'] == 0]
	mmd_comp_plot_data = mmd_comp_plot_data.loc[mmd_comp_plot_data['rules'] != 'islearn']
	mmd_comp_plot_data = mmd_comp_plot_data.loc[mmd_comp_plot_data['coverage'] == 0.3]
	#sort rows correctly
	mmd_comp_plot_data = mmd_comp_plot_data.sort_values(by=['rules', 'blackbox']).reset_index()
	#define color palette
	palette = sns.light_palette('green', reverse=False)
	#create box plot
	ax = sns.boxplot(x='blackbox', y='fscore_without', hue='rules', data=mmd_comp_plot_data, width=0.8, dodge=True, palette=palette, medianprops={'color' : 'blue'})
	#change labels
	ax.set_xlabel('black boxes')
	ax.set_ylabel('weighted F-score')
	ax.legend_.set_title('rule construction approach')
	#set y axis scale
	ax.set_ylim(-0.05,1.05)
	#save figure
	fig = ax.get_figure()
	fig.savefig('evaluation/graph/performance_solo_fscore_other.png', bbox_inches='tight')
	#clear figure
	plt.clf()
	
	#plot for mmd/tree/bayesian (specificity)
	#remove unused rows
	mmd_comp_plot_data = results.loc[results['learner'] == 0]
	mmd_comp_plot_data = mmd_comp_plot_data.loc[mmd_comp_plot_data['rules'] != 'islearn']
	mmd_comp_plot_data = mmd_comp_plot_data.loc[mmd_comp_plot_data['coverage'] == 0.3]
	#sort rows correctly
	mmd_comp_plot_data = mmd_comp_plot_data.sort_values(by=['rules', 'blackbox']).reset_index()
	#define color palette
	palette = sns.light_palette('green', reverse=False)
	#create box plot
	ax = sns.boxplot(x='blackbox', y='specificity_without', hue='rules', data=mmd_comp_plot_data, width=0.8, dodge=True, palette=palette, medianprops={'color' : 'blue'})
	#change labels
	ax.set_xlabel('black boxes')
	ax.set_ylabel('specificity')
	ax.legend_.set_title('rule construction approach')
	#move legend
	sns.move_legend(ax, "lower left")
	#set y axis scale
	ax.set_ylim(-0.05,1.05)
	#save figure
	fig = ax.get_figure()
	fig.savefig('evaluation/graph/performance_solo_specificity_other.png', bbox_inches='tight')
	#clear figure
	plt.clf()

	#plot for islearn (fscore)
	#remove unused rows
	mmd_comp_plot_data = results.loc[results['learner'] == 0]
	mmd_comp_plot_data = mmd_comp_plot_data.loc[mmd_comp_plot_data['rules'] == 'islearn']
	#sort rows correctly
	mmd_comp_plot_data = mmd_comp_plot_data.sort_values(by=['coverage', 'blackbox']).reset_index()
	#define color palette
	palette = sns.light_palette('green', reverse=False)
	#create box plot
	ax = sns.boxplot(x='blackbox', y='fscore_without', hue='coverage', data=mmd_comp_plot_data, width=0.8, dodge=True, palette=palette, medianprops={'color' : 'blue'})
	#change labels
	ax.set_xlabel('black boxes')
	ax.set_ylabel('weighted F-score')
	ax.legend_.set_title('desired coverage')
	#set y axis scale
	ax.set_ylim(-0.05,1.05)
	#save figure
	fig = ax.get_figure()
	fig.savefig('evaluation/graph/performance_solo_fscore_islearn.png', bbox_inches='tight')
	#clear figure
	plt.clf()
	
	#plot for islearn (specificity)
	#remove unused rows
	mmd_comp_plot_data = results.loc[results['learner'] == 0]
	mmd_comp_plot_data = mmd_comp_plot_data.loc[mmd_comp_plot_data['rules'] == 'islearn']
	#sort rows correctly
	mmd_comp_plot_data = mmd_comp_plot_data.sort_values(by=['coverage', 'blackbox']).reset_index()
	#define color palette
	palette = sns.light_palette('green', reverse=False)
	#create box plot
	ax = sns.boxplot(x='blackbox', y='specificity_without', hue='coverage', data=mmd_comp_plot_data, width=0.8, dodge=True, palette=palette, medianprops={'color' : 'blue'})
	#change labels
	ax.set_xlabel('black boxes')
	ax.set_ylabel('specificity')
	#set y axis scale
	ax.set_ylim(-0.05,1.05)
	#set legend font size
	legend = plt.legend(title = 'desired coverage')
	current_title = legend.get_title().get_fontsize()
	current_text = legend.get_texts()[0].get_fontsize()
	legend.get_title().set_fontsize(current_title - 2)
	for text in legend.get_texts():
		text.set_fontsize(current_text - 2)
	#move legend
	sns.move_legend(ax, "lower left")
	#save figure
	fig = ax.get_figure()
	fig.savefig('evaluation/graph/performance_solo_specificity_islearn.png', bbox_inches='tight')
	#clear figure
	plt.clf()

	#-------------------------------------------------------------
	#plot for single rule performance with 5 important features (fscore and specificity)
	
	#plot for mmd/tree/bayesian (fscore)
	#remove unused rows
	mmd_comp_plot_data = results.loc[results['learner'] == 5]
	mmd_comp_plot_data = mmd_comp_plot_data.loc[mmd_comp_plot_data['rules'] != 'islearn']
	mmd_comp_plot_data = mmd_comp_plot_data.loc[mmd_comp_plot_data['coverage'] == 0.3]
	#sort rows correctly
	mmd_comp_plot_data = mmd_comp_plot_data.sort_values(by=['rules', 'blackbox']).reset_index()
	#define color palette
	palette = sns.light_palette('green', reverse=False)
	#create box plot
	ax = sns.boxplot(x='blackbox', y='fscore_without', hue='rules', data=mmd_comp_plot_data, width=0.8, dodge=True, palette=palette, medianprops={'color' : 'blue'})
	#change labels
	ax.set_xlabel('black boxes')
	ax.set_ylabel('weighted F-score')
	ax.legend_.set_title('rule construction approach')
	#set y axis scale
	ax.set_ylim(-0.05,1.05)
	#save figure
	fig = ax.get_figure()
	fig.savefig('evaluation/graph/performance_solo_important_fscore_other.png', bbox_inches='tight')
	#clear figure
	plt.clf()
	
	#plot for mmd/tree/bayesian (specificity)
	#remove unused rows
	mmd_comp_plot_data = results.loc[results['learner'] == 5]
	mmd_comp_plot_data = mmd_comp_plot_data.loc[mmd_comp_plot_data['rules'] != 'islearn']
	mmd_comp_plot_data = mmd_comp_plot_data.loc[mmd_comp_plot_data['coverage'] == 0.3]
	#sort rows correctly
	mmd_comp_plot_data = mmd_comp_plot_data.sort_values(by=['rules', 'blackbox']).reset_index()
	#define color palette
	palette = sns.light_palette('green', reverse=False)
	#create box plot
	ax = sns.boxplot(x='blackbox', y='specificity_without', hue='rules', data=mmd_comp_plot_data, width=0.8, dodge=True, palette=palette, medianprops={'color' : 'blue'})
	#change labels
	ax.set_xlabel('black boxes')
	ax.set_ylabel('specificity')
	ax.legend_.set_title('rule construction approach')
	#set y axis scale
	ax.set_ylim(-0.05,1.05)
	#move legend
	sns.move_legend(ax, "lower left")
	#save figure
	fig = ax.get_figure()
	fig.savefig('evaluation/graph/performance_solo_important_specificity_other.png', bbox_inches='tight')
	#clear figure
	plt.clf()

	#plot for islearn (fscore)
	#remove unused rows
	mmd_comp_plot_data = results.loc[results['learner'] == 5]
	mmd_comp_plot_data = mmd_comp_plot_data.loc[mmd_comp_plot_data['rules'] == 'islearn']
	#sort rows correctly
	mmd_comp_plot_data = mmd_comp_plot_data.sort_values(by=['coverage', 'blackbox']).reset_index()
	#define color palette
	palette = sns.light_palette('green', reverse=False)
	#create box plot
	ax = sns.boxplot(x='blackbox', y='fscore_without', hue='coverage', data=mmd_comp_plot_data, width=0.8, dodge=True, palette=palette, medianprops={'color' : 'blue'})
	#change labels
	ax.set_xlabel('black boxes')
	ax.set_ylabel('weighted F-score')
	ax.legend_.set_title('desired coverage')
	#set y axis scale
	ax.set_ylim(-0.05,1.05)
	#save figure
	fig = ax.get_figure()
	fig.savefig('evaluation/graph/performance_solo_important_fscore_islearn.png', bbox_inches='tight')
	#clear figure
	plt.clf()
	
	#plot for islearn (specificity)
	#remove unused rows
	mmd_comp_plot_data = results.loc[results['learner'] == 5]
	mmd_comp_plot_data = mmd_comp_plot_data.loc[mmd_comp_plot_data['rules'] == 'islearn']
	#sort rows correctly
	mmd_comp_plot_data = mmd_comp_plot_data.sort_values(by=['coverage', 'blackbox']).reset_index()
	#define color palette
	palette = sns.light_palette('green', reverse=False)
	#create box plot
	ax = sns.boxplot(x='blackbox', y='specificity_without', hue='coverage', data=mmd_comp_plot_data, width=0.8, dodge=True, palette=palette, medianprops={'color' : 'blue'})
	#change labels
	ax.set_xlabel('black boxes')
	ax.set_ylabel('specificity')
	#set y axis scale
	ax.set_ylim(-0.05,1.05)
	#move legend
	sns.move_legend(ax, "lower left")
	#set legend font size
	legend = plt.legend(title = 'desired coverage')
	current_title = legend.get_title().get_fontsize()
	current_text = legend.get_texts()[0].get_fontsize()
	legend.get_title().set_fontsize(current_title - 1)
	for text in legend.get_texts():
		text.set_fontsize(current_text - 1)
	#save figure
	fig = ax.get_figure()
	fig.savefig('evaluation/graph/performance_solo_important_specificity_islearn.png', bbox_inches='tight')
	#clear figure
	plt.clf()
	
	#-------------------------------------------------------------
	#plot for single rule performance with different number of important features and recall 0.6 (fscore and specificity)
	
	#plot for islearn (fscore)
	#remove unused rows
	mmd_comp_plot_data = results.loc[(results['learner'] > 0) & (results['learner'] != 5)]
	mmd_comp_plot_data = mmd_comp_plot_data.loc[mmd_comp_plot_data['rules'] == 'islearn']
	#sort rows correctly
	mmd_comp_plot_data = mmd_comp_plot_data.sort_values(by=['learner', 'blackbox']).reset_index()
	#define color palette
	palette = sns.light_palette('green', reverse=False)
	#create box plot
	ax = sns.boxplot(x='blackbox', y='fscore_without', hue='learner', data=mmd_comp_plot_data, width=0.8, dodge=True, palette=palette, medianprops={'color' : 'blue'})
	#change labels
	ax.set_xlabel('black boxes')
	ax.set_ylabel('weighted F-score')
	ax.legend_.set_title('number of used features')
	#set y axis scale
	ax.set_ylim(-0.05,1.05)
	#save figure
	fig = ax.get_figure()
	fig.savefig('evaluation/graph/performance_solo_recall_fscore_islearn.png', bbox_inches='tight')
	#clear figure
	plt.clf()
	
	#plot for islearn (specificity)
	#remove unused rows
	mmd_comp_plot_data = results.loc[(results['learner'] > 0) & (results['learner'] != 5)]
	mmd_comp_plot_data = mmd_comp_plot_data.loc[mmd_comp_plot_data['rules'] == 'islearn']
	#sort rows correctly
	mmd_comp_plot_data = mmd_comp_plot_data.sort_values(by=['learner', 'blackbox']).reset_index()
	#define color palette
	palette = sns.light_palette('green', reverse=False)
	#create box plot
	ax = sns.boxplot(x='blackbox', y='specificity_without', hue='learner', data=mmd_comp_plot_data, width=0.8, dodge=True, palette=palette, medianprops={'color' : 'blue'})
	#change labels
	ax.set_xlabel('black boxes')
	ax.set_ylabel('specificity')
	ax.legend_.set_title('number of used features')
	#set y axis scale
	ax.set_ylim(-0.05,1.05)
	#move legend
	sns.move_legend(ax, "lower left")
	#save figure
	fig = ax.get_figure()
	fig.savefig('evaluation/graph/performance_solo_recall_specificity_islearn.png', bbox_inches='tight')
	#clear figure
	plt.clf()

	#plot for mmd (fscore) 
	#remove unused rows
	mmd_comp_plot_data = results.loc[(results['learner'] > 0) & (results['learner'] != 5)]
	mmd_comp_plot_data = mmd_comp_plot_data.loc[mmd_comp_plot_data['rules'] == 'induction']
	#sort rows correctly
	mmd_comp_plot_data = mmd_comp_plot_data.sort_values(by=['learner', 'blackbox']).reset_index()
	#define color palette
	palette = sns.light_palette('green', reverse=False)
	#create box plot
	ax = sns.boxplot(x='blackbox', y='fscore_without', hue='learner', data=mmd_comp_plot_data, width=0.8, dodge=True, palette=palette, medianprops={'color' : 'blue'})
	#change labels
	ax.set_xlabel('black boxes')
	ax.set_ylabel('weighted F-score')
	ax.legend_.set_title('number of used features')
	#set y axis scale
	ax.set_ylim(-0.05,1.05)
	#save figure
	fig = ax.get_figure()
	fig.savefig('evaluation/graph/performance_solo_recall_fscore_mmd.png', bbox_inches='tight')
	#clear figure
	plt.clf()
	
	#plot for mmd (specificity)
	#remove unused rows
	mmd_comp_plot_data = results.loc[(results['learner'] > 0) & (results['learner'] != 5)]
	mmd_comp_plot_data = mmd_comp_plot_data.loc[mmd_comp_plot_data['rules'] == 'induction']
	#sort rows correctly
	mmd_comp_plot_data = mmd_comp_plot_data.sort_values(by=['learner', 'blackbox']).reset_index()
	#define color palette
	palette = sns.light_palette('green', reverse=False)
	#create box plot
	ax = sns.boxplot(x='blackbox', y='specificity_without', hue='learner', data=mmd_comp_plot_data, width=0.8, dodge=True, palette=palette, medianprops={'color' : 'blue'})
	#change labels
	ax.set_xlabel('black boxes')
	ax.set_ylabel('specificity')
	ax.legend_.set_title('number of used features')
	#set y axis scale
	ax.set_ylim(-0.05,1.05)
	#move legend
	sns.move_legend(ax, "lower left")
	#save figure
	fig = ax.get_figure()
	fig.savefig('evaluation/graph/performance_solo_recall_specificity_mmd.png', bbox_inches='tight')
	#clear figure
	plt.clf()
	
	#plot for tree (fscore) 
	#remove unused rows
	mmd_comp_plot_data = results.loc[(results['learner'] > 0) & (results['learner'] != 5)]
	mmd_comp_plot_data = mmd_comp_plot_data.loc[mmd_comp_plot_data['rules'] == 'tree']
	#sort rows correctly
	mmd_comp_plot_data = mmd_comp_plot_data.sort_values(by=['learner', 'blackbox']).reset_index()
	#define color palette
	palette = sns.light_palette('green', reverse=False)
	#create box plot
	ax = sns.boxplot(x='blackbox', y='fscore_without', hue='learner', data=mmd_comp_plot_data, width=0.8, dodge=True, palette=palette, medianprops={'color' : 'blue'})
	#change labels
	ax.set_xlabel('black boxes')
	ax.set_ylabel('weighted F-score')
	ax.legend_.set_title('number of used features')
	#set y axis scale
	ax.set_ylim(-0.05,1.05)
	#save figure
	fig = ax.get_figure()
	fig.savefig('evaluation/graph/performance_solo_recall_fscore_tree.png', bbox_inches='tight')
	#clear figure
	plt.clf()
	
	#plot for tree (specificity)
	#remove unused rows
	mmd_comp_plot_data = results.loc[(results['learner'] > 0) & (results['learner'] != 5)]
	mmd_comp_plot_data = mmd_comp_plot_data.loc[mmd_comp_plot_data['rules'] == 'tree']
	#sort rows correctly
	mmd_comp_plot_data = mmd_comp_plot_data.sort_values(by=['learner', 'blackbox']).reset_index()
	#define color palette
	palette = sns.light_palette('green', reverse=False)
	#create box plot
	ax = sns.boxplot(x='blackbox', y='specificity_without', hue='learner', data=mmd_comp_plot_data, width=0.8, dodge=True, palette=palette, medianprops={'color' : 'blue'})
	#change labels
	ax.set_xlabel('black boxes')
	ax.set_ylabel('specificity')
	ax.legend_.set_title('number of used features')
	#set y axis scale
	ax.set_ylim(-0.05,1.05)
	#move legend
	sns.move_legend(ax, "lower left")
	#save figure
	fig = ax.get_figure()
	fig.savefig('evaluation/graph/performance_solo_recall_specificity_tree.png', bbox_inches='tight')
	#clear figure
	plt.clf()
	
	#plot for bayesian (fscore) 
	#remove unused rows
	mmd_comp_plot_data = results.loc[(results['learner'] > 0) & (results['learner'] != 5)]
	mmd_comp_plot_data = mmd_comp_plot_data.loc[mmd_comp_plot_data['rules'] == 'bayesian']
	#sort rows correctly
	mmd_comp_plot_data = mmd_comp_plot_data.sort_values(by=['learner', 'blackbox']).reset_index()
	#define color palette
	palette = sns.light_palette('green', reverse=False)
	#create box plot
	ax = sns.boxplot(x='blackbox', y='fscore_without', hue='learner', data=mmd_comp_plot_data, width=0.8, dodge=True, palette=palette, medianprops={'color' : 'blue'})
	#change labels
	ax.set_xlabel('black boxes')
	ax.set_ylabel('weighted F-score')
	ax.legend_.set_title('number of used features')
	#set y axis scale
	ax.set_ylim(-0.05,1.05)
	#save figure
	fig = ax.get_figure()
	fig.savefig('evaluation/graph/performance_solo_recall_fscore_bayesian.png', bbox_inches='tight')
	#clear figure
	plt.clf()
	
	#plot for bayesian (specificity)
	#remove unused rows
	mmd_comp_plot_data = results.loc[(results['learner'] > 0) & (results['learner'] != 5)]
	mmd_comp_plot_data = mmd_comp_plot_data.loc[mmd_comp_plot_data['rules'] == 'bayesian']
	#sort rows correctly
	mmd_comp_plot_data = mmd_comp_plot_data.sort_values(by=['learner', 'blackbox']).reset_index()
	#define color palette
	palette = sns.light_palette('green', reverse=False)
	#create box plot
	ax = sns.boxplot(x='blackbox', y='specificity_without', hue='learner', data=mmd_comp_plot_data, width=0.8, dodge=True, palette=palette, medianprops={'color' : 'blue'})
	#change labels
	ax.set_xlabel('black boxes')
	ax.set_ylabel('specificity')
	ax.legend_.set_title('number of used features')
	#set y axis scale
	ax.set_ylim(-0.05,1.05)
	#move legend
	sns.move_legend(ax, "lower left")
	#save figure
	fig = ax.get_figure()
	fig.savefig('evaluation/graph/performance_solo_recall_specificity_bayesian.png', bbox_inches='tight')
	#clear figure
	plt.clf()
	
	#-------------------------------------------------------------
	#plot for computational demand single rule with all features
	
	#plot for mmd/bayes/trees
	#remove unused rows
	mmd_comp_plot_data = results.loc[results['learner'] == 0]
	mmd_comp_plot_data = mmd_comp_plot_data.loc[mmd_comp_plot_data['rules'] != 'islearn']
	mmd_comp_plot_data = mmd_comp_plot_data.loc[mmd_comp_plot_data['coverage'] == 0.3]
	#sort rows correctly
	mmd_comp_plot_data = mmd_comp_plot_data.sort_values(by=['rules', 'blackbox']).reset_index()
	#define color palette
	palette = sns.light_palette('green', reverse=False)
	#create box plot
	ax = sns.boxplot(x='blackbox', y='time_without', hue='rules', data=mmd_comp_plot_data, width=0.8, dodge=True, palette=palette, medianprops={'color' : 'blue'})
	#change labels
	ax.set_xlabel('black boxes')
	ax.set_ylabel('CPU time in seconds')
	ax.legend_.set_title('rule construction approach')
	#save figure
	fig = ax.get_figure()
	fig.savefig('evaluation/graph/time_solo_other.png', bbox_inches='tight')
	#clear figure
	plt.clf()
	
	#plot for islearn
	#remove unused rows
	mmd_comp_plot_data = results.loc[results['learner'] == 0]
	mmd_comp_plot_data = mmd_comp_plot_data.loc[mmd_comp_plot_data['rules'] == 'islearn']
	#sort rows correctly
	mmd_comp_plot_data = mmd_comp_plot_data.sort_values(by=['coverage', 'blackbox']).reset_index()
	#define color palette
	palette = sns.light_palette('green', reverse=False)
	#create box plot
	ax = sns.boxplot(x='blackbox', y='time_without', hue='coverage', data=mmd_comp_plot_data, width=0.8, dodge=True, palette=palette, medianprops={'color' : 'blue'})
	#change labels
	ax.set_xlabel('black boxes')
	ax.set_ylabel('CPU time in seconds')
	#set legend font size
	legend = plt.legend(title = 'desired coverage')
	current_title = legend.get_title().get_fontsize()
	current_text = legend.get_texts()[0].get_fontsize()
	legend.get_title().set_fontsize(current_title - 2)
	for text in legend.get_texts():
		text.set_fontsize(current_text - 2)
	#move legend
	sns.move_legend(ax, "lower left", bbox_to_anchor=(0.65, 0))
	#save figure
	fig = ax.get_figure()
	fig.savefig('evaluation/graph/time_solo_islearn.png', bbox_inches='tight')
	#clear figure
	plt.clf()
	
	#-------------------------------------------------------------
	#plot for computational demand single rule with 5 important features
	
	#plot for mmd/bayes/trees
	#remove unused rows
	mmd_comp_plot_data = results.loc[results['learner'] == 5]
	mmd_comp_plot_data = mmd_comp_plot_data.loc[mmd_comp_plot_data['rules'] != 'islearn']
	mmd_comp_plot_data = mmd_comp_plot_data.loc[mmd_comp_plot_data['coverage'] == 0.3]
	#sort rows correctly
	mmd_comp_plot_data = mmd_comp_plot_data.sort_values(by=['rules', 'blackbox']).reset_index()
	#define color palette
	palette = sns.light_palette('green', reverse=False)
	#create box plot
	ax = sns.boxplot(x='blackbox', y='time_without', hue='rules', data=mmd_comp_plot_data, width=0.8, dodge=True, palette=palette, medianprops={'color' : 'blue'})
	#change labels
	ax.set_xlabel('black boxes')
	ax.set_ylabel('CPU time in seconds')
	ax.legend_.set_title('rule construction approach')
	#save figure
	fig = ax.get_figure()
	fig.savefig('evaluation/graph/time_solo_important_other.png', bbox_inches='tight')
	#clear figure
	plt.clf()
	
	#plot for islearn
	#remove unused rows
	mmd_comp_plot_data = results.loc[results['learner'] == 5]
	mmd_comp_plot_data = mmd_comp_plot_data.loc[mmd_comp_plot_data['rules'] == 'islearn']
	#sort rows correctly
	mmd_comp_plot_data = mmd_comp_plot_data.sort_values(by=['coverage', 'blackbox']).reset_index()
	#define color palette
	palette = sns.light_palette('green', reverse=False)
	#create box plot
	ax = sns.boxplot(x='blackbox', y='time_without', hue='coverage', data=mmd_comp_plot_data, width=0.8, dodge=True, palette=palette, medianprops={'color' : 'blue'})
	#change labels
	ax.set_xlabel('black boxes')
	ax.set_ylabel('CPU time in seconds')
	ax.legend_.set_title('desired coverage')
	#move legend
	sns.move_legend(ax, "upper left", bbox_to_anchor=(0.08, 1))
	#save figure
	fig = ax.get_figure()
	fig.savefig('evaluation/graph/time_solo_important_islearn.png', bbox_inches='tight')
	#clear figure
	plt.clf()
	
	#-------------------------------------------------------------
	#plot for computational demand single rule with differnt numbber of important features and recall = 0.6
	
	#plot for mmd
	#remove unused rows
	mmd_comp_plot_data = results.loc[(results['learner'] > 0) & (results['learner'] != 5)]
	mmd_comp_plot_data = mmd_comp_plot_data.loc[mmd_comp_plot_data['rules'] == 'induction']
	#sort rows correctly
	mmd_comp_plot_data = mmd_comp_plot_data.sort_values(by=['learner', 'blackbox']).reset_index()
	#define color palette
	palette = sns.light_palette('green', reverse=False)
	#create box plot
	ax = sns.boxplot(x='blackbox', y='time_without', hue='learner', data=mmd_comp_plot_data, width=0.8, dodge=True, palette=palette, medianprops={'color' : 'blue'})
	#change labels
	ax.set_xlabel('black boxes')
	ax.set_ylabel('CPU time in seconds')
	ax.legend_.set_title('number of used features')
	#move legend
	sns.move_legend(ax, "upper left")
	#save figure
	fig = ax.get_figure()
	fig.savefig('evaluation/graph/time_solo_recall_mmd.png', bbox_inches='tight')
	#clear figure
	plt.clf()
	
	#plot for islearn
	#remove unused rows
	mmd_comp_plot_data = results.loc[(results['learner'] > 0) & (results['learner'] != 5)]
	mmd_comp_plot_data = mmd_comp_plot_data.loc[mmd_comp_plot_data['rules'] == 'islearn']
	#sort rows correctly
	mmd_comp_plot_data = mmd_comp_plot_data.sort_values(by=['learner', 'blackbox']).reset_index()
	#define color palette
	palette = sns.light_palette('green', reverse=False)
	#create box plot
	ax = sns.boxplot(x='blackbox', y='time_without', hue='learner', data=mmd_comp_plot_data, width=0.8, dodge=True, palette=palette, medianprops={'color' : 'blue'})
	#change labels
	ax.set_xlabel('black boxes')
	ax.set_ylabel('CPU time in seconds')
	#set legend font size
	legend = plt.legend(title = 'number of used features')
	current_title = legend.get_title().get_fontsize()
	current_text = legend.get_texts()[0].get_fontsize()
	legend.get_title().set_fontsize(current_title - 2)
	for text in legend.get_texts():
		text.set_fontsize(current_text - 2)
	#save figure
	fig = ax.get_figure()
	fig.savefig('evaluation/graph/time_solo_recall_islearn.png', bbox_inches='tight')
	#clear figure
	plt.clf()
	
	#plot for decision trees
	#remove unused rows
	mmd_comp_plot_data = results.loc[(results['learner'] > 0) & (results['learner'] != 5)]
	mmd_comp_plot_data = mmd_comp_plot_data.loc[mmd_comp_plot_data['rules'] == 'tree']
	#sort rows correctly
	mmd_comp_plot_data = mmd_comp_plot_data.sort_values(by=['learner', 'blackbox']).reset_index()
	#define color palette
	palette = sns.light_palette('green', reverse=False)
	#create box plot
	ax = sns.boxplot(x='blackbox', y='time_without', hue='learner', data=mmd_comp_plot_data, width=0.8, dodge=True, palette=palette, medianprops={'color' : 'blue'})
	#change labels
	ax.set_xlabel('black boxes')
	ax.set_ylabel('CPU time in seconds')
	ax.legend_.set_title('number of used features')
	#save figure
	fig = ax.get_figure()
	fig.savefig('evaluation/graph/time_solo_recall_tree.png', bbox_inches='tight')
	#clear figure
	plt.clf()

	#plot for bayesian
	#remove unused rows
	mmd_comp_plot_data = results.loc[(results['learner'] > 0) & (results['learner'] != 5)]
	mmd_comp_plot_data = mmd_comp_plot_data.loc[mmd_comp_plot_data['rules'] == 'bayesian']
	#sort rows correctly
	mmd_comp_plot_data = mmd_comp_plot_data.sort_values(by=['learner', 'blackbox']).reset_index()
	#define color palette
	palette = sns.light_palette('green', reverse=False)
	#create box plot
	ax = sns.boxplot(x='blackbox', y='time_without', hue='learner', data=mmd_comp_plot_data, width=0.8, dodge=True, palette=palette, medianprops={'color' : 'blue'})
	#change labels
	ax.set_xlabel('black boxes')
	ax.set_ylabel('CPU time in seconds')
	ax.legend_.set_title('number of used features')
	#set y axis start
	ax.set_ylim(bottom = -5)
	#save figure
	fig = ax.get_figure()
	fig.savefig('evaluation/graph/time_solo_recall_bayesian.png', bbox_inches='tight')
	#clear figure
	plt.clf()
