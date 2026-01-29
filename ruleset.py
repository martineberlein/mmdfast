import numpy as np
import pandas as pd

#rewrite ruleset output for mmd
def mmd_ruleset_string(mmd_string):
	
	mmd_output_string = ""
	
	#split string into singel rules
	or_split = [[x] for x in mmd_string.split(" | ")]
	and_split = [[[z for z in y.split(" & ")] for y in x] for x in or_split]

	#build new ruleset string
	for i in range(len(and_split)):
		for j in range(len(and_split[i])):
			for k in range(len(and_split[i][j])):
				and_split[i][j][k] = and_split[i][j][k].replace("(", "")
				and_split[i][j][k] = and_split[i][j][k].replace(")", "")
				and_split[i][j][k] = and_split[i][j][k].replace(">", " > ")
				and_split[i][j][k] = and_split[i][j][k].replace("<=", " <= ")
				mmd_output_string += and_split[i][j][k]
				if (k+1 == len(and_split[i][j])) and (i+1 != len(and_split)):
					mmd_output_string += " or\n"
				elif k+1 != len(and_split[i][j]):
					mmd_output_string += " and "
					
	return mmd_output_string

#rewrite ruleset output for islearn	
def islearn_ruleset_string(islearn_string, original):
	
	print(islearn_string)
	islearn_output_string = ""
	#print(islearn_string)
	#build new ruleset string
	and_split = [[x] for x in islearn_string.split(" and\n")]
	or_split = [[[y.split(" or\n")] for y in x] for x in and_split]

	for i in range(len(or_split)):
		for j in range(len(or_split[i])):
			for k in range(len(or_split[i][j])):
				for l in range(len(or_split[i][j][k])):		
									
					or_split[i][j][k][l] = or_split[i][j][k][l].replace("(", "")
					or_split[i][j][k][l] = or_split[i][j][k][l].replace(")", "")

					or_split[i][j][k][l] = or_split[i][j][k][l].replace("> elem_0 in start:\n", "")		
					or_split[i][j][k][l] = or_split[i][j][k][l].replace("> elem_1 in start:\n", "")
					or_split[i][j][k][l] = or_split[i][j][k][l].replace("> elem_2 in start:\n", "")
					or_split[i][j][k][l] = or_split[i][j][k][l].replace("> elem_3 in start:\n", "")
					or_split[i][j][k][l] = or_split[i][j][k][l].replace("> elem_4 in start:\n", "")
					or_split[i][j][k][l] = or_split[i][j][k][l].replace("> elem in start:\n", "")

					or_split[i][j][k][l] = or_split[i][j][k][l].replace("str.to.int elem_0", "")		
					or_split[i][j][k][l] = or_split[i][j][k][l].replace("str.to.int elem_1", "")
					or_split[i][j][k][l] = or_split[i][j][k][l].replace("str.to.int elem_2", "")
					or_split[i][j][k][l] = or_split[i][j][k][l].replace("str.to.int elem_3", "")
					or_split[i][j][k][l] = or_split[i][j][k][l].replace("str.to.int elem_4", "")
					or_split[i][j][k][l] = or_split[i][j][k][l].replace("str.to.int elem", "")			
					or_split[i][j][k][l] = or_split[i][j][k][l].replace("str.to.int", "")

					or_split[i][j][k][l] = or_split[i][j][k][l].replace("forall ", "")
					or_split[i][j][k][l] = or_split[i][j][k][l].replace("\"", "")
					or_split[i][j][k][l] = or_split[i][j][k][l].replace(" ", "")
					or_split[i][j][k][l] = or_split[i][j][k][l][1:]

					or_split[i][j][k][l] = or_split[i][j][k][l].replace("<=", " <= ")
					or_split[i][j][k][l] = or_split[i][j][k][l].replace(">", " > ")
	
	for i in range(len(or_split[0][0][0])):
		
		#recreate original value range before normalization etc.
		parts1 = or_split[0][0][0][i].split(" ")
		divided1 = int(parts1[2]) / 100000000
		min_value1 = original.loc[:, parts1[0]].min()
		max_value1 = original.loc[:, parts1[0]].max()
		unnormalized1 = round(divided1 * (max_value1 - min_value1) + min_value1, 2)
		
		#if no and
		if len(or_split) == 1:
			
			islearn_output_string += parts1[0] + " " + parts1[1] + " " + str(unnormalized1)
			
		else:
			for j in range(len(or_split[1][0][0])):
				
				#recreate original value range before normalization etc.
				parts2 = or_split[1][0][0][j].split(" ")
				divided2 = int(parts2[2]) / 100000000
				min_value2 = original.loc[:, parts2[0]].min()
				max_value2 = original.loc[:, parts2[0]].max()
				unnormalized2 = round(divided2 * (max_value2 - min_value2) + min_value2, 2)
		
				islearn_output_string += parts1[0] + " " + parts1[1] + " " + str(unnormalized1) + " and "  + parts2[0] + " " + parts2[1] + " " + str(unnormalized2)

				#add or if not last rule
				if j+1 != len(or_split[1][0][0]):
					islearn_output_string += " or\n"
		
		#add or if not last rule
		if i+1 != len(or_split[0][0][0]):
			islearn_output_string += " or\n"	
					
	return islearn_output_string
	
#holds and handels rulesets
class RuleSet:
	
	"""
	The RuleSet class holds and contains methods to handel rule sets.
	"""

	def __init__(self):
		
		"""
		Constructs a new RuleSet object.
		"""
		self._rules = []
		self._samples_overall = []
		self._mispred_left = []
		self._samples_rules = []
		self._values_rules = []
		self._ruleset = ""
		self._ruleset_covered = 0
		self._ruleset_overall_covered = 0
	
	#extract rules from decision tree
	def extract_rules(self, model, names, decimal_places):
		
		extraction_run = len(self._rules)
		self._rules.append([])
		self._samples_rules.append([])
		self._values_rules.append([]) 
		child_left = model.tree_.children_left	
		child_right = model.tree_.children_right
		feature = model.tree_.feature
		threshold =	model.tree_.threshold
		samples = model.tree_.n_node_samples
		values = model.tree_.value
		stack = [(0, "")]
		self._samples_overall.append(samples[0])
		self._mispred_left.append(int(values[0][0][1]))

		#traverse tree while building rules
		while len(stack) > 0:
			
			nodenum, rule = stack.pop()
			
			#if children different then node is splitnode
			if child_left[nodenum] != child_right[nodenum]:
				
				if nodenum != 0:
					rule = rule + " and "
					
				#extend rules and add next nodes to stack
				if child_left[nodenum] != -1:

					extended_rule = rule + names[feature[nodenum]] + " <= " + str(round(threshold[nodenum], decimal_places))
					stack.append((child_left[nodenum], extended_rule))
				
				if child_right[nodenum] != -1:
					extended_rule = rule + names[feature[nodenum]] + " > " + str(round(threshold[nodenum], decimal_places))
					stack.append((child_right[nodenum], extended_rule))
			
			#if leaf node save rule and information about samples that are covered by that rule	
			else:
				self._rules[extraction_run].append(rule)
				self._samples_rules[extraction_run].append(samples[nodenum])
				self._values_rules[extraction_run].append(values[nodenum][0].tolist()) 				
				for i in range(len(self._values_rules[extraction_run])):
					for j in range(len(self._values_rules[extraction_run][i])):
						self._values_rules[extraction_run][i][j] = int(self._values_rules[extraction_run][i][j])				

	#add best rule from latest extract_rules() to ruleset
	def add_best(self):
		
		best = 0
		index = 0
		#print("\n")

		#check precision and coverage of each rule
		for i in range(len(self._rules[-1])):
			
			prec = self._values_rules[-1][i][1] / self._samples_rules[-1][i] 
			cov = self._values_rules[-1][i][1] / self._mispred_left[-1]
				
			#hyperparamerter to evaluate best rule
			#weighted f score (prec more important)
			if prec + cov == 0:
				eval_value = 0
			else:
				eval_value = ((1+(0.5 ** 2)) * prec * cov) / (((0.5 ** 2) * prec) + cov)
			
			#eval_value = prec * 7 + cov * 1
						
			#prints for checks
			#print(self._rules[-1][i])
			#print("prec: " + str(prec))
			#print("rec: " + str(cov))
			#print("eval: " + str(eval_value) + "\n")
			
			if eval_value > best:
				best = eval_value
				index = i
		
		#get best rule and add to ruleset
		best_rule = self._rules[-1][index]
		
		if self._ruleset != "":
			self._ruleset += " or "
			
		self._ruleset += best_rule
		self._ruleset_covered += self._values_rules[-1][index][1]
		self._ruleset_overall_covered += self._samples_rules[-1][index]

	#checks amount of mispredictions covered by ruleset
	def check_coverage(self):
		
		coverage = self._ruleset_covered / self._mispred_left[0]

		return coverage
	
	#check coverage of a ruleset on a given dataframe
	def check_coverage_df(self, inputs):
			
		true_positives = 0
		overall_positives = len(inputs.loc[inputs["misprediction"] == 1])
		cover_index = pd.Index([], dtype='int64')
		#print(self._ruleset)		
		or_split = [[x] for x in self._ruleset.split(" or ")]
		and_split = [[[z.split(" ") for z in y.split(" and ")] for y in x] for x in or_split]
		#print(and_split)
		#build strings for eval function
		for x in and_split:	
			to_filter = "inputs.loc["	
			for y in x:
				for i, z in enumerate(y):
					to_filter += "("
					for r in z:
						if r == z[0]:
							to_filter += "inputs."
							to_filter += r
						else:				
							to_filter += r
					to_filter += ")"
					if i != len(y)-1:
						to_filter += " & "
			to_filter += "].index"
			
			#print(to_filter)
			
			#eval string to generate index
			filter_index = eval(to_filter)
			cover_index = cover_index.union(filter_index)

			
		#filter for covered inputs and count true_positives
		if len(cover_index) > 0:
			true_positives += len(inputs.filter(items = cover_index, axis=0).loc[inputs["misprediction"] == 1])
		
		cov = true_positives / overall_positives
		
		return cov
		
	#returns inputs not covered by ruleset
	def get_uncovered(self, inputs):
		
		uncovered = inputs.copy(deep = True)

		or_split = [[x] for x in self._ruleset.split(" or ")]
		and_split = [[[z.split(" ") for z in y.split(" and ")] for y in x] for x in or_split]
		
		#build strings for eval function
		for x in and_split:	
			to_drop = "uncovered["	
			for y in x:
				for i, z in enumerate(y):
					to_drop += "("
					for r in z:
						if r == z[0]:
							to_drop += "uncovered."
							to_drop += r
						else:				
							to_drop += r
					to_drop += ")"				
					if i != len(y)-1:
						to_drop += " & "
					#if z != y[-1]:
						#to_drop += " & "
			to_drop += "].index"

			#drop rows covered by rules
			uncovered.drop(eval(to_drop), inplace=True)
			uncovered = uncovered.reset_index(drop=True)
				
		return uncovered
	
	#calculate specificity of a ruleset
	def eval_ruleset_spec(self, all_inputs):
		
		false_positives = 0
		cover_index = pd.Index([], dtype='int64')
				
		or_split = [[x] for x in self._ruleset.split(" or ")]
		and_split = [[[z.split(" ") for z in y.split(" and ")] for y in x] for x in or_split]
	
		#build strings for eval function
		for x in and_split:	
			to_filter = "all_inputs.loc["	
			for y in x:
				for i, z in enumerate(y):
					to_filter += "("
					for r in z:
						if r == z[0]:
							to_filter += "all_inputs."
							to_filter += r
						else:				
							to_filter += r
					to_filter += ")"
					if i != len(y)-1:
						to_filter += " & "
			to_filter += "].index"
			
			#eval string to generate index
			filter_index = eval(to_filter)
			cover_index = cover_index.union(filter_index)

			
		#filter for covered inputs and count false_positives
		if len(cover_index) > 0:
			false_positives += len(all_inputs.filter(items = cover_index, axis=0).loc[all_inputs["misprediction"] == 0])
		
		#print(all_inputs[all_inputs.index.isin(cover_index)])
		
		#calculate specificity
		true_negatives = len(all_inputs) - len(all_inputs.loc[all_inputs["misprediction"] == 1])
		specificity = (true_negatives - false_positives) / true_negatives

		return specificity
		
	#calculates precision of a ruleset
	def eval_ruleset_prec(self, all_inputs):
		
		true_positives = 0
		cover_index = pd.Index([], dtype='int64')
				
		or_split = [[x] for x in self._ruleset.split(" or ")]
		and_split = [[[z.split(" ") for z in y.split(" and ")] for y in x] for x in or_split]
	
		#build strings for eval function
		for x in and_split:	
			to_filter = "all_inputs.loc["	
			for y in x:
				for i, z in enumerate(y):
					to_filter += "("
					for r in z:
						if r == z[0]:
							to_filter += "all_inputs."
							to_filter += r
						else:				
							to_filter += r
					to_filter += ")"
					if i != len(y)-1:
						to_filter += " & "
			to_filter += "].index"

			#eval string to generate index
			filter_index = eval(to_filter)
			cover_index = cover_index.union(filter_index)
			
		#filter for covered inputs and count true_positives
		if len(cover_index) > 0:
			true_positives += len(all_inputs.filter(items = cover_index, axis=0).loc[all_inputs["misprediction"] == 1])
			num_covered = len(all_inputs[all_inputs.index.isin(cover_index)])
			
			#calculate precision		
			precision = true_positives / num_covered
			
		else:
			
			precision = 0.0
			
		#print(all_inputs[all_inputs.index.isin(cover_index)])
			
		return precision
		
	#returns full ruleset
	def get_ruleset(self):
		
		ruleset = self._ruleset.replace(" or ", " or\n")
		
		return ruleset
		
	#set ruleset from string
	def set_ruleset(self, ruleset):
		
		self._ruleset = ruleset.replace(" or\n", " or ")
		
	#add string to ruleset 
	def add_ruleset(self, rule_string):
		
		self._ruleset += rule_string

