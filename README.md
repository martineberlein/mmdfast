# Efficient Understanding of Machine Learning Model Mispredictions
This repository contains data sets, trained black boxes, evaluation code and results for our paper "Efficient Understanding of Machine Learning Model Mispredictions".

## Contents of Folders and Python Scripts
-dataset: contains data sets used  
-evaluation: contains experimental results in .csv files and graphs  
-mmd: contains MMD implementation from Cito et al.  
-output: contains the raw output of all experiments  
-tree: contains decision trees trained to extract rules  
  
-dataload.py: code to read/prepare data sets, read saved test sets/models  
-evaluation.py: code to create evaluation check lists, execute experiments, extract results from outputs and create graphs  
-misprediction.py: code for the different approaches we evaluate  
-modelevaluation.py: code to calculate performance metrics for black box models and to label mispredictions  
-ruleset.py: code to handle rule sets and to format output from MMD   
-trainmodel.py: code to train black boxes from data sets and train decision trees for rule extraction  

## How to Run
The required inputs for every script can be determined by executing the script with "-h".  
When a single experiment should be run execute misprediction.py with the desired parameters.  
For multiple different runs, edit evaluation.py to create a checklist containing all the runs and then execute all of them.  
  
## How to add new data set
1) Add data set to dataset folder.  
2) Add as possible input argument in misprediciton.py.  
3) Add to dataload.py getdata() and do prepocessing if needed.  
4) Execute misprediction.py to train and save black box models and test sets.  
5) Use misprediction.py or evaluation.py to do single or many experiments.  