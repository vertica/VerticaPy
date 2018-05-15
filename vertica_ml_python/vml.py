# (c) Copyright [2018] Micro Focus or one of its affiliates. 
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

############################################################################################################ 
#  __ __   ___ ____  ______ ____   __  ____      ___ ___ _          ____  __ __ ______ __ __  ___  ____    #
# |  |  | /  _|    \|      |    | /  ]/    |    |   |   | |        |    \|  |  |      |  |  |/   \|    \   #
# |  |  |/  [_|  D  |      ||  | /  /|  o  |    | _   _ | |        |  o  |  |  |      |  |  |     |  _  |  #
# |  |  |    _|    /|_|  |_||  |/  / |     |    |  \_/  | |___     |   _/|  ~  |_|  |_|  _  |  O  |  |  |  #
# |  :  |   [_|    \  |  |  |  /   \_|  _  |    |   |   |     |    |  |  |___, | |  | |  |  |     |  |  |  #
#  \   /|     |  .  \ |  |  |  \     |  |  |    |   |   |     |    |  |  |     | |  | |  |  |     |  |  |  #
#   \_/ |_____|__|\_| |__| |____\____|__|__|    |___|___|_____|    |__|  |____/  |__| |__|__|\___/|__|__|  #
#                                                                                                          #
############################################################################################################
# Vertica-ML-Python allows user to create  RVD (Resilient Vertica Dataset).         #
# RVD  simplifies data exploration, data cleaning and machine learning in  Vertica. #
# It is an object which keeps in it all the actions that the user wants to achieve  # 
# and execute them when they are needed.                                            #
#####################################################################################
#                    #
# Author: Badr Ouali #
#                    #
######################
# Libraries
import vertica_ml_python.rvd as rvd
import vertica_ml_python.rvc as rvc
from vertica_ml_python.fun import isnotebook
import sys,os
import shutil
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
import time

# Return model accuracy
def accuracy(model,threshold=0.5,input_class=1):
	return 1-error_rate(model=model,threshold=threshold,input_class=input_class).data_columns[1][-1]
# Return model auc
def auc(model,input_class=1):
	return roc(model=model,num_bins=30000,show=False,input_class=input_class)[0].data_columns[1][1]
# Champion Challenger for Binomial Model
# ALL THE COLUMNS OF THE INPUT RELATION MUST BE NUMERICAL
def champion_challenger_binomial(input_relation,response_column,predictor_columns,cursor,
								fold_count=3,max_iterations=100,logit_optimizer='Newton',logit_regularization='None',
								logit_alpha=0.5,rf_ntree=20,rf_mtry=None,rf_max_depth=5,rf_sampling_size=0.632):
	if (not(isinstance(input_relation,str))):
		raise TypeError("The parameter 'input_relation' must be a varchar")
	if (not(isinstance(response_column,str))):
		raise TypeError("The parameter 'response_column' must be a varchar")
	if (not(isinstance(predictor_columns,list))):
		raise TypeError("The parameter 'predictor_columns' must be a list of varchar")
	else:
		for column in predictor_columns:
			if (not(isinstance(column,str))):
				raise TypeError("The parameter 'predictor_columns' must be a list of varchar")
	if (not(isinstance(fold_count,int)) or (fold_count<0)):
		raise TypeError("The parameter 'fold_count' must be a positive integer")
	if (not(isinstance(max_iterations,int)) or (max_iterations<0)):
		raise TypeError("The parameter 'max_iterations' must be a positive integer")
	if not(logit_optimizer in ["Newton","BFGS"]):
		raise TypeError("The parameter 'logit_optimizer' must be in Newton|BFGS")
	if not(logit_regularization in ["L2","None"]):
		raise TypeError("The parameter 'logit_regularization' must be in L2|None")
	if (not(isinstance(logit_alpha,float)) or (logit_alpha<0) or (logit_alpha>1)):
		raise TypeError("The parameter 'logit_alpha' must be in [0,1]")
	if (not(isinstance(rf_ntree,int)) or (rf_ntree<0)):
		raise TypeError("The parameter 'rf_ntree' must be a positive integer")
	if (not(isinstance(rf_max_depth,int)) or (rf_max_depth<0)):
		raise TypeError("The parameter 'rf_max_depth' must be a positive integer")
	if (not(isinstance(rf_sampling_size,float)) or (rf_sampling_size<0) or (rf_sampling_size>1)):
		raise TypeError("The parameter 'rf_sampling_size' must be in [0,1]")
	colors=['dodgerblue','seagreen','indianred','gold']
	temp_rvd=rvd.RVD(input_relation,cursor)
	test_name=input_relation+"_cc_binomial_test_"+str(np.random.randint(10000000))
	train_name=input_relation+"_cc_binomial_train_"+str(np.random.randint(10000000))
	all_logit_info=[[],[],[],[],['logistic_reg']]
	all_svm_info=[[],[],[],[],['svm_classifier']]
	all_nb_info=[[],[],[],[],['naive_bayes']]
	all_rf_info=[[],[],[],[],['rf_classifier']]
	for i in range(fold_count):
		rvd.drop_table("random_vpython_table_"+input_relation,cursor,print_info=False)
		temp_rvd.train_test_split(test_name=test_name,train_name=train_name,print_info=False)
		# Logit
		model_name="model_"+str(np.random.randint(10000000))
		start_time=time.time()
		logistic_reg(model_name=model_name,input_relation=train_name,response_column=response_column,
						predictor_columns=predictor_columns,cursor=cursor,optimizer=logit_optimizer,
						regularization=logit_regularization,alpha=logit_alpha,max_iterations=max_iterations)
		all_logit_info[3]+=[time.time()-start_time]
		logit=load_model(model_name,cursor,input_relation=test_name)
		roc=logit.roc(show=False)
		all_logit_info[0]+=[roc[0].data_columns[1][1]]
		all_logit_info[1]+=[roc[1].data_columns]
		error=logit.error_rate()
		all_logit_info[2]+=[1-error.data_columns[1][-1]]
		drop_model(model_name,cursor,print_info=False)
		# SVM
		model_name="model_"+str(np.random.randint(10000000))
		start_time=time.time()
		svm_classifier(model_name=model_name,input_relation=train_name,response_column=response_column,
						predictor_columns=predictor_columns,cursor=cursor,max_iterations=max_iterations)
		all_svm_info[3]+=[time.time()-start_time]
		svm=load_model(model_name,cursor,input_relation=test_name)
		roc=svm.roc(show=False)
		all_svm_info[0]+=[roc[0].data_columns[1][1]]
		all_svm_info[1]+=[roc[1].data_columns]
		error=svm.error_rate()
		all_svm_info[2]+=[1-error.data_columns[1][-1]]
		drop_model(model_name,cursor,print_info=False)
		# Naive Bayes
		model_name="model_"+str(np.random.randint(10000000))
		start_time=time.time()
		naive_bayes(model_name=model_name,input_relation=train_name,response_column=response_column,
						predictor_columns=predictor_columns,cursor=cursor)
		all_nb_info[3]+=[time.time()-start_time]
		svm=load_model(model_name,cursor,input_relation=test_name)
		roc=svm.roc(show=False)
		all_nb_info[0]+=[roc[0].data_columns[1][1]]
		all_nb_info[1]+=[roc[1].data_columns]
		error=svm.error_rate()
		all_nb_info[2]+=[1-error.data_columns[1][-1]]
		drop_model(model_name,cursor,print_info=False)
		# Random Forest
		model_name="model_"+str(np.random.randint(10000000))
		start_time=time.time()
		cursor.execute("create view "+train_name+"_rf as select "+", ".join(predictor_columns)+", "+response_column+"::varchar(1) from "+train_name)
		rf_classifier(model_name=model_name,input_relation=train_name+"_rf",response_column=response_column,
						predictor_columns=predictor_columns,cursor=cursor,ntree=rf_ntree,
						mtry=rf_mtry,sampling_size=rf_sampling_size,max_depth=rf_max_depth)
		all_rf_info[3]+=[time.time()-start_time]
		rf=load_model(model_name,cursor,input_relation=test_name)
		roc=rf.roc(show=False)
		all_rf_info[0]+=[roc[0].data_columns[1][1]]
		all_rf_info[1]+=[roc[1].data_columns]
		error=rf.error_rate()
		all_rf_info[2]+=[1-error.data_columns[1][-1]]
		drop_model(model_name,cursor,print_info=False)
		# End
		rvd.drop_view(train_name+"_rf",cursor,print_info=False)
		rvd.drop_view(test_name,cursor,print_info=False)
		rvd.drop_view(train_name,cursor,print_info=False)
	# DRAW THE METRICS
	sample=list(range(fold_count))
	# accuracy
	plt.figure(figsize=(10,6))
	plt.plot(sample,all_logit_info[2],label=all_logit_info[4][0]+" avg_accuracy="+str(np.mean(all_logit_info[2])),color=colors[0])
	plt.plot(sample,all_svm_info[2],label=all_svm_info[4][0]+" avg_accuracy="+str(np.mean(all_svm_info[2])),color=colors[1])
	plt.plot(sample,all_nb_info[2],label=all_nb_info[4][0]+" avg_accuracy="+str(np.mean(all_nb_info[2])),color=colors[2])
	plt.plot(sample,all_rf_info[2],label=all_rf_info[4][0]+" avg_accuracy="+str(np.mean(all_rf_info[2])),color=colors[3])
	plt.ylabel('accuracy')
	plt.xlabel('sample')
	plt.legend()
	plt.grid()
	plt.xticks(sample,sample)
	plt.title('Binomial Champion Challenger on ["'+input_relation+'" Dataset]')
	plt.show()
	# time
	plt.figure(figsize=(10,6))
	plt.plot(sample,all_logit_info[3],label=all_logit_info[4][0]+" avg_time="+str(np.mean(all_logit_info[3])),color=colors[0])
	plt.plot(sample,all_svm_info[3],label=all_svm_info[4][0]+" avg_time="+str(np.mean(all_svm_info[3])),color=colors[1])
	plt.plot(sample,all_nb_info[3],label=all_nb_info[4][0]+" avg_time="+str(np.mean(all_nb_info[3])),color=colors[2])
	plt.plot(sample,all_rf_info[3],label=all_rf_info[4][0]+" avg_time="+str(np.mean(all_rf_info[3])),color=colors[3])
	plt.ylabel('time')
	plt.xlabel('sample')
	plt.legend()
	plt.grid()
	plt.xticks(sample,sample)
	plt.title('Binomial Champion Challenger on ["'+input_relation+'" Dataset]')
	plt.show()
	# auc
	plt.figure(figsize=(10,6))
	plt.plot(sample,all_logit_info[0],label=all_logit_info[4][0]+" avg_auc="+str(np.mean(all_logit_info[0])),color=colors[0])
	plt.plot(sample,all_svm_info[0],label=all_svm_info[4][0]+" avg_auc="+str(np.mean(all_svm_info[0])),color=colors[1])
	plt.plot(sample,all_nb_info[0],label=all_nb_info[4][0]+" avg_auc="+str(np.mean(all_nb_info[0])),color=colors[2])
	plt.plot(sample,all_rf_info[0],label=all_rf_info[4][0]+" avg_auc="+str(np.mean(all_rf_info[0])),color=colors[3])
	plt.ylabel('auc')
	plt.xlabel('sample')
	plt.legend() 
	plt.grid()
	plt.xticks(sample,sample)
	plt.title('Binomial Champion Challenger on ["'+input_relation+'" Dataset]')
	plt.show()
	return(rvd.column_matrix([['','logistic_reg','svm_classifier','naive_bayes','rf_classifier'],
							['avg_time',np.mean(all_logit_info[3]),np.mean(all_svm_info[3]),np.mean(all_nb_info[3]),np.mean(all_rf_info[3])],
							['avg_auc',np.mean(all_logit_info[0]),np.mean(all_svm_info[0]),np.mean(all_nb_info[0]),np.mean(all_rf_info[0])],
							['avg_accuracy',np.mean(all_logit_info[2]),np.mean(all_svm_info[2]),np.mean(all_nb_info[2]),np.mean(all_rf_info[2])],
							['std_accuracy',np.std(all_logit_info[2]),np.std(all_svm_info[2]),np.std(all_nb_info[2]),np.std(all_rf_info[2])]]))
# Return the Confusion Matrix of the model
def confusion_matrix(model,threshold=0.5,input_class=1):
	if (input_class==None):
		use_input_class=False
	else:
		use_input_class=True
	if (not(isinstance(model,(rf_classifier,rf_regressor,svm_classifier,svm_regressor,linear_reg,logistic_reg,naive_bayes)))):
		raise TypeError("This function is not available with this model")
	if (not(isinstance(threshold,float)) or (threshold<0) or (threshold>1)):
		raise TypeError("The parameter 'threshold' must be in [0,1]")
	if ((model.category=="binomial") or (use_input_class and ((model.category=="multinomial") and ((input_class in model.classes) or (str(input_class) in model.classes))))):
		query=("select confusion_matrix(obs, response using parameters num_classes=2) over() from (select (case when {}='{}' then 1 else 0 end) as obs, (case when predict_{}"+ 
				"({} using parameters model_name='{}',type='probability',class='{}',match_by_pos='true')::float>{} then 1 else 0 end) as response from {}) x")
		query=query.format(model.response_column,input_class,model.model_type,",".join(model.predictor_columns),model.model_name,
							input_class,threshold,model.input_relation)
		model.cursor.execute(query)
		query_result=model.cursor.fetchall()
		matrix=[['',0,1],[0,query_result[0][1],query_result[1][1]],[1,query_result[0][2],query_result[1][2]]]
		if (model.model_type in ["naive_bayes","rf_classifier"]):
			title=("Confusion Matrix with as positive '"+str(input_class)+"'")
		else:
			title=("Confusion Matrix")
		return rvd.column_matrix(matrix,title=title)
	elif (model.category=="multinomial"):
		classes=model.classes
		num_classes=str(len(classes))
		query="select confusion_matrix(obs, response using parameters num_classes="+num_classes+") over() from (select decode("+model.response_column 
		for idx,item in enumerate(classes):
			query+=",'"+str(item)+"',"+str(idx)
		query+=") as obs, decode(predict_"+model.model_type+"("+",".join(model.predictor_columns)+" using parameters model_name='{}'".format(model.model_name)
		query+=",type='response',match_by_pos='true')"
		for idx,item in enumerate(classes):
			query+=",'"+str(item)+"',"+str(idx)
		query+=") as response from {}) x".format(model.input_relation)
		model.cursor.execute(query)
		query_result=model.cursor.fetchall()
		matrix=[[['']]*(len(classes)+1)]
		matrix[0]=['']+classes
		for idx in range(1,len(query_result[0])-1):
			matrix+=[[classes[idx-1]]+[item[idx] for item in query_result]]
		return rvd.column_matrix(matrix)
	else:
		raise Exception("The Confusion Matrix is only available for multinomial/binomial models.")
# Return all the details of the model
def details(model_name,cursor,attr_name="details",model_type=None,header=None):
	if (not(isinstance(model_name,str))):
		raise TypeError("The parameter 'model_name' must be a varchar")
	if (not(isinstance(attr_name,str))):
		raise TypeError("The parameter 'attr_name' must be a varchar")
	query="select get_model_attribute(using parameters model_name='{}',attr_name='{}')".format(model_name,attr_name)
	cursor.execute(query)
	query_result=cursor.fetchall()
	if (attr_name=="prior"):
		header=['class','value']
	elif (model_type=="rf"):
		header=['column','type']
	else:
		header=['','coefficient','std_error','t_value','p_value']
	columns=[]
	for i in range(len(query_result[0])):
			columns+=[[item[i] for item in query_result]]
	for i in range(len(columns)):
		columns[i]=[header[i]]+columns[i]
	return rvd.column_matrix(columns,first_element=header[0])
# Drop Model if it exists
def drop_model(model_name,cursor,print_info=True):
	cursor.execute("select 1;")
	try:
		query="drop model {};".format(model_name)
		cursor.execute(query)
		if (print_info):
			print("The model {} was successfully dropped.".format(model_name))
	except:
		print("/!\\ Warning: The model {} doesn't exist !".format(model_name))
# Draw the Elbow curve: Help to find the best K for Kmeans
def elbow(input_relation,input_columns,cursor,min_num_cluster=1,max_num_cluster=15,max_iterations=10,
						epsilon=1e-4,init_method="kmeanspp",print_each_round=False):
	if (not(isinstance(input_columns,list))):
		raise TypeError("The parameter 'input_columns' must be a list of varchar")
	else:
		for column in input_columns:
			if (not(isinstance(column,str))):
				raise TypeError("The parameter 'input_columns' must be a list of varchar")
	if (not(isinstance(max_num_cluster,int)) or (max_num_cluster<0)):
		raise TypeError("The parameter 'max_num_cluster' must be a positive integer")
	if (not(isinstance(max_iterations,int)) or (max_iterations<0)):
		raise TypeError("The parameter 'max_iterations' must be a positive integer")
	if (not(isinstance(epsilon,float)) or (epsilon<0)):
		raise TypeError("The parameter 'epsilon' must be a positive float")
	if (not(isinstance(input_relation,str))):
		raise TypeError("The parameter 'input_relation' must be a varchar")
	all_within_cluster_SS=[]
	for i in range(min_num_cluster,max_num_cluster):
		if (print_each_round):
			print("Round "+str(i)+" begins")
		name="_vpython_kmeans_"+str(np.random.randint(1000000))
		query="drop model if exists {};".format(name)
		cursor.execute(query)
		all_within_cluster_SS+=[kmeans(name,input_relation,input_columns,i,cursor,
						max_iterations=max_iterations,epsilon=epsilon,init_method=init_method).within_cluster_SS()]
		if (print_each_round):
			print("Round "+str(i)+" ends")
		query="drop model if exists {};".format(name)
	cursor.execute(query)
	num_clusters=range(min_num_cluster,max_num_cluster)
	plt.rcParams['axes.facecolor']='#F4F4F4'
	plt.grid()
	plt.plot(num_clusters,all_within_cluster_SS,marker="s",color="dodgerblue")
	plt.title("Elbow Curve")
	plt.xlabel('Number of Clusters')
	plt.ylabel('Within-Cluster SS')
	plt.subplots_adjust(left=0.2)
	plt.show()
	return rvd.column_matrix([['num_clusters']+list(num_clusters),['all_within_cluster_SS']+all_within_cluster_SS],first_element='num_clusters')
# Return the Error Rate
def error_rate(model,threshold=0.5,input_class=1):
	if (input_class==None):
		use_input_class=False
	else:
		use_input_class=True
	if (not(isinstance(model,(rf_classifier,rf_regressor,svm_classifier,svm_regressor,linear_reg,logistic_reg,naive_bayes)))):
		raise TypeError("This function is not available with this model")
	if (not(isinstance(threshold,float)) or (threshold<0) or (threshold>1)):
		raise TypeError("The parameter 'threshold' must be in [0,1]")
	if ((model.category=="binomial") or (use_input_class and ((model.category=="multinomial") and ((input_class in model.classes) or (str(input_class) in model.classes))))):
		query=("select error_rate(obs, response using parameters num_classes=2) over() from (select (case when {}='{}' then 1 else 0 end) as obs, (case when predict_{}"+ 
				"({} using parameters model_name='{}',type='probability',class='{}',match_by_pos='true')::float>{} then 1 else 0 end) as response from {}) x")
		query=query.format(model.response_column,input_class,model.model_type,",".join(model.predictor_columns),model.model_name,
							input_class,threshold,model.input_relation)
		model.cursor.execute(query)
		query_result=model.cursor.fetchall()
		matrix=[['',0,1,'total'],['error_rate',query_result[0][1],query_result[1][1],query_result[2][1]]]
		if (model.model_type in ["naive_bayes","rf_classifier"]):
			title=("Error Rate Table with as positive '"+str(input_class)+"'")
		else:
			title=("Error Rate")
		return rvd.column_matrix(matrix,title=title)
	elif (model.category=="multinomial"):
		classes=model.classes
		num_classes=str(len(classes))
		query="select error_rate(obs, response using parameters num_classes="+num_classes+") over() from (select decode("+model.response_column 
		for idx,item in enumerate(classes):
			query+=",'"+str(item)+"',"+str(idx)
		query+=") as obs, decode(predict_"+model.model_type+"("+",".join(model.predictor_columns)+" using parameters model_name='{}',match_by_pos='true'".format(model.model_name)
		query+=",type='response')"
		for idx,item in enumerate(classes):
			query+=",'"+str(item)+"',"+str(idx)
		query+=") as response from {}) x".format(model.input_relation)
		model.cursor.execute(query)
		query_result=model.cursor.fetchall()
		matrix=[['']+classes+['total'],['error_rate']+[item[1] for item in query_result]]
		return rvd.column_matrix(matrix)
	else:
		raise Exception("The Error Rate is only available for multinomial/binomial models")
# Return the Features Importance
def features_importance(model,show=True,with_intercept=False):
	if (not(isinstance(model,(rf_classifier,rf_regressor,svm_classifier,svm_regressor,linear_reg,logistic_reg,naive_bayes,kmeans)))):
		raise TypeError("This function is not available with this model")
	if (not(isinstance(show,bool))):
		raise TypeError("The parameter 'show' must be a bool")
	if (not(isinstance(with_intercept,bool))):
		raise TypeError("The parameter 'with_intercept' must be a bool")
	if (model.model_type in ["linear_reg","svm_regressor","svm_classifier","logistic_reg"]):
		columns=model.details().data_columns
		if (type(model.all_importances)!=list):
			coefficients=columns[0]
			all_importances=columns[1]
			del coefficients[0]
			del all_importances[0]
			del coefficients[0]
			query=[]
			for item in coefficients:
				query+=["avg({}),stddev({})".format(item,item)]
			query="select "+",".join(query)+" from "+model.input_relation
			model.cursor.execute(query)
			avg_std=model.cursor.fetchone()
			all_avg=[]
			all_std=[]
			for i in range(len(avg_std)):
				if ((i%2)==0):
					all_avg+=[avg_std[i]]
				else:
					all_std+=[avg_std[i]]
			coefficients=['Intercept']+coefficients
			all_importances[0]+=sum([all_avg[i]*columns[1][i+1] for i in range(len(all_avg))])
			for i in range(1,len(all_importances)):
				all_importances[i]=all_importances[i]*all_std[i-1]
			if not(with_intercept):
				del all_importances[0]
				del coefficients[0]
			all_sign=[item>0 for item in all_importances]
			all_importances=[abs(item) for item in all_importances]
			all_importances=[item/sum(all_importances) for item in all_importances]
			model.all_importances=[coefficients,all_importances,all_sign]
		else:
			coefficients=model.all_importances[0]
			all_importances=model.all_importances[1]
			all_sign=model.all_importances[2]
	elif (model.model_type=="kmeans"):
		query=[]
		if (type(model.all_importances)!=list):
			try:
				for item in model.input_columns:
					for i in range(0,model.num_clusters):
						query_item="corr((case when apply_kmeans({} using parameters model_name='{}', match_by_pos='True') = {} then 1 else 0 end),{})"
						query_item=query_item.format(",".join(model.input_columns),model.model_name,i,item)
						query+=[query_item]
				query="select "+",".join(query)+" from "+model.input_relation
				model.cursor.execute(query)
				query_result=model.cursor.fetchone()
			except:
				query_result=[]
				for item in model.input_columns:
					for i in range(0,model.num_clusters):
						query_item="corr((case when apply_kmeans({} using parameters model_name='{}', match_by_pos='True') = {} then 1 else 0 end),{})"
						query_item=query_item.format(",".join(model.input_columns),model.model_name,i,item)
						query="select "+query_item+" from "+model.input_relation
						model.cursor.execute(query)
						query_result+=[model.cursor.fetchone()[0]]
			all_importances=[]
			k=0
			importance=0
			all_sign=[]
			for item in query_result:
				if (k==model.num_clusters):
					all_importances+=[importance]
					importance=0
					k=0
				k+=1
				all_sign+=[item>0]
				importance+=abs(item)
			all_importances=all_importances+[importance]
			all_importances=[item/sum(all_importances) for item in all_importances]
			model.all_importances=[all_importances,all_sign]
		else:
			all_importances=model.all_importances[0]
			all_sign=model.all_importances[1]
		coefficients=model.input_columns
	else:
		raise Exception("The features_importance function is not yet implemented for '{}' model.".format(model.model_type))
	all_importances,coefficients,all_sign=zip(*sorted(zip(all_importances,coefficients,all_sign)))
	coefficients=[item for item in coefficients]
	all_importances=[item for item in all_importances]
	all_sign=[item for item in all_sign]
	if (show):
		plt.figure(figsize=(7,5))
		plt.rcParams['axes.facecolor']='#F5F5F5'
		color=[]
		for item in all_sign:
			if (item):
				color+=['dodgerblue']
			else:
				color+=['mediumseagreen']
		plt.barh(range(0,len(all_importances)),all_importances,0.9,color=color,alpha=0.86)
		orange=mpatches.Patch(color='mediumseagreen', label='sign -')
		blue=mpatches.Patch(color='dodgerblue', label='sign +')
		plt.legend(handles=[orange,blue],loc="lower right")
		plt.ylabel("Features")
		plt.xlabel("Importance")
		plt.title("Model {}: '{}'".format(model.model_type,model.model_name))
		plt.gca().xaxis.grid()
		plt.gca().set_axisbelow(True)
		plt.yticks(range(0,len(all_importances)),coefficients)
		plt.show()
	return rvd.column_matrix([[""]+coefficients,['Importance']+all_importances])
# Return the Lift table
def lift_table(model,num_bins=200,color=["dodgerblue","#444444"],show=True,input_class=1):
	if (not(isinstance(model,(rf_classifier,rf_regressor,svm_classifier,svm_regressor,linear_reg,logistic_reg,naive_bayes)))):
		raise TypeError("This function is not available with this model")
	if (not(isinstance(num_bins,int)) or (num_bins<0)):
		raise TypeError("The parameter 'num_bins' must be a positive integer")
	if (not(isinstance(show,bool))):
		raise TypeError("The parameter 'show' must be a bool")
	if ((model.category=="binomial") or ((model.category=="multinomial") and ((input_class in model.classes) or (str(input_class) in model.classes)))):
		query=("select lift_table(obs, prob using parameters num_bins={}) over() from (select (case when {}='{}' then 1 else 0 end) as obs, predict_{}"+ 
				"({} using parameters model_name='{}',type='probability',class='{}',match_by_pos='true')::float as prob from {}) as prediction_output")
		query=query.format(num_bins,model.response_column,input_class,model.model_type,",".join(model.predictor_columns),model.model_name,
							input_class,model.input_relation)
		model.cursor.execute(query)
		query_result=model.cursor.fetchall()
		decision_boundary=[item[0] for item in query_result]
		positive_prediction_ratio=[item[1] for item in query_result]
		lift=[item[2] for item in query_result]
		decision_boundary.reverse()
		if (show):
			plt.figure(figsize=(7,5),facecolor='white')
			plt.rcParams['axes.facecolor']='#F5F5F5'
			plt.xlabel('Cumulative Data Fraction')
			plt.plot(decision_boundary,lift,color=color[0])
			plt.plot(decision_boundary,positive_prediction_ratio,color=color[1])
			if (model.category=="multinomial"):
				plt.title(model.model_name+" Lift Table of class '{}'".format(input_class))
			else:
				plt.title(model.model_name+" Lift Table")
			plt.gca().set_axisbelow(True)
			plt.grid()
			color1=mpatches.Patch(color=color[0], label='Cumulative Lift')
			color2=mpatches.Patch(color=color[1], label='Cumulative Capture Rate')
			plt.legend(handles=[color1,color2])
			plt.show()
		return rvd.column_matrix([['decision_boundary']+decision_boundary,['positive_prediction_ratio']+positive_prediction_ratio,['lift']+lift],repeat_first_column=False)
	else:
		raise Exception("The Lift Table is only available for multinomial/binomial models with a correct class.")
# Load the Model
def load_model(model_name,cursor,input_relation=None):
	if (not(isinstance(model_name,str))):
		raise TypeError("The parameter 'model_name' must be a varchar")
	cursor.execute("select model_type from models where model_name='"+model_name+"'")
	model_type=cursor.fetchone()[0].lower()
	if (model_type=="kmeans"):
		query="select attr_fields from (select get_model_attribute(using parameters model_name='{}')) x limit 1;"
		query=query.format(model_name)
		cursor.execute(query)
		input_columns=cursor.fetchone()[0]
		input_columns=input_columns.split(", ")
		num_clusters=len(input_columns)
		if (type(input_relation)!=str):
			summarize=summarize_model(model_name,cursor).lower()
			input_relation=summarize.split("kmeans(")[1].split(",")[1].split("'")[1]
		return kmeans(cursor,model_name,input_relation,input_columns,num_clusters,load=True)
	elif (model_type=="cross_validation"):
		return cross_validate("","","","",cursor,model_name=model_name,load=True)
	else:
		query="select predictor from (select get_model_attribute(using parameters model_name='{}',attr_name='details')) x;"
		query=query.format(model_name)
		cursor.execute(query)
		predictor_columns=cursor.fetchall()
		predictor_columns=[item for sublist in predictor_columns for item in sublist]
		if (model_type in ["linear_regression","svm_regressor","svm_classifier","logistic_regression","naive_bayes"]):
			predictor_columns=predictor_columns[1:len(predictor_columns)]
		query="select get_model_attribute(using parameters model_name='{}',attr_name='call_string');"
		query=query.format(model_name)
		cursor.execute(query)
		query_result=cursor.fetchone()[0]
		response_column=query_result.split(',')[2].replace("'","").replace('"',"").replace(' ',"")
		if (type(input_relation)!=str):
			input_relation=query_result.split(',')[1].replace("'","").replace('"',"").replace(' ',"")
		if (model_type=="linear_regression"):
			return linear_reg(model_name,input_relation,response_column,predictor_columns,cursor,load=True)
		elif (model_type=="svm_regressor"):
			return svm_regressor(model_name,input_relation,response_column,predictor_columns,cursor,load=True)
		elif (model_type=="svm_classifier"):
			return svm_classifier(model_name,input_relation,response_column,predictor_columns,cursor,load=True)
		elif (model_type=="naive_bayes"):
			return naive_bayes(model_name,input_relation,response_column,predictor_columns,cursor,load=True)
		elif (model_type=="logistic_regression"):
			return logistic_reg(model_name,input_relation,response_column,predictor_columns,cursor,load=True)
		elif (model_type=="rf_classifier"):
			return rf_classifier(model_name,input_relation,response_column,predictor_columns,cursor,load=True)
		elif (model_type=="rf_regressor"):
			return rf_regressor(model_name,input_relation,response_column,predictor_columns,cursor,load=True)
		else:
			raise Exception("The model '{}' is not took in charge.".format(model_type))
# return the Log loss for multinomial model
def logloss(model):
	if (not(isinstance(model,(rf_classifier,rf_regressor,svm_classifier,svm_regressor,linear_reg,logistic_reg,naive_bayes)))):
		raise TypeError("This function is not available with this model")
	if ((model.category in ["binomial","multinomial"])):
		if (model.model_type in ["svm_classifier","logistic_reg"]):
			query=("select avg(case when {}=1 then -log(predict_{}({} using parameters model_name='{}',type='probability',class='1',match_by_pos='true')::float+0.000001)"+
					" else -log(1-predict_{}({} using parameters model_name='{}',type='probability',class='1',match_by_pos='true')::float+0.000001) end) from {};")
			query=query.format(model.response_column,model.model_type,",".join(model.predictor_columns),model.model_name,model.model_type,
								",".join(model.predictor_columns),model.model_name,model.input_relation)
			model.cursor.execute(query)
			logloss_value=model.cursor.fetchone()[0]
		else:
			logloss_value=0
			for current_class in model.classes:
				query=("select avg(case when {}='{}' then -log(predict_{}({} using parameters model_name='{}',type='probability',class='{}',match_by_pos='true')::float+0.000001)"+
						" else -log(1-predict_{}({} using parameters model_name='{}',type='probability',class='{}',match_by_pos='true')::float+0.000001) end) from {};")
				query=query.format(model.response_column,current_class,model.model_type,",".join(model.predictor_columns),model.model_name,current_class,model.model_type,
									",".join(model.predictor_columns),model.model_name,current_class,model.input_relation)
				model.cursor.execute(query)
				logloss_value+=model.cursor.fetchone()[0]
			logloss_value=logloss_value/len(model.classes)
		model.logloss_value=logloss_value
		return logloss_value
	else:
		raise Exception("The logloss is only available for multinomial/binomial models")
# Return the ntree metric curve for a rf_classifier algorithm
def metric_rf_curve_ntree(input_relation,test_relation,response_column,predictor_columns,cursor,
						mode='logloss',ntree_begin=1,ntree_end=20,mtry=None,sampling_size=0.632,
						max_depth=5,max_breadth=32,min_leaf_size=5,min_info_gain=0.0,nbins=32,
						test_only=True):
	if (not(isinstance(predictor_columns,list))):
		raise TypeError("The parameter 'predictor_columns' must be a list of varchar")
	else:
		for column in predictor_columns:
			if (not(isinstance(column,str))):
				raise TypeError("The parameter 'predictor_columns' must be a list of varchar")
	if (not(isinstance(input_relation,str))):
		raise TypeError("The parameter 'input_relation' must be a varchar")
	if (not(isinstance(test_relation,str))):
		raise TypeError("The parameter 'test_relation' must be a varchar")
	if (not(isinstance(response_column,str))):
		raise TypeError("The parameter 'response_column' must be a varchar")
	if (not(isinstance(ntree_begin,int)) or (ntree_begin<0)):
		raise TypeError("The parameter 'ntree_begin' must be a positive integer")
	if (not(isinstance(ntree_end,int)) or (ntree_end<0)):
		raise TypeError("The parameter 'ntree_end' must be a positive integer")
	if (not(isinstance(max_depth,int)) or (max_depth<0)):
		raise TypeError("The parameter 'max_depth' must be a positive integer")
	if (not(isinstance(max_breadth,int)) or (max_breadth<0)):
		raise TypeError("The parameter 'max_breadth' must be a positive integer")
	if (not(isinstance(min_leaf_size,int)) or (min_leaf_size<0)):
		raise TypeError("The parameter 'min_leaf_size' must be a positive integer")
	if (not(isinstance(nbins,int)) or (nbins<0)):
		raise TypeError("The parameter 'nbins' must be a positive integer")
	if (not(isinstance(sampling_size,float)) or (sampling_size<0) or (sampling_size>1)):
		raise TypeError("The parameter 'sampling_size' must be in [0,1]")
	if (not(isinstance(min_info_gain,float)) or (min_info_gain<0)):
		raise TypeError("The parameter 'min_info_gain' must be a positive float")
	marker='s'
	if (ntree_end-ntree_begin>20):
		marker=None
	all_error_test=[]
	all_error_train=[]
	if (mode not in ['logloss','accuracy','error_rate','auc']):
		raise TypeError("Mode must be in logloss|accuracy|error_rate|auc.")
	for i in range(ntree_begin,ntree_end+1):
		name="_vpython_error_"+str(np.random.randint(1000000))
		query="drop model if exists {};".format(name)
		cursor.execute(query)
		rf_classifier(name,input_relation,response_column,predictor_columns,cursor,ntree=i,
				mtry=mtry,sampling_size=sampling_size,max_depth=max_depth,max_breadth=max_breadth,min_leaf_size=min_leaf_size,
				min_info_gain=min_info_gain,nbins=nbins)
		model=load_model(name,cursor,test_relation)
		if (mode=='logloss'):
			all_error_test+=[model.logloss()]
		elif (mode=='error_rate'):
			all_error_test+=[model.error_rate().data_columns[1][3]]
		elif (mode=='accuracy'):
			all_error_test+=[1-model.error_rate().data_columns[1][3]]
		elif (mode=='auc'):
			all_error_test+=[model.roc(show=False)[0].data_columns[1][1]]
		model=load_model(name,cursor,input_relation)
		if (mode=='logloss'):
			all_error_train+=[model.logloss()]
		elif (mode=='error_rate'):
			all_error_train+=[model.error_rate().data_columns[1][3]]
		elif (mode=='accuracy'):
			all_error_train+=[1-model.error_rate().data_columns[1][3]]
		elif (mode=='auc'):
			all_error_train+=[model.roc(show=False)[0].data_columns[1][1]]
		query="drop model if exists {};".format(name)
		cursor.execute(query)
	ntrees=range(ntree_begin,ntree_end+1)
	plt.rcParams['axes.facecolor']='#F4F4F4'
	plt.grid()
	plt.plot(ntrees,all_error_test,marker=marker,color='dodgerblue')
	if not(test_only):
		plt.plot(ntrees,all_error_train,marker=marker,color='mediumseagreen')
	plt.title(mode+" curve")
	plt.xlabel('ntree')
	plt.ylabel(mode)
	if not(test_only):
		orange=mpatches.Patch(color='mediumseagreen', label='train')
		blue=mpatches.Patch(color='dodgerblue', label='test')
		plt.legend(handles=[orange,blue])
	plt.xlim(ntree_begin,ntree_end)
	plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
	plt.subplots_adjust(left=0.2)
	plt.show()
	return (rvd.column_matrix([["ntree"]+list(ntrees),[mode+"_test"]+all_error_test],first_element="ntree"),
			rvd.column_matrix([["ntree"]+list(ntrees),[mode+"_train"]+all_error_train],first_element="ntree"))
# Return the depth metric curve for the rf_classifier algorithm
def metric_rf_curve_depth(input_relation,test_relation,response_column,predictor_columns,cursor,mode='logloss',
							ntree=20,mtry=None,sampling_size=0.632,max_depth_begin=1,max_depth_end=12,
							max_breadth=32,min_leaf_size=5,min_info_gain=0.0,nbins=32,test_only=True):
	if (not(isinstance(predictor_columns,list))):
		raise TypeError("The parameter 'predictor_columns' must be a list of varchar")
	else:
		for column in predictor_columns:
			if (not(isinstance(column,str))):
				raise TypeError("The parameter 'predictor_columns' must be a list of varchar")
	if (not(isinstance(input_relation,str))):
		raise TypeError("The parameter 'input_relation' must be a varchar")
	if (not(isinstance(test_relation,str))):
		raise TypeError("The parameter 'test_relation' must be a varchar")
	if (not(isinstance(response_column,str))):
		raise TypeError("The parameter 'response_column' must be a varchar")
	if (not(isinstance(ntree,int)) or (ntree<0)):
		raise TypeError("The parameter 'ntree' must be a positive integer")
	if (not(isinstance(max_depth_begin,int)) or (max_depth_begin<0)):
		raise TypeError("The parameter 'max_depth_begin' must be a positive integer")
	if (not(isinstance(max_depth_end,int)) or (max_depth_end<0)):
		raise TypeError("The parameter 'max_depth_end' must be a positive integer")
	if (not(isinstance(max_breadth,int)) or (max_breadth<0)):
		raise TypeError("The parameter 'max_breadth' must be a positive integer")
	if (not(isinstance(min_leaf_size,int)) or (min_leaf_size<0)):
		raise TypeError("The parameter 'min_leaf_size' must be a positive integer")
	if (not(isinstance(nbins,int)) or (nbins<0)):
		raise TypeError("The parameter 'nbins' must be a positive integer")
	if (not(isinstance(sampling_size,float)) or (sampling_size<0) or (sampling_size>1)):
		raise TypeError("The parameter 'sampling_size' must be in [0,1]")
	if (not(isinstance(min_info_gain,float)) or (min_info_gain<0)):
		raise TypeError("The parameter 'min_info_gain' must be a positive float")
	marker='s'
	if (max_depth_end-max_depth_begin>20):
		marker=None
	all_error_test=[]
	all_error_train=[]
	if (mode not in ['logloss','accuracy','error_rate','auc']):
		raise TypeError("Mode must be in logloss|accuracy|error_rate|auc.")
	for i in range(max_depth_begin,max_depth_end+1):
		name="_vpython_error_"+str(np.random.randint(1000000))
		query="drop model if exists {};".format(name)
		cursor.execute(query)
		model=rf_classifier(name,input_relation,response_column,predictor_columns,cursor,ntree=ntree,
				mtry=mtry,sampling_size=sampling_size,max_depth=i,max_breadth=max_breadth,min_leaf_size=min_leaf_size,
				min_info_gain=min_info_gain,nbins=nbins)
		model=load_model(name,cursor,test_relation)
		if (mode=='logloss'):
			all_error_test+=[model.logloss()]
		elif (mode=='error_rate'):
			all_error_test+=[model.error_rate().data_columns[1][3]]
		elif (mode=='accuracy'):
			all_error_test+=[1-model.error_rate().data_columns[1][3]]
		elif (mode=='auc'):
			all_error_test+=[model.roc(show=False)[0].data_columns[1][1]]
		model=load_model(name,cursor,input_relation)
		if (mode=='logloss'):
			all_error_train+=[model.logloss()]
		elif (mode=='error_rate'):
			all_error_train+=[model.error_rate().data_columns[1][3]]
		elif (mode=='accuracy'):
			all_error_train+=[1-model.error_rate().data_columns[1][3]]
		elif (mode=='auc'):
			all_error_train+=[model.roc(show=False)[0].data_columns[1][1]]
		query="drop model if exists {};".format(name)
		cursor.execute(query)
	max_depth=range(max_depth_begin,max_depth_end+1)
	plt.rcParams['axes.facecolor']='#F4F4F4'
	plt.grid()
	plt.plot(max_depth,all_error_test,marker=marker,color='dodgerblue')
	if not(test_only):
		plt.plot(max_depth,all_error_train,marker=marker,color='mediumseagreen')
	plt.title(mode+" curve")
	plt.xlabel('max_depth')
	plt.ylabel(mode)
	if not(test_only):
		orange=mpatches.Patch(color='mediumseagreen', label='train')
		blue=mpatches.Patch(color='dodgerblue', label='test')
		plt.legend(handles=[orange,blue])
	plt.xlim(max_depth_begin,max_depth_end)
	plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
	plt.subplots_adjust(left=0.2)
	plt.show()
	return (rvd.column_matrix([["max_depth"]+list(max_depth),[mode+"_test"]+all_error_test],first_element="max_depth"),
			rvd.column_matrix([["max_depth"]+list(max_depth),[mode+"_train"]+all_error_train],first_element="max_depth"))
# Return the mse of the model for a reg model
def mse(model):
	if (not(isinstance(model,(rf_classifier,rf_regressor,svm_classifier,svm_regressor,linear_reg,logistic_reg,naive_bayes)))):
		raise TypeError("This function is not available with this model")
	if (model.mse_val==None):
		query=("select mse(obs,prediction) over () from (select "+model.response_column+
				" as obs, predict_"+model.model_type+"("+",".join(model.predictor_columns)+" using parameters "+
				"model_name='"+model.model_name+"',match_by_pos='true') as prediction from "+model.input_relation+") x;")
		model.cursor.execute(query)
		model.mse_val=model.cursor.fetchone()[0]
	return abs(model.mse_val)
# Return the value of the concerned model parameter: use * to see them all
def parameter_value(model,parameter_name):
	if (not(isinstance(model,(rf_classifier,rf_regressor,svm_classifier,svm_regressor,linear_reg,logistic_reg,naive_bayes)))):
		raise TypeError("This function is not available with this model")
	if (parameter_name in ["regularization","rejected_row_count","accepted_row_count","iteration_count","alpha","tree_count"]):
		query="select get_model_attribute(using parameters model_name='{}',attr_name='{}')".format(model.model_name,parameter_name)
		model.cursor.execute(query)
		return model.cursor.fetchone()[0]
	elif ((parameter_name=="l") or (parameter_name=="lambda")):
		query="select get_model_attribute(using parameters model_name='{}',attr_name='regularization')".format(model.model_name,parameter_name)
		model.cursor.execute(query)
		return model.cursor.fetchone()[1]
	else:
		print("/!\\ Warning: The parameter doesn't exist.")
# Plot for regressions
def plot_reg(model,color=None,projection=None,max_nb_points=1000,show=True):
	if (not(isinstance(model,(rf_classifier,rf_regressor,svm_classifier,svm_regressor,linear_reg,logistic_reg,naive_bayes)))):
		raise TypeError("This function is not available with this model")
	if (not(isinstance(max_nb_points,int)) or (max_nb_points<0)):
		raise TypeError("The parameter 'max_nb_points' must be a positive integer")
	if (not(isinstance(show,bool))):
		raise TypeError("The parameter 'show' must be a varchar")
	if (model.category=="regression"):
		coefficients=model.details().data_columns
		a0=float(coefficients[1][1])
		if (type(projection)==list):
			if (len(projection)>len(model.predictor_columns)):
				return model.plot(max_nb_points=max_nb_points,show=show)
			else:
				idx=coefficients[0].index(projection[0])
				a1=coefficients[1][idx]
				if (len(projection)>1):
					idx=coefficients[0].index(projection[1])
					a2=coefficients[1][idx]
		else:
			a1=float(coefficients[1][2])
			if (len(coefficients[1]))>=4:
				a2=float(coefficients[1][3])
			projection=model.predictor_columns
		if (len(projection)==1):
			if ((type(color)!=list) or (len(color)!=2)):
				color=["dodgerblue","black"]
			query="select {},{},random() from {} where {} is not null and {} is not null order by 3 limit {}".format(projection[0],
				model.response_column,model.input_relation,projection[0],model.response_column,max_nb_points)
			model.cursor.execute(query)
			all_points=model.cursor.fetchall()
			column1=[float(item[0]) for item in all_points]
			column2=[float(item[1]) for item in all_points]
			min_r=float(min(column1))
			max_r=float(max(column1))
			if (show):
				plt.figure(figsize=(7,5),facecolor='white')
				plt.gca().grid()
				plt.gca().set_axisbelow(True)
				plt.scatter(column1,column2,color=color[0],s=15)
				plt.plot([min_r,max_r],[a0+a1*min_r,a0+a1*max_r],color=color[1])
				plt.xlabel(projection[0])
				plt.ylabel(model.response_column)
				plt.title(model.model_type+': '+model.response_column+'='+str(round(a0,3))+"+("+str(round(a1,3))+")*"+projection[0])
				plt.show()
			return [column1,column2,[a0,a1]]
		elif (len(projection)==2):
			if ((type(color)!=list) or (len(color)!=2)):
				color=["dodgerblue","gray"]
			query="select {},{},{},random() from {} where {} is not null and {} is not null and {} is not null order by 3 limit {}".format(
				projection[0],projection[1],model.response_column,model.input_relation,projection[0],projection[1],model.response_column,max_nb_points)
			model.cursor.execute(query)
			all_points=model.cursor.fetchall()
			column1=[float(item[0]) for item in all_points]
			column2=[float(item[1]) for item in all_points]
			column3=[float(item[2]) for item in all_points]
			min_r1=float(min(column1))
			max_r1=float(max(column1))
			min_r2=float(min(column2))
			max_r2=float(max(column2))
			if (show):
				fig=plt.figure(figsize=(7,5),facecolor='white')
				ax=fig.add_subplot(111,projection='3d')
				X=np.arange(min_r1,max_r1,(max_r1-min_r1)/5.0)
				Y=np.arange(min_r2,max_r2,(max_r2-min_r2)/5.0)
				X,Y=np.meshgrid(X, Y)
				Z=a0+a1*X+a2*Y
				ax.scatter(column1,column2,column3,color[0],s=15)
				ax.plot_surface(X,Y,Z, rstride=1, cstride=1, alpha=0.8,color=color[1])
				ax.set_xlabel(projection[0])
				ax.set_ylabel(projection[1])
				ax.set_zlabel(model.response_column)
				plt.title(model.model_type+': '+model.response_column+'='+str(round(a0,3))+"+("+str(round(a1,3))+
					")*"+projection[0]+"+("+str(round(a2,3))+")*"+projection[1])
				plt.show()
		else:
			print("/!\\ Warning: The dimension is too big.")
			print("Please use the 'projection' parameter to see a projection of your figure")
	else:
		raise Exception("The model must be a regression to use this function")
# Return model metrics for a reg model
def reg_metrics(model):
	if (not(isinstance(model,(rf_classifier,rf_regressor,svm_classifier,svm_regressor,linear_reg,logistic_reg,naive_bayes)))):
		raise TypeError("This function is not available with this model")
	mse_val=model.mse()
	rsquared_val=model.rsquared()
	return rvd.column_matrix([['','mse','rsquared'],['value',mse_val,rsquared_val]])
# Plot the ROC curve
def roc(model,num_bins=1000,color=["dodgerblue","#444444"],show=True,input_class=1):
	if (not(isinstance(model,(rf_classifier,rf_regressor,svm_classifier,svm_regressor,linear_reg,logistic_reg,naive_bayes)))):
		raise TypeError("This function is not available with this model")
	if (not(isinstance(num_bins,int)) or (num_bins<0)):
		raise TypeError("The parameter 'num_bins' must be a positive integer")
	if (not(isinstance(show,bool))):
		raise TypeError("The parameter 'show' must be a varchar")
	if ((model.category=="binomial") or ((model.category=="multinomial") and ((input_class in model.classes) or (str(input_class) in model.classes)))):
		query=("select roc(obs, prob using parameters num_bins={}) over() from (select (case when {}='{}' then 1 else 0 end) as obs, predict_{}"+ 
				"({} using parameters model_name='{}',type='probability',class='{}',match_by_pos='true')::float as prob from {}) as prediction_output")
		query=query.format(num_bins,model.response_column,input_class,model.model_type,",".join(model.predictor_columns),model.model_name,
							input_class,model.input_relation)
		model.cursor.execute(query)
		query_result=model.cursor.fetchall()
		threshold=[item[0] for item in query_result]
		false_positive=[item[1] for item in query_result]
		true_positive=[item[2] for item in query_result]
		auc=0
		for i in range(len(false_positive)-1):
			if (false_positive[i+1]-false_positive[i]!=0.0):
				a=(true_positive[i+1]-true_positive[i])/(false_positive[i+1]-false_positive[i])
				b=true_positive[i+1]-a*false_positive[i+1]
				auc=auc+a*(false_positive[i+1]*false_positive[i+1]-false_positive[i]*false_positive[i])/2+b*(false_positive[i+1]-false_positive[i]);
		auc=-auc
		best_threshold_arg=np.argmax([abs(y-x) for x,y in zip(false_positive,true_positive)])
		best_threshold=threshold[best_threshold_arg]
		if (model.model_type!="svm_classifier"):
			metrics=rvd.column_matrix([['','auc','best_threshold'],['value',auc,best_threshold]])
		if (show):
			plt.figure(figsize=(7,5),facecolor='white')
			plt.rcParams['axes.facecolor']='#F5F5F5'
			plt.xlabel('False Positive Rate (1-Specificity)')
			plt.ylabel('True Positive Rate (Sensitivity)')
			plt.plot(false_positive,true_positive,color=color[0])
			plt.plot([0,1],[0,1],color=color[1])
			plt.ylim(0,1)
			plt.xlim(0,1)
			if (model.category=="multinomial"):
				plt.title(model.model_name+" ROC Curve of class '{}'\nAUC=".format(input_class)+str(auc))
			else:
				plt.title(model.model_name+" ROC Curve\nAUC="+str(auc))
			plt.gca().set_axisbelow(True)
			plt.grid()
			plt.show()
		if (model.model_type!="svm_classifier"):
			return metrics,rvd.column_matrix([['threshold']+threshold,['false_positive']+false_positive,['true_positive']+true_positive],repeat_first_column=False)
		else:
			return rvd.column_matrix([['','auc'],['value',auc]]),rvd.column_matrix([['threshold']+threshold,['false_positive']+false_positive,['true_positive']+true_positive],repeat_first_column=False)
	else:
		raise Exception("The ROC Curve is only available for multinomial/binomial models with a correct class.")
# Return the rsquared of the model for a reg model
def rsquared(model):
	if (not(isinstance(model,(rf_classifier,rf_regressor,svm_classifier,svm_regressor,linear_reg,logistic_reg,naive_bayes)))):
		raise TypeError("This function is not available with this model")
	if (model.rsquared_val==None):
		query=("select rsquared(obs,prediction) over () from (select "+model.response_column+
				" as obs, predict_"+model.model_type+"("+",".join(model.predictor_columns)+" using parameters "+
				"model_name='"+model.model_name+"',match_by_pos='true') as prediction from "+model.input_relation+") x;")
		model.cursor.execute(query)
		model.rsquared_val=model.cursor.fetchone()[0]
	return abs(model.rsquared_val)
# Summarize the Model
def summarize_model(model_name,cursor):
	query="select summarize_model('"+model_name+"')"
	cursor.execute(query)
	query_result=cursor.fetchone()[0]
	return query_result
# Print Tree Id n for a RF
def tree(model,n=0):
	if (not(isinstance(model,(rf_classifier,rf_regressor)))):
		raise TypeError("Tree is only available with Random Forest models")
	if (not(isinstance(n,int)) or (n<0)):
		raise TypeError("The parameter 'n' must be a positive integer")
	if (model.model_type=="rf_classifier"):
		leaf_info="probability"
	elif (model.model_type=="rf_regressor"):
		leaf_info="variance"
	query=("select read_tree(using parameters model_name='{}', treeid='{}')")
	query=query.format(model.model_name,n)
	model.cursor.execute(query)
	query_result=model.cursor.fetchone()[0]
	query_result=query_result.replace('TreeID: ','')
	query_result=query_result.replace(' #node: ','')
	query_result=query_result.replace(' Tree Depth: ','')
	query_result=query_result.replace(' Tree Breadth: ','')
	query_result=query_result.replace(' rootID: ','')
	query_result=query_result.replace('NodeID: ','')
	query_result=query_result.replace(' Node Depth: ','')
	query_result=query_result.replace(' isLeaf: ','')
	query_result=query_result.replace(' threshold: ','')
	query_result=query_result.replace(' split on: ','')
	query_result=query_result.replace(' isSplitCategorical: ','')
	query_result=query_result.replace(' leftCategory: ','')
	query_result=query_result.replace(' leftChild ID: ','')
	query_result=query_result.replace(' rightChild ID: ','')
	query_result=query_result.replace(' prediction: ','')
	query_result=query_result.replace(' probability: ','')
	query_result=query_result.split('\n')
	del query_result[-1]
	information=query_result[0]
	tree0=[item.split(',') for item in query_result]
	del tree0[0]
	for item in tree0:
		if (len(item)==8):
			item[4]=model.predictor_columns[int(item[4])]
	information=information.split(",")
	try:
		screen_columns=shutil.get_terminal_size().columns
	except:
		screen_rows, screen_columns = os.popen('stty size', 'r').read().split()
	print("-"*int(screen_columns))
	print("Tree Id: "+information[0])
	print("Number of Nodes: "+information[1])
	print("Tree Depth: "+information[2])
	print("Tree Breadth: "+information[3])
	print("-"*int(screen_columns))
	try:
		from anytree import Node, RenderTree
		if (tree0[0][2]=="1"):
			tree_node_1=Node("[1] => {} (probability={})".format(tree0[0][3],tree0[0][4]))
		else:
			tree_node_1=Node("[1]")
		leaf_nb=0
		for idx,item in enumerate(tree0):
			if (idx!=0):
				for item1 in tree0:
					if ((len(item1)==8) and (item[0] in [item1[6],item1[7]])):
						parent=item1
				if (parent[3]=="1"):
					if (parent[6]==item[0]):
						op="="
					else:
						op="!="
				elif (parent[6]==item[0]):
					op="<"
				else:
					op=">="
				if (item[2]=="0"):
					exec("tree_node_{}=Node('[{}] ({}{}{}){}',parent=tree_node_{})".format(item[0],item[0],parent[4],op,parent[5],"",parent[0]))
				else:
					exec("tree_node_{}=Node('[{}] ({}{}{}){}',parent=tree_node_{})".format(item[0],item[0],parent[4],op,parent[5],"",parent[0]))
					exec("tree_leaf_{}=Node('{} (probability={})',parent=tree_node_{})".format(leaf_nb,item[3],item[4],item[0]))
					leaf_nb+=1
		for pre, fill, node in RenderTree(tree_node_1):
			print("%s%s" % (pre,node.name))
		from anytree.dotexport import RenderTreeGraph 
		try: 
			RenderTreeGraph(tree_node_1).to_picture("anytree/"+model.model_name+"_tree"+str(n)+".png")
			if rvd.isnotebook():
				from IPython.core.display import HTML,display
				display(HTML("<img src='anytree/"+model.model_name+"_tree"+str(n)+".png'>"))
		except:
			print("/!\\ Warning: Please create a folder 'anytree' where you execute the code in order to export a png of the tree.")
	except:
		print("/!\\ Warning: Please install the anytree package to print the tree in the terminal.")
	all_infos=['NodeID','Node Depth','isLeaf','isSplitCategorical','split on','threshold','leftChildID','rightChildID','prediction',leaf_info]
	for idx,item in enumerate(tree0):
		if (len(item)==8):
			tree0[idx]+=['-','-']
		else:
			tree0[idx]=[item[0],item[1],item[2],'-','-','-','-','-',item[3],item[4]]
	data_columns=[]
	for i in range(0,10):
		data_columns+=[[all_infos[i]]+[item[i] for item in tree0]]
	return rvd.column_matrix(data_columns=data_columns,first_element="NodeID",title="Tree"+information[0])
#
#############################
#  __      ____  __ _       #
#  \ \    / /  \/  | |      #
#   \ \  / /| \  / | |      #
#    \ \/ / | |\/| | |      #
#     \  /  | |  | | |____  #
#      \/   |_|  |_|______| #
#                           #
#############################                         
#                        
#############################
#                           #
# Vertica Machine Learning  #
#                           #
#############################
#
##
#####################
#                   #
# Cross Validation  #
#                   #
#####################
class cross_validate:
	#
	# Initialization
	#
	def  __init__(self,algorithm,input_relation,response_column,predictor_columns,cursor,
					model_name=None,fold_count=5,hyperparams=None,prediction_cutoff=0.5,load=False):
		drop_at_the_end=False
		if not(load):
			if (type(model_name)!=str):
				drop_at_the_end=True
				model_name="_vpython_cv_"+str(np.random.randint(10000))
			query="select cross_validate('{}','{}','{}','{}' using parameters cv_model_name='{}',cv_metrics='accuracy,error_rate'"
			query=query.format(algorithm,input_relation,response_column,",".join(predictor_columns),model_name)
			query+=",cv_fold_count={}".format(fold_count)
			if (type(hyperparams)==str):
				query+=",cv_hyperparams='{}'".format(hyperparams)
			if (algorithm=="logistic_reg"):
				query+=",cv_prediction_cutoff={}".format(prediction_cutoff)
			query+=")"
			cursor.execute(query)
		self.cursor=cursor
		self.model_name=model_name
		self.model_type="cross_validation"
		print(self)
		if (drop_at_the_end):
			drop_model(model_name,cursor)
	# Object Representation
	def __repr__(self):
		formatted_text="model_type='{}'\nmodel_name='{}'\nCounters:\n"+rvd.print_table(self.get_model_attribute("counters"))
		formatted_text+="Fold Info:\n"+rvd.print_table(self.get_model_attribute("fold_info"))
		formatted_text+="Details:\n"+rvd.print_table(self.get_model_attribute("run_details"))
		formatted_text+="Averages:\n"+rvd.print_table(self.get_model_attribute("run_average"))[0:-2]
		formatted_text=formatted_text.format(self.model_type,self.model_name)
		if isnotebook():
			return "<cross_validate>"
		else:
			return formatted_text
	#
	###########
	#         #
	# Methods #
	#         #
	###########
	# 
	def get_model_attribute(self,attr_name="run_details"):
		if (attr_name in ["run_details","run_average","fold_info","counters","call_string"]):
			query="select get_model_attribute(using parameters model_name='{}',attr_name='{}')".format(
						self.model_name,attr_name)
			self.cursor.execute(query)
			columns=[column[0] for column in self.cursor.description]
			all_columns=[]
			query_result=self.cursor.fetchall()
			for i in range(len(query_result[0])):
				all_columns+=[[columns[i]]+[item[i] for item in query_result]]
			return all_columns
		else:
			raise TypeError("'attr_name' must be in run_details|run_average|fold_info|counters|call_string")
###########
#         #
# Kmeans  #
#         #
###########
class kmeans:
	#
	# Initialization
	#
	def  __init__(self,model_name,input_relation,input_columns,num_clusters,cursor,max_iterations=10,
				epsilon=1e-4,init_method="kmeanspp",initial_centers=None,load=False):
		if not(load):
			query="select kmeans('{}','{}','{}',{} using parameters max_iterations={},epsilon={}"
			query=query.format(model_name,input_relation,",".join(input_columns),num_clusters,max_iterations,epsilon)
			name="_vpython_kmeans_initial_centers_table_"+str(np.random.randint(1000000))
			if (type(initial_centers)==list):
				query0="drop table if exists "+name
				cursor.execute(query0)
				if (len(initial_centers)!=num_clusters):
					print("/!\\ Warning: 'initial_centers' must be a list of 'num_clusters'={} points".format(num_clusters))
					print("The 'initial_centers' will be choosed using the 'init_method'="+init_method)
				else:
					wrong_initial_centers=False
					for item in initial_centers:
						if (len(input_columns)!=len(item)):
							wrong_initial_centers=True
							break
					if (wrong_initial_centers):
						print("/!\\ Warning: Each points of 'initial_centers' must be of size len({})={}".format(
									input_columns,len(input_columns)))
						print("The 'initial_centers' will be choosed using the 'init_method' "+init_method)
					else:
						temp_initial_centers=[item for item in initial_centers]
						for item in initial_centers:
							del temp_initial_centers[0]
							if (item in temp_initial_centers):
								wrong_initial_centers=True
								break
						if (wrong_initial_centers):
							print("/!\\ Warning: All the points of 'initial_centers' must be different")
							print("The 'initial_centers' will be choosed using the 'init_method' "+init_method)
						else:
							query0=[]
							for i in range(len(initial_centers)):
								line=[]
								for j in range(len(initial_centers[0])):
									line+=[str(initial_centers[i][j])+" as "+input_columns[j]]
								line=",".join(line)
								query0+=["select "+line]
							query0=" union ".join(query0)
							query0="create table "+name+" as "+query0
							cursor.execute(query0)
							query+=",initial_centers_table='"+name+"'"
			else:
				query+=",init_method='"+init_method+"'"
			query+=")"
			cursor.execute(query)
			query="drop table if exists "+name
			cursor.execute(query)
		self.cursor=cursor
		self.model_name=model_name
		self.input_relation=input_relation
		self.input_columns=input_columns
		self.num_clusters=num_clusters
		self.model_type="kmeans"
		self.category="clustering"
		self.all_importances=None
	# Object Representation
	def __repr__(self):
		query="select get_model_attribute(using parameters model_name='{}',attr_name='centers')".format(self.model_name)
		self.cursor.execute(query)
		query_result=self.cursor.fetchall()
		columns=[]
		for i in range(0,len(self.input_columns)):
			columns+=[[self.input_columns[i]]+[item[i] for item in query_result]]
		if (isnotebook()):
			rvd.print_table(columns)[0:-2]
			formatted_text=""
		else:
			formatted_text="Clusters:\n"+rvd.print_table(columns)[0:-2]
		formatted_text="model_type='{}'\nmodel_name='{}'\ninput_relation='{}'\ninput_columns='{}'\n"+formatted_text
		formatted_text=formatted_text.format(self.model_type,self.model_name,self.input_relation,",".join(self.input_columns))
		return formatted_text
	#
	###########
	#         #
	# Methods #
	#         #
	###########
	# 
	# Add the Kmeans prediction to the rvd
	def add_to_rvd(self,rvd,name="kmeans_cluster"+str(np.random.randint(10000))):
		imputation="apply_kmeans("+",".join(self.input_columns)+" using parameters model_name='"+self.model_name+"'"+",match_by_pos='True')"
		rvd.add_feature(name,imputation)
		return name
	# Return True if the model Converged
	def converged(self):
		query="select get_model_attribute(using parameters model_name='{}', attr_name='metrics')".format(self.model_name)
		self.cursor.execute(query)
		query_result=self.cursor.fetchone()[0]
		if (query_result.split("Converged: ")[1].split("\n")[0]=="True"):
			return True
		return False
	# Features Importance
	def features_importance(self,show=True):
		return features_importance(self,show=show)
	# Sum of Squares
	def between_cluster_SS(self):
		query="select get_model_attribute(using parameters model_name='{}', attr_name='metrics')".format(self.model_name)
		self.cursor.execute(query)
		query_result=self.cursor.fetchone()[0]
		return float(query_result.split("Between-Cluster Sum of Squares: ")[1].split("\n")[0])
	def cluster_SS(self,show=True,display=True):
		query="select get_model_attribute(using parameters model_name='{}', attr_name='metrics')".format(self.model_name)
		self.cursor.execute(query)
		query_result=self.cursor.fetchone()[0]
		all_cluster_SS=[]
		for i in range(0,self.num_clusters):
			all_cluster_SS+=[float(query_result.split("Cluster "+str(i)+": ")[1].split("\n")[0])]
		if (show):
			all_cluster_SS_pr=['Cluster SS']+all_cluster_SS
			if not(isnotebook()):
				print(rvd.print_table([all_cluster_SS_pr])[0:-2])
			else:
				rvd.print_table([all_cluster_SS_pr])[0:-2]
		if (display):
			plt.figure(figsize=(7,5))
			plt.rcParams['axes.facecolor']='#E5E5E5'
			clusters=range(0,self.num_clusters)
			all_cluster_SS_pr=all_cluster_SS
			plt.barh(clusters,all_cluster_SS_pr,0.9,alpha=0.86,color="dodgerblue")
			plt.ylabel("Clusters")
			plt.xlabel("Sum of Squares")
			plt.title("Clusters Sum of Squares")
			plt.gca().xaxis.grid()
			plt.gca().set_axisbelow(True)
			plt.yticks(clusters)
			plt.show()
		return all_cluster_SS
	def total_SS(self):
		query="select get_model_attribute(using parameters model_name='{}', attr_name='metrics')".format(self.model_name)
		self.cursor.execute(query)
		query_result=self.cursor.fetchone()[0]
		return float(query_result.split("Total Sum of Squares: ")[1].split("\n")[0])
	def within_cluster_SS(self):
		query="select get_model_attribute(using parameters model_name='{}', attr_name='metrics')".format(self.model_name)
		self.cursor.execute(query)
		query_result=self.cursor.fetchone()[0]
		return float(query_result.split("Total Within-Cluster Sum of Squares: ")[1].split("\n")[0])
######################
#                    #
# Linear Regression  #
#                    #
######################
class linear_reg:
	#
	# Initialization
	#
	def  __init__(self,model_name,input_relation,response_column,predictor_columns,cursor,optimizer='Newton',
				epsilon=1e-6,max_iterations=100,regularization="None",l=0.0,alpha=0.5,load=False):
		if not(load):
			query="select linear_reg('{}','{}','{}','{}' using parameters optimizer='{}',epsilon={},max_iterations={}"
			query=query.format(model_name,input_relation,response_column,",".join(predictor_columns),
					optimizer,epsilon,max_iterations)
			query+=",regularization='{}',lambda={}".format(regularization,l)
			if (regularization=='ENet'):
				query+=",alpha={}".format(alpha)
			query+=")"
			cursor.execute(query)
		self.cursor=cursor
		self.model_name=model_name
		self.input_relation=input_relation
		self.response_column=response_column
		self.predictor_columns=predictor_columns
		self.model_type="linear_reg"
		self.category="regression"
		self.all_importances=None
		self.mse_val=None
		self.rsquared_val=None
	# Object Representation
	def __repr__(self):
		object_repr=self.details().__repr__()
		formatted_text=("model_type='{}'\nmodel_name='{}'\ninput_relation='{}'\nresponse_column='{}'\npredictor_columns='{}'\n"+
							self.parameter_value(show=False)[4])
		if not(isnotebook()):
			formatted_text=formatted_text+"\nParameters:\n"+object_repr
		formatted_text=formatted_text.format(self.model_type,self.model_name,self.input_relation,self.response_column,",".join(self.predictor_columns))
		return formatted_text
	#
	###########
	#         #
	# Methods #
	#         #
	###########
	# 
	# Add the linear_reg prediction to the rvd
	def add_to_rvd(self,rvd,name="linear_reg_pred"+str(np.random.randint(10000))):
		imputation="predict_linear_reg("+",".join(self.predictor_columns)+" using parameters model_name='"+self.model_name+"'"+",match_by_pos='True')"
		rvd.add_feature(name,imputation)
		return name
	# All the details of the model: p-value,t-value,coeffs...
	def details(self):
		return details(self.model_name,self.cursor)
	# Features Importance
	def features_importance(self,show=True,with_intercept=False):
		return features_importance(self,show=show,with_intercept=with_intercept)
	# MSE and RSQUARED
	def metrics(self):
		return reg_metrics(self)
	# Return the mse of the model
	def mse(self):
		return mse(self)
	# Return the value of the concerned parameter: use * to see them all
	def parameter_value(self,parameter_name="*",show=True):
		if (parameter_name=="*"):
			r=self.parameter_value("regularization")
			l=self.parameter_value("lambda")
			r_row=self.parameter_value("rejected_row_count")
			a_row=self.parameter_value("accepted_row_count")
			formatted_text="regularization: "+str(r)
			formatted_text+="\nlambda: "+str(l)
			formatted_text+="\nrejected_row_count: "+str(r_row)
			formatted_text+="\naccepted_row_count: "+str(a_row)
			if (show):
				print(formatted_text)
			return r,l,r_row,a_row,formatted_text
		elif (parameter_name in ["regularization","l","lambda","rejected_row_count","accepted_row_count"]):
			return parameter_value(self,parameter_name=parameter_name)
		else:
			print("Please use a correct parameter value: regularization|lambda|rejected_row_count|accepted_row_count|*")
			return False
	def plot(self,color=None,projection=None,max_nb_points=1000,show=True):
		plot_reg(self,color=color,projection=projection,max_nb_points=max_nb_points,show=show)
	# Return the rsquared of the model
	def rsquared(self):
		return rsquared(self)
########################
#                      #
# Logistic Regression  #
#                      #
########################
class logistic_reg:
	#
	# Initialization
	#
	def  __init__(self,model_name,input_relation,response_column,predictor_columns,cursor,optimizer='Newton',epsilon=1e-6,
					max_iterations=100,regularization='None',l=1,alpha=0.5,load=False):
		if not(load):
			query="select logistic_reg('{}','{}','{}','{}' using parameters optimizer='{}',epsilon={},max_iterations={}"
			query=query.format(model_name,input_relation,response_column,",".join(predictor_columns),
					optimizer,epsilon,max_iterations)
			query+=",regularization='{}',lambda={}".format(regularization,l)
			if (regularization=='ENet'):
				query+=",alpha={}".format(alpha)
			query+=")"
			cursor.execute(query)
		self.cursor=cursor
		self.model_name=model_name
		self.input_relation=input_relation
		self.response_column=response_column
		self.predictor_columns=predictor_columns
		self.model_type="logistic_reg"
		self.category="binomial"
		self.all_importances=None
		self.logloss_value=None
	# Object Representation
	def __repr__(self):
		object_repr=self.details().__repr__()
		formatted_text=("model_type='{}'\nmodel_name='{}'\ninput_relation='{}'\nresponse_column='{}'\npredictor_columns='{}'\n"+
							self.parameter_value(show=False)[4])
		if not(isnotebook()):
			formatted_text=formatted_text+"\nParameters:\n"+object_repr
		formatted_text=formatted_text.format(self.model_type,self.model_name,self.input_relation,self.response_column,",".join(self.predictor_columns))
		return formatted_text
	#
	###########
	#         #
	# Methods #
	#         #
	###########
	# 
	# model accuracy
	def accuracy(self,threshold=0.5):
		return accuracy(self,threshold,1)
	# Add the logistic_reg prediction to the rvd
	def add_to_rvd(self,rvd,name="logistic_reg_pred"+str(np.random.randint(10000)),prediction_type='response',cutoff=0.5):
		if (prediction_type in ['response','probability']):
			imputation=("predict_logistic_reg("+",".join(self.predictor_columns)+" using parameters model_name='"
							+self.model_name+"',"+"type='{}',cutoff={},match_by_pos='True')".format(prediction_type,cutoff))
			rvd.add_feature(name,imputation)
			return name
		else:
			raise TypeError("Please use a correct prediction_type: response|probability")
	# model auc
	def auc(self):
		return auc(self,1)
	# Confusion Matrix
	def confusion_matrix(self,threshold=0.5):
		return confusion_matrix(self,threshold=threshold)
	# All the details of the model: p-value,t-value,coeffs...
	def details(self):
		return details(self.model_name,self.cursor)
	# Error Rate
	def error_rate(self,threshold=0.5):
		return error_rate(self,threshold=threshold)
	# Features Importance
	def features_importance(self,show=True,with_intercept=False):
		return features_importance(self,show=show,with_intercept=with_intercept)
	# Lift Table
	def lift_table(self,num_bins=100,color=["dodgerblue","#444444"],show=True,input_class=1):
		return lift_table(self,num_bins=num_bins,color=color,show=show)
	# Log Loss
	def logloss(self):
		if (self.logloss_value==None):
			return logloss(self)
		return self.logloss_value
	# Return the value of the concerned parameter: use * to see them all
	def parameter_value(self,parameter_name="*",show=True):
		if (parameter_name=="*"):
			r=self.parameter_value("regularization")
			l=self.parameter_value("lambda")
			r_row=self.parameter_value("rejected_row_count")
			a_row=self.parameter_value("accepted_row_count")
			formatted_text="regularization: "+str(r)
			formatted_text+="\nlambda: "+str(l)
			formatted_text+="\nrejected_row_count: "+str(r_row)
			formatted_text+="\naccepted_row_count: "+str(a_row)
			if (show):
				print(formatted_text)
			return r,l,r_row,a_row,formatted_text
		elif (parameter_name in ["regularization","l","lambda","rejected_row_count","accepted_row_count"]):
			return parameter_value(self,parameter_name=parameter_name)
		else:
			raise TypeError("Please use a correct parameter value: regularization|lambda|rejected_row_count|accepted_row_count|*")
	def plot(self,marker=["o","^"],color=None,projection=None,max_nb_points=None):
		def logit(x):
			return 1/(1+math.exp(-x))
		coefficients=self.details().data_columns
		a0=float(coefficients[1][1])
		alpha=[0.9,0.7]
		if (type(projection)==list):
			if (len(projection)>len(self.predictor_columns)):
				return self.plot(max_nb_points=max_nb_points,show=show)
			else:
				idx=coefficients[0].index(projection[0])
				a1=coefficients[1][idx]
				if (len(projection)>1):
					idx=coefficients[0].index(projection[1])
					a2=coefficients[1][idx]
		else:
			a1=float(coefficients[1][2])
			if (len(coefficients[1]))>=4:
				a2=float(coefficients[1][3])
			projection=self.predictor_columns
		if ((type(max_nb_points)!=int) or (max_nb_points<0)):
			if (len(projection)==1):
				max_nb_points=500
			else:
				max_nb_points=1000
		if (len(projection)==1):
			if ((type(color)!=list) or (len(color)!=3)):
				color=["mediumseagreen","dodgerblue","black"]
			columns=[]
			for i in range(2):
				query="select {},random() from {} where {} is not null and {}={} order by 2 limit {}".format(projection[0],
						self.input_relation,projection[0],self.response_column,i,int(max_nb_points/2))
				self.cursor.execute(query)
				all_points=self.cursor.fetchall()
				columns+=[[float(item[0]) for item in all_points]]
			plt.figure(figsize=(7,5),facecolor='#EFEFEF')
			all_scatter=[]
			min_f=min(columns[0]+columns[1])
			max_f=max(columns[0]+columns[1])
			x=np.linspace(min_f,max_f,num=1000)
			y=[logit(a0+a1*item) for item in x]
			plt.plot(x,y,alpha=0.5,color=color[2])
			for i in range(2):
				all_scatter+=[plt.scatter(columns[i],[logit(a0+a1*item) for item in columns[i]],alpha=alpha[i],marker=marker[i],color=color[i])]
			plt.gca().grid()
			plt.gca().set_axisbelow(True)
			plt.xlabel(projection[0])
			plt.ylabel("logit")
			plt.legend(all_scatter,[0,1],scatterpoints=1,loc="upper right",ncol=4,title=self.response_column,fontsize=8)
			plt.title(self.model_type+': '+self.response_column+'=logit('+str(round(a0,3))+"+("+str(round(a1,3))+")*"+projection[0]+")")
			plt.show()
		elif (len(projection)==2):
			if ((type(color)!=list) or (len(color)!=3)):
				color=["mediumseagreen","dodgerblue","gray"]
			columns=[]
			for i in range(2):
				query="select {},{},random() from {} where {} is not null and {} is not null and {}={} order by 3 limit {}".format(
					projection[0],projection[1],self.input_relation,projection[0],projection[1],self.response_column,i,int(max_nb_points/2))
				self.cursor.execute(query)
				all_points=self.cursor.fetchall()
				columns+=[[[float(item[0]) for item in all_points],[float(item[1]) for item in all_points]]]
			min_f1=float(min(columns[0][0]+columns[1][0]))
			max_f1=float(max(columns[0][0]+columns[1][0]))
			min_f2=float(min(columns[0][1]+columns[1][1]))
			max_f2=float(max(columns[0][1]+columns[1][1]))
			X=np.arange(min_f1,max_f1,(max_f1-min_f1)/40.0)
			Y=np.arange(min_f2,max_f2,(max_f2-min_f2)/40.0)
			X,Y=np.meshgrid(X, Y)
			Z=1/(1+np.exp(-(a0+a1*X+a2*Y)))
			fig=plt.figure(figsize=(7,5),facecolor='white')
			ax=fig.add_subplot(111,projection='3d')
			ax.plot_surface(X,Y,Z, rstride=1, cstride=1, alpha=0.5,color=color[2])
			all_scatter=[]
			logit3D=[[],[]]
			for i in range(2):
				for j in range(len(columns[i][0])):
					logit3D[i]+=[a0+columns[i][0][j]*a1+columns[i][1][j]*a2]
			for i in range(2):
				logit3D[i]=[logit(item) for item in logit3D[i]]
				all_scatter+=[ax.scatter(columns[i][0],columns[i][1],logit3D[i],alpha=alpha[i],marker=marker[i],color=color[i])]
			ax.set_xlabel(projection[0])
			ax.set_ylabel(projection[1])
			ax.set_zlabel("logit")
			ax.legend(all_scatter,[0,1],scatterpoints=1,loc="lower left",ncol=4,title=self.response_column,fontsize=8,bbox_to_anchor=(0.9,1))
			plt.title(self.model_type+': '+self.response_column+'=logit('+str(round(a0,3))+"+("+str(round(a1,3))+
							")*"+projection[0]+"+("+str(round(a2,3))+")*"+projection[1]+")")
			plt.show()
		else:
			print("/!\\ Warning: The dimension is too big.")
			print("Please use the 'projection' parameter to see a projection of your figure")
	def roc(self,num_bins=100,color=["dodgerblue","#444444"],show=True):
		return roc(self,num_bins=num_bins,color=color,show=show)
################
#              #
# Naive Bayes  #
#              #
################
class naive_bayes:
	#
	# Initialization
	#
	def  __init__(self,model_name,input_relation,response_column,predictor_columns,cursor,alpha=1.0,load=False):
		if not(load):
			query="select naive_bayes('{}','{}','{}','{}' using parameters alpha={}"
			query=query.format(model_name,input_relation,response_column,",".join(predictor_columns),
					alpha)+")"
			cursor.execute(query)
		self.cursor=cursor
		self.model_name=model_name
		self.input_relation=input_relation
		self.response_column=response_column
		self.predictor_columns=predictor_columns
		self.model_type="naive_bayes"
		self.logloss_value=None
		query="select {} from {} group by {}".format(response_column,input_relation,response_column)
		cursor.execute(query)
		query_result=cursor.fetchall()
		classes=[item for sublist in query_result for item in sublist]
		classes.sort()
		if (len(classes)>2):
			self.category="multinomial"
		else:
			self.category="binomial"
		self.classes=classes
	# Object Representation
	def __repr__(self):
		formatted_text=self.details().__repr__()
		formatted_text=("model_type='{}'\nmodel_name='{}'\ninput_relation='{}'\nresponse_column='{}'\npredictor_columns='{}'\n"+
							self.parameter_value(show=False)[3]+"\nProbabilities:\n"+formatted_text)
		formatted_text=formatted_text.format(self.model_type,self.model_name,self.input_relation,self.response_column,",".join(self.predictor_columns))
		return formatted_text
	#
	###########
	#         #
	# Methods #
	#         #
	###########
	# 
	# model accuracy
	def accuracy(self,threshold=0.5,input_class=None):
		if (len(self.classes)==2) and (input_class==None):
			input_class=self.classes[1]
		return accuracy(self,threshold,input_class)
	# Add the naive_bayes prediction to the rvd
	def add_to_rvd(self,rvd,name="naive_bayes_pred"+str(np.random.randint(10000)),prediction_type='response',input_class=None):
		if (prediction_type in ['response','probability']):
			imputation=("predict_naive_bayes("+",".join(self.predictor_columns)+" using parameters model_name='"
							+self.model_name+"',"+"type='{}'".format(prediction_type))
			if ((prediction_type=="probability") and ((input_class in self.classes) or (int(input_class) in self.classes))):
				imputation+=",class='{}'".format(input_class)
			imputation+=",match_by_pos='True')+0"
			rvd.add_feature(name,imputation)
			return name
		else:
			raise TypeError("Please use a correct prediction_type: response|probability")
	# model auc
	def auc(self,input_class=None):
		if not((input_class in self.classes) or (str(input_class) in self.classes)):
			input_class=self.classes[1]
		return auc(self,input_class)
	# Confusion Matrix
	def confusion_matrix(self,threshold=0.5,input_class=None):
		if (len(self.classes)==2) and (input_class==None):
			input_class=self.classes[1]
		return confusion_matrix(self,threshold=threshold,input_class=input_class)
	# All the details of the model: probabilities
	def details(self):
		return details(self.model_name,self.cursor,attr_name="prior")
	# Error Rate
	def error_rate(self,threshold=0.5,input_class=None):
		if (len(self.classes)==2) and (input_class==None):
			input_class=self.classes[1]
		return error_rate(self,threshold=threshold,input_class=input_class)
	# Lift Table
	def lift_table(self,num_bins=100,color=["dodgerblue","#444444"],show=True,input_class=None):
		if (self.category=="binomial"):
			input_class=self.classes[1]
		elif (input_class==None):
			input_class=self.classes[0]
		return lift_table(self,num_bins=num_bins,color=color,show=show,input_class=input_class)
	# Log Loss
	def logloss(self):
		if (self.logloss_value==None):
			return logloss(self)
		return self.logloss_value
	# Return the value of the concerned parameter: use * to see them all
	def parameter_value(self,parameter_name="*",show=True):
		if (parameter_name=="*"):
			alpha=self.parameter_value("alpha")
			r_row=self.parameter_value("rejected_row_count")
			a_row=self.parameter_value("accepted_row_count")
			formatted_text="alpha: "+str(alpha)
			formatted_text+="\nrejected_row_count: "+str(r_row)
			formatted_text+="\naccepted_row_count: "+str(a_row)
			if (show):
				print(formatted_text)
			return alpha,r_row,a_row,formatted_text
		elif (parameter_name in ["alpha","rejected_row_count","accepted_row_count"]):
			return parameter_value(self,parameter_name=parameter_name)
		else:
			print("Please use a correct parameter value: alpha|rejected_row_count|accepted_row_count|*")
			return False
	# ROC
	def roc(self,num_bins=100,color=["dodgerblue","#444444"],show=True,input_class=None):
		if (self.category=="binomial"):
			input_class=self.classes[1]
		elif (input_class==None):
			input_class=self.classes[0]
		return roc(self,num_bins=num_bins,color=color,show=show,input_class=input_class)
#############################
#                           #
# Random Forest Classifier  #
#                           #
#############################
class rf_classifier:
	#
	# Initialization
	#
	def  __init__(self,model_name,input_relation,response_column,predictor_columns,cursor,ntree=20,
				mtry=None,sampling_size=0.632,max_depth=5,max_breadth=32,min_leaf_size=1,min_info_gain=0.0,
				nbins=32,load=False):
		if not(load):
			if (mtry==None):
				mtry=max(int(len(predictor_columns)/3),1)
			query=("select rf_classifier('{}','{}','{}','{}' using parameters ntree={},mtry={},sampling_size={},"+
					"max_depth={},max_breadth={},min_leaf_size={},min_info_gain={},nbins={}")
			query=query.format(model_name,input_relation,response_column,",".join(predictor_columns),
					ntree,mtry,sampling_size,max_depth,max_breadth,min_leaf_size,min_info_gain,nbins)+")"
			cursor.execute(query)
		self.cursor=cursor
		self.model_name=model_name
		self.input_relation=input_relation
		self.response_column=response_column
		self.predictor_columns=predictor_columns
		self.model_type="rf_classifier"
		self.logloss_value=None
		query="select {} from {} group by {}".format(response_column,input_relation,response_column)
		cursor.execute(query)
		query_result=cursor.fetchall()
		classes=[item for sublist in query_result for item in sublist]
		classes.sort()
		if (len(classes)>2):
			self.category="multinomial"
		else:
			self.category="binomial"
		self.classes=classes
	# Object Representation
	def __repr__(self):
		object_repr=self.details().__repr__()
		formatted_text=("model_type='{}'\nmodel_name='{}'\ninput_relation='{}'\nresponse_column='{}'\npredictor_columns='{}'\n"+
							self.parameter_value(show=False)[3])
		if not(isnotebook()):
			formatted_text=formatted_text+"\n"+object_repr
		formatted_text=formatted_text.format(self.model_type,self.model_name,self.input_relation,self.response_column,",".join(self.predictor_columns))
		return formatted_text
	#
	###########
	#         #
	# Methods #
	#         #
	###########
	# 
	# model accuracy
	def accuracy(self,threshold=0.5,input_class=None):
		if (len(self.classes)==2) and (input_class==None):
			input_class=self.classes[1]
		return accuracy(self,threshold,input_class)
	# Add the rf_classifier prediction to the rvd
	def add_to_rvd(self,rvd,name="rf_classifier_pred"+str(np.random.randint(10000)),prediction_type='response',input_class=None):
		if (prediction_type in ['response','probability']):
			imputation=("predict_rf_classifier("+",".join(self.predictor_columns)+" using parameters model_name='"
							+self.model_name+"',"+"type='{}'".format(prediction_type))
			if ((prediction_type=="probability") and ((input_class in self.classes) or (int(input_class) in self.classes))):
				imputation+=",class='{}'".format(input_class)
			imputation+=",match_by_pos='True')+0"
			rvd.add_feature(name,imputation)
			return name
		else:
			raise TypeError("Please use a correct prediction_type: response|probability")
	# model auc
	def auc(self,input_class=None):
		if not((input_class in self.classes) or (str(input_class) in self.classes)):
			input_class=self.classes[1]
		return auc(self,input_class)
	# Confusion Matrix
	def confusion_matrix(self,threshold=0.5,input_class=None):
		if (len(self.classes)==2) and (input_class==None):
			input_class=self.classes[1]
		return confusion_matrix(self,threshold=threshold,input_class=input_class)
	# All the details of the model: probabilities
	def details(self):
		return details(self.model_name,self.cursor,model_type="rf")
	# Error Rate
	def error_rate(self,threshold=0.5,input_class=None):
		if (len(self.classes)==2) and (input_class==None):
			input_class=self.classes[1]
		return error_rate(self,threshold=threshold,input_class=input_class)
	# Lift Table
	def lift_table(self,num_bins=100,color=["dodgerblue","#444444"],show=True,input_class=None):
		if (self.category=="binomial"):
			input_class=self.classes[1]
		elif (input_class==None):
			input_class=self.classes[0]
		return lift_table(self,num_bins=num_bins,color=color,show=show,input_class=input_class)
	# Log Loss
	def logloss(self):
		if (self.logloss_value==None):
			return logloss(self)
		return self.logloss_value
	# Return the value of the concerned parameter: use * to see them all
	def parameter_value(self,parameter_name="*",show=True):
		if (parameter_name=="*"):
			t_count=self.parameter_value("tree_count")
			r_row=self.parameter_value("rejected_row_count")
			a_row=self.parameter_value("accepted_row_count")
			formatted_text="tree_count: "+str(t_count)
			formatted_text+="\nrejected_row_count: "+str(r_row)
			formatted_text+="\naccepted_row_count: "+str(a_row)
			if (show):
				print(formatted_text)
			return t_count,r_row,a_row,formatted_text
		elif (parameter_name in ["tree_count","rejected_row_count","accepted_row_count"]):
			return parameter_value(self,parameter_name=parameter_name)
		else:
			raise TypeError("Please use a correct parameter value: tree_count|rejected_row_count|accepted_row_count|*")
	# ROC
	def roc(self,num_bins=100,color=["dodgerblue","#444444"],show=True,input_class=None):
		if (self.category=="binomial"):
			input_class=self.classes[1]
		elif (input_class==None):
			input_class=self.classes[0]
		return roc(self,num_bins=num_bins,color=color,show=show,input_class=input_class)
	# Print Tree Id n
	def tree(self,n=0):
		return tree(self,n)
############################
#                          #
# Random Forest Regressor  #
#                          #
############################
class rf_regressor:
	#
	# Initialization
	#
	def  __init__(self,model_name,input_relation,response_column,predictor_columns,cursor,ntree=20,
				mtry=None,sampling_size=0.632,max_depth=5,max_breadth=32,min_leaf_size=5,min_info_gain=0.0,
				nbins=32,load=False):
		if not(load):
			if (mtry==None):
				mtry=max(int(len(predictor_columns)/3),1)
			query=("select rf_regressor('{}','{}','{}','{}' using parameters ntree={},mtry={},sampling_size={},"+
					"max_depth={},max_breadth={},min_leaf_size={},min_info_gain={},nbins={}")
			query=query.format(model_name,input_relation,response_column,",".join(predictor_columns),
					ntree,mtry,sampling_size,max_depth,max_breadth,min_leaf_size,min_info_gain,nbins)+")"
			cursor.execute(query)
		self.cursor=cursor
		self.model_name=model_name
		self.input_relation=input_relation
		self.response_column=response_column
		self.predictor_columns=predictor_columns
		self.model_type="rf_regressor"
		self.category="regression"
		self.mse_val=None
		self.rsquared_val=None
	# Object Representation
	def __repr__(self):
		object_repr=self.details().__repr__()
		formatted_text=("model_type='{}'\nmodel_name='{}'\ninput_relation='{}'\nresponse_column='{}'\npredictor_columns='{}'\n"+
							self.parameter_value(show=False)[3])
		if not(isnotebook()):
			formatted_text=formatted_text+"\n"+object_repr
		formatted_text=formatted_text.format(self.model_type,self.model_name,self.input_relation,self.response_column,",".join(self.predictor_columns))
		return formatted_text
	#
	###########
	#         #
	# Methods #
	#         #
	###########
	# 
	# Add the rf_regressor prediction to the rvd
	def add_to_rvd(self,rvd,name="rf_regressor_pred"+str(np.random.randint(10000))):
		imputation="predict_rf_regressor("+",".join(self.predictor_columns)+" using parameters model_name='"+self.model_name+"'"+",match_by_pos='True')"
		rvd.add_feature(name,imputation)
		return name
	# All the details of the model: p-value,t-value,coeffs...
	def details(self):
		return details(self.model_name,self.cursor,model_type="rf")
	# MSE and RSQUARED
	def metrics(self):
		return reg_metrics(self)
	# Return the mse of the model
	def mse(self):
		return mse(self)
	# Return the value of the concerned parameter: use * to see them all
	def parameter_value(self,parameter_name="*",show=True):
		if (parameter_name=="*"):
			t_count=self.parameter_value("tree_count")
			r_row=self.parameter_value("rejected_row_count")
			a_row=self.parameter_value("accepted_row_count")
			formatted_text="tree_count: "+str(t_count)
			formatted_text+="\nrejected_row_count: "+str(r_row)
			formatted_text+="\naccepted_row_count: "+str(a_row)
			if (show):
				print(formatted_text)
			return t_count,r_row,a_row,formatted_text
		elif (parameter_name in ["tree_count","rejected_row_count","accepted_row_count"]):
			return parameter_value(self,parameter_name=parameter_name)
		else:
			raise TypeError("Please use a correct parameter value: regularization|lambda|rejected_row_count|accepted_row_count|*")
	# Return the rsquared of the model
	def rsquared(self):
		return rsquared(self)
	# Print Tree Id n
	def tree(self,n=0):
		return tree(self,n)
###################
#                 #
# SVM Classifier  #
#                 #
###################
class svm_classifier:
	#
	# Initialization
	#
	def  __init__(self,model_name,input_relation,response_column,predictor_columns,cursor,C=1.0,epsilon=1e-3,
					max_iterations=100,load=False):
		if not(load):
			query="select svm_classifier('{}','{}','{}','{}' using parameters C={},epsilon={},max_iterations={}"
			query=query.format(model_name,input_relation,response_column,",".join(predictor_columns),
					C,epsilon,max_iterations)+")"
			cursor.execute(query)
		self.cursor=cursor
		self.model_name=model_name
		self.input_relation=input_relation
		self.response_column=response_column
		self.predictor_columns=predictor_columns
		self.model_type="svm_classifier"
		self.category="binomial"
		self.all_importances=None
	# Object Representation
	def __repr__(self):
		object_repr=self.details().__repr__()
		formatted_text=("model_type='{}'\nmodel_name='{}'\ninput_relation='{}'\nresponse_column='{}'\npredictor_columns='{}'\n"+
							self.parameter_value(show=False)[3])
		if not(isnotebook()):
			formatted_text=formatted_text+"\nParameters:\n"+object_repr
		formatted_text=formatted_text.format(self.model_type,self.model_name,self.input_relation,self.response_column,",".join(self.predictor_columns))
		return formatted_text
	#
	###########
	#         #
	# Methods #
	#         #
	###########
	# 
	# model accuracy
	def accuracy(self):
		return accuracy(self,0.5,1)
	# Add the svm_classifier prediction to the rvd
	def add_to_rvd(self,rvd,name="svm_classifier_pred"+str(np.random.randint(10000))):
		imputation="predict_svm_classifier("+",".join(self.predictor_columns)+" using parameters model_name='"+self.model_name+"'"+",match_by_pos='True')"
		rvd.add_feature(name,imputation)
		return name
	# model auc
	def auc(self):
		return auc(self,0.5,1)
	# Confusion Matrix
	def confusion_matrix(self):
		return confusion_matrix(self)
	# All the details of the model: p-value,t-value,coeffs...
	def details(self):
		return details(self.model_name,self.cursor)
	# Error Rate
	def error_rate(self):
		return error_rate(self)
	# Features Importance
	def features_importance(self,show=True,with_intercept=False):
		return features_importance(self,show=show,with_intercept=with_intercept)
	# Lift Table
	def lift_table(self,num_bins=100,color=["dodgerblue","#444444"],show=True):
		return lift_table(self,num_bins=num_bins,color=color,show=show)
	# Log Loss
	def logloss(self):
		if (self.logloss_value==None):
			return logloss(self)
		return self.logloss_value
	# Return the value of the concerned parameter: use * to see them all
	def parameter_value(self,parameter_name="*",show=True):
		if (parameter_name=="*"):
			iteration=self.parameter_value("iteration_count")
			r_row=self.parameter_value("rejected_row_count")
			a_row=self.parameter_value("accepted_row_count")
			formatted_text="iteration_count: "+str(iteration)
			formatted_text+="\nrejected_row_count: "+str(r_row)
			formatted_text+="\naccepted_row_count: "+str(a_row)
			if (show):
				print(formatted_text)
			return iteration,r_row,a_row,formatted_text
		elif (parameter_name in ["iteration_count","rejected_row_count","accepted_row_count"]):
			return parameter_value(self,parameter_name=parameter_name)
		else:
			raise TypeError("Please use a correct parameter value: iteration_count|rejected_row_count|accepted_row_count|*")
	# Plot the model
	def plot(self,marker=["o","^"],color=None,projection=None,max_nb_points=None):
		coefficients=self.details().data_columns
		a0=float(coefficients[1][1])
		alpha=[0.9,0.7]
		if (type(projection)==list):
			if (len(projection)>len(self.predictor_columns)):
				self.plot(max_nb_points=max_nb_points,show=show)
			else:
				idx=coefficients[0].index(projection[0])
				a1=coefficients[1][idx]
				if (len(projection)>1):
					idx=coefficients[0].index(projection[1])
					a2=coefficients[1][idx]
				if (len(projection)>2):
					idx=coefficients[0].index(projection[2])
					a3=float(coefficients[1][4])
		else:
			a1=float(coefficients[1][2])
			if (len(coefficients[1]))>=4:
				a2=float(coefficients[1][3])
			if (len(coefficients[1]))>=5:
				a3=float(coefficients[1][4])
			projection=self.predictor_columns
		if ((type(max_nb_points)!=int) or (max_nb_points<0)):
			if (len(projection)==1):
				max_nb_points=40
			elif (len(projection)==2):
				max_nb_points=400
			else:
				max_nb_points=1000
		if (len(projection)==1):
			if ((type(color)!=list) or (len(color)!=3)):
				color=["mediumseagreen","dodgerblue","black"]
			columns=[]
			for i in range(2):
				query="select {},random() from {} where {} is not null and {}={} order by 2 limit {}".format(projection[0],
						self.input_relation,projection[0],self.response_column,i,int(max_nb_points/2))
				self.cursor.execute(query)
				all_points=self.cursor.fetchall()
				columns+=[[float(item[0]) for item in all_points]]
			border=-a0/a1
			plt.figure(figsize=(7,5),facecolor='white')
			all_scatter=[]
			for i in range(2):
				all_scatter+=[plt.scatter(columns[i],[0]*len(columns[i]),alpha=alpha[i],marker=marker[i],color=color[i])]
			plt.plot([border,border],[-1,1],color=color[2])
			plt.gca().grid()
			plt.gca().set_axisbelow(True)
			plt.xlabel(self.predictor_columns[0])
			plt.gca().get_yaxis().set_ticks([])
			plt.legend(all_scatter,[0,1],scatterpoints=1,loc="upper right",ncol=4,title="response",fontsize=8)
			plt.title(self.model_type+': '+self.response_column+'=sign('+str(round(a0,3))+"+("+str(round(a1,3))+")*"+projection[0]+")")
			plt.show()
		elif (len(projection)==2):
			if ((type(color)!=list) or (len(color)!=3)):
				color=["mediumseagreen","dodgerblue","black"]
			columns=[]
			for i in range(2):
				query="select {},{},random() from {} where {} is not null and {} is not null and {}={} order by 3 limit {}".format(
					projection[0],projection[1],self.input_relation,projection[0],projection[1],self.response_column,i,int(max_nb_points/2))
				self.cursor.execute(query)
				all_points=self.cursor.fetchall()
				columns+=[[[float(item[0]) for item in all_points],[float(item[1]) for item in all_points]]]
			plt.figure(figsize=(7,5),facecolor='white')
			all_scatter=[]
			for i in range(2):
				all_scatter+=[plt.scatter(columns[i][0],columns[i][1],alpha=alpha[i],marker=marker[i],color=color[i])]
			min_f=min(columns[0][0]+columns[1][0])
			max_f=max(columns[0][0]+columns[1][0])
			plt.plot([min_f,max_f],[-(a0+a1*min_f)/a2,-(a0+a1*max_f)/a2],color=color[2])
			plt.gca().grid()
			plt.gca().set_axisbelow(True)
			plt.xlabel(projection[0])
			plt.ylabel(projection[1])
			plt.legend(all_scatter,[0,1],scatterpoints=1,loc="upper right",ncol=4,title="response",fontsize=8)
			plt.title(self.model_type+': '+self.response_column+'=sign('+str(round(a0,3))+"+("+str(round(a1,3))+
							")*"+projection[0]+"+("+str(round(a2,3))+")*"+projection[1]+")")
			plt.show()
		elif (len(projection)==3):
			if ((type(color)!=list) or (len(color)!=3)):
				color=["mediumseagreen","dodgerblue","gray"]
			columns=[]
			for i in range(2):
				query="select {},{},{},random() from {} where {} is not null and {} is not null and {} is not null and {}={} order by 4 limit {}".format(
					projection[0],projection[1],projection[2],self.input_relation,projection[0],projection[1],projection[2],self.response_column,i,int(max_nb_points/2))
				self.cursor.execute(query)
				all_points=self.cursor.fetchall()
				columns+=[[[float(item[0]) for item in all_points],[float(item[1]) for item in all_points],[float(item[2]) for item in all_points]]]
			min_f1=float(min(columns[0][0]+columns[1][0]))
			max_f1=float(max(columns[0][0]+columns[1][0]))
			min_f2=float(min(columns[0][1]+columns[1][1]))
			max_f2=float(max(columns[0][1]+columns[1][1]))
			X=np.arange(min_f1,max_f1,(max_f1-min_f1)/20.0)
			Y=np.arange(min_f2,max_f2,(max_f2-min_f2)/20.0)
			X,Y=np.meshgrid(X, Y)
			Z=a0+a1*X+a2*Y
			fig=plt.figure(figsize=(7,5),facecolor='white')
			ax=fig.add_subplot(111,projection='3d')
			ax.plot_surface(X,Y,Z, rstride=1, cstride=1, alpha=0.5,color=color[2])
			all_scatter=[]
			for i in range(2):
				all_scatter+=[ax.scatter(columns[i][0],columns[i][1],columns[i][2],alpha=alpha[i],marker=marker[i],color=color[i])]
			ax.set_xlabel(projection[0])
			ax.set_ylabel(projection[1])
			ax.set_zlabel(projection[2])
			ax.legend(all_scatter,[0,1],scatterpoints=1,loc="lower left",ncol=4,title="response",fontsize=8,bbox_to_anchor=(0.9,1))
			plt.title(self.model_type+': '+self.response_column+'=sign('+str(round(a0,3))+"+("+str(round(a1,3))+
							")*"+projection[0]+"+("+str(round(a2,3))+")*"+projection[1]+
							"+("+str(round(a3,3))+")*"+projection[2]+")")
			plt.show()
		else:
			print("/!\\ Warning: The dimension is too big.")
			print("Please use the 'projection' parameter to see a projection of your figure")
	# ROC
	def roc(self,color=["dodgerblue","#444444"],show=True):
		return roc(self,num_bins=4,color=color,show=show)
##################
#                #
# SVM Regressor  #
#                #
##################
class svm_regressor:
	#
	# Initialization
	#
	def  __init__(self,model_name,input_relation,response_column,predictor_columns,cursor,error_tolerance=0.1,
				C=1.0,epsilon=1e-3,max_iterations=100,load=False):
		if not(load):
			query="select svm_regressor('{}','{}','{}','{}' using parameters error_tolerance={},C={},epsilon={},max_iterations={}"
			query=query.format(model_name,input_relation,response_column,",".join(predictor_columns),
					error_tolerance,C,epsilon,max_iterations)+")"
			cursor.execute(query)
		self.cursor=cursor
		self.model_name=model_name
		self.input_relation=input_relation
		self.response_column=response_column
		self.predictor_columns=predictor_columns
		self.model_type="svm_regressor"
		self.category="regression"
		self.all_importances=None
		self.mse_val=None
		self.rsquared_val=None
	# Object Representation
	def __repr__(self):
		object_repr=self.details().__repr__()
		formatted_text=("model_type='{}'\nmodel_name='{}'\ninput_relation='{}'\nresponse_column='{}'\npredictor_columns='{}'\n"+
							self.parameter_value(show=False)[3])
		if not(isnotebook()):
			formatted_text=formatted_text+"\nParameters:\n"+object_repr
		formatted_text=formatted_text.format(self.model_type,self.model_name,self.input_relation,self.response_column,",".join(self.predictor_columns))
		return formatted_text
	#
	###########
	#         #
	# Methods #
	#         #
	###########
	# 
	# Add the linear_reg prediction to the rvd
	def add_to_rvd(self,rvd,name="svm_regressor_pred"+str(np.random.randint(10000))):
		imputation="predict_svm_regressor("+",".join(self.predictor_columns)+" using parameters model_name='"+self.model_name+"'"+",match_by_pos='True')"
		rvd.add_feature(name,imputation)
		return name
	# All the details of the model: p-value,t-value,coeffs...
	def details(self):
		return details(self.model_name,self.cursor)
	# Features Importance
	def features_importance(self,show=True,with_intercept=False):
		return features_importance(self,show=show,with_intercept=with_intercept)
	# MSE and RSQUARED
	def metrics(self):
		return reg_metrics(self)
	# Return the mse of the model
	def mse(self):
		return mse(self)
	# Return the value of the concerned parameter: use * to see them all
	def parameter_value(self,parameter_name="*",show=True):
		if (parameter_name=="*"):
			iteration=self.parameter_value("iteration_count")
			r_row=self.parameter_value("rejected_row_count")
			a_row=self.parameter_value("accepted_row_count")
			formatted_text="iteration_count: "+str(iteration)
			formatted_text+="\nrejected_row_count: "+str(r_row)
			formatted_text+="\naccepted_row_count: "+str(a_row)
			if (show):
				print(formatted_text)
			return iteration,r_row,a_row,formatted_text
		elif (parameter_name in ["iteration_count","rejected_row_count","accepted_row_count"]):
			return parameter_value(self,parameter_name=parameter_name)
		else:
			print("Please use a correct parameter value: iteration_count|rejected_row_count|accepted_row_count|*")
			return False
	# Plot the model
	def plot(self,color=None,projection=None,max_nb_points=1000,show=True):
		plot_reg(self,color=color,projection=projection,max_nb_points=max_nb_points)
	# Return the rsquared of the model
	def rsquared(self):
		return rsquared(self)






		
