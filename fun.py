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
import numpy as np
import os
import math
import shutil
import time

######################
#    __              #
#   / _|             #
#  | |_ _   _ _ __   #
#  |  _| | | | '_ \  #
#  | | | |_| | | | | #
#  |_|  \__,_|_| |_| #
#                    #
#############################################################                 
#                                                           #
# Functions and Class used to simplify the RVC and RVD code #
#                                                           #
#############################################################
#
#
# Column Matrix
#
##
# column matrix is a matrix seen as a succession of columns instead of a  succession of rows.
# all the columns are stored in the data_columns attribute. This class is mainly used to have
# a beautiful output instead of an unreadable list.
##
class column_matrix:
	# Initialization
	def  __init__(self,data_columns,repeat_first_column=True,first_element="",table_info="",title="",is_finished=True):
		self.data_columns=data_columns
		self.repeat_first_column=repeat_first_column
		self.first_element=first_element
		self.table_info=table_info
		self.title=title
		self.is_finished=is_finished
	# Representation
	def __repr__(self):
		formatted_text=print_table(self.data_columns,repeat_first_column=self.repeat_first_column,first_element=self.first_element,is_finished=self.is_finished)[0:-2]
		if (isnotebook()):
			return "<column_matrix>"
		if (self.title!=""):
			formatted_text=self.title+"\n"+formatted_text
		if (self.table_info!=""):
			formatted_text+="\n"+self.table_info
		return formatted_text
# Return if the user is currently using an ipython notebook
def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False # Terminal running IPython
        else:
            return False # Other type (?)
    except NameError:
        return False # Probably standard Python interpreter
# Print the table on the screen using a columnar representation
def print_table(data_columns,is_finished=True,offset=0,repeat_first_column=False,first_element=""):
	data_columns_rep=[]+data_columns
	if (repeat_first_column):
		del data_columns_rep[0]
		columns_ljust_val=min(len(max([str(item) for item in data_columns[0]],key=len))+4,40)
	else:
		columns_ljust_val=len(str(len(data_columns[0])))+2
	try:
		screen_columns=shutil.get_terminal_size().columns
	except:
		screen_rows, screen_columns = os.popen('stty size', 'r').read().split()
	formatted_text=""
	rjust_val=[]
	for idx in range(0,len(data_columns_rep)):
		rjust_val+=[min(len(max([str(item) for item in data_columns_rep[idx]],key=len))+2,40)]
	total_column_len=len(data_columns_rep[0])
	while (rjust_val!=[]):
		columns_to_print=[data_columns_rep[0]]
		columns_rjust_val=[rjust_val[0]]
		max_screen_size=int(screen_columns)-14-int(rjust_val[0])
		del data_columns_rep[0]
		del rjust_val[0]
		while ((max_screen_size>0) and (rjust_val!=[])):
			columns_to_print+=[data_columns_rep[0]]
			columns_rjust_val+=[rjust_val[0]]
			max_screen_size=max_screen_size-7-int(rjust_val[0])
			del data_columns_rep[0]
			del rjust_val[0]
		if (repeat_first_column):
			columns_to_print=[data_columns[0]]+columns_to_print
		else:
			columns_to_print=[[i-1+offset for i in range(0,total_column_len)]]+columns_to_print
		columns_to_print[0][0]=first_element
		columns_rjust_val=[columns_ljust_val]+columns_rjust_val
		column_count=len(columns_to_print)
		for i in range(0,total_column_len):
			for k in range(0,column_count):
				val=columns_to_print[k][i]
				if len(str(val))>40:
					val=str(val)[0:37]+"..."
				if (k==0):
					formatted_text+=str(val).ljust(columns_rjust_val[k])
				else:
					formatted_text+=str(val).rjust(columns_rjust_val[k])+"  "
			if ((rjust_val!=[])):
				formatted_text+=" \\\\"
			formatted_text+="\n"	
		if (not(is_finished) and (i==total_column_len-1)):
			for k in range(0,column_count):
				if (k==0):
					formatted_text+="...".ljust(columns_rjust_val[k])
				else:
					formatted_text+="...".rjust(columns_rjust_val[k])+"  "
			if (rjust_val!=[]):
				formatted_text+=" \\\\"
			formatted_text+="\n"
	try:	
		if (isnotebook()):
			from IPython.core.display import HTML,display
			if not(repeat_first_column):
				data_columns=[[""]+list(range(0+offset,len(data_columns[0])-1+offset))]+data_columns
			m=len(data_columns)
			n=len(data_columns[0])
			html_table="<table style=\"border-collapse: collapse; border: 2px solid rgb(200,200,200)\">"
			for i in range(n):
				html_table+="<tr style=\"{border: 1px solid rgb(200,200,200);}\">"
				for j in range(m):
					if (j==0):
						html_table+="<td style=\"border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#DDD\"><b>"+str(data_columns[j][i])+"</b></td>"
					elif (i==0):
						html_table+="<td style=\"font-size:1.02em;background-color:#DDD\"><b>"+str(data_columns[j][i])+"</b></td>"
					else:
						html_table+="<td style=\"border: 1px solid rgb(200,200,200);\">"+str(data_columns[j][i])+"</td>"
				html_table+="</tr>"
			if not(is_finished):
				html_table+="<tr>"
				for j in range(m):
					if (j==0):
						html_table+="<td style=\"border-top: 1px solid #DDD;background-color:#DDD\"></td>"
					else:
						html_table+="<td style=\"border: 1px solid rgb(200,200,200);\">...</td>"
				html_table+="</tr>"
			html_table+="</table>"
			display(HTML(html_table))
	except:
		pass
	return formatted_text
# Return the column_matrix of a query
def run_query(query,cursor,limit=1000):
	query0="select * from ("+str(query)+") new_table_vpython limit "+str(limit)
	start_time=time.time()
	cursor.execute(query0)
	elapsed_time=time.time()-start_time
	all_columns=[[column[0]] for column in cursor.description]
	for k in range(limit):
		raw=cursor.fetchone()
		try:
			for i in range(len(raw)):
				all_columns[i]+=[raw[i]]
		except:
			break;
	query1="select count(*) from ("+str(query)+") new_table_vpython"
	cursor.execute(query1)
	count=cursor.fetchone()[0]
	is_finished=(len(all_columns[0])>=count)
	table_info="count="+str(count)+" rows, elapsed_time="+str(elapsed_time)
	return column_matrix(all_columns,repeat_first_column=False,is_finished=is_finished,table_info=table_info)