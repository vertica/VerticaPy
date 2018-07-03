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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import shutil
import time
import matplotlib.colors as colors
from random import shuffle
from vertica_ml_python.rvc import RVC
from vertica_ml_python.fun import print_table
from vertica_ml_python.fun import isnotebook
from vertica_ml_python.fun import run_query
from vertica_ml_python.fun import column_matrix

# Drop Table if it exists
def drop_table(input_relation,cursor,print_info=True):
	cursor.execute("select 1;")
	try:
		query="drop table {};".format(input_relation)
		cursor.execute(query)
		if (print_info):
			print("The table {} was successfully dropped.".format(input_relation))
	except:
		print("/!\\ Warning: The table {} doesn't exist !".format(input_relation))
# Drop View if it exists
def drop_view(view_name,cursor,print_info=True):
	cursor.execute("select 1;")
	try:
		query="drop view {};".format(view_name)
		cursor.execute(query)
		if (print_info):
			print("The view {} was successfully dropped.".format(view_name))
	except:
		print("/!\\ Warning: The view {} doesn't exist !".format(view_name))
# Create a RVD from a csv file (= Vertica CSV parser using Flex Tables)
def read_csv(path,cursor,local=True,input_relation=None,delimiter=',',columns=None,types=None,
			null='',enclosed_by='"',escape='\\',skip=1,temporary=False,skip_all=False,
			split=False,split_name='vpython_split'):
	if (not(isinstance(skip,int)) or (skip<0)):
		raise TypeError("The parameter 'skip' must be a positive integer")
	if (not(isinstance(temporary,bool))):
		raise TypeError("The parameter 'temporary' must be a bool")
	if (not(isinstance(skip_all,bool))):
		raise TypeError("The parameter 'skip_all' must be a bool")
	if (not(isinstance(local,bool))):
		raise TypeError("The parameter 'local' must be a bool")
	if (not(isinstance(split_name,str))):
		raise TypeError("The parameter 'split_name' must be a varchar")
	if (not(isinstance(escape,str))):
		raise TypeError("The parameter 'escape' must be a varchar")
	if (not(isinstance(enclosed_by,str))):
		raise TypeError("The parameter 'enclosed_by' must be a varchar")
	if (not(isinstance(null,str))):
		raise TypeError("The parameter 'null' must be a varchar")
	if (not(isinstance(delimiter,str))):
		raise TypeError("The parameter 'delimiter' must be a varchar")
	if (not(isinstance(path,str))):
		raise TypeError("The parameter 'path' must be a varchar")
	if (local):
		local=" local "
	else:
		local=""
	if (type(input_relation)!=str):
			input_relation=path.split("/")[-1].split(".csv")[0]
	if (temporary):
		temporary="temporary"
	else:
		temporary=""
	schema_input_relation=input_relation.split(".")
	if (len(schema_input_relation)==1):
		schema=None
	else:
		input_relation=schema_input_relation[1]
		schema=schema_input_relation[0]
	query="select column_name from columns where table_name='{}'".format(input_relation)
	if (schema!=None):
		query+=" and table_schema='{}'".format(schema)
	cursor.execute(query)
	query_result=cursor.fetchall()
	if (query_result!=[]):
		print("/!\\ Warning: The table {} already exists !".format(input_relation))
		return
	else:
		if (columns==None):
			flex_name="_vpython"+str(np.random.randint(10000000))+"_flex_"
			query="drop table if exists "+flex_name
			cursor.execute(query)
			query="create flex table if not exists "+flex_name+"()"
			cursor.execute(query)
			query="copy "+flex_name+" from"+local+"'{}' parser fcsvparser(delimiter='{}',"
			query+="enclosed_by='{}',escape='{}') null '{}'"
			query=query.format(path,delimiter,enclosed_by,escape,null)
			cursor.execute(query)
			query="select compute_flextable_keys('"+flex_name+"');"
			cursor.execute(query)
			query="select key_name,data_type_guess from "+flex_name+"_keys"
			cursor.execute(query)
			query_result=cursor.fetchall()
			columns=[]
			for column in query_result:
				columns+=[[item for item in column]]
			print("The parser guess the following columns and types:")
			for column,column_type in columns:
				print(column+": "+column_type)
			print("Illegal characters in the columns names will be erased.")
			if not(skip_all):
				next=False
				while not(next):
					print("Is any type wrong?\nIf one of the types is not correct, it will be considered as Varchar(100).")
					print("0 - There is one type that I want to modify.")
					print("1 - I wish to continue.")
					print("2 - I wish to see the columns and their types again.")
					next=input()
					if (next==0 or next=='0'):
						print("please write ['column_name','column_type'] to modify the type of the corresponding column.")
						try:
							column_name,column_type=eval(input())
							for column in columns:
								if (column[0]==column_name):
									column[1]=column_type
									print("type of "+column_name+" has been successfully changed.")
									break
						except:
							print("Failed to change type. Try again.")
						next=False
					elif (next==2 or next=='2'):
						for column,column_type in columns:
							print(column+": "+column_type)
						next=False
					elif (next!=1 and next!='1'):
						print("Please enter a value between 0 and 2.")
						next=False
			for column in columns:
				try:
					if (column[1]=="Interval"):
						column[1]="Varchar(100)"
						print("/!\\ Warning: Type of {} was changed to Varchar(100) [Interval type is not supported]".format(column[0]))
					elif ("Varchar" not in column[1]):
						query='select (case when "'+column[0]+'"=\''+null+'\' then null else "'+column[0]+'" end)::'+column[1]+' as "'+column[0]+'"'
						query+=" from "+flex_name+" where "+column[0]+" is not null limit 1000"
						cursor.execute(query)
				except:
					print("/!\\ Warning: Type of {} was changed to Varchar(100)".format(column[0]))
					column[1]="Varchar(100)"
			columns=['(case when "'+item[0]+'"=\''+null+'\' then null else "'+item[0]+'" end)::'+item[1]+' as "'+
					item[0].replace('.','').replace('-','').replace('+','').replace('=','').replace('*','')
					+'"' for item in columns]
			if (split):
				columns+=['random() as '+split_name]
			query=("create {} table {} as select ".format(temporary,input_relation)+",".join(columns)+
					" from "+flex_name)
			query=query.format(input_relation)
			cursor.execute(query)
			query="drop table "+flex_name
			cursor.execute(query)
		else:
			if (type(columns)!=list) or (type(types)!=list) or (len(types)!=len(columns)):
				raise TypeError("The parameters 'types' and 'columns' must be two lists having the same size")
			query="create table {}(".format(input_relation)
			try:
				for i in range(len(columns)):
					query+=columns[i]+" "+types[i]+", "
				if (split):
					query+=" "+split_name+" float default random()"+");"
				else:
					query=query[0:-2]
					query+=");"
			except:
				raise TypeError("The parameters 'types' and 'columns' must be two lists containing only varchars")
			cursor.execute(query)
			query="copy {}({}) from {} '{}' delimiter '{}' null '{}' enclosed by '{}' escape as '{}' skip {};".format(
				input_relation,", ".join(columns),local,path,delimiter,null,enclosed_by,escape,skip)
			cursor.execute(query)
		print("The table {} has been successfully created.".format(input_relation))
		return RVD(input_relation,cursor)
#
############################
#   _______      _______   #
#  |  __ \ \    / /  __ \  #
#  | |__) \ \  / /| |  | | #
#  |  _  / \ \/ / | |  | | #
#  | | \ \  \  /  | |__| | #
#  |_|  \_\  \/   |_____/  #
#                          #
#############################
#                           #
# Resilient Vertica Dataset #
#                           #
#############################
#
##
class RVD:
	###################
	#                 #
	# Special Methods #
	#                 #
	###################
	#
	# Initialization
	#
	# RVD has 7 main attributes: input_relation, cursor, dsn, columns, where, offset and limit
	# It has also 7 other attributes to simplify the code and to have easy interaction with
	# sql and matplotlib. 
	def  __init__(self,input_relation,cursor=None,dsn=None,columns=None):
		if ((isinstance(cursor,type(None))) and (isinstance(dsn,type(None)))):
			raise Exception("At least one of the two parameters (dsn or cursor) must receive an input for the RVD creation")
		if ((isinstance(cursor,type(None))) and not(isinstance(dsn,str))):
			raise Exception("If the cursor is not informed, the dsn must be a varchar corresponding to a Vertica DSN")
		elif (isinstance(dsn,str)):
			import pyodbc
			cursor=pyodbc.connect("DSN="+dsn).cursor()
			self.dsn=dsn
		schema_input_relation=input_relation.split(".")
		if (len(schema_input_relation)==1):
			# Name of the concerned table
			self.schema=None
			self.input_relation=input_relation
		else:
			self.input_relation=schema_input_relation[1]
			self.schema=schema_input_relation[0]
		# Cursor to the Vertica Database
		self.cursor=cursor
		# All the columns of the RVD
		if (type(columns)!=list):
			query="select column_name from columns where table_name='{}'".format(self.input_relation)
			if (self.schema!=None):
				query+=" and table_schema='{}'".format(self.schema)
			cursor.execute(query)
			columns=cursor.fetchall()
			columns=[str(item) for sublist in columns for item in sublist]
		if (columns!=[]):
			self.columns=columns
			view=False
		else:
			view=True
		if (view):
			query="select * from views where table_name='{}'".format(self.input_relation)
			if (self.schema!=None):
				query+=" and table_schema='{}'".format(self.schema)
			cursor.execute(query)
			columns=cursor.fetchall()
			if (columns==[]):
				print("/!\\ Warning: No table or views '{}' found.\nNothing was created.".format(self.input_relation))
				del self
				return None
			name="_vpython"+str(np.random.randint(10000000))+"_tt_"
			query="drop table if exists "+name
			cursor.execute(query)
			query="create temporary table "+name+" as select * from "+input_relation+" limit 1000"
			cursor.execute(query)
			query="select column_name from columns where table_name='"+name+"'"
			cursor.execute(query)
			columns=cursor.fetchall()
			self.columns=[str(item) for sublist in columns for item in sublist]
			self.input_relation=name
		for column in self.columns:
			new_rvc=RVC(column,parent=self)
			setattr(self,column,new_rvc)
		# Table Limitation
		self.limit=None 
		# Table Offset
		self.offset=0
		# Rules for the cleaned data
		self.where=[]
		# Display the elapsed time during the query
		self.time_on=False
		# Display or not the sequal queries that are used during the RVD manipulation
		self.query_on=False
		# Use sqlparse to reindent the query
		self.reindent=False
		# Label Location and figure size
		self.legend_loc=(None,None,None)
		if (isnotebook()):
			self.figsize=(9,7)
		else:
			self.figsize=(7,5)
		# Figure color
		rvd_colors=['dodgerblue','seagreen','indianred','gold','tan','pink','darksalmon','lightskyblue','lightgreen',
					'palevioletred','coral']
		all_colors=[item for item in colors.cnames]
		shuffle(all_colors)
		for c in all_colors:
			if c not in rvd_colors:
				rvd_colors+=[c]
		self.colors=rvd_colors
		# RVD history
		self.rvd_history=[]
		if (view):
			self.input_relation=input_relation
			query="drop table if exists "+name
			cursor.execute(query)
	# Get and Set item
	def __getitem__(self,index):
		return getattr(self,index)
	def __setitem__(self,index,val):
		setattr(self,index,val)
	# Object Representation
	def __repr__(self,limit=30,table_info=True):
		if ((self.limit!=None) and (self.limit<limit)):
			is_finished=True
		else:
			is_finished=False
		query="select * from {} limit {}".format(self._table_transf_(),limit)
		self._display_query_(query)
		start_time = time.time()
		self.cursor.execute(query)
		self._display_time_(elapsed_time=time.time()-start_time)
		query_result=self.cursor.fetchall()
		data=[item for item in query_result]
		formatted_text=""
		if (data!=[]):
			data_columns=[[item] for item in self.columns]
			for row in data:
				for idx,val in enumerate(row):
					data_columns[idx]+=[val]
			formatted_text+=print_table(data_columns,is_finished=is_finished,offset=max(self.offset,0))
		else:
			for column in self.columns:
				formatted_text+=column+"  "
			formatted_text+="\n"
		if (table_info):
			formatted_text+="Name: {}, Number of rows: {}, Number of columns: {}".format(
				self.input_relation,self.count(),len(self.columns))
		if isnotebook():
			formatted_text="Name: {}, Number of rows: {}, Number of columns: {}".format(
				self.input_relation,self.count(),len(self.columns))
		return formatted_text
	# Object attr affectation
	def __setattr__(self,attr,val):
		# input_relation
		if (attr=="input_relation"):
			if not(isinstance(val,str)):
				print("/!\\ Warning: attribute 'input_relation' must be a string corresponding to a "
					+"table or view inside your Vertica DB.\nYou are not allowed to manually change"
					+ "this attribute, it can destroy the RVD robustness.\nNothing was changed.")
			else:
				self.__dict__[attr]=val
		# schema
		if (attr=="schema"):
			if not(isinstance(val,(str,type(None)))):
				print("/!\\ Warning: attribute 'schema' must be a string corresponding to a "
					+"schema inside your Vertica DB.\nYou are not allowed to manually change"
					+ "this attribute, it can destroy the RVD robustness.\nNothing was changed.")
			else:
				self.__dict__[attr]=val
		# cursor
		elif (attr=="cursor"):
			try:
				val.execute("select 1;")
				result=int(val.fetchone()[0])
				if (result==1):
					self.__dict__[attr]=val
				else:
					print("/!\\ Warning: attribute 'cursor' must be a cursor to a Vertica DB having "
						+"the fetchone and fetchall methods.\nNothing was changed.")
			except:
				print("/!\\ Warning: attribute 'cursor' must be a cursor to a Vertica DB. Use pyodbc or jaydebeapi for "
					+"respectively ODBC and JDBC connection using Python.\nNothing was changed.")
		# columns
		elif (attr=="columns"):
			error=False
			if not(isinstance(val,list)):
				error=True
			else:
				for item in val:
					if not(isinstance(item,str)):
						error=True
			if (error):
				print("/!\\ Warning: attribute 'columns' must be the list of the different table/view columns."
					+"\nNothing was changed.")
			else:
				self.__dict__[attr]=val
		# offset
		elif (attr=="offset"):
			if (not(isinstance(val,int)) or (val<0)):
				print("/!\\ Warning: attribute '"+attr+"' must be a positive integer.\nNothing was changed.")
			else:
				self.__dict__[attr]=val
		# limit
		elif (attr =="dsn"):
			if not(isinstance(val,str)):
				print("/!\\ Warning: attribute '"+attr+"' must be a varchar corresponding to a Vertica DSN.\nNothing was changed.")
			else:
				self.__dict__[attr]=val
		# limit
		elif (attr =="limit"):
			if (not(isinstance(val,(int,type(None)))) or ((isinstance(val,int)) and (val<0))):
				print("/!\\ Warning: attribute '"+attr+"' must be a positive integer or null (no limit).\nNothing was changed.")
			else:
				self.__dict__[attr]=val
		# where
		elif (attr=="where"):
			error=False
			if not(isinstance(val,list)):
				error=True
			else:
				for item in val:
					if ((type(item)!=tuple) or (len(item)!=2) or (type(item[0])!=str) or (type(item[1])!=int or item[1]<0)):
						error=True
			if (error):
				print("/!\\ Warning: attribute 'where' must be a list of 2-tuple of the form"
					+ "(filter,filter_pos). Changing this attribute can destroy the RVD robustness."
					+ "\nNothing was changed.")
			else:
				self.__dict__[attr]=val
		# time_on/query_on/reindent
		elif (attr in ["time_on","query_on","reindent"]):
			if not(isinstance(val,bool)):
				print("/!\\ Warning: attribute '"+attr+"' must be a bool.\nNothing was changed.")
			else:
				self.__dict__[attr]=val
		# legend_loc
		elif (attr=="legend_loc"):
			if (not(isinstance(val,tuple)) or (len(val)!=3)):
				print("/!\\ Warning: attribute '"+attr+"' must be a tuple of length 3.\nNothing was changed.")
			else:
				self.__dict__[attr]=val
		# rvd_history
		elif (attr=="rvd_history"):
			error=False
			if not(isinstance(val,list)):
				error=True
			else:
				for item in val:
					if (type(item)!=str):
						error=True
			if (error):
				print("/!\\ Warning: attribute 'rvd_history' must be a list of varchar. "
					+ "Changing this attribute manually can destroy the RVD robustness."
					+ "\nNothing was changed.")
			else:
				self.__dict__[attr]=val
		# colors
		elif (attr=="colors"):
			if isinstance(val,str):
				val=[val]
			if not(isinstance(val,list)):
				print("/!\\ Warning: attribute '"+attr+"' must be a list of colors.\nNothing was changed.")
			else:
				all_colors=[item for item in colors.cnames]
				correct_colors=[]
				error=False
				for color in val:
					if color not in all_colors:
						if (color[0]!='#'):
							correct_elem=['0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f','A','B','C','D','E','F']
							for idx in range(1,len(color)):
								if (color[idx] not in correct_elem):
									error=True
					if not(error):
						correct_colors+=[color]
					else:
						error=False
						print("/!\\ Warning: the color '"+color+"' doesn't exist.\nIt was not added to the RVD colors attribute.")
				shuffle(all_colors)
				for c in all_colors:
					if c not in correct_colors:
						correct_colors+=[c]
				self.__dict__[attr]=correct_colors*5
		# other attributes
		else:
			self.__dict__[attr]=val
	#
	########################
	#                      #
	# Semi Special Methods #
	#                      #
	########################
	#
	# These methods are used to simplify the code
	#
	# 
	# Display the query if the attribute query_on is True
	def _display_query_(self,query,title=""):
		if (self.query_on):
			try:
				screen_columns=shutil.get_terminal_size().columns
			except:
				screen_rows, screen_columns = os.popen('stty size', 'r').read().split()
			try:
				import sqlparse
				query_print=sqlparse.format(query,reindent=self.reindent)
			except:
				query_print=query
			if (isnotebook()):
				from IPython.core.display import HTML,display
				display(HTML("<h4 style='color:#444444;text-decoration:underline;'>"+title+"</h4>"))
				query_print=query_print.lower()
				sql_syntax=["select","from","where","insert","update","delete","order",
							"offset","limit","group","not","union","null"]
				sql_syntax2=["by","top","fetch","next","rows","only","distinct","between",
							"in","like","is","null","having","as","join","left","right",
							"full","self","union","any","all","exists","into","over","partition",
							"and","or","else","when","end","xor"]
				sql_analytics_fun=["avg","count","cume_dist","dense_rank","exponential_moving_average",
									"first_value","lag","last_value","lead","max","median","min","ntile",
									"nth_value","percent_rank","percentile_cont","percentile_disc","rank",
									"row_number","stddev","stddev_pop","stddev_samp","sum","var_pop","var_samp",
									"variance","floor","round","corr","decode","coalesce"]
				query_print=query_print.replace('\n',' <br>')
				query_print=query_print.replace('  ',' &emsp; ')
				for item in sql_syntax:
					query_print=query_print.replace(item+" ",' <b> '+item+' </b> ')
				for item in sql_syntax2:
					query_print=query_print.replace(" "+item+" ",' <b> '+item+' </b> ')
					query_print=query_print.replace("("+item+" ",'(<b> '+item+' </b> ')
				for item in sql_analytics_fun:
					query_print=query_print.replace(item+"(",'<b>'+item+'</b>(')
				display(HTML(query_print))
				display(HTML("<div style='border:1px dashed black;width:100%'></div>"))
			else:
				print("$ "+title+" $\n")
				print(query_print)
				print("-"*int(screen_columns)+"\n")
	# Display the elapsed query time if the attribute time_on is True
	def _display_time_(self,elapsed_time):
		if (self.time_on):
			try:
				screen_columns=shutil.get_terminal_size().columns
			except:
				screen_rows, screen_columns = os.popen('stty size', 'r').read().split()
			if (isnotebook()):
				from IPython.core.display import HTML,display
				display(HTML("<div><b>Elapsed Time:</b> "+str(elapsed_time)+"</div>"))
				display(HTML("<div style='border:1px dashed black;width:100%'></div>"))
			else:
				print("Elapsed Time: "+str(elapsed_time))
				print("-"*int(screen_columns)+"\n")
	# Return the Label Loc Initialization
	def _legend_loc_init_(self):
		if ((type(self.legend_loc)!=list) or (len(self.legend_loc)!=3)):
			self.legend_loc=(None,None,None)
		legend_loc=[]
		# bbox_to_anchor
		if (type(self.legend_loc[0])!=tuple):
			if (isnotebook()):
				legend_loc+=[(1.04,0.5)]
			else:
				legend_loc+=[(1,1)]
		else:
			legend_loc+=[self.legend_loc[0]]
		# ncol
		if ((type(self.legend_loc[1])!=int) or (self.legend_loc[1]<0)):
			if (isnotebook()):
				legend_loc+=[3]
			else:
				legend_loc+=[2]
		else:
			legend_loc+=[self.legend_loc[1]]
		# loc
		if (type(self.legend_loc[2])!=str):
			if (isnotebook()):
				legend_loc+=["center left"]
			else:
				legend_loc+=["upper right"]
		else:
			legend_loc+=[self.legend_loc[2]]
		return (legend_loc[0],legend_loc[1],legend_loc[2])
	# Display the columnar matrix using appropriate colors
	def _show_matrix_(self,matrix,columns_x,columns_y,n,m,vmax,vmin,cmap='PRGn',title="",
						colorbar="",x_label="",y_label="",with_numbers=True,mround=3):
		matrix_array=np.ndarray(shape=(n,m),dtype=float)
		for i in range(n):
			for j in range(m):
				try:
					matrix_array[i][j]=matrix[j+1][i+1]
				except:
					matrix_array[i][j]=None
		plt.figure(figsize=(self.figsize[0]+1,self.figsize[1]+1))
		plt.title(title)
		plt.imshow(matrix_array,cmap=cmap,interpolation='nearest',vmax=vmax,vmin=vmin)
		plt.colorbar().set_label(colorbar)
		plt.gca().set_xlabel(x_label)
		plt.gca().set_ylabel(y_label)
		plt.gca().set_yticks([i for i in range(0,n)])
		plt.gca().set_xticks([i for i in range(0,m)])
		plt.yticks(rotation=0)
		plt.xticks(rotation=90)
		plt.subplots_adjust(bottom=max(0.2,len(max([str(item) for item in columns_y],key=len))/90.0))	
		plt.gca().set_xticklabels(columns_y)
		plt.gca().set_yticklabels(columns_x)
		x_positions=np.linspace(start=0,stop=m,num=m,endpoint=False)
		y_positions=np.linspace(start=0,stop=n,num=n,endpoint=False)
		if (with_numbers):
			for y_index,y in enumerate(y_positions):
			    for x_index,x in enumerate(x_positions):
			        label=round(matrix_array[y_index,x_index],mround)
			        plt.gca().text(x,y,label,color='black',ha='center',va='center')
		plt.show()
	# Return the string corresponding to the new table used by all the method: [The most important method]
	def _table_transf_(self,tablesample=None):
		if (tablesample==None):
			tablesample=""
		else:
			tablesample=" tablesample({})".format(tablesample)
		# We save all the imputation grammar in a single list in order to find the max floor
		all_imputations_grammar=[]
		for column in self.columns:
		    all_imputations_grammar+=[[item[0] for item in self[column].transformations]]
		# The max floor is the one of the column having the biggest number of transformations
		max_len=len(max(all_imputations_grammar,key=len))
		# Complete the imputations of the columns having a len < max_len
		for imputations in all_imputations_grammar:
		    diff=max_len-len(imputations)
		    if diff>0:
		        imputations+=["{}"]*diff
		# filtering positions
		where_positions=[item[1] for item in self.where]
		max_where_pos=max(where_positions+[0])
		all_where=[[] for item in range(max_where_pos+1)]
		for i in range(0,len(self.where)):
			all_where[where_positions[i]]+=[self.where[i][0]]
		all_where=[" and ".join(item) for item in all_where]
		for i in range(len(all_where)):
			if (all_where[i]!=''):
				all_where[i]=" where "+all_where[i]
		# first floor
		first_values=[item[0] for item in all_imputations_grammar]
		for i in range(0,len(first_values)):
		    first_values[i]=first_values[i]+" as "+self.columns[i]
		table="select "+", ".join(first_values)+" from "+self.input_relation+tablesample
		# all the other floors
		for i in range(1,max_len):
		    values=[item[i] for item in all_imputations_grammar]
		    for j in range(0,len(values)):
		        values[j]=values[j].replace("{}",self.columns[j])+" as "+self.columns[j]
		    table="select "+", ".join(values)+" from ("+table+") t"+str(i)
		    try:
		    	table+=all_where[i-1]
		    except:
		    	pass
		# add the limit and the offset in the end of the query
		if (type(self.offset)==int) and (self.offset>0):
			table+=" offset "+str(self.offset)
		if (type(self.limit)==int) and (self.limit>=0):
			table+=" limit "+str(self.limit)
		try:
			if (all_where[max_len-1]==""):
				table="("+table+") new_table"
			else:
				table="("+table+") t"+str(max_len)
				table+=all_where[max_len-1]
				table="(select * from "+table+") new_table"
		except:
			table="("+table+") new_table"
		return table
	#
	###########
	#         #
	# Methods #
	#         #
	###########
	# 
	# add a new RVC to the rvd
	def to_pandas(self,limit=30,table_info=True):
		query="select * from {} limit {}".format(self._table_transf_(),limit)
		self._display_query_(query)
		start_time = time.time()
		self.cursor.execute(query)
		self._display_time_(elapsed_time=time.time()-start_time)
		column_names=[column[0] for column in self.cursor.description]
		query_result=self.cursor.fetchall()
		#print("The value of query_result is {}".format(query_result))
		#print("----------------------------------------------------")
		data=[list(item) for item in query_result]
		#print("The value of data is {}".format(data))
		df=pd.DataFrame(data)
		df.columns = column_names
		return df


	def add_feature(self,alias,imputation):
		if not(isinstance(alias,str)):
			raise TypeError("The parameter 'alias' must be a varchar")
		if not(isinstance(imputation,str)):
			raise TypeError("The parameter 'imputation' must be a varchar")
		try:
			name="_vpython"+str(np.random.randint(10000000))+"_"
			query="drop table if exists "+name
			self._display_query_(query,title="Drop the existing generated table")
			start_time = time.time()
			self.cursor.execute(query)
			self._display_time_(elapsed_time=time.time()-start_time)
			query="create temporary table "+name+" as select {} as {} from {} limit 20".format(
				imputation,alias,self.input_relation)
			self._display_query_(query,title="Create a temporary table to test if the new feature is correct")
			start_time = time.time()
			self.cursor.execute(query)
			self._display_time_(elapsed_time=time.time()-start_time)
			query="select data_type from columns where column_name='{}' and table_name='{}'".format(
				alias,name)
			self._display_query_(query,title="Catch the type of the new feature")
			start_time = time.time()
			self.cursor.execute(query)
			self._display_time_(elapsed_time=time.time()-start_time)
			ctype=self.cursor.fetchone()
			query="drop table if exists "+name
			self._display_query_(query,title="Drop the temporary table")
			start_time = time.time()
			self.cursor.execute(query)
			self._display_time_(elapsed_time=time.time()-start_time)
			if not(ctype==[]):
				ctype=ctype[0]
				# RVC category (date|int|float|text)
				if (ctype[0:4]=="date") or (ctype[0:4]=="time") or (ctype[0:8]=="interval"):
					category="date"
				elif ((ctype[0:3]=="int") or (ctype[0:4]=="bool")):
					category="int"
				elif ((ctype[0:7]=="numeric") or (ctype[0:5]=="float")):
					category="float"
				else:
					category="text"
			else:
				ctype="undefined"
				category="undefined"
			new_rvc=RVC(alias,parent=self,first_transformation=(imputation,ctype,category))
			setattr(self,alias,new_rvc)
			self.columns+=[alias]
			print("The new RVC '{}' was added to the RVD.".format(
				alias))
			self.rvd_history+=["{"+time.strftime("%c")+"} "+"[Add Feature]: A new RVC '{}' was added to the RVD.".format(
								alias)]
		except:
			raise Exception("An error occurs during the creation of the new feature")
	# Draw a 2D bar
	def bar(self,columns,method="density",of=None,max_cardinality=[6,6],h=[None,None],color=None,limit_distinct_elements=200,stacked=False):
		if (color==None):
			color=self.colors
		if not(isinstance(stacked,bool)):
			raise TypeError("The parameter 'stacked' must be a bool")
		bbox_to_anchor,ncol,loc=self._legend_loc_init_()
		if (type(columns)==str):
			self[columns].bar(method=method,of=of,max_cardinality=max_cardinality[0],h=h[0],
				color=color[0])
		elif ((type(columns)==list) and (len(columns)==1)):
			return self[columns[0]].bar(method=method,of=of,max_cardinality=max_cardinality[0],h=h[0],
				color=color[0])
		else:
			all_columns=self.pivot_table(columns,method=method,of=of,h=h,max_cardinality=max_cardinality,show=False,
				limit_distinct_elements=limit_distinct_elements).data_columns
			plt.figure(figsize=self.figsize,facecolor='white')
			plt.rcParams['axes.facecolor']='#F5F5F5'
			n=len(all_columns)
			m=len(all_columns[0])
			n_groups=m-1
			index=np.arange(n_groups)
			bar_width=0.5
			for i in range(1,n):
				current_column=all_columns[i][1:m]
				for idx,item in enumerate(current_column):
					try:
						current_column[idx]=float(item)
					except:
						current_column[idx]=0
				current_label=str(all_columns[i][0])
				if (stacked):
					if (i==1):
						last_column=[0 for item in all_columns[i][1:m]]
					else:
						for idx,item in enumerate(all_columns[i-1][1:m]):
							try:
								last_column[idx]+=float(item)
							except:
								last_column[idx]+=0
					plt.barh(index,current_column,bar_width,alpha=0.86,
						color=color[i-1],label=current_label,left=last_column)
				else:
					plt.barh(index+(i-1)*bar_width/(n-1),current_column,bar_width/(n-1),alpha=0.86,
						color=color[i-1],label=current_label)
			if (stacked):
				plt.yticks(index,all_columns[0][1:m])
			else:
				plt.yticks(index+bar_width/2-bar_width/2/(n-1),all_columns[0][1:m])
			plt.subplots_adjust(left=max(0.3,len(max([str(item) for item in all_columns[0][1:m]],key=len))/140.0))
			plt.ylabel(columns[0])
			if (method=="mean"):
				method="avg"
			if (method=="density"):
				plt.xlabel('Density')
				plt.title('Distribution of {} group by {}'.format(columns[0],columns[1]))
			elif ((method in ["avg","min","max","sum"]) and (of!=None)):
				plt.xlabel("{}({})".format(method,of))
				plt.title('{}({}) of {} group by {}'.format(method,of,columns[0],columns[1]))
			else:
				plt.xlabel('Frequency')
				plt.title('Count by {} group by {}'.format(columns[0],columns[1]))
			plt.legend(title=columns[1],loc=loc,ncol=ncol,bbox_to_anchor=bbox_to_anchor)
			plt.gca().set_axisbelow(True)
			plt.gca().xaxis.grid()
			plt.show()
	# case columns size=2: Return the correlation between two elements of the RVD
	# else: Return the correlation matrix 
	def corr(self,columns=[],cmap="PRGn",show=True):
		if not(isinstance(show,bool)):
			raise TypeError("The parameter 'show' must be a bool")
		if not(isinstance(cmap,str)):
			raise TypeError("The parameter 'cmap' must be a varchar")
		if not(isinstance(columns,(list,str))):
			raise TypeError("The parameter 'columns' must be a list of different RVD columns")
		else:
			if (isinstance(columns,str)):
				columns=[columns]
			for item in columns:
				if not(item in self.columns):
					raise Exception("The parameter 'columns' must be a list of different RVD columns")
		if (type(columns)==str) or (len(columns)==1):
			return 1
		elif (len(columns)==2):
			query="select round(corr({},{}),3) from {}".format(columns[0],columns[1],self._table_transf_())
			self._display_query_(query,title="Compute the Correlation between the two variables")
			start_time = time.time()
			self.cursor.execute(query)
			self._display_time_(elapsed_time=time.time()-start_time)
			result=self.cursor.fetchone()[0]
			return result
		elif (len(columns)>=2):
			all_corr=[]
			n=len(columns)
			for i in range(1,n):
				for j in range(0,i):
					all_corr+=["round(corr("+columns[i]+","+columns[j]+"),3)"]
			all_corr=",".join(all_corr)
			query="select {} from {}".format(all_corr,self._table_transf_())
			self._display_query_(query,title="Compute all the Correlations in a single query")
			start_time = time.time()
			self.cursor.execute(query)
			self._display_time_(elapsed_time=time.time()-start_time)
			result=self.cursor.fetchone()
			matrix=[[1 for i in range(0,n+1)] for i in range(0,n+1)]
			matrix[0]=[""]+columns
			for i in range(0,n+1):
				matrix[i][0]=columns[i-1]
			k=0
			for i in range(1,n):
				for j in range(0,i):
					current_corr=result[k]
					k+=1
					if (current_corr==None):
						current_corr=0
					matrix[i+1][j+1]=current_corr
					matrix[j+1][i+1]=current_corr
			if (show):
				with_numbers=True
				matrix_to_draw=matrix
				mround=3
				if (n>8) and (n<=12):
					mround=2
				elif (n>12) and (n<19):
					mround=1
				elif n>=19:
					with_numbers=False
				self._show_matrix_(matrix_to_draw,columns,columns,n,n,vmax=1,vmin=-1,cmap=cmap,
					title='Correlation Matrix of {} RVD'.format(self.input_relation),
					mround=mround,with_numbers=with_numbers)
			return column_matrix(matrix)
		else:
			numerical_bool_columns=[]
			for column in self.columns:
				if (self[column].category() in ["bool","int","float"]):
					numerical_bool_columns+=[column]
			if (len(numerical_bool_columns)==0):
			 	raise Exception("No numerical columns found in the RVD.")
			else:
				return self.corr(columns=numerical_bool_columns,show=show)
	# case columns size=2: Return the correlation between the first element and the log of the second one
	# else: Return the correlation log matrix 
	def corr_log(self,columns=[],cmap="PRGn",epsilon=1e-8,show=True):
		if not(isinstance(show,bool)):
			raise TypeError("The parameter 'show' must be a bool")
		if not(isinstance(cmap,str)):
			raise TypeError("The parameter 'cmap' must be a varchar")
		if not(isinstance(columns,(list,str))):
			raise TypeError("The parameter 'columns' must be a list of different RVD columns")
		else:
			if (isinstance(columns,str)):
				columns=[columns]
			for item in columns:
				if not(item in self.columns):
					raise Exception("The parameter 'columns' must be a list of different RVD columns")
		if (type(columns)==str) or (len(columns)==1):
			query="select round(corr({},log({}+"+str(epsilon)+")),3) from {}".format(columns[0],columns[0],self._table_transf_())
			self._display_query_(query,title="Compute the Correlation between the two variables")
			start_time = time.time()
			self.cursor.execute(query)
			self._display_time_(elapsed_time=time.time()-start_time)
			result=self.cursor.fetchone()[0]
			return result
		elif (len(columns)==2):
			query="select round(corr({},log({}+"+str(epsilon)+")),3) from {}".format(columns[0],columns[1],self._table_transf_())
			self._display_query_(query,title="Compute the Correlation between the two variables")
			start_time = time.time()
			self.cursor.execute(query)
			self._display_time_(elapsed_time=time.time()-start_time)
			result=self.cursor.fetchone()[0]
			return result
		elif (len(columns)>=2):
			all_corr=[]
			n=len(columns)
			for i in range(0,n):
				for j in range(0,n):
					all_corr+=["round(corr("+columns[i]+",log("+columns[j]+"+"+str(epsilon)+")),3)"]
			all_corr=",".join(all_corr)
			query="select {} from {}".format(all_corr,self._table_transf_())
			self._display_query_(query,title="Compute all the Correlations in a single query")
			start_time = time.time()
			self.cursor.execute(query)
			self._display_time_(elapsed_time=time.time()-start_time)
			result=self.cursor.fetchone()
			matrix=[[1 for i in range(0,n+1)] for i in range(0,n+1)]
			matrix[0]=[""]+columns
			for i in range(0,n+1):
				matrix[i][0]="log("+columns[i-1]+")"
			for i in range(0,n):
				for j in range(0,n):
					matrix[i+1][j+1]=result[i+n*j]
			if (show):
				with_numbers=True
				matrix_to_draw=matrix
				mround=3
				if (n>8) and (n<=12):
					mround=2
				elif (n>12) and (n<19):
					mround=1
				elif n>=19:
					with_numbers=False
				self._show_matrix_(matrix_to_draw,columns,["log("+item+")" for item in columns],n,n,vmax=1,vmin=-1,cmap=cmap,
					title='Correlation Log Matrix of {} RVD using epsilon={}'.format(self.input_relation,epsilon),
					mround=mround,with_numbers=with_numbers)
			return column_matrix(matrix)
		else:
			numerical_bool_columns=[]
			for column in self.columns:
				if (self[column].category() in ["bool","int","float"]):
					numerical_bool_columns+=[column]
			if (len(numerical_bool_columns)==0):
			 	raise Exception("No numerical columns found in the RVD.")
			else:
				return self.corr_log(columns=numerical_bool_columns,show=show)
	# Return the number of elements in the RVD
	def count(self):
		query="select count(*) from {}".format(self._table_transf_())
		self.cursor.execute(query)
		total=self.cursor.fetchone()[0]
		return total
	# Return the current table, we are working on
	def current_table(self):
		return self._table_transf_()
	# Generates descriptive statistics that summarize the Table
	# mode can be auto (only numerical values are printed),all (it will describe all the column one per one)
	# ,categorical (only categorical variables are described (cardinality<=6)) or date
	def describe(self,mode="auto",columns=None,include_cardinality=True):
		if not(isinstance(columns,list)):
			columns=self.columns
		else:
			for column in columns:
				if column not in self.columns:
					raise TypeError("RVC '"+column+"' doesn't exist")
		if not(isinstance(include_cardinality,bool)):
			raise TypeError("The parameter 'include_cardinality' must be a bool")
		if (mode=="auto"):
			try:
				if (type(include_cardinality)!=bool):
					include_cardinality_temp=True
				query="select summarize_numcol("
				for column in columns:
					if ((self[column].category()=="float") or (self[column].category()=="int")):
						if (self[column].transformations[-1][1]=="boolean"):
							query+=column+"::int,"
						else:
							query+=column+","
				query=query[:-1]
				query+=") over () from {}".format(self._table_transf_())
				self._display_query_(query,title="Compute the descriptive statistics of all the numerical columns")
				start_time = time.time()
				self.cursor.execute(query)
				self._display_time_(elapsed_time=time.time()-start_time)
				query_result=self.cursor.fetchall()
				data=[item for item in query_result]
				data_columns=[['column'],['count'],['mean'],['std'],['min'],['25%'],['50%'],['75%'],['max']]
				for row in data:
					for idx,val in enumerate(row):
						data_columns[idx]+=[val]
				if (include_cardinality):
					query=[]
					try:
						for column in data_columns[0][1:]:
							query+=["count(distinct {})".format(column)]
						query="select "+",".join(query)+" from "+self._table_transf_()
						self._display_query_(query,title="Compute the cardinalities of all the elements in a single query")
						start_time = time.time()
						self.cursor.execute(query)
						self._display_time_(elapsed_time=time.time()-start_time)
						cardinality=self.cursor.fetchone()
						cardinality=[item for item in cardinality]
					except:
						cardinality=[]
						for column in data_columns[0][1:]:
							query="select count(distinct {}) from {}".format(column,self._table_transf_())
							self._display_query_(query,title="Fail: Compute one per one all the cardinalities")
							start_time = time.time()
							self.cursor.execute(query)
							self._display_time_(elapsed_time=time.time()-start_time)
							cardinality+=[self.cursor.fetchone()[0]]
					data_columns+=[['cardinality']+cardinality]
				return column_matrix(data_columns)
			except:
				return self.describe(mode="all")
		elif (mode=="all"):
			for column in columns:
				print(self[column].describe())
				print("-"*(len(column)+len(self[column].ctype())+15))
		elif (mode=="categorical"):
			for column in columns:
				if ((self[column].cardinality() <= 6) or (self[column].category()=="text")):
					print(self[column].describe())
					print("-"*(len(column)+len(self[column].ctype())+15))
		elif (mode=="date"):
			for column in columns:
				if ((self[column].category()=="date")):
					print(self[column].describe())
					print("-"*(len(column)+len(self[column].ctype())+15))
		else:
			raise TypeError("The parameter 'mode' must be in auto|all|categorical|date")
	# Drop the RVD columns
	def drop_columns(self,columns=[]):
		if not(isinstance(columns,list)):
			if not(isinstance(columns,str)):
				raise TypeError("The parameter 'columns' must be a list of different RVD columns")
			else:
				columns=[columns]
		for column in columns:
			if (column in self.columns):
				self[column].drop_column()
			else:
				print("/!\\ Warning: Column '{}' is not in the RVD.".format(column))
	# Restart the cursor using the Vertica DSN (pyodbc must be installed)
	def dsn_restart(self):
		import pyodbc
		self.cursor=pyodbc.connect("DSN="+self.dsn).cursor()
	# Return the RVD's columns types
	def dtypes(self):
		ljust_val=len(max([str(item) for item in self.columns],key=len))+2
		ctypes=[]
		all_types=[]
		for column in self.columns:
			all_types+=[self[column].ctype()]
		formatted_text=print_table([[""]+self.columns,["type"]+all_types],repeat_first_column=True,first_element="")[0:-2]
		if not(isnotebook()):
			print(formatted_text)
		print("Name: {},Number of rows: {},Number of columns: {}".format(self.input_relation,self.count(),len(self.columns)))
	# Filter the values of the RVD (adding a where clause to the RVD) following the conditions
	def filter(self,conditions):
		count=self.count()
		if (type(conditions)!=str):
			if (isinstance(conditions,list)):
				for item in conditions:
					self.filter(item)
				return True
			else:
				print("/!\\ Warning: 'conditions' must be a varchar (found {}).\nNothing was filtered.".format(type(item)))
				return False
		max_pos=0
		for column in self.columns:
			if (column in conditions):
				max_pos=max(max_pos,len(self[column].transformations)-1)
		self.where+=[(conditions,max_pos)]
		try:
			count-=self.count()
		except:
			del self.where[-1]
			print("/!\\ Warning: The condition '{}' is incorrect.\nNothing was filtered.".format(conditions))
			return False
		if (count>1):
			print("{} elements were filtered".format(count))
			self.rvd_history+=["{"+time.strftime("%c")+"} "+"[Filter]: {} elements were filtered using the filter '{}'".format(count,conditions)]
		elif (count==1):
			print("{} element was filtered".format(count))
			self.rvd_history+=["{"+time.strftime("%c")+"} "+"[Filter]: {} element was filtered using the filter '{}'".format(count,conditions)]
		else:
			del self.where[-1]
			print("Nothing was filtered.")
	# Fully Stacked Bar
	def fully_stacked_bar(self,columns,max_cardinality=[6,6],h=[None,None],color=None,limit_distinct_elements=200):
		if (color==None):
			color=self.colors
		bbox_to_anchor,ncol,loc=self._legend_loc_init_()
		if (type(columns)==str) or ((type(columns)==list) and (len(columns)==1)):
			print("/!\\ Warning: Fully Stacked Bar is only available with two variables.")
		else:
			all_columns=self.pivot_table(columns,method="density",h=h,max_cardinality=max_cardinality,show=False,
				limit_distinct_elements=limit_distinct_elements).data_columns
			plt.figure(figsize=self.figsize,facecolor='white')
			plt.rcParams['axes.facecolor']='#F5F5F5'
			n=len(all_columns)
			m=len(all_columns[0])
			n_groups=m-1
			index=np.arange(n_groups)
			bar_width=0.5
			total=[0 for item in range(1,m)]
			for i in range(1,n):
				for j in range(1,m):
					if not(type(all_columns[i][j]) in [str]):
						total[j-1]+=float(all_columns[i][j])
			for i in range(1,n):
				for j in range(1,m):
					if not(type(all_columns[i][j]) in [str]):
						all_columns[i][j]=float(all_columns[i][j])/total[j-1]
			for i in range(1,n):
				current_column=all_columns[i][1:m]
				for idx,item in enumerate(current_column):
					try:
						current_column[idx]=float(item)
					except:
						current_column[idx]=0
				current_label=str(all_columns[i][0])
				if (i==1):
					last_column=[0 for item in all_columns[i][1:m]]
				else:
					for idx,item in enumerate(all_columns[i-1][1:m]):
						try:
							last_column[idx]+=float(item)
						except:
							last_column[idx]+=0
				plt.barh(index,current_column,bar_width,alpha=0.86,
					color=color[i-1],label=current_label,left=last_column)
			plt.yticks(index,all_columns[0][1:m])
			plt.subplots_adjust(left=max(0.3,len(max([str(item) for item in all_columns[0][1:m]],key=len))/140.0))
			plt.ylabel(columns[0])
			plt.xlabel('Density per category')
			plt.title('Distribution per category of {} group by {}'.format(columns[0],columns[1]))
			plt.legend(title=columns[1],loc=loc,ncol=ncol,bbox_to_anchor=bbox_to_anchor)
			plt.gca().set_axisbelow(True)
			plt.gca().xaxis.grid()
			plt.show()
	# Group by the elements
	def group_by(self,columns,aggregations,order_by=None,limit=1000):
		if (not(isinstance(limit,int)) or (limit<1)):
			raise TypeError("The parameter 'limit' must be a strictly positive integer")
		if (type(columns)==str):
			columns=[columns]
		elif (type(columns)!=list):
			raise TypeError("The parameter 'columns' must be a list of varchar")
		if (type(aggregations)==str):
			aggregations=[aggregations]
		elif (type(aggregations)!=list):
			raise TypeError("The parameter 'aggregations' must be a list of varchar")
		if (type(order_by)==str):
			order_by=" order by "+order_by
		elif (type(order_by)==list):
			order_by=" order by "+", ".join(order_by)
		else:
			order_by=" order by "+", ".join(columns)
		aggregations_alias=[]
		for item in aggregations:
			if "(*)" in item:
				aggregations_alias+=[item.replace('(','').replace(')','').replace('*','')]
			else:
				aggregations_alias+=[item.replace('(','_').replace(')','').replace('*','').replace('/','_').replace('+','_').replace('-','_')]
		query=("select "+", ".join(columns+[aggregations[i]+" as "+aggregations_alias[i] for i in range(len(aggregations))])+" from "+
				self._table_transf_()+" group by "+", ".join(columns)+order_by)
		return run_query(query,self.cursor,limit=limit)
	# Print the first n rows of the rvd
	def head(self,n=5):
		print(self.__repr__(limit=n))
	# Hexbin
	def hexbin(self,columns,method="count",of=None,cmap='Blues',gridsize=10,color="white"):
		if (color==None):
			color=self.colors[0]
		if (not(isinstance(columns,list))):
			raise TypeError("The parameter 'columns' must be a list of different RVD columns")
		if (len(columns)!=2):
			raise TypeError("The parameter 'columns' must be exactly of size 2 for drawing the hexbin")
		if (method=="mean"):
			method="avg"
		if ((method in ["avg","min","max","sum"]) and (type(of)==str) and (of in self.columns)):
			aggregate="{}({})".format(method,of)
			of=[of]
			others_aggregate=method
			if (method=="avg"):
				reduce_C_function=np.mean
			elif (method=="min"):
				reduce_C_function=min
			elif (method=="max"):
				reduce_C_function=max
			elif (method=="sum"):
				reduce_C_function=sum
		else:
			aggregate="count(*)"
			of=[]
			reduce_C_function=sum
		count=self.count()
		if (method=="density"):
			over="/"+str(float(count))
		else:
			over=""
		query="select {},{},{}{} from {} group by {},{}".format(
				columns[0],columns[1],aggregate,over,self._table_transf_(),columns[0],columns[1])
		self._display_query_(query,title="Group all the elements for the Hexbin Plot")
		start_time = time.time()
		self.cursor.execute(query)
		self._display_time_(elapsed_time=time.time()-start_time)
		query_result=self.cursor.fetchall()
		column1=[]
		column2=[]
		column3=[]
		for item in query_result:
			if ((item[0]!=None) and (item[1]!=None) and (item[2]!=None)):
				if (reduce_C_function in [min,max,np.mean]):
					column1+=[float(item[0])]*2
					column2+=[float(item[1])]*2
					column3+=[float(item[2])]*2
				else:
					column1+=[float(item[0])]*2
					column2+=[float(item[1])]*2
					column3+=[float(item[2])/2]*2
		plt.figure(figsize=self.figsize,facecolor='white')
		plt.rcParams['axes.facecolor']='white'
		plt.title('Hexbin of {} vs {}'.format(columns[0],columns[1]))
		plt.ylabel(columns[1])
		plt.xlabel(columns[0])
		plt.hexbin(column1,column2,C=column3,reduce_C_function=reduce_C_function,gridsize=gridsize,color=color,cmap=cmap,mincnt=1)
		if (method=="density"):
			plt.colorbar().set_label(method)
		else:
			plt.colorbar().set_label(aggregate)
		plt.show()
	# 2D hist
	def hist(self,columns,method="density",of=None,max_cardinality=[6,6],h=[None,None],color=None,limit_distinct_elements=200,stacked=False):
		if (color==None):
			color=self.colors
		if not(isinstance(stacked,bool)):
			raise TypeError("The parameter 'stacked' must be a bool")
		bbox_to_anchor,ncol,loc=self._legend_loc_init_()
		if (type(columns)==str):
			self[columns].hist(method=method,of=of,max_cardinality=max_cardinality[0],h=h[0],
				color=color[0])
		elif ((type(columns)==list) and (len(columns)==1)):
			return self[columns[0]].hist(method=method,of=of,max_cardinality=max_cardinality[0],h=h[0],
				color=color[0])
		else:
			all_columns=self.pivot_table(columns,method=method,of=of,h=h,max_cardinality=max_cardinality,show=False,
				limit_distinct_elements=limit_distinct_elements).data_columns
			plt.figure(figsize=self.figsize,facecolor='white')
			plt.rcParams['axes.facecolor']='#F5F5F5'
			n=len(all_columns)
			m=len(all_columns[0])
			n_groups=m-1
			index=np.arange(n_groups)
			bar_width=0.5
			for i in range(1,n):
				current_column=all_columns[i][1:m]
				for idx,item in enumerate(current_column):
					try:
						current_column[idx]=float(item)
					except:
						current_column[idx]=0
				current_label=str(all_columns[i][0])
				if (stacked):
					if (i==1):
						last_column=[0 for item in all_columns[i][1:m]]
					else:
						for idx,item in enumerate(all_columns[i-1][1:m]):
							try:
								last_column[idx]+=float(item)
							except:
								last_column[idx]+=0
					plt.bar(index,current_column,bar_width,alpha=0.86,
						color=color[i-1],label=current_label,bottom=last_column)
				else:
					plt.bar(index+(i-1)*bar_width/(n-1),current_column,bar_width/(n-1),alpha=0.86,
						color=color[i-1],label=current_label)
			if (stacked):
				plt.xticks(index,all_columns[0][1:m],rotation=90)
			else:
				plt.xticks(index+bar_width/2-bar_width/2/(n-1),all_columns[0][1:m],rotation=90)
			plt.subplots_adjust(bottom=max(0.3,len(max([str(item) for item in all_columns[0][1:m]],key=len))/140.0))
			plt.xlabel(columns[0])
			if (method=="mean"):
				method="avg"
			if (method=="density"):
				plt.ylabel('Density')
				plt.title('Distribution of {} group by {}'.format(columns[0],columns[1]))
			elif ((method in ["avg","min","max","sum"]) and (of!=None)):
				plt.ylabel("{}({})".format(method,of))
				plt.title('{}({}) of {} group by {}'.format(method,of,columns[0],columns[1]))
			else:
				plt.ylabel('Frequency')
				plt.title('Count by {} group by {}'.format(columns[0],columns[1]))
			plt.legend(title=columns[1],loc=loc,ncol=ncol,bbox_to_anchor=bbox_to_anchor)
			plt.gca().set_axisbelow(True)
			plt.gca().yaxis.grid()
			plt.show()
	# Resume all the modifications made on the RVD
	def history(self):
		if (type(self.rvd_history)!=list or len(self.rvd_history)==0):
			self.rvd_history=[]
			print("The RVD was never modified.")
		elif (len(self.rvd_history)==1):
			print("The RVD was modified with only one action: ")
			print(" * "+self.rvd_history[0])
		else:
			print("The RVD was modified many times: ")
			for modif in self.rvd_history:
				print(" * "+modif)
	# Resume the number of null elements
	def missing(self):
		count=self.count()
		query=[]
		for column in self.columns:
			query+=["(select '{}',count(*) as count from {} where {} is null)".format(
						column,self._table_transf_(),column)]
		query=" union ".join(query)+" order by count desc"
		self._display_query_(query,title="Compute all the missing elements for each feature")
		start_time = time.time()
		self.cursor.execute(query)
		self._display_time_(elapsed_time=time.time()-start_time)
		query_result=self.cursor.fetchall()
		missing_values_count=[item[1] for item in query_result]
		missing_array=[[""]+[item[0] for item in query_result],["total"]+missing_values_count,
						["percent"]+[round(item/float(count),3) for item in missing_values_count]]
		return column_matrix(missing_array)
	# Multiple Histograms
	def multiple_hist(self,columns,method="density",of=None,h=None,color=None):
		if (color==None):
			color=self.colors
		if (not(isinstance(columns,list))):
			raise TypeError("The parameter 'columns' must be a list of different RVD columns")
		if (len(columns)>5):
			raise Exception("The number of column must be <= 5 to use 'multiple_hist' method")
		else:
			bbox_to_anchor,ncol,loc=self._legend_loc_init_()
			plt.figure(figsize=self.figsize,facecolor='white')
			plt.rcParams['axes.facecolor']='#F5F5F5'
			alpha=1
			all_columns=[]
			all_h=[]
			if (type(h) not in [int,float]):
				for idx,column in enumerate(columns):
					is_numeric=(self[column].category()=="float") or (self[column].category()=="int")
					if (is_numeric):
						all_h+=[self[column]._best_hist_interval_()]
				h=min(all_h)
			for idx,column in enumerate(columns):
				is_numeric=(self[column].category()=="float") or (self[column].category()=="int")
				if (is_numeric):
					[x,y,z,h,is_categorical]=self[column]._hist_(method=method,of=of,max_cardinality=1,h=h)
					h=h/0.94
					plt.bar(x,y,h,color=color[idx],alpha=alpha,label=column)
					alpha-=0.2
					all_columns+=[columns[idx]]
				else:
					print("/!\\ Warning: {} is not numerical. Its histogram will not be draw.")
			plt.xlabel(", ".join(all_columns))
			plt.gca().set_axisbelow(True)
			plt.gca().yaxis.grid()
			if (method=="density"):
				plt.ylabel('Density')
			elif ((method in ["avg","min","max","sum"]) and (of!=None)):
				plt.ylabel(method+"("+of+")")
			else:
				plt.ylabel('Frequency')
			plt.title("Multiple Histograms")
			plt.legend(title="columns",loc=loc,ncol=1,bbox_to_anchor=bbox_to_anchor)
			plt.show()
	# Normalize the column 
	def normalize(self,method="zscore",with_int=False):
		if not(isinstance(with_int,bool)):
			raise TypeError("The parameter 'with_int' must be a bool")
		for column in self.columns:
			if (self[column].category()=="float") or ((self[column].category()=="int") and with_int):
				self[column].normalize(method=method)
	# Return the Pivot Table of the RVD on the corresponding column and raw [One of the most important function for Data Exploration]
	def pivot_table(self,columns,method="count",of=None,h=[None,None],max_cardinality=[20,20],show=True,
			cmap='Blues',limit_distinct_elements=1000,with_numbers=True):
		if (not(isinstance(cmap,str))):
			raise TypeError("The parameter 'cmap' must be a varchar")
		if (not(isinstance(limit_distinct_elements,int)) or (limit_distinct_elements<1)):
			raise TypeError("The parameter 'limit_distinct_elements' must be a list of two strictly positive numerical integers or null")
		if not(isinstance(with_numbers,bool)):
			raise TypeError("The parameter 'with_numbers' must be a bool")
		if not(isinstance(show,bool)):
			raise TypeError("The parameter 'show' must be a bool")
		if not(isinstance(h,list)):
			raise TypeError("The parameter 'h' must be a list of two strictly positive numerical numbers or null")
		if not(isinstance(max_cardinality,list)):
			raise TypeError("The parameter 'max_cardinality' must be a list of two strictly positive numerical integers or null")
		# aggregation used for the bins height
		if (method=="mean"):
			method="avg"
		if ((method in ["avg","min","max","sum"]) and (type(of)==str)):
			if (of in self.columns):
				aggregate="{}({})".format(method,of)
				of=[of]
				others_aggregate=method
			else:
				raise Exception("None RVC named '"+of+"' found")
		elif (method in ["density","count"]):
			aggregate="count(*)"
			others_aggregate="sum"
			of=[]
		else:
			raise TypeError("The parameter 'method' must be in avg|mean|min|max|sum")
		for i in range(2):
			if (not(isinstance(max_cardinality[i],int)) or (max_cardinality[i]<1)):
				raise TypeError("The parameter 'max_cardinality' must be a list of two strictly positive numerical integers or null")
		for i in range(2):
			if (not(isinstance(h[i],type(None)))):
				if (not(isinstance(h[i],(int,float))) or (h[i]<=0)):
					raise TypeError("The parameter 'h' must be a list of two strictly positive numbers or null")
		if ((type(columns)!=list) or (len(columns)!=2)):
			raise TypeError("The parameter 'columns' must be a list of size 2 in order to plot the Pivot Table")
		all_columns=[]
		is_column_date=[False,False]
		timestampadd=["",""]
		for idx,column in enumerate(columns):
			is_numeric=(self[column].category()=="float") or (self[column].category()=="int")
			is_date=(self[column].category()=="date")
			if (is_numeric):
				if (h[idx]==None):
					interval=self[column]._best_hist_interval_()
					interval=round(interval,2)
				else:
					interval=h[idx]
				if (self[column].category()=="int"):
					floor_end="-1"
					interval=int(max(math.floor(interval),1))
				else:
					floor_end=""
				if (interval>1) or (self[column].category()=="float"):
					all_columns+=["'[' || floor({}/{})*{} ||';'|| (floor({}/{})*{}+{}{}) || ']'".format(
						column,interval,interval,column,interval,interval,interval,floor_end)]
				else:
					all_columns+=["floor({}) || ''".format(column)]
			elif (is_date):
				if (h[idx]==None):
					interval=self[column]._best_hist_interval_()
				else:
					interval=max(math.floor(h[idx]),1)
				min_date=self[column].min()
				all_columns+=["floor(datediff('second','"+str(min_date)+"',"+column+")/"+str(interval)+")*"+str(interval)]
				is_column_date[idx]=True
				timestampadd[idx]="timestampadd('second',"+columns[idx]+"::int,'"+str(min_date)+"'::timestamp)"
			else:
				all_columns+=[column]
		if (type(of)==str or (type(of)==list and (len(of)>0) and type(of[0])==str)):
			if (type(of)==list):
				of=of[0]
			subtable=("(select "+all_columns[0]+" as "+columns[0]+", "+all_columns[1]+" as "+columns[1]+", "+of+" as "+of+
						" from "+self._table_transf_()+") pivot_table")
			if (is_column_date[0] and not(is_column_date[1])):
				subtable=("(select "+timestampadd[0]+" as "+columns[0]+", "+columns[1]+", "+of+" from "+subtable+") pivot_table_date")
			elif (is_column_date[1] and not(is_column_date[0])):
				subtable=("(select "+columns[0]+", "+timestampadd[1]+" as "+columns[1]+", "+of+" from "+subtable+") pivot_table_date")
			elif (is_column_date[1] and is_column_date[0]):
				subtable=("(select "+timestampadd[0]+" as "+columns[0]+", "+timestampadd[1]+" as "+columns[1]+", "+of+" from "+subtable+") pivot_table_date")
		else:
			subtable=("(select "+all_columns[0]+" as "+columns[0]+", "+all_columns[1]+" as "+columns[1]+
						" from "+self._table_transf_()+") pivot_table")
			if (is_column_date[0] and not(is_column_date[1])):
				subtable=("(select "+timestampadd[0]+" as "+columns[0]+", "+columns[1]+" from "+subtable+") pivot_table_date")
			elif (is_column_date[1] and not(is_column_date[0])):
				subtable=("(select "+columns[0]+", "+timestampadd[1]+" as "+columns[1]+" from "+subtable+") pivot_table_date")
			elif (is_column_date[1] and is_column_date[0]):
				subtable=("(select "+timestampadd[0]+" as "+columns[0]+", "+timestampadd[1]+" as "+columns[1]+" from "+subtable+") pivot_table_date")
		if (len(columns)==1):
			return self[columns[0]].describe(method=method,of=of)
		else:
			is_finished=limit_distinct_elements
			limit_distinct_elements=" limit "+str(limit_distinct_elements)
			if (method=="density"):
				over="/"+str(self.count())
			else:
				over=""
			query="select {},{},{}{} from {} where {} is not null and {} is not null group by {},{} order by {},{} asc"
			query=query.format(columns[0],columns[1],aggregate,over,subtable,columns[0],columns[1],
					columns[0],columns[1],columns[0],columns[1])+limit_distinct_elements
			self._display_query_(query,title="Group the features to compute the pivot table")
			start_time = time.time()
			self.cursor.execute(query)
			self._display_time_(elapsed_time=time.time()-start_time)
			query_result=self.cursor.fetchall()
			# Column0 sorted categories
			all_column0_categories=list(set([str(item[0]) for item in query_result]))
			all_column0_categories.sort()
			try:
				try:
					order=[]
					for item in all_column0_categories:
						order+=[float(item.split(";")[0].split('[')[1])]
				except:
					order=[float(item) for item in all_column0_categories]
				all_column0_categories = [x for _,x in sorted(zip(order,all_column0_categories))]
			except:
				pass
			# Column1 sorted categories
			all_column1_categories=list(set([str(item[1]) for item in query_result])) 
			all_column1_categories.sort()
			try:
				try:
					order=[]
					for item in all_column1_categories:
						order+=[float(item.split(";")[0].split('[')[1])]
				except:
					order=[float(item) for item in all_column1_categories]
				all_column1_categories = [x for _,x in sorted(zip(order,all_column1_categories))]
			except:
				pass
			all_columns=[['' for item in all_column0_categories] for item in all_column1_categories]
			is_finished=(is_finished>=len(all_column0_categories)*len(all_column1_categories))
			for item in query_result:
				j=all_column0_categories.index(str(item[0]))
				i=all_column1_categories.index(str(item[1]))
				all_columns[i][j]=item[2]
			all_columns=[[all_column1_categories[i]]+all_columns[i] for i in range(0,len(all_columns))]
			all_columns=[[columns[0]+"/"+columns[1]]+all_column0_categories]+all_columns
			if (show):
				all_count=[item[2] for item in query_result]
				self._show_matrix_(all_columns,all_column0_categories,all_column1_categories,len(all_column0_categories),
					len(all_column1_categories),vmax=max(all_count),vmin=min(all_count),
					cmap=cmap,title="Pivot Table of "+columns[0]+" vs "+columns[1],
					colorbar=aggregate,x_label=columns[1],y_label=columns[0],with_numbers=with_numbers)
			return column_matrix(all_columns,first_element=columns[0]+"/"+columns[1])
	# Save the RVD by creating a view/a temporary table or a table in order to make the computations
	# faster or simply to apply ML algorithm on it
	def save(self,name,columns=None,mode="view",affect=True):
		if not(isinstance(name,str)):
			raise TypeError("The parameter 'name' must be a varchar")
		if (mode not in ["view","temporary table","table"]):
			raise TypeError("The parameter mode must be in view|temporary table|table\nNothing was saved.")
		if (type(columns)!=list):
			columns="*"
		else:
			for column in columns:
				if not(column in self.columns):
					raise Exception("The RVC '{}' doesn't exist".format(column))
			columns=",".join(columns)
		query="create {} {} as select {} from {}".format(mode,name,columns,self._table_transf_())
		self._display_query_(query,title="Create a new "+mode+" to save the RVD")
		start_time = time.time()
		self.cursor.execute(query)
		self._display_time_(elapsed_time=time.time()-start_time)
		self.rvd_history+=["{"+time.strftime("%c")+"} "+"[Save]: The RVD was saved into a {} named '{}'.".format(mode,name)]
		if (affect):
			query_on=self.query_on
			self.query_on=False
			history=self.rvd_history
			time_on=self.time_on
			self.__init__(name,self.cursor)
			self.rvd_history=history
			self.query_on=query_on
			self.time_on=time_on
			print("The RVD was successfully saved.")
	# Draw the scatter plot between 2/3 columns
	def scatter(self,columns,max_cardinality=3,cat_priority=None,with_others=True,color=None,marker=["^","o","+","*","h","x","D","1"]*10,max_nb_points=1000):
		if (color==None):
			color=self.colors
		try:
			return self.scatter2D(columns,max_cardinality,cat_priority,with_others,color,marker,max_nb_points)
		except:
			try:
				return self.scatter3D(columns,max_cardinality,cat_priority,with_others,color,marker,max_nb_points)
			except:
				raise Exception("An error occured during the execution of the 'scatter' method.\nPlease use the"
								+" methods 'scatter2D' or 'scatter3D' for more details.")
	# Draw the scatter plot between 2 columns
	def scatter2D(self,columns,max_cardinality=3,cat_priority=None,with_others=True,color=None,marker=["^","o","+","*","h","x","D","1"]*10,max_nb_points=1000):
		if (color==None):
			color=self.colors
		if (not(isinstance(max_cardinality,int)) or (max_cardinality<1)):
			raise TypeError("The parameter 'max_cardinality' must be a strictly positive integer")
		if (not(isinstance(with_others,bool))):
			raise TypeError("The parameter 'with_others' must be a bool")
		if (not(isinstance(max_nb_points,int)) or (max_nb_points<1)):
			raise TypeError("The parameter 'max_nb_points' must be a strictly positive integer")
		if (not(isinstance(cat_priority,(list,type(None))))):
			raise TypeError("The parameter 'cat_priority' must be a list of categories or null")
		bbox_to_anchor,ncol,loc=self._legend_loc_init_()
		if (type(columns)!=list):
			raise TypeError("The parameter 'columns' must be a list of columns")
		if ((len(columns)<2) or (len(columns)>3)):
			raise Exception("2D Scatter plot can only be done with at least two columns and maximum with three columns")
		else:
			for column in columns:
				if (column not in self.columns):
					raise Exception("The RVC '{}' doesn't exist".format(column))
			if (self[columns[0]].category() not in ["int","float"]) or (self[columns[1]].category() not in ["int","float"]):
				raise TypeError("The two first value of 'columns' must be numerical")
			if (len(columns)==2):
				tablesample=max_nb_points/self.count()
				query="select {},{} from {} where {} is not null and {} is not null limit {}".format(columns[0],columns[1],
					self._table_transf_(tablesample),columns[0],columns[1],max_nb_points)
				self._display_query_(query, title="Select random points for the scatter plot")
				start_time = time.time()
				self.cursor.execute(query)
				self._display_time_(elapsed_time=time.time()-start_time)
				query_result=self.cursor.fetchall()
				column1=[item[0] for item in query_result]
				column2=[item[1] for item in query_result]
				plt.figure(figsize=self.figsize)
				plt.gca().grid()
				plt.gca().set_axisbelow(True)
				plt.title('Scatter Plot of {} vs {}'.format(columns[0],columns[1]))
				plt.ylabel(columns[1])
				plt.xlabel(columns[0])
				plt.scatter(column1,column2,color=color[0],s=14)
				plt.show()
			else:
				column_groupby=columns[2]
				count=self.count()
				if (type(cat_priority)==list):
					query_result=cat_priority
				else:
					query="select {} from {} where {} is not null group by {} order by count(*) desc limit {}".format(
							column_groupby,self._table_transf_(),column_groupby,column_groupby,max_cardinality)
					self._display_query_(query,title="Select all the category of the column "+column_groupby)
					start_time = time.time()
					self.cursor.execute(query)
					self._display_time_(elapsed_time=time.time()-start_time)
					query_result=self.cursor.fetchall()
					query_result=[item for sublist in query_result for item in sublist]
				all_columns=[query_result]
				all_scatter=[]
				all_categories=query_result
				fig=plt.figure(figsize=self.figsize,facecolor="white")
				ax=plt
				others=[]
				groupby_cardinality=self[column_groupby].cardinality()
				count=self.count()
				if (count>10000):
					tablesample=10
				else:
					tablesample=90
				for idx,category in enumerate(all_categories):
					if ((max_cardinality<groupby_cardinality) or ((type(cat_priority)==list) and len(cat_priority)<groupby_cardinality)):
							others+=["{}!='{}'".format(column_groupby,category)]
					query="select {},{} from {} where {}='{}' and {} is not null and {} is not null limit {}"
					query=query.format(columns[0],columns[1],self._table_transf_(tablesample),
						columns[2],category,columns[0],columns[1],int(max_nb_points/len(all_categories))) 
					self._display_query_(query,title="Select random points for the scatter plot (category='"+str(category)+"')")
					start_time = time.time()
					self.cursor.execute(query)
					self._display_time_(elapsed_time=time.time()-start_time)
					query_result=self.cursor.fetchall()
					column1=[float(item[0]) for item in query_result]
					column2=[float(item[1]) for item in query_result]
					all_columns+=[[column1,column2]]
					all_scatter+=[ax.scatter(column1,column2,alpha=0.8,marker=marker[idx],color=color[idx])]
				if (len(others)>0 and with_others):
					all_categories+=["others"]
					query=("select {},{} from {} where {} and {} is not null and {} is not null limit {}")
					query=query.format(columns[0],columns[1],self._table_transf_(tablesample),
						" and ".join(others),columns[0],columns[1],int(max_nb_points/len(all_categories)))
					self._display_query_(query,title="Select random points for the scatter plot (category='others')")
					start_time=time.time()
					self.cursor.execute(query)
					self._display_time_(elapsed_time=time.time()-start_time)
					query_result=self.cursor.fetchall()
					column1=[float(item[0]) for item in query_result]
					column2=[float(item[1]) for item in query_result]
					all_columns+=[[column1,column2]]
					all_scatter+=[ax.scatter(column1,column2,alpha=0.8,marker=marker[idx+1],color=color[idx+1])]
				for idx,item in enumerate(all_categories):
					if (len(str(item))>10):
						all_categories[idx]=str(item)[0:10]+"..."
				plt.gca().grid()
				plt.gca().set_axisbelow(True)
				plt.title('Scatter Plot of {} vs {}'.format(columns[0],columns[1]))
				ax.xlabel(columns[0])
				ax.ylabel(columns[1])
				ax.legend(all_scatter,all_categories,scatterpoints=1,loc=loc,ncol=4,
							title=column_groupby,bbox_to_anchor=bbox_to_anchor,fontsize=8)
				plt.show()
	# Draw the scatter plot between 3 columns
	def scatter3D(self,columns,max_cardinality=3,cat_priority=None,with_others=True,color=None,marker=["^","o","+","*","h","x","D","1"]*10,max_nb_points=1000):
		if (color==None):
			color=self.colors
		if (not(isinstance(max_cardinality,int)) or (max_cardinality<1)):
			raise TypeError("The parameter 'max_cardinality' must be a strictly positive integer")
		if (not(isinstance(with_others,bool))):
			raise TypeError("The parameter 'with_others' must be a bool")
		if (not(isinstance(max_nb_points,int)) or (max_nb_points<1)):
			raise TypeError("The parameter 'max_nb_points' must be a strictly positive integer")
		if (not(isinstance(cat_priority,(list,type(None))))):
			raise TypeError("The parameter 'cat_priority' must be a list of categories or null")
		bbox_to_anchor,ncol,loc=self._legend_loc_init_()
		if (type(columns)!=list):
			raise TypeError("The parameter 'columns' must be a list of columns")
		if ((len(columns)<3) or (len(columns)>4)):
			raise Exception("3D Scatter plot can only be done with at least two columns and maximum with four columns")
		else:
			for column in columns:
				if (column not in self.columns):
					raise Exception("The RVC '{}' doesn't exist".format(column))
			for i in range(3):
				if (self[columns[i]].category() not in ["int","float"]):
					raise TypeError("The three first value of 'columns' must be numerical")
			if (len(columns)==3):
				tablesample=max_nb_points/self.count()
				query="select {},{},{} from {} where {} is not null and {} is not null and {} is not null limit {}".format(
							columns[0],columns[1],columns[2],self._table_transf_(tablesample),columns[0],
							columns[1],columns[2],max_nb_points)
				self._display_query_(query,title="Select random points for the scatter plot")
				start_time = time.time()
				self.cursor.execute(query)
				self._display_time_(elapsed_time=time.time()-start_time)
				query_result=self.cursor.fetchall()
				column1=[float(item[0]) for item in query_result]
				column2=[float(item[1]) for item in query_result]
				column3=[float(item[2]) for item in query_result]
				fig=plt.figure(figsize=self.figsize,facecolor='white')
				ax=fig.add_subplot(111,projection='3d')
				plt.title('Scatter Plot of {} vs {} vs {}'.format(columns[0],columns[1],columns[2]))
				ax.scatter(column1,column2,column3,color=color[0])
				ax.set_xlabel(columns[0])
				ax.set_ylabel(columns[1])
				ax.set_zlabel(columns[2])
				ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
				ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
				ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
				plt.show()
			else:
				column_groupby=columns[3]
				count=self.count()
				if (type(cat_priority)==list):
					query_result=cat_priority
				else:
					query="select {} from {} where {} is not null group by {} order by count(*) desc limit {}".format(
							column_groupby,self._table_transf_(),column_groupby,column_groupby,max_cardinality)
					self._display_query_(query,title="Select all the category of the column "+column_groupby)
					start_time = time.time()
					self.cursor.execute(query)
					self._display_time_(elapsed_time=time.time()-start_time)
					query_result=self.cursor.fetchall()
					query_result=[item for sublist in query_result for item in sublist]
				all_columns=[query_result]
				all_scatter=[]
				all_categories=query_result
				fig=plt.figure(figsize=self.figsize,facecolor="white")
				ax=fig.add_subplot(111,projection='3d')
				others=[]
				groupby_cardinality=self[column_groupby].cardinality()
				if (count>10000):
					tablesample=10
				else:
					tablesample=90
				for idx,category in enumerate(all_categories):
					if ((max_cardinality<groupby_cardinality) or ((type(cat_priority)==list) and len(cat_priority)<groupby_cardinality)):
							others+=["{}!='{}'".format(column_groupby,category)]
					query=("select {},{},{} from {} where {}='{}' and {} is not null and {} is not null " +
						"and {} is not null limit {}")
					query=query.format(columns[0],columns[1],columns[2],self._table_transf_(tablesample),
						columns[3],category,columns[0],columns[1],columns[2],int(max_nb_points/len(all_categories))) 
					self._display_query_(query,title="Select random points for the scatter plot (category='"+str(category)+"')")
					start_time = time.time()
					self.cursor.execute(query)
					self._display_time_(elapsed_time=time.time()-start_time)
					query_result=self.cursor.fetchall()
					column1=[float(item[0]) for item in query_result]
					column2=[float(item[1]) for item in query_result]
					column3=[float(item[2]) for item in query_result]
					all_columns+=[[column1,column2,column3]]
					all_scatter+=[ax.scatter(column1,column2,column3,alpha=0.8,marker=marker[idx],
						color=color[idx])]
				if (len(others)>0 and with_others):
					all_categories+=["others"]
					query=("select {},{},{} from {} where {} and {} is not null and {} is not null " +
						"and {} is not null limit {}")
					query=query.format(columns[0],columns[1],columns[2],self._table_transf_(tablesample)," and ".join(others),
						columns[0],columns[1],columns[2],int(max_nb_points/len(all_categories)))
					self._display_query_(query,title="Select random points for the scatter plot (category='others')")
					start_time = time.time()
					self.cursor.execute(query)
					self._display_time_(elapsed_time=time.time()-start_time)
					query_result=self.cursor.fetchall()
					column1=[float(item[0]) for item in query_result]
					column2=[float(item[1]) for item in query_result]
					all_columns+=[[column1,column2]]
					all_scatter+=[ax.scatter(column1,column2,alpha=0.8,marker=marker[idx+1],color=color[idx+1])]
				for idx,item in enumerate(all_categories):
					if (len(str(item))>10):
						all_categories[idx]=str(item)[0:10]+"..."
				plt.title('Scatter Plot of {} vs {} vs {}'.format(columns[0],columns[1],columns[2]))
				ax.set_xlabel(columns[0])
				ax.set_ylabel(columns[1])
				ax.set_zlabel(columns[2])
				ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
				ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
				ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
				if (bbox_to_anchor==(1,1)):
					bbox_to_anchor=(1,1.15)
				ax.legend(all_scatter,all_categories,scatterpoints=1,loc=loc,ncol=4,
							title=column_groupby,bbox_to_anchor=bbox_to_anchor,fontsize=8)
				plt.show()
	# Draw the Scatter Plot Matrix of the RVD
	def scatter_matrix(self, columns=None, color=None):
		if (color==None):
			if (type(self.colors) == list):
				color=self.colors[0]
			else:
				color=self.colors
		if not(isinstance(columns,type(None))):
			for column in columns:
				if (column not in self.columns):
					raise Exception("The RVC '{}' doesn't exist".format(column))
		if (type(columns)!=list):
			columns=[]
			for column in self.columns:
				if (self[column].category() in ["bool","int","float"]):
					columns+=[column]
		elif (len(columns)==1):	
			self[columns[0]].hist(color=color)
		n=len(columns)
		fig,axes=plt.subplots(nrows=n,ncols=n)
		query=("select "+",".join(columns)+",random() as rand from {} ".format(self._table_transf_())+
				"order by rand limit 1000")
		self._display_query_(query,title="Select random points for the scatter plot")
		start_time = time.time()
		self.cursor.execute(query)
		self._display_time_(elapsed_time=time.time()-start_time)
		all_scatter_points=self.cursor.fetchall()
		all_scatter_columns=[]
		all_h=[]
		for idx,column in enumerate(columns):
			is_numeric=(self[column].category()=="float") or (self[column].category()=="int")
			if (is_numeric):
				all_h+=[self[column]._best_hist_interval_()]
		h=min(all_h)
		for i in range(n):
			all_scatter_columns+=[[item[i] for item in all_scatter_points]]
		for i in range(n):
			x = columns[i]
			axes[-1][i].set_xlabel(x,rotation=90)
			axes[i][0].set_ylabel(x,rotation=0)
			axes[i][0].yaxis.get_label().set_ha('right')
			for j in range(n):
				axes[i][j].get_xaxis().set_ticks([])
				axes[i][j].get_yaxis().set_ticks([])
				axes[i][j].set_facecolor("#F0F0F0")
				y=columns[j]
				if (x==y):
					x0,y0,z0,h0,is_categorical=self[x]._hist_(method="density",h=h,max_cardinality=1)
					axes[i,j].bar(x0,y0,h0/0.94,color=color)
				else:
					axes[i,j].scatter(all_scatter_columns[j],all_scatter_columns[i],color=color,s=4,marker='o')
		fig.suptitle('Scatter Plot Matrix of {}'.format(self.input_relation))
		plt.show()
	# Select some columns
	def select(self,columns,order_by=None,asc=True,limit=100):
		if (not(isinstance(asc,bool))):
			raise TypeError("The parameter 'asc' must be a bool")
		if (not(isinstance(limit,int)) or (limit<1)):
			raise TypeError("The parameter 'limit' must be a strictly positive integer")
		if (type(columns)==str):
			columns=[columns]
		elif (type(columns)!=list):
			raise TypeError("The parameter 'columns' must be a list of varchar")
		if (asc):
			order=" asc "
		else:
			order=" desc "
		if (type(order_by)!=list):
			if (type(order_by)!=str):
				order_by=""
			else:
				order_by=" order by "+order_by+order
		else:
			order_by=" order by "+", ".join(order_by)+order
		query="select "+", ".join(columns)+" from "+self._table_transf_()+order_by
		return(run_query(query,self.cursor,limit=limit))
	# Set new rvd cursors
	def set_colors(self,colors):
		self.colors=colors
	# Set a new cursor
	def set_cursor(self,cursor):
		self.cursor=cursor
	# Set a new dsn
	def set_dsn(self,dsn):
		self.dsn=dsn
	# Set the figure size
	def set_figure_size(self,figsize=(7,5)):
		self.figsize=figsize
	# Set the label location
	def set_legend_loc(self,bbox_to_anchor=None,ncol=None,loc=None):
		self.legend_loc=(bbox_to_anchor,ncol,loc)
	# Set the RVD limit
	def set_limit(self,limit=None):
		self.limit=limit
	# Set the RVD offset
	def set_offset(self,offset=0):
		self.offset=offset
	# Print all the SQL queries in the terminal
	def sql_on_off(self,reindent=False):
		self.query_on=not(self.query_on)
		self.reindent=reindent
	# Draw the Stacked bar
	def stacked_bar(self,columns,method="density",of=None,max_cardinality=[6,6],h=[None,None],color=None,limit_distinct_elements=200):
		self.bar(columns,method=method,of=of,max_cardinality=max_cardinality,h=h,color=color,limit_distinct_elements=limit_distinct_elements,
					stacked=True)
	# Draw the Stacked Histogram
	def stacked_hist(self,columns,method="density",of=None,max_cardinality=[6,6],h=[None,None],color=None,limit_distinct_elements=200):
		self.hist(columns,method=method,of=of,max_cardinality=max_cardinality,h=h,color=color,limit_distinct_elements=limit_distinct_elements,
					stacked=True)
	# Display all the queries elapsed time
	def time_on_off(self):
		self.time_on=not(self.time_on)
	# Split the rvd into two relations using a split column
	# If the split column does not exist, a column of random float will be created
	# The split column must be a columns containing random float in [0,1] 
	def train_test_split(self,split=None,test_name=None,train_name=None,columns="*",test_size=0.33,mode="view",print_info=True):
		if (mode not in ["view","temporary table","table"]):
			raise Exception("The parameter 'mode' must be in view|temporary table|table\nNothing was saved.")
		if not(isinstance(test_name,(str,type(None)))):
			raise TypeError("The parameter 'test_name' must be a varchar or null")
		if not(isinstance(train_name,(str,type(None)))):
			raise TypeError("The parameter 'train_name' must be a varchar or null")
		if not(isinstance(split,(str,type(None)))):
			raise TypeError("The parameter 'split' must be a varchar or null")
		if (not(isinstance(test_size,float)) or (test_size<=0) or (test_size>=1)):
			raise TypeError("The parameter 'test_size' must be in ]0,1[")
		if not(isinstance(print_info,bool)):
			raise TypeError("The parameter 'print_info' must be a bool")
		if columns!="*":
			for column in columns:
				if (column not in self.columns):
					raise Exception("The RVC '{}' is not in the RVD".format(column))
		else:
			columns=self.columns
		if (type(test_name)!=str):
			test_name="test_"+self.input_relation+"0"+str(int(test_size*100))
			drop_test=True
		else:
			drop_test=False
		if (type(train_name)!=str):
			train_name="train_"+self.input_relation+"0"+str(int(100-test_size*100))
			drop_train=True
		else:
			drop_train=False
		if (mode=="view"):
			if (drop_test):
				query="drop view if exists "+test_name
				self._display_query_(query,title="Drop the test view")
				start_time = time.time()
				self.cursor.execute(query)
				self._display_time_(elapsed_time=time.time()-start_time)
			if (drop_train):
				query="drop view if exists "+train_name
				self._display_query_(query,title="Drop the train view")
				start_time = time.time()
				self.cursor.execute(query)
				self._display_time_(elapsed_time=time.time()-start_time)
		else:
			if (drop_test):
				query="drop table if exists "+test_name
				self._display_query_(query,title="Drop the test table")
				start_time = time.time()
				self.cursor.execute(query)
				self._display_time_(elapsed_time=time.time()-start_time)
			if (drop_train):
				query="drop table if exists "+train_name
				self._display_query_(query,title="Drop the train table")
				start_time = time.time()
				self.cursor.execute(query)
				self._display_time_(elapsed_time=time.time()-start_time)
		if (type(split)!=str):
			random_name="random_vpython_table_"+self.input_relation
			try:
				query="select split from "+random_name
				self._display_query_(query,title="Try to see if the random table exists and contains a column split")
				start_time = time.time()
				self.cursor.execute(query)
				self._display_time_(elapsed_time=time.time()-start_time)
			except:
				query="drop table if exists "+random_name
				self._display_query_(query,title="The random doesn't exist or does not have the good format: drop the random table")
				start_time = time.time()
				self.cursor.execute(query)
				self._display_time_(elapsed_time=time.time()-start_time)
				query=("create table "+random_name+" as select row_number() over() as row_number,random() as split from "
						+self._table_transf_())
				self._display_query_(query,title="Create the random table")
				start_time = time.time()
				self.cursor.execute(query)
				self._display_time_(elapsed_time=time.time()-start_time)
				if (print_info):
					print("The random table "+random_name+" was successfully created.")
			query="create {} {} as select {} from (select row_number() over() as row_number,* from {}) z natural join {} where split<{}"
			query=query.format(mode,test_name,",".join(columns),self._table_transf_(),random_name,test_size)
			self._display_query_(query,title="Create the test "+mode)
			start_time = time.time()
			self.cursor.execute(query)
			self._display_time_(elapsed_time=time.time()-start_time)
			query="create {} {} as select {} from (select row_number() over() as row_number,* from {}) z natural join {} where split>={}"
			query=query.format(mode,train_name,",".join(columns),self._table_transf_(),random_name,test_size)
			self._display_query_(query,title="Create the train "+mode)
			start_time = time.time()
			self.cursor.execute(query)
			self._display_time_(elapsed_time=time.time()-start_time)
		else:
			query="create {} {} as select {} from {} where {}<{}"
			query=query.format(mode,test_name,",".join(columns),self._table_transf_(),split,test_size)
			self._display_query_(query,title="Create the test "+mode+" using the corresponding split")
			start_time = time.time()
			self.cursor.execute(query)
			self._display_time_(elapsed_time=time.time()-start_time)
			query="create {} {} as select {} from {} where {}>={}"
			query=query.format(mode,train_name,",".join(columns),self._table_transf_(),split,test_size)
			self._display_query_(query,title="Create the train "+mode+" using the corresponding split")
			start_time = time.time()
			self.cursor.execute(query)
			self._display_time_(elapsed_time=time.time()-start_time)
		if (print_info):
			print("The "+mode+"s "+test_name+" and "+train_name+" were successfully created.")
		self.rvd_history+=["{"+time.strftime("%c")+"} "+"[Train Test Split]: The "+mode+"s '"+test_name+"' and '"+train_name+"' were created."]
		return RVD(train_name,self.cursor),RVD(test_name,self.cursor)
	# Undo all the filters
	def undo_all_filters(self):
		self.where=[]
		self.rvd_history+=["{"+time.strftime("%c")+"} "+"[Undo All Filters]: All the filters were deleted."]
	# Undo the last filter
	def undo_filter(self):
		if (len(self.where)>0):
			del self.where[-1]
			self.rvd_history+=["{"+time.strftime("%c")+"} "+"[Undo Filter]: The last filter was removed."]
	#
	#######################
	#                     #
	# Information Methods #
	#                     #
	#######################
	# 
	# Help: return some RVD info 
	def help(self):
		print("############################")
		print("#   _______      _______   #")
		print("#  |  __ \ \    / /  __ \  #")
		print("#  | |__) \ \  / /| |  | | #")
		print("#  |  _  / \ \/ / | |  | | #")
		print("#  | | \ \  \  /  | |__| | #")
		print("#  |_|  \_\  \/   |_____/  #")
		print("#                          #")
		print("#############################")
		print("#                           #")
		print("# Resilient Vertica Dataset #")
		print("#                           #")
		print("#############################")
		print("")
		print("The RVD is a Python object which will keep in mind all the user modifications in order "
				+"to use an optimized SQL query. It will send the query to the database which will use its "
				+"aggregations to compute fast results. It is created using a view or a table stored in the "
				+"user database and a database cursor. It will create for each column of the table a RVC (Resilient"
				+" Vertica Column) which will store for each column its name, its imputations and allows to do easy "
				+"modifications and explorations.")
		print("")
		print("RVC and RVD coexist and one can not live without the other. RVC will use the RVD information and reciprocally." 
				+" It is imperative to understand both structures to know how to use the entire object.")
		print("")
		print("When the user imputes or filters the data, the RVD gets in memory all the transformations to select for each query "
				+"the needed data in the input relation.")
		print("")
		print("As the RVD will try to keep in mind where the transformations occurred in order to use the appropriate query," 
				+" it is highly recommended to save the RVD when the user has done a lot of transformations in order to gain in efficiency" 
				+" (using the save method). We can also see all the modifications using the history method.")
		print("")
		print("If you find any difficulties using vertica_ml_python, please contact me: badr.ouali@microfocus.com / I'll be glad to help.")
		print("")
		print("For more information about the different methods or the entire RVD structure, please see the entire documentation")
	# Return the vertica Version and what is available
	def version(self):
		query="select version();"
		self.cursor.execute(query)
		version=self.cursor.fetchone()[0]
		print("############################################################################################################") 
		print("#  __ __   ___ ____  ______ ____   __  ____      ___ ___ _          ____  __ __ ______ __ __  ___  ____    #")
		print("# |  |  | /  _|    \|      |    | /  ]/    |    |   |   | |        |    \|  |  |      |  |  |/   \|    \   #")
		print("# |  |  |/  [_|  D  |      ||  | /  /|  o  |    | _   _ | |        |  o  |  |  |      |  |  |     |  _  |  #")
		print("# |  |  |    _|    /|_|  |_||  |/  / |     |    |  \_/  | |___     |   _/|  ~  |_|  |_|  _  |  O  |  |  |  #")
		print("# |  :  |   [_|    \  |  |  |  /   \_|  _  |    |   |   |     |    |  |  |___, | |  | |  |  |     |  |  |  #")
		print("#  \   /|     |  .  \ |  |  |  \     |  |  |    |   |   |     |    |  |  |     | |  | |  |  |     |  |  |  #")
		print("#   \_/ |_____|__|\_| |__| |____\____|__|__|    |___|___|_____|    |__|  |____/  |__| |__|__|\___/|__|__|  #")
		print("#                                                                                                          #")
		print("############################################################################################################")
		print("#")
		print("# Author: Badr Ouali, Datascientist at Vertica")
		print("#")
		print("# You are currently using "+version)
		print("#")
		version=version.split("Database v")
		version_id=int(version[1][0])
		version_release=int(version[1][2])
		if (version_id>8):
			print("# You have a perfectly adapted version for using RVD and Vertica ML")
		elif (version_id==8):
			if (version_release>0):
				print("# You have a perfectly adapted version for using RVD and Vertica ML except some algorithms")
				print("# Go to your Vertica version documentation for more information")
				print("# Unavailable algorithms: rf_regressor and cross_validate")
			else:
				print("# Your Vertica version is adapted for using RVD but you are quite limited for Vertica ML")
				print("# Go to your Vertica version documentation for more information")
				print("# Unavailable algorithms: rf, svm and cross_validate")
				print("# /!\\ Some RVD queries can be really big because of the unavailability of a lot of functions")
		else:
			print("# Your Vertica version is adapted for using RVD but you can not use Vertica ML")
			print("# Go to your Vertica version documentation for more information")
			print("# /!\\ Some RVD queries can be really big because of the unavailability of a lot of functions")
			print("# /!\\ Some RVD functions could not work")
		print("#")
		print("# For more information about the RVD you can use the help() method")
		return (version_id,version_release)

		




