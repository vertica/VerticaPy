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
import math
import matplotlib.pyplot as plt
import time
import datetime
from vertica_ml_python.fun import column_matrix
from vertica_ml_python.fun import print_table
from vertica_ml_python.fun import isnotebook

############################
#                          #
#   _______      _______   #
#  |  __ \ \    / / ____|  #
#  | |__) \ \  / / |       #
#  |  _  / \ \/ /| |       #
#  | | \ \  \  / | |____   #
#  |_|  \_\  \/   \_____|  #
#                          #
#                          #
############################
#                          #
# Resilient Vertica Column #
#                          #
############################
#
##
class RVC:
	###################
	#                 #
	# Special Methods #
	#                 #
	###################
	#
	# Initialization
	#
	# RVC has 3 attributes: parent, alias and transformations
	def  __init__(self,alias,first_transformation=None,parent=None):
		# Pointer to the RVC parent
		self.parent=parent
		# Alias of the concerned column
		self.alias=alias
		if (first_transformation==None):
			# Compute the column's type
			ctype,category=self._cat_type_()
			# Keep in mind all the user modifications
			self.transformations=[(alias,ctype,category)]
		else:
			self.transformations=[first_transformation]
	# RVC[index] returns the corresponding raw(s)
	def __getitem__(self,index):
		# Case when index is an integer
		try:
			if ((type(index)==int) and (index<0)):
				index+=self.parent.count()
			query="select {} from {} limit 1 offset {}".format(self.alias,self.parent._table_transf_(),index)
			self.parent.cursor.execute(query)
			return self.parent.cursor.fetchone()
		except:
			# Case when index is a list of two integers
			try:
				x,y=index
				count=self.parent.count()
				if ((type(x)==int) and (x<0)):
					x+=count
				if ((type(y)==int) and (y<0)):
					y+=count
				if (y-x>0):
					limit=y-x+1
					query="select {} from {} limit {} offset {}".format(self.alias,self.parent._table_transf_(),limit,x)
					self.parent.cursor.execute(query)
					return [item[0] for item in self.parent.cursor.fetchall()]
			except:
				raise Exception('The RVC index must be an integer or a list of two ordered integers')
	# Object Representation when using the print function
	def __repr__(self,limit=30):
		return self.parent.select(columns=[self.alias],limit=limit).__repr__()
	# Object attr affectation to deal with all the modifications (also to avoid users unadapted modifications)
	def __setattr__(self,attr,val):
		if (attr=="alias"):
			if not(isinstance(val,str)):
				print("/!\\ Warning: attribute 'alias' must be a string which is exactly equal to the"
					+"parent attribute having the same name.\nYou are not allowed to manually change"
					+ "this attribute, it can destroy the RVD robustness.\nNothing was changed.")
			else:
				self.__dict__[attr]=val
		elif (attr=="parent"):
			from vertica_ml_python.rvd import RVD
			if not(isinstance(val,RVD)):
				print("/!\\ Warning: attribute 'parent' must be an RVD.\nYou are not allowed to manually change"
					+ "this attribute, it can destroy the RVD robustness.\nNothing was changed.")
			else:
				self.__dict__[attr]=val
		elif (attr=="transformations"):
			error=False
			if (not(isinstance(val,list)) or (len(val)==0)):
				error=True
			for item in val:
				if (not(isinstance(item[0],str)) or not(isinstance(item[1],str)) or not(item[2] in ["int","float","date","text","undefined"])):
					error=True
			if (error):
				print("/!\\ Warning: attribute 'transformations' must be a list of 3-tuples of the form "
					+ "(imputation,ctype,category) with category in int | float | date | text | undefined."
					+ " Changing this attribute can destroy the RVD robustness."
					+ "\nNothing was changed.")
			else:
				self.__dict__[attr]=val
		else:
			print("/!\\ Warning: RVC has only 3 attributes:\n\t- parent (the parent RVD)\n\t- cursor (the Vertica DB cursor)"
						+ "\n\t- transformations (all the RVC transformations).\nOther attributes are useless and will not be"
						+ " added to the structure.")
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
	# Compute the RVC best histogram interval
	def _best_hist_interval_(self):
		if (self.category() in ["int","float"]):
			summarize=self._summarize_num_col_()
			count=summarize[0]
			rvc_min=summarize[3]
			rvc_025=summarize[4]
			rvc_075=summarize[6]
			rvc_max=summarize[7]
			# Freedman-Diaconis/Sturges depending on the max
			return max(2.0*(rvc_075-rvc_025)/(count)**(1.0/3.0),float(rvc_max-rvc_min)/int(math.floor(math.log(count,2)+1)))
		elif (self.category() in ["date"]):
			min_date=self.min()
			table="(select datediff('second','"+str(min_date)+"'::timestamp,"+self.alias+") as "+self.alias+" from "+self.parent._table_transf_()
			table+=") best_h_date_table"
			query=("select (select count(*) from {} where {} is not null) as NAs, min({}) as min, (select percentile_cont(0.25)"
					+ " within group (order by {}) over () from {} limit 1) as Q1, (select percentile_cont(0.75) within group" 
					+ " (order by {}) over () from {} limit 1) as Q3, max({}) as max from {} group by Q1,Q3,NAs")
			query = query.format(table,self.alias,self.alias,self.alias,table,self.alias,table,self.alias,table,self.alias,table)
			self.parent._display_query_(query,title="Compute the date best interval of ("+self.alias+")")
			start_time=time.time()
			self.parent.cursor.execute(query)
			self.parent._display_time_(elapsed_time=time.time()-start_time)
			query_result=self.parent.cursor.fetchone()
			count=query_result[0]
			rvc_min=query_result[1]
			rvc_025=query_result[2]
			rvc_075=query_result[3]
			rvc_max=query_result[4]
			# Freedman-Diaconis/Sturges depending on the max
			return max(math.floor(2.0*(rvc_075-rvc_025)/(count)**(1.0/3.0))+1,math.floor(float(rvc_max-rvc_min)/int(math.floor(math.log(count,2)+1)))+1)
		else:
			raise Exception("The best hist interval can only be computed with numerical RVC or dates")
	# Return the RVC category and the type
	def _cat_type_(self):
		query="select data_type from columns where table_name='{}' and (column_name='{}' or column_name='{}');".format(
			self.parent.input_relation,self.alias,self.alias[1:-1])
		self.parent.cursor.execute(query)
		query_result=self.parent.cursor.fetchall()
		if not(query_result==[]):
			ctype=str(query_result[0][0])
			# Column's category (date|int|float|text)
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
		return (ctype,category)
	# Return all the RVC information needed to draw an histogram
	def _hist_(self,method="density",of=None,max_cardinality=6,bins=None,h=None,pie=False):
		# aggregation used for the bins height
		if (method=="mean"):
			method="avg"
		if ((method in ["avg","min","max","sum"]) and (type(of)==str)):
			if (of in self.parent.columns):
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
		if (not(isinstance(max_cardinality,int)) or (max_cardinality<1)):
			raise TypeError("The parameter 'max_cardinality' must be a strictly positive integer")
		if (not(isinstance(bins,int)) or (bins<1)):
			if (not(isinstance(bins,type(None)))):
				raise TypeError("The parameter 'bins' must be a strictly positive integer or null")
		if (not(isinstance(h,(float,int))) or (h<0)):
			if (not(isinstance(h,type(None)))):
				raise TypeError("The parameter 'h' must be a strictly positive number or null")
			elif (h==0):
				raise ZeroDivisionError("The parameter 'h' can not be zero")
		if (not(isinstance(pie,bool))):
			raise TypeError("The parameter 'pie' must be a bool")
		# depending on the cardinality, the type, the RVC can be treated as categorical or not
		cardinality=self.cardinality()
		count=self.parent.count()
		is_numeric=(self.category()=="float") or (self.category()=="int")
		is_date=(self.category()=="date")
		is_categorical=False
		if ((is_numeric) and (cardinality > max_cardinality)):
			rotation=0
		else:
			rotation=90
		# case when categorical
		if ((((cardinality<=max_cardinality) or not(is_numeric)) or pie) and not(is_date)):
			if ((is_numeric) and not(pie)):
				query="select {},{} from {} where {} is not null group by {} order by {} asc limit {}".format(
					self.alias,aggregate,self.parent._table_transf_(),self.alias,
					self.alias,self.alias,max_cardinality)
			else:
				table=self.parent._table_transf_()
				if ((pie) and (is_numeric)):
					enum_trans=self._enum_trans_(h)[0].replace("{}",self.alias)+" as "+self.alias
					if (len(of)==1):
						enum_trans+=" , "+of[0]
					table="(select "+enum_trans+" from "+table+") enum_table"
				query="(select {}||'',{} from {} group by {} order by {} desc limit {})".format(
					self.alias,aggregate,table,self.alias,aggregate,max_cardinality)
				if (cardinality>max_cardinality):
					query+=(" union (select 'Others',{}(count) from (select {} as count from {} " + 
						"group by {} order by {} desc offset {}) y limit 1) order by count desc")
					query=query.format(others_aggregate,aggregate,table,self.alias,aggregate,max_cardinality)
			self.parent._display_query_(query,title="Compute the histogram heights")
			start_time = time.time()
			self.parent.cursor.execute(query)
			self.parent._display_time_(elapsed_time=time.time()-start_time)
			query_result=self.parent.cursor.fetchall()
			if (method=="density"):
				y=[item[1]/float(count) for item in query_result]
			else:
				y=[item[1] for item in query_result]
			z=[item[0] for item in query_result]
			x=[0.4*i+0.2 for i in range(0,len(y))]
			h=0.39
			is_categorical=True
		# case when date
		elif (is_date):
			if ((h==None) and (bins==None)):
				h=self._best_hist_interval_()
			elif (bins!=None):
				query="select datediff('second',min("+self.alias+"),max("+self.alias+")) from "
				query+=self.parent._table_transf_()
				self.parent._display_query_(query,title="Compute the histogram interval")
				start_time=time.time()
				self.parent.cursor.execute(query)
				self.parent._display_time_(elapsed_time=time.time()-start_time)
				query_result=self.parent.cursor.fetchone()
				h=float(query_result[0])/bins
			min_date=self.min()
			converted_date="datediff('second','"+str(min_date)+"',"+self.alias+")"
			query=("select floor({}/{})*{},{} from {} where {} is not null " +
						"group by 1 order by 1")
			query=query.format(converted_date,h,h,aggregate,self.parent._table_transf_(),self.alias)
			self.parent._display_query_(query,title="Compute the histogram heights")
			start_time=time.time()
			self.parent.cursor.execute(query)
			self.parent._display_time_(elapsed_time=time.time()-start_time)
			query_result=self.parent.cursor.fetchall()
			if (method=="density"):
				y=[item[1]/float(count) for item in query_result]
			else:
				y=[item[1] for item in query_result]
			x=[float(item[0]) for item in query_result]
			query=""
			for idx,item in enumerate(query_result):
				query+=" union (select timestampadd('second',"+str(math.floor(h*idx))+",'"+str(min_date)+"'::timestamp))"
			query=query[7:-1]+")"
			h=0.94*h
			self.parent.cursor.execute(query)
			query_result=self.parent.cursor.fetchall()
			z=[item[0] for item in query_result]
			z.sort()
			is_categorical=True
		# case when numerical
		else:
			if ((h==None) and (bins==None)):
				h=self._best_hist_interval_()
			elif (bins!=None):
				h=float(self.max()-self.min())/bins
			if (self.ctype=="int"):
				h=max(1.0,h)
			query=("select floor({}/{})*{},{} from {} where {} is not null " +
						"group by 1 order by 1")
			query=query.format(self.alias,h,h,aggregate,self.parent._table_transf_(),self.alias)
			self.parent._display_query_(query,title="Compute the histogram heights")
			start_time=time.time()
			self.parent.cursor.execute(query)
			self.parent._display_time_(elapsed_time=time.time()-start_time)
			query_result=self.parent.cursor.fetchall()
			if (method=="density"):
				y=[item[1]/float(count) for item in query_result]
			else:
				y=[item[1] for item in query_result]
			x=[float(item[0])+h/2 for item in query_result]
			h=0.94*h
			z=None
		return [x,y,z,h,is_categorical]
	# Return descriptive statistics of the RVC
	def _summarize_num_col_(self):
		# For Vertica 9.0 and higher
		try:
			query="select summarize_numcol("+self.alias+") over () from {}".format(self.parent._table_transf_())
			self.parent._display_query_(query,title="Compute a direct summarize_numcol("+self.alias+")")
			start_time=time.time()
			self.parent.cursor.execute(query)
			self.parent._display_time_(elapsed_time=time.time()-start_time)
			query_result=self.parent.cursor.fetchone()
			return [float(query_result[i]) for i in range(1,len(query_result))]
		# For all versions of Vertica
		except:
			try:
				query=("select (select count(*) from {} where {} is not null) as NAs, avg({}) as mean, stddev({}) as std, "
						+ "min({}) as min, (select percentile_cont(0.25) within group (order by {}) over () from {} limit 1)"
						+ " as Q1, (select percentile_cont(0.50) within group (order by {}) over () from {} limit 1) as Median, "
						+ "(select percentile_cont(0.75) within group (order by {}) over () from {} limit 1) as Q3, max({}) as max "
						+ " from {} group by Q1,Median,Q3,NAs")
				query = query.format(self.parent._table_transf_(),self.alias,self.alias,self.alias,self.alias,self.alias,self.parent._table_transf_(),
								self.alias,self.parent._table_transf_(),self.alias,self.parent._table_transf_(),self.alias,self.parent._table_transf_())
				self.parent._display_query_(query,title="Compute a manual summarize_numcol("+self.alias+")")
				start_time=time.time()
				self.parent.cursor.execute(query)
				self.parent._display_time_(elapsed_time=time.time()-start_time)
				query_result=self.parent.cursor.fetchone()
				return [float(item) for item in query_result]
			except:
				return [self.count(),self.mean(),self.std(),self.min(),self.percentile_cont(0.25),self.median(),self.percentile_cont(0.75),self.max()]
	# Return the enum transformation of the RVC
	def _enum_trans_(self,h=None):
		is_numeric=(self.category()=="float") or (self.category()=="int")
		if (is_numeric):
			if (h==None):
				h=self._best_hist_interval_()
				h=round(h,2)
			elif (not(isinstance(h,(int,float))) or (h<0)):
				raise TypeError("The parameter 'h' must be a strictly positive number or null")
			elif (h==0):
				ZeroDivisionError("The parameter 'h can not be equal to 0")
			if (self.category=="int"):
				floor_end="-1"
				h=int(max(math.floor(h),1))
			else:
				floor_end=""
			if (h>1) or (self.category=="float"):
				return ("'[' || floor({}/{})*{} ||';'|| (floor({}/{})*{}+{}{}) || ']'".format(
					"{}",h,h,"{}",h,h,h,floor_end),"varchar","text")
			else:
				return ("floor({}) || ''","varchar","text")
		else:
			return ("{} || ''","varchar","text")
	#
	###########
	#         #
	# Methods #
	#         #
	###########
	# 
	# abs: RVC=abs(RVC)
	def abs(self):
		if (self.category() not in ['float','int']):
			raise Exception("The RVC must be numerical in order to apply abs")
		else:
			self.transformations+=[("abs({})",self.ctype(),self.category())]
			self.parent.rvd_history+=["{"+time.strftime("%c")+"} "+"[Abs]: The RVC '{}' was transformed with the function 'abs'.".format(self.alias)]
	# acos: RVC=acos(RVC)
	def acos(self):
		if (self.category() not in ['float','int']):
			raise Exception("The RVC must be numerical in order to apply acos")
		else:
			self.transformations+=[("acos({})","float","float")]
			self.parent.rvd_history+=["{"+time.strftime("%c")+"} "+"[Acos]: The RVC '{}' was transformed with the function 'acos'.".format(self.alias)]
	# add: RVC=RVC+x
	def add(self,x):
		if (self.category() not in ['float','int']):
			raise Exception("The RVC must be numerical in order to apply add")
		elif (type(x) not in [int,float]):
			raise TypeError("The parameter 'x' must be numerical")
		elif (x<0):
			raise TypeError("The parameter 'x' must be positive")
		else:
			if (type(x)==float):
				self.transformations+=[("{}+{}".format("{}",x),"float","float")]
			else:
				self.transformations+=[("{}+{}".format("{}",x),self.ctype(),self.category())]
			self.parent.rvd_history+=["{"+time.strftime("%c")+"} "+"[Add]: {} was added to the RVC '{}'.".format(x,self.alias)]
	# asin: RVC=asin(RVC)
	def asin(self):
		if (self.category() not in ['float','int']):
			raise Exception("The RVC must be numerical in order to apply asin")
		else:
			self.transformations+=[("asin({})","float","float")]
			self.parent.rvd_history+=["{"+time.strftime("%c")+"} "+"[Asin]: The RVC '{}' was transformed with the function 'asin'.".format(self.alias)]
	# atan: RVC=atan(RVC)
	def atan(self):
		if (self.category() not in ['float','int']):
			raise Exception("The RVC must be numerical in order to apply atan")
		else:
			self.transformations+=[("atan({})","float","float")]
			self.parent.rvd_history+=["{"+time.strftime("%c")+"} "+"[Atan]: The RVC '{}' was transformed with the function 'atan'.".format(self.alias)]
	# Return the RVC mean
	def avg(self):
		return self.mean()
	# Draw the RVC bar chart
	def bar(self,method="density",of=None,max_cardinality=6,bins=None,h=None,color=None):
		if (color==None):
			color=self.parent.colors[0]
		x,y,z,h,is_categorical=self._hist_(method=method,of=of,max_cardinality=max_cardinality,bins=bins,h=h)
		plt.figure(figsize=self.parent.figsize)
		plt.rcParams['axes.facecolor']='#F5F5F5'
		plt.barh(x,y,h,color=color,alpha=0.86)
		plt.ylabel(self.alias)
		plt.gca().xaxis.grid()
		plt.gca().set_axisbelow(True)
		if (is_categorical):
			if (self.category=="text"):
				new_z=[]
				for item in z:
					if (len(str(item))>20):
						new_z+=[item[0:17] + "..."]
					else:
						new_z+=[item]
			else:
				new_z=z
			plt.yticks(x,new_z)
			plt.subplots_adjust(left=max(0.1,min(len(max([str(item) for item in z],key=len)),20)/80.0))
		if (method=="density"):
			plt.xlabel('Density')
			plt.title('Distribution of {}'.format(self.alias))
		elif ((method in ["avg","min","max","sum"]) and (of!=None)):
			aggregate="{}({})".format(method,of)
			plt.ylabel(aggregate)
			plt.title('{} group by {}'.format(aggregate,self.alias))
		else:
			plt.xlabel('Frequency')
			plt.title('Count by {}'.format(self.alias))
		plt.show()
	# Return the RVC boxplot
	def boxplot(self,by=None,h=None,max_cardinality=8,cat_priority=None):
		if (not(isinstance(max_cardinality,int)) or (max_cardinality<1)):
			raise TypeError("The parameter 'max_cardinality' must be a strictly positive integer")
		if not(isinstance(by,(str,type(None)))):
			raise TypeError("The parameter 'by' must be a varchar or null")
		# case with single box plot
		if (type(by)!=str):
			if (self.category() not in ["int","float"]):
				raise Exception("The RVC must be numerical in order to draw a boxplot")
			summarize=self._summarize_num_col_()
			for i in range(0,2):
				del summarize[0]
			plt.figure(figsize=(7,4))
			plt.rcParams['axes.facecolor']='#F5F5F5'
			plt.xlabel(self.alias)
			box=plt.boxplot(summarize,notch=False,sym='',whis=np.inf,vert=False,widths=0.7,labels=[""],
				patch_artist=True)
			for median in box['medians']:
				median.set(color='black',linewidth=1,)
			for patch in box['boxes']:
				patch.set_facecolor("dodgerblue")
			plt.gca().xaxis.grid()
			plt.gca().set_axisbelow(True)
			plt.title('BoxPlot of {}'.format(self.alias))
			plt.show()
		# case with multiple box plot: only available with Vertica 9.0 and higher
		else:
			try:
				if (self.alias==by):
					raise Exception("The RVC and the groupby RVC can not be the same")
				elif (by not in self.parent.columns):
					raise Exception("None RVC named '"+by+"' found")
				count=self.parent.count()
				cardinality=self.parent[by].cardinality()
				is_numeric=(self.parent[by].category()=="float") or (self.parent[by].category()=="int")
				is_categorical=(cardinality<=max_cardinality) or not(is_numeric)
				table=self.parent._table_transf_()
				if not(is_categorical):
					enum_trans=self.parent[by]._enum_trans_(h)[0].replace("{}",by)+" as "+by
					enum_trans+=" , "+self.alias
					table="(select "+enum_trans+" from "+table+") enum_table"
					query="select count(distinct {}) from {}".format(by,table)
					self.parent._display_query_(query,title="Compute the cardinality of the enum feature "+by)
					start_time = time.time()
					self.parent.cursor.execute(query)
					self.parent._display_time_(elapsed_time=time.time()-start_time)
					cardinality=self.parent.cursor.fetchone()[0]
				if (type(cat_priority)!=list):
					query="select {} from {} where {} is not null group by {} order by count(*) desc limit {}".format(
							by,table,self.alias,by,max_cardinality)
					self.parent._display_query_(query,title="Compute the category of the feature "+by)
					start_time = time.time()
					self.parent.cursor.execute(query)
					self.parent._display_time_(elapsed_time=time.time()-start_time)
					query_result=self.parent.cursor.fetchall()
					cat_priority=[item for sublist in query_result for item in sublist]
				try:
					with_summarize=True
					query=""
					for idx,category in enumerate(cat_priority):
						if (category==None or category=="None"):
							query+=" union (select *,'None' from (select summarize_numcol({}) over() from {} where {} is null) boxplot_info_cat)".format(
								self.alias,table,by)
						else:
							query+=" union (select *,'{}' from (select summarize_numcol({}) over() from {} where {}='{}') boxplot_info_cat)".format(
								category,self.alias,table,by,category)
					if (cardinality>max_cardinality):
						where=[]
						for item in cat_priority:
							where+=[by+"!='"+str(item)+"'"]
						where=" and ".join(where)
						query+=" union (select *,'others' from (select summarize_numcol({}) over() from {} where {}) boxplot_info_cat)".format(
								self.alias,table,where)
					query=query[7:len(query)]
					self.parent._display_query_(query,title="Compute all the descriptive statistics for each category to draw the box plot")
					start_time = time.time()
					self.parent.cursor.execute(query)
					query_result=self.parent.cursor.fetchall()
				except:
					try:
						raise
						with_summarize=True
						query_result=[]
						for idx,category in enumerate(cat_priority):
							if (category==None or category=="None"):
								query="select *,'None' from (select summarize_numcol({}) over() from {} where {} is null) boxplot_info_cat".format(
									self.alias,table,by)
							else:
								query="select *,'{}' from (select summarize_numcol({}) over() from {} where {}='{}') boxplot_info_cat".format(
									category,self.alias,table,by,category)
							self.parent._display_query_(query,title="Compute manually all the descriptive statistics for each category to draw the box plot")
							start_time = time.time()
							self.parent.cursor.execute(query)
							query_result+=[self.parent.cursor.fetchone()]
						if (cardinality>max_cardinality):
							where=[]
							for item in cat_priority:
								where+=[by+"!='"+str(item)+"'"]
							where=" and ".join(where)
							query="select *,'others' from (select summarize_numcol({}) over() from {} where {}) boxplot_info_cat".format(
									self.alias,table,where)
							self.parent._display_query_(query,title="Compute manually all the descriptive statistics for each category to draw the box plot")
							start_time = time.time()
							self.parent.cursor.execute(query)
							query_result+=[self.parent.cursor.fetchone()]
					except:
						try:
							raise
							with_summarize=False
							query=""
							for idx,category in enumerate(cat_priority):
								if (category==None or category=="None"):
									tmp_query=(" union (select min({}) as min, (select percentile_cont(0.25) within group (order by {}) over () from {} limit 1)"
										+ " as Q1, (select percentile_cont(0.50) within group (order by {}) over () from {} limit 1) as Median, "
										+ "(select percentile_cont(0.75) within group (order by {}) over () from {} limit 1) as Q3, max({}) as max, 'None' "
										+ " from {} where {} is null group by Q1,Median,Q3)")
									query+=tmp_query.format(self.alias,self.alias,table,self.alias,table,self.alias,table,self.alias,table,by)
								else:
									tmp_query=(" union (select min({}) as min, (select percentile_cont(0.25) within group (order by {}) over () from {} limit 1)"
										+ " as Q1, (select percentile_cont(0.50) within group (order by {}) over () from {} limit 1) as Median, "
										+ "(select percentile_cont(0.75) within group (order by {}) over () from {} limit 1) as Q3, max({}) as max, '{}' "
										+ " from {} where {}='{}' group by Q1,Median,Q3)")
									query+=tmp_query.format(self.alias,self.alias,table,self.alias,table,self.alias,table,self.alias,category,table,by,category)
							if (cardinality>max_cardinality):
								where=[]
								for item in cat_priority:
									where+=[by+"!='"+str(item)+"'"]
								where=" and ".join(where)
								tmp_query=(" union (select min({}) as min, (select percentile_cont(0.25) within group (order by {}) over () from {} limit 1)"
									+ " as Q1, (select percentile_cont(0.50) within group (order by {}) over () from {} limit 1) as Median, "
									+ "(select percentile_cont(0.75) within group (order by {}) over () from {} limit 1) as Q3, max({}) as max, 'Others' "
									+ " from {} where {} group by Q1,Median,Q3)")
								query+=tmp_query.format(self.alias,self.alias,table,self.alias,table,self.alias,table,self.alias,table,where)
							query=query[7:len(query)]
							self.parent._display_query_(query,title="Compute all the descriptive statistics for each category to draw the box plot")
							start_time = time.time()
							self.parent.cursor.execute(query)
							self.parent._display_time_(elapsed_time=time.time()-start_time)
							query_result=self.parent.cursor.fetchall()
						except:
							with_summarize=False
							query_result=[]
							for idx,category in enumerate(cat_priority):
								if (category==None or category=="None"):
									tmp_query=("select min({}) as min, (select percentile_cont(0.25) within group (order by {}) over () from {} limit 1)"
										+ " as Q1, (select percentile_cont(0.50) within group (order by {}) over () from {} limit 1) as Median, "
										+ "(select percentile_cont(0.75) within group (order by {}) over () from {} limit 1) as Q3, max({}) as max, 'None' "
										+ " from {} where {} is null group by Q1,Median,Q3")
									query=tmp_query.format(self.alias,self.alias,table,self.alias,table,self.alias,table,self.alias,table,by)
								else:
									tmp_query=("select min({}) as min, (select percentile_cont(0.25) within group (order by {}) over () from {} limit 1)"
										+ " as Q1, (select percentile_cont(0.50) within group (order by {}) over () from {} limit 1) as Median, "
										+ "(select percentile_cont(0.75) within group (order by {}) over () from {} limit 1) as Q3, max({}) as max, '{}' "
										+ " from {} where {}='{}' group by Q1,Median,Q3")
									query=tmp_query.format(self.alias,self.alias,table,self.alias,table,self.alias,table,self.alias,category,table,by,category)
								self.parent._display_query_(query,title="Compute manually all the descriptive statistics for each category to draw the box plot")
								start_time = time.time()
								self.parent.cursor.execute(query)
								query_result+=[self.parent.cursor.fetchone()]
							if (cardinality>max_cardinality):
								where=[]
								for item in cat_priority:
									where+=[by+"!='"+str(item)+"'"]
								where=" and ".join(where)
								tmp_query=("select min({}) as min, (select percentile_cont(0.25) within group (order by {}) over () from {} limit 1)"
									+ " as Q1, (select percentile_cont(0.50) within group (order by {}) over () from {} limit 1) as Median, "
									+ "(select percentile_cont(0.75) within group (order by {}) over () from {} limit 1) as Q3, max({}) as max, 'Others' "
									+ " from {} where {} group by Q1,Median,Q3")
								query=tmp_query.format(self.alias,self.alias,table,self.alias,table,self.alias,table,self.alias,table,where)
								self.parent._display_query_(query,title="Compute manually all the descriptive statistics for each category to draw the box plot")
								start_time = time.time()
								self.parent.cursor.execute(query)
								query_result+=[self.parent.cursor.fetchone()]
				cat_priority=[item[-1] for item in query_result]
				if (with_summarize):
					result=[[float(item[i]) for i in range(4,9)] for item in query_result]
				else:
					result=[[float(item[i]) for i in range(0,5)] for item in query_result]
				result.reverse()
				cat_priority.reverse()
				if (self.parent[by].category()=="text"):
					labels=[]
					for item in cat_priority:
						if (len(str(item))>18):
							labels+=[item[0:15] + "..."]
						else:
							labels+=[item]
				else:
					labels=cat_priority
				plt.figure(figsize=(7,6))
				plt.rcParams['axes.facecolor']='#F8F8F8'
				plt.ylabel(self.alias)
				plt.xlabel(by)
				plt.xticks(rotation=90)
				plt.gca().yaxis.grid()
				other_labels=[]
				other_result=[]
				all_idx=[]
				if not(is_categorical):
					for idx,item in enumerate(labels):
						try:
							math.floor(int(item))
						except:
							try:
								math.floor(float(item))
							except:
								try:
									math.floor(float(labels[idx][1:-1].split(';')[0]))
								except:
									other_labels+=[labels[idx]]
									other_result+=[result[idx]]
									all_idx+=[idx]
					for idx in all_idx:
						del labels[idx]
						del result[idx]
				if not(is_categorical):
					sorted_boxplot=sorted([[float(labels[i][1:-1].split(';')[0]),labels[i],result[i]] for i in range(len(labels))])
					labels=[item[1] for item in sorted_boxplot]+other_labels
					result=[item[2] for item in sorted_boxplot]+other_result
				else:
					sorted_boxplot=sorted([(labels[i],result[i]) for i in range(len(labels))])
					labels=[item[0] for item in sorted_boxplot]
					result=[item[1] for item in sorted_boxplot]
				box=plt.boxplot(result,notch=False,sym='',whis=np.inf,widths=0.5,labels=labels,patch_artist=True)
				plt.title('BoxPlot of {} group by {}'.format(self.alias,by))
				plt.subplots_adjust(bottom=max(0.3,len(max([str(item) for item in labels],key=len))/90.0))
				colors=['dodgerblue','seagreen','palevioletred','tan','pink','darksalmon','lightskyblue']
				colors+=['lightgreen','coral','indianred']
				colors=colors*5
				for median in box['medians']:
					median.set(color='black',linewidth=1,)
				for patch,color in zip(box['boxes'],colors):
					patch.set_facecolor(color)
				plt.show()
			except:
				print("/!\\ Warning: An error occured during the BoxPlot creation")
				raise
	# Return the RVC cardinality
	def cardinality(self):
		query="select count(distinct {}) from {}".format(self.alias,self.parent._table_transf_())
		self.parent._display_query_(query,title="Compute the feature "+self.alias+" cardinality")
		start_time = time.time()
		self.parent.cursor.execute(query)
		self.parent._display_time_(elapsed_time=time.time()-start_time)
		distinct_count=self.parent.cursor.fetchall()[0][0]
		return distinct_count
	# Return the current RVC category
	def category(self):
		return self.transformations[-1][2]
	# Try to convert a text variable to num
	def convert_to_num(self):
		if (self.category() in ["int","float"]):
			print("/!\\ Warning: "+self.alias+" is already numerical.\nNothing was done.")
		else:
			try:
				query="select {}+0 as {} from {} where {} is not null limit 20".format(
					self.alias,self.alias,self.parent._table_transf_(),self.alias)
				start_time = time.time()
				self.parent.cursor.execute(query)
				self.parent._display_time_(elapsed_time=time.time()-start_time)
				self.transformations+=[("{}+0","float","float")]
				self.parent.rvd_history+=["{"+time.strftime("%c")+"} "+"[Convert to Num]: The RVC '{}' was converted to num.".format(self.alias)]
			except:
				raise Exception("The type "+self.ctype()+" can not be converted to numerical")
	# Return the number of non-null elements in the RVC
	def count(self):
		query="select count({}) from {}".format(self.alias,self.parent._table_transf_())
		self.parent._display_query_(query,title="Compute the RVC '"+self.alias+"' number of non-missing elements")
		start_time = time.time()
		self.parent.cursor.execute(query)
		self.parent._display_time_(elapsed_time=time.time()-start_time)
		missing_data=self.parent.cursor.fetchall()[0][0]
		return missing_data
	# cos: RVC=cos(RVC)
	def cos(self):
		if (self.category() not in ['float','int']):
			raise Exception("The RVC must be numerical in order to apply cos")
		else:
			self.transformations+=[("cos({})","float","float")]
			self.parent.rvd_history+=["{"+time.strftime("%c")+"} "+"[Cos]: The RVC '{}' was transformed with the function 'cos'.".format(self.alias)]
	# cosh: RVC=cosh(RVC)
	def cosh(self):
		if (self.category() not in ['float','int']):
			raise Exception("ctype must be a numerical in order to apply cosh")
		else:
			self.transformations+=[("cosh({})","float","float")]
			self.parent.rvd_history+=["{"+time.strftime("%c")+"} "+"[Cosh]: The RVC '{}' was transformed with the function 'cosh'.".format(self.alias)]
	# cot: RVC=cot(RVC)
	def cot(self):
		if (self.category() not in ['float','int']):
			raise Exception("ctype must be a numerical in order to apply cos")
		else:
			self.transformations+=[("cot({})","float","float")]
			self.parent.rvd_history+=["{"+time.strftime("%c")+"} "+"[Cot]: The RVC '{}' was transformed with the function 'cot'.".format(self.alias)]
	# Return the current RVC type
	def ctype(self):
		return self.transformations[-1][1]
	# Extract the date part of the RVC
	def date_part(self,field="month"):
		if (self.category()=="date"):
			valid_values=["century","day","decade","doq","dow","doy","epoch","hour","isodow","isoweek",
							"isoyear","microseconds","millenium","milliseconds","minute","month","quarter",
							"second","timezone","timezone_hour","timezone_minute","week","year"]
			if (field not in valid_values):
				raise Exception("The parameter 'field' must be in "+"|".join(valid_values))
			else:
				self.transformations+=[("date_part('{}',{})".format(field,"{}"),"int","int")]
			self.parent.rvd_history+=["{"+time.strftime("%c")+"} "+"[Date Part]: The field '{}' was extracted from the RVC '{}'.".format(field,self.alias)]
		else:
			raise Exception("The RVC category must be a date in order to apply date_part")
	# Use the decode function to change the RVC categories into others
	def decode(self,categories,values,others=None):
		if ((type(categories)!=list) or (type(values)!=list) or (len(values)!=len(categories))):
			raise Exception("The parameters 'categories' and 'values' must be lists of the same size")
		else:
			# decode treatment
			if (others==None):
				others="NULL"
			else:
				others="'"+str(others)+"'"
			new_transformation="decode({},"
			for i in range(0,len(categories)):
				if (categories[i]==None) and (values[i]==None):
					new_transformation+="NULL,NULL,"
				elif (categories[i]==None):
					new_transformation+="NULL,'{}',".format(values[i])
				elif (values[i]==None):
					new_transformation+="'{}',NULL,".format(categories[i])
				else:
					new_transformation+="'{}','{}',".format(categories[i],values[i])
			new_transformation+="{})".format(others)
			rvd_history="{"+time.strftime("%c")+"} "+"[Decode]: The RVC '{}' was transformed with the function 'decode'.".format(self.alias)
			for i in range(len(categories)):
				rvd_history+="\n\t"+str(categories[i])+" => "+str(values[i])
			rvd_history+="\n\tothers => "+str(others)
			# new RVC type
			num_float=0
			num_int=0
			num_text=0
			for item in values:
				if (isinstance(item,int)):
					num_int+=1
				elif (isinstance(item,float)):
					num_float+=1
				else:
					try:
						int(item)
						num_int+=1
					except:
						try:
							float(item)
							num_float+=1
						except:
							num_text+=1
			if (num_text>0):
				ctype="varchar("+str(max(len(max([str(item) for item in values],key=len)),len(str(others))))+")"
				category="text"
			elif (num_float>0):
				ctype="float"
				category="float"
			elif (num_int>0):
				ctype="int"
				category="int"
			else:
				ctype="undefined"
				category="undefined"
			self.parent.rvd_history+=[rvd_history]
			self.transformations+=[(new_transformation,ctype,category)]
	# degrees: RVC=degrees(RVC)
	def degrees(self):
		if (self.category() not in ['float','int']):
			raise Exception("The RVC must be numerical in order to apply degrees")
		else:
			self.transformations+=[("degrees({})","float","float")]
			self.parent.rvd_history+=["{"+time.strftime("%c")+"} "+"[Degrees]: The RVC '{}' was transformed with the function 'degrees'.".format(self.alias)]
	# Draw the RVC density plot
	def density(self,a=None,kernel="gaussian",smooth=200,color=None):
		if (color==None):
			color=self.parent.colors[0]
		if (not(isinstance(smooth,int)) or (smooth<1)):
			raise TypeError("The parameter 'smooth' must be a strictly positive integer")
		if (not(isinstance(a,(float,int))) or (a<0)):
			if (not(isinstance(a,type(None)))):
				raise TypeError("The parameter 'a' must be a strictly positive number or null")
		if (kernel=="gaussian"):
			def fkernel(x):
				return math.exp(-1/2*((x)**2))/(math.sqrt(2*math.pi))
		elif (kernel=="logistic"):
			def fkernel(x):
				return 1/(2+math.exp(x)+math.exp(-x))
		elif (kernel=="sigmoid"):
			def fkernel(x):
				return 2/(math.pi*(math.exp(x)+math.exp(-x)))
		elif (kernel=="silverman"):
			def fkernel(x):
				return math.exp(-1/math.sqrt(2)*abs(x))/(2)*math.sin(abs(x)/math.sqrt(2)+math.pi/4)
		else:
			raise TypeError("The parameter 'kernel' must be in gaussian|logistic|sigmoid|silverman")
		if (a==None):
			a=1.06*self.std()/self.count()**(1.0/5.0)
		if (self.category() not in ["int","float"]):
			raise Exception("Cannot draw a density plot for non-numerical RVC")
		x,y,z,h,is_categorical=self._hist_(method="count",max_cardinality=1) 
		x=[item-h/2/0.94 for item in x]
		N=sum(y)
		y_smooth=[]
		x_smooth=[(max(x)-min(x))*i/smooth+min(x) for i in range(0,smooth+1)]
		n=len(y)
		for x0_smooth in x_smooth:
			K=sum([y[i]*fkernel(((x0_smooth-x[i])/a)**2)/(a*N) for i in range(0,len(x))])
			y_smooth+=[K]
		plt.figure(figsize=self.parent.figsize)
		plt.rcParams['axes.facecolor']='#F5F5F5'
		plt.plot(x_smooth,y_smooth,color="#222222")
		plt.xlim(min(x),max(x))
		plt.ylim(0,max(y_smooth)*1.1)
		plt.grid()
		plt.gca().set_axisbelow(True)
		plt.fill_between(x_smooth,y_smooth,facecolor=color,alpha=0.7)
		plt.ylabel("density")
		plt.title('Distribution of {} ({} kernel)'.format(self.alias,kernel))
		plt.show()
	# Generates descriptive statistics that summarize the Column
	# mode can be auto|categorical|numerical
	def describe(self,mode="auto",max_cardinality=6):
		if (not(isinstance(max_cardinality,int)) or (max_cardinality<1)):
			raise TypeError("The parameter 'max_cardinality' must be a strictly positive integer")
		if (mode not in ["auto","numerical","categorical"]):
			raise TypeError("The parameter 'mode' must be in auto|categorical|numerical")
		distinct_count=self.cardinality()
		is_numeric=((self.category()=="int") or (self.category()=="float"))
		is_date=(self.category()=="date")
		if ((is_date) and mode!="categorical"):
			query=("select (select count(*) from {} where {} is not null) as count,min({}) as min," +
						" max({}) as max from {} group by count;").format(
						self.parent._table_transf_(),self.alias,self.alias,self.alias,
						self.parent._table_transf_())
			self.parent._display_query_(query,title="Compute the descriptive statistics of "+self.alias)
			start_time = time.time()
			self.parent.cursor.execute(query)
			self.parent._display_time_(elapsed_time=time.time()-start_time)
			result=self.parent.cursor.fetchall()
			result=[item for sublist in result for item in sublist]
			index=['count','min','max']
		elif (((distinct_count < max_cardinality+1) and (mode!="numerical")) or not(is_numeric) or (mode=="categorical")):
			query="(select {}||'',count(*) from {} group by {} order by count(*) desc limit {})".format(
					self.alias,self.parent._table_transf_(),self.alias,max_cardinality)
			if (distinct_count > max_cardinality):
				query+=("union (select 'Others',sum(count) from (select count(*) as count from {} where {} is not null group by {}" +
							" order by count(*) desc offset {}) x) order by count desc").format(
							self.parent._table_transf_(),self.alias,self.alias,max_cardinality+1)
			self.parent._display_query_(query,title="Compute the descriptive statistics of "+self.alias)
			start_time = time.time()
			self.parent.cursor.execute(query)
			self.parent._display_time_(elapsed_time=time.time()-start_time)
			query_result=self.parent.cursor.fetchall()
			result=[item[1] for item in query_result]
			index=[item[0] for item in query_result]
			if (distinct_count > max_cardinality):
				index.append('cardinality')
				result.append(distinct_count)
		else:
			result=self._summarize_num_col_()
			result+=[distinct_count]
			index=['count','mean','std','min','25%','50%','75%','max','cardinality']
		return column_matrix(data_columns=[['']+index,['value']+result],table_info="Name: {},dtype: {}".format(self.alias,self.ctype()))
	# Return the list of distinct elements of the RVC
	def distinct(self):
		query="select {} from {} where {} is not null group by {} order by {}".format(
				self.alias,self.parent._table_transf_(),self.alias,self.alias,self.alias)
		start_time = time.time()
		self.parent.cursor.execute(query)
		self.parent._display_time_(elapsed_time=time.time()-start_time)
		query_result=self.parent.cursor.fetchall()
		return [item for sublist in query_result for item in sublist]
	# div: RVC=RVC/x
	def div(self,x):
		if (self.category() not in ['float','int']):
			raise Exception("The RVC must be numerical in order to apply div")
		elif (type(x) not in [int,float]):
			raise TypeError("The parameter 'x' must be numerical")
		elif (x==0):
			raise ZeroDivisionError("The parameter 'x' must be != 0")
		else:
			self.transformations+=[("{}/({})".format("{}",x),"float","float")]
			self.parent.rvd_history+=["{"+time.strftime("%c")+"} "+"[Add]: {} was added to the RVC '{}'.".format(x,self.alias)]
	# Draw the RVC donut chart
	def donut(self,method="density",of=None,max_cardinality=6,h=None,colors=['dodgerblue','seagreen','indianred','gold','tan','pink',
				'darksalmon','lightskyblue','lightgreen','palevioletred','coral']*10):
		return self.pie(method=method,of=of,max_cardinality=max_cardinality,h=h,colors=colors,donut=True)
	# Drop the RVC
	def drop_column(self,add_history=True):
		if not(isinstance(add_history,bool)):
			raise TypeError("The parameter 'add_history' must be a bool")
		self.parent.columns.remove(self.alias)
		delattr(self.parent,self.alias)
		if (add_history):
			print("RVC '{}' deleted from the RVD.".format(self.alias))
			self.parent.rvd_history+=["{"+time.strftime("%c")+"} "+"[Drop Column]: RVC '{}' was deleted from the RVD.".format(self.alias)]
	# Drop the NAs values of the column (adding a where clause to the RVD = equivalent to filter)
	def dropna(self):
		count=self.parent.count()
		self.parent.where+=[("{} is not null".format(self.alias),len(self.transformations)-1)]
		count=abs(count-self.parent.count())
		if (count>1):
			print("{} elements were dropped".format(count))
			self.parent.rvd_history+=["{"+time.strftime("%c")+"} "+"[Dropna]: The {} missing elements of column '{}' were dropped from the RVD.".format(count,self.alias)]
		elif (count==1):
			print("{} element was dropped".format(count))
			self.parent.rvd_history+=["{"+time.strftime("%c")+"} "+"[Dropna]: The only missing element of column '{}' was dropped from the RVD.".format(self.alias)]
		else:
			del self.parent.where[-1]
			print("/!\\ Warning: Nothing was dropped")
	# Return the RVC type
	def dtype(self):
		print("col".ljust(6) + self.ctype().rjust(12))
		print("dtype: object")
	# Duplicate the column
	def duplicate(self,name=None,add_history=True,print_info=True):
		if not(isinstance(add_history,bool)):
			raise TypeError("The parameter 'add_history' must be a bool")
		if (type(name)!=str):
			name=self.alias+str(np.random.randint(10000))
		new_rvc=RVC(name,parent=self.parent)
		new_rvc.transformations=self.transformations
		setattr(self.parent,name,new_rvc)
		self.parent.columns+=[name]
		if (print_info):
			print("The RVC '{}' was added to the RVD.".format(name))
		if (add_history):
			self.parent.rvd_history+=["{"+time.strftime("%c")+"} "+"[Duplicate]: Duplication of '{}' was added to the RVC with the name '{}'.".format(self.alias,name)]
		return name
	# Transform the column to enum
	def enum(self,h=None):
		if (self.category() not in ['float','int']):
			raise Exception("The RVC must be numerical in order to apply enum")
		else:
			self.transformations+=[self._enum_trans_(h)]
			self.parent.rvd_history+=["{"+time.strftime("%c")+"} "+"[Enum]: The RVC '{}' was converted to enum.".format(self.alias)]
	# exp: RVC=exp(RVC)
	def exp(self):
		if (self.category() not in ['float','int']):
			raise Exception("The RVC must be numerical in order to apply exp")
		else:
			self.transformations+=[("exp({})","float","float")]
			self.parent.rvd_history+=["{"+time.strftime("%c")+"} "+"[Exp]: The RVC '{}' was transformed with the function 'exp'.".format(self.alias)]
	# Impute missing values
	def fillna(self,val=None,method=None,by=[],compute_before=True):
		if not(isinstance(compute_before,bool)):
			raise TypeError("The parameter 'compute_before' must be a bool")
		total=self.count()
		if (val!=None):
			new_column="coalesce({},{})".format("{}",val)
		elif (method!=None):
			if ((method=="mean") or (method=="avg")):
				fun="avg"
			elif ((method=="median") or (method=="lead") or (method=="lag")):
				fun=method
			else:
				raise Exception("The method '{}' does not exist or is not available.".format(
							method) + "\nPlease use a method in mean|median|lead|lag.\nNothing was transformed.")
			if ((by==[]) and compute_before):
				if (fun=="avg"):
					new_column="coalesce({},{}".format("{}",self.mean())
				elif (fun=="median"):
					new_column="coalesce({},{}".format("{}",self.median())
			else:
				new_column="coalesce({},{}({}) over (".format("{}",fun,"{}")
			if ((by!=[]) and (method!="lead") and (method!="lag")):
				exist=True
				partition_by=[]
				columns=self.parent.columns
				for column in by:
					if not(column in columns):
						exist=False
						false_column=column
						break
				if exist:
					new_column+="partition by "
					new_column+=",".join(by)
					new_column+="))"
				else:
					raise Exception("column '{}' is not in the RVD.\nNothing was transformed.".format(
							false_column))
			elif ((by!=[]) and ((method=="lead") or (method=="lag"))):
				exist=True
				order_by=[]
				columns=self.parent.columns
				for column in by:
					if not(column in columns):
						exist=False
						false_column=column
				if exist:
					new_column+="order by "
					new_column+=",".join(by)
					new_column+="))"
				else:
					raise Exception("column '{}' is not in the RVD.\nNothing was transformed.".format(
							false_column))
			elif ((by==[]) and ((method=="lead") or (method=="lag"))):
				raise Exception("'method' lead|lag must contain an order by clause.\nNothing was transformed.")
			else:
				if ((by==[]) and compute_before):
					new_column+=")"
				else:
					new_column+="))"
		try:
			self.transformations+=[(new_column,"float","float")]
			total=abs(self.count()-total)
			if (total>1):
				print("{} elements were filled".format(total))
				self.parent.rvd_history+=["{"+time.strftime("%c")+"} "+"[Fillna]: {} missing values of the RVC '{}' were filled.".format(
										total,self.alias)]
			else:
				print("{} element was filled".format(total))
				self.parent.rvd_history+=["{"+time.strftime("%c")+"} "+"[Fillna]: {} missing value of the RVC '{}' was filled.".format(
										total,self.alias)]
		except:
			raise TypeError("The method 'fillna' takes at least one argument (val or method)")
	# return the final transformation
	def final_transformation(self):
		all_transformations_grammar=[item[0] for item in self.transformations]
		all_transformations_grammar.reverse()
		ftrans=all_transformations_grammar[0]
		for i in range(1,len(all_transformations_grammar)):
			ftrans=ftrans.format(all_transformations_grammar[i])
		return ftrans
	# floor: RVC=floor(RVC)
	def floor(self):
		if (self.category() not in ['float']):
			raise Exception("The RVC type must be float in order to apply floor")
		else:
			self.transformations+=[("floor({})","int","int")]
			self.parent.rvd_history+=["{"+time.strftime("%c")+"} "+"[Floor]: The RVC '{}' was transformed with the function 'floor'.".format(self.alias)]
	# Print the first n rows of the RVC
	def head(self,n=5):
		print(self.__repr__(limit=n))
	# Draw the RVC histogram
	def hist(self,method="density",of=None,max_cardinality=6,bins=None,h=None,color=None):
		if (color==None):
			color=self.parent.colors[0]
		[x,y,z,h,is_categorical]=self._hist_(method=method,of=of,max_cardinality=max_cardinality,bins=bins,h=h)
		is_numeric=(self.category()=="float") or (self.category()=="int")
		if ((is_numeric) and not(is_categorical)):
			rotation=0
		else:
			rotation=90
		plt.figure(figsize=self.parent.figsize,facecolor='white')
		plt.rcParams['axes.facecolor']='#F5F5F5'
		plt.bar(x,y,h,color=color,alpha=0.86)
		plt.xlabel(self.alias)
		plt.gca().set_axisbelow(True)
		plt.gca().yaxis.grid()
		if (is_categorical):
			if not(is_numeric):
				new_z=[]
				for item in z:
					if (len(str(item))>20):
						new_z+=[item[0:17] + "..."]
					else:
						new_z+=[item]
			else:
				new_z=z
			plt.xticks(x,new_z,rotation=rotation)
			plt.subplots_adjust(bottom=max(0.3,len(max([str(item) for item in z],key=len))/140.0))
		if (method=="density"):
			plt.ylabel('Density')
			plt.title('Distribution of {}'.format(self.alias))
		elif ((method in ["avg","min","max","sum"]) and (of!=None)):
			plt.ylabel(aggregate)
			plt.title('{} group by {}'.format(aggregate,self.alias))
		else:
			plt.ylabel('Frequency')
			plt.title('Count by {}'.format(self.alias))
		plt.show()
	# Use the label encode to encode the variable
	def label_encode(self,cat_priority=None,values=None,others=None,order_by="count",force_encoding=False,show=True):
		if not(isinstance(force_encoding,bool)):
			raise TypeError("The parameter 'force_encoding' must be a bool")
		if not(isinstance(show,bool)):
			raise TypeError("The parameter 'show' must be a bool")
		if ((self.category()=="text") or force_encoding):
			if (order_by not in ["count"]):
				order_by=""
			else:
				order_by="order by count(*)"
			if (type(cat_priority)!=list):
				query="select {} from {} group by {} {}".format(
						self.alias,self.parent._table_transf_(),self.alias,order_by)
				self.parent._display_query_(query,title="Compute all the categories of feature "+self.alias)
				start_time=time.time()
				self.parent.cursor.execute(query)
				self.parent._display_time_(elapsed_time=time.time()-start_time)
				query_result=self.parent.cursor.fetchall()
				cat_priority=[item for sublist in query_result for item in sublist]
				for idx,item in enumerate(cat_priority):
					if (item==None):
						del cat_priority[idx]
						break
				values=range(0,len(cat_priority))
				others="NULL"
			if (len(cat_priority)!=len(values)):
				print("/!\\ Warning: 'cat_priority' and 'values' must have the same length.")
				print("'values' will be choose as an increase sequence of integers")
				values=range(0,len(cat_priority))
			if (others==None):
				others="NULL"
			new_transformation=[]
			for i in range(0,len(values)):
				if (cat_priority[i]==None):
					new_transformation+=["NULL",str(values[i])]
				else:
					new_transformation+=["'"+cat_priority[i].replace("'","''")+"'",str(values[i])]
			new_transformation="decode({},"+",".join(new_transformation)+","+str(others)+")"
			try:
				all_transformations_grammar=[item[0] for item in self.transformations]
				all_transformations_grammar.reverse()
				new_transformation_test=new_transformation
				for item in all_transformations_grammar:
					new_transformation_test=new_transformation_test.replace("{}",item)
				query="select "+new_transformation_test+" as "+self.alias+" from "+self.parent.input_relation+" limit 10"
				self.parent.cursor.execute(query)
				query_result=self.parent.cursor.fetchall()
				self.transformations+=[(new_transformation,"int","int")]
				rvd_history="{"+time.strftime("%c")+"} "+"[Label Encode]: The RVC '{}' was transformed with the 'label encoding'.".format(self.alias)
				for i in range(len(cat_priority)):
					rvd_history+="\n\t"+str(cat_priority[i])+" => "+str(values[i])
				rvd_history+="\n\tothers => "+str(others)
				self.parent.rvd_history+=[rvd_history]
			except:
				raise Exception("The label encoding failed.\nMaybe one of the elements inside one of the two lists 'cat_priority' and 'values' is not correct. Nothing was changed.")
			if (show):
				formatted_text=print_table([[self.alias]+cat_priority,['encoding']+list(values)],repeat_first_column=True,first_element=self.alias)[0:-2]
				if not(isnotebook()):
					print(formatted_text)
			print("The label encoding was successfully done.")
		else:
			print("/!\\ Warning: 'label_encoding' is only for variables of type string.")
			print("You can force the encoding using the 'force_encoding' variable")
	# log: RVC=log(RVC)
	def log(self):
		if (self.category() not in ['float','int']):
			raise Exception("The RVC must be numerical in order to apply log")
		else:
			self.transformations+=[("log({})","float","float")]
			self.parent.rvd_history+=["{"+time.strftime("%c")+"} "+"[Log]: The RVC '{}' was transformed with the function 'log'.".format(self.alias)]
	# Return the RVC max
	def max(self):
		query="select max({}) from {}".format(
			self.alias,self.parent._table_transf_())
		self.parent._display_query_(query,title="Compute the max of "+self.alias)
		start_time = time.time()
		self.parent.cursor.execute(query)
		self.parent._display_time_(elapsed_time=time.time()-start_time)
		query_result=self.parent.cursor.fetchone()
		return query_result[0]
	# Return the RVC mean
	def mean(self):
		query="select avg({}) from {};".format(
				self.alias,self.parent._table_transf_())
		self.parent._display_query_(query,title="Compute the avg of "+self.alias)
		start_time = time.time()
		self.parent.cursor.execute(query)
		self.parent._display_time_(elapsed_time=time.time()-start_time)
		mean=self.parent.cursor.fetchall()[0][0]
		return mean
	# Use a mean encode according to a response column
	def mean_encode(self,response_column):
		if (response_column not in self.parent.columns):
			raise Exception("The response RVC must be inside the parent RVD to use a mean encoding")
		elif (self.parent[response_column].category() not in ["int","float"]):
			raise Exception("The response RVC must be numerical to use a mean encoding")
		else:
			self.transformations+=[("avg({}) over (partition by {})".format(response_column,"{}"),"int","float")]
			self.parent.rvd_history+=["{"+time.strftime("%c")+"} "+"[Mean Encode]: The RVC '{}' was transformed using a mean encoding with as response column '{}'.".format(
											self.alias,response_column)]
			print("The mean encoding was successfully done.")
	# Return the RVC median
	def median(self):
		return self.percentile_cont(0.5)
	# Return the RVC min
	def min(self):
		query="select min({}) from {}".format(
			self.alias,self.parent._table_transf_())
		self.parent._display_query_(query,title="Compute the min of "+self.alias)
		start_time = time.time()
		self.parent.cursor.execute(query)
		self.parent._display_time_(elapsed_time=time.time()-start_time)
		query_result=self.parent.cursor.fetchone()
		return query_result[0]
	# mod: RVC=RVC%n
	def mod(self,n):
		if (self.category() not in ['float','int']):
			raise Exception("The RVC must be numerical in order to apply mod")
		else:
			if (type(n) in [int,float]):
				self.transformations+=[("mod({},{})".format("{}",n),self.ctype(),self.category())]
				self.parent.rvd_history+=["{"+time.strftime("%c")+"} "+"[Mod{}]: The RVC '{}' was transformed with the function 'mod{}'.".format(n,self.alias,n)]
			else:
				raise TypeError("The parameter 'n' must be numerical in order to apply mod")
	# mult: RVC=RVC*x
	def mult(self,x):
		if (self.category() not in ['float','int']):
			raise Exception("The RVC must be numerical in order to apply mult")
		elif (type(x) not in [int,float]):
			raise TypeError("The parameter 'x' must be numerical in order to apply mult")
		else:
			self.transformations+=[("{}*({})".format("{}",x),self.ctype(),self.category())]
			self.parent.rvd_history+=["{"+time.strftime("%c")+"} "+"[Mult]: The RVC '{}' was multiplied by {}.".format(self.alias,x)]
	# Normalize the RVC 
	def normalize(self,method="zscore",compute_before=True):
		if not(isinstance(compute_before,bool)):
			raise TypeError("The parameter 'compute_before' must be a bool")
		if (self.category() in ["int","float"]):
			if (method=="zscore"):
				if (compute_before):
					query="select avg("+self.alias+"),stddev("+self.alias+") from "+self.parent._table_transf_()
					self.parent._display_query_(query,title="Compute the avg and stddev of "+self.alias+" for normalization")
					start_time = time.time()
					self.parent.cursor.execute(query)
					self.parent._display_time_(elapsed_time=time.time()-start_time)
					avg,stddev=self.parent.cursor.fetchone()
					self.transformations+=[("({}-{})/({})".format("{}",avg,stddev),"float","float")]
				else:
					self.transformations+=[("({}-avg({}) over())/(stddev({}) over ())","float","float")]
			elif (method=="robust_zscore"):
				if (compute_before):
					query="select avg("+self.alias+"),stddev("+self.alias+") from "+self.parent._table_transf_()
					self.parent._display_query_(query,title="Compute the avg and stddev of "+self.alias+" for normalization")
					start_time = time.time()
					self.parent.cursor.execute(query)
					self.parent._display_time_(elapsed_time=time.time()-start_time)
					avg,stddev=self.parent.cursor.fetchone()
					transformations+=[("abs({}-{})/({})".format("{}",avg,stddev),"float","float")]
				else:
					transformations+=[("abs({}-avg({}) over())/(stddev({}) over ())","float","float")]
			elif (method=="minmax"):
				if (compute_before):
					query="select min("+self.alias+"),max("+self.alias+") from "+self.parent._table_transf_()
					self.parent._display_query_(query,title="Compute the min and max of "+self.alias+" for normalization")
					start_time = time.time()
					self.parent.cursor.execute(query)
					self.parent._display_time_(elapsed_time=time.time()-start_time)
					cmin,cmax=self.parent.cursor.fetchone()
					transformations+=[("({}-{})/({}-{})".format("{}",cmin,cmax,cmin),"float","float")]
				else:
					transformations+=[("({}-min({}) over ())/(max({}) over ()-min({}) over ())","float","float")]
			else:
				raise Exception("The method '{}' doesn't exist.\nPlease use a method in zscore|robust_zscore|minmax".format(
					method))
			print("The RVC '"+self.alias+"' was successfully normalized.")
			self.parent.rvd_history+=["{"+time.strftime("%c")+"} "+"[Normalize]: The RVC '{}' was normalized with the method '{}'.".format(self.alias,method)]
		else:
			raise Exception("The RVC must be numerical for Normalization.\nNothing changed.")
	# Add the one hot encoder RVC in the RVD
	def one_hot_encoder(self):
		if (self.category() in ["date","float"]):
			print("/!\\ Warning: one_hot_encoder is not available for date or float.")
			print("Please convert the column to enum using the enumerate function.")
		elif (self.cardinality()<3):
			print("/!\\ Warning: The column has already a limited number of elements.")
			print("Please use the label_encode function in order to code the elements if it is needed.")
		else:
			distinct_elements=self.distinct()
			all_new_features=[]
			for element in distinct_elements:
				# forbidden characters
				alias=self.alias+"_"+str(element)
				alias=alias.replace("-","_")
				alias=alias.replace("+","_")
				alias=alias.replace("/","_")
				alias=alias.replace(".","_")
				alias=alias.replace("=","_")
				alias=alias.replace(" ","_")
				transformations=self.transformations+[("decode({},'{}',1,0)".format("{}",element),"int","int")]
				self.duplicate(name=alias,add_history=False,print_info=False)
				self.parent[alias].transformations=transformations
				all_new_features+=[alias]
			print("{} new features: {}".format(len(all_new_features),", ".join(all_new_features)))
			self.parent.rvd_history+=["{"+time.strftime("%c")+"} "+"[One Hot Encoder]: One hot encoder was applied to the RVC '{}' and {} features were created: {}".format(
										self.alias,len(all_new_features),", ".join(all_new_features))+"."]
	# Return the RVC outliers
	def outliers(self,max_number=10,threshold=3):
		if (not(isinstance(max_number,int)) or (max_number<1)):
			raise TypeError("The parameter 'max_number' must be a strictly positive integer")
		if (not(isinstance(threshold,(int,float))) or (threshold<=0)):
			raise TypeError("The parameter 'threshold' must be a strictly positive number")
		if (self.category() in ["int","float"]):
			query=("select * from (select {},({}-(avg({}) over ()))/(stddev({}) over ()) as normalized from {}) y " + 
						" where abs(normalized)>{} order by abs(normalized) desc limit {}")
			query=query.format(self.alias,self.alias,self.alias,self.alias,
						self.parent._table_transf_(),threshold,max_number)
			self.parent._display_query_(query, title="Compute the feature "+self.alias+" outliers")
			start_time = time.time()
			self.parent.cursor.execute(query)
			self.parent._display_time_(elapsed_time=time.time()-start_time)
			result=self.parent.cursor.fetchall()
			if (result==[]):
				print("No outlier detected.")
				return []
			values=[self.alias]+[item[0] for item in result]
			normalized_values=["normalized_"+self.alias]+[item[1] for item in result]
			result=[values,normalized_values]
			outliers_count=len(values)-1
			if (len(values)==11):
				is_finished=False
			else:
				is_finished=True
			formatted_text=print_table(result,is_finished=is_finished) + "Total of {} outliers detected.".format(outliers_count)
			if not(isnotebook()):
				print(formatted_text)
			else:
				print("Total of {} outliers detected.".format(outliers_count))
		else:
			raise Exception("The method 'outliers' is only possible with numerical RVC")
	# Return the percentile_cont of the RVC
	def percentile_cont(self,x):
		if (not(isinstance(x,(int,float))) or (x<0) or (x>0)):
			raise TypeError("The parameter 'x' must be in [0,1]")
		query="select percentile_cont({}) within group (order by {}) over () from {} limit 1".format(
			x,self.alias,self.parent._table_transf_())
		self.parent._display_query_(query,title="Compute the percentile cont of "+self.alias)
		start_time = time.time()
		self.parent.cursor.execute(query)
		self.parent._display_time_(elapsed_time=time.time()-start_time)
		query_result=self.parent.cursor.fetchone()
		return query_result[0]
	# Draw the RVC pie chart
	def pie(self,method="density",of=None,max_cardinality=6,h=None,colors=['dodgerblue','seagreen','indianred','gold','tan','pink',
				'darksalmon','lightskyblue','lightgreen','palevioletred','coral']*10,donut=False):
		if not(isinstance(donut,bool)):
			raise TypeError("The parameter 'donut' must be a bool")
		bbox_to_anchor,ncol,loc=self.parent._legend_loc_init_()
		count=self.parent.count()
		cardinality=self.cardinality()
		is_numeric=(self.category()=="float") or (self.category()=="int")
		is_categorical=(cardinality<=max_cardinality) or not(is_numeric)
		x,y,z,h,is_categorical=self._hist_(max_cardinality=max_cardinality,method=method,of=of,pie=True)
		z.reverse()
		y.reverse()
		explode=[0 for i in y]
		explode[np.argmax(y)]=0.13
		current_explode=0.15
		total_count=sum(y)
		for idx,item in enumerate(y):
			if ((item<0.05) or ((item>1) and (item/float(total_count)<0.05))):
				current_explode=min(0.9,current_explode*1.4) 
				explode[idx]=current_explode
		if (method=="density"):
			autopct='%1.1f%%'
		else:
			def make_autopct(values,category):
			    def my_autopct(pct):
			        total=sum(values)
			        val=pct*total/100.0
			        if (category=="int"):
			        	val=int(round(val))
			        	return '{v:d}'.format(v=val)
			        else:
			        	return '{v:f}'.format(v=val)
			    return my_autopct
			if ((method in ["sum","count"]) or ((method in ["min","max"]) and (self.parent[of].category=="int"))):
				category="int"
			else:
				category=None
			autopct=make_autopct(y,category)
		if (donut):
			plt.figure(figsize=self.parent.figsize)
			explode=None
			centre_circle=plt.Circle((0,0),0.72,color='#666666',fc='white',linewidth=1.25)
			fig=plt.gcf()
			fig.gca().add_artist(centre_circle)
		else:
			plt.figure(figsize=self.parent.figsize)
		plt.pie(y,labels=z,autopct=autopct,colors=colors,shadow=True,startangle=290,explode=explode)
		plt.legend(bbox_to_anchor=bbox_to_anchor,ncol=ncol,loc=loc)
		plt.subplots_adjust(bottom=0.2)
		if (method=="density"):
			plt.title('Distribution of {}'.format(self.alias))
		elif ((method in ["avg","min","max","sum"]) and (of!=None)):
			aggregate="{}({})".format(method,of)
			plt.title('{} group by {}'.format(aggregate,self.alias))
		else:
			plt.title('Count by {}'.format(self.alias))
		plt.show()
	# pow: RVC=RVC^n
	def pow(self,n):
		if (self.category() not in ['float','int']):
			raise Exception("The RVC must be numerical in order to apply pow")
		else:
			if (type(n) in [int,float]):
				if (((type(n)==int) and (n>=0)) or (self.category()=="float")):
					ctype=self.ctype()
					category=self.category()
				else:
					ctype="float"
					category="float"
				self.transformations+=[("power({},{})".format("{}",n),ctype,category)]
				self.parent.rvd_history+=["{"+time.strftime("%c")+"} "+"[Pow{}]: Pow{} was applied on the RVC '{}'.".format(
											n,n,self.alias)]
			else:
				raise TypeError("The parameter 'n' must be numerical")
	# Print the RVC n first rows
	def print_rows(self,n=30):
		print(self.__repr__(limit=n))
	# radians: RVC=radians(RVC)
	def radians(self):
		if (self.category() not in ['float','int']):
			raise Exception("The RVC must be numerical in order to apply radians")
		else:
			self.transformations+=[("radians({})","float","float")]
			self.parent.rvd_history+=["{"+time.strftime("%c")+"} "+"[Radians]: The RVC '{}' was transformed with the function 'radians'.".format(self.alias)]
	# regexp_substr: RVC=regexp_substr(RVC)
	def regexp_substr(self,expression):
		if (self.category() not in ["text"]):
			raise Exception("The RVC type must be varchar in order to apply regexp_substr")
		elif (type(expression)!=str):
			raise TypeError("The parameter 'expression' must be a string")
		else:
			self.transformations+=[("regexp_substr({},'{}')".format("{}",expression),self.ctype(),self.category())]
			self.parent.rvd_history+=["{"+time.strftime("%c")+"} "+"[Regexp Substr]: '{}' was extracted from the RVC '{}'.".format(
										expression,self.alias)]
	# Change the RVC alias
	def rename(self,new_name):
		if (type(new_name)!=str):
			raise TypeError("The parameter 'new_name' must be a string")
		new_rvc=RVC(new_name,parent=self.parent)
		new_rvc.transformations=self.transformations
		setattr(self.parent,new_name,new_rvc)
		self.parent.columns+=[new_name]
		self.drop_column(add_history=False)
		self.parent.rvd_history+=["{"+time.strftime("%c")+"} "+"[Rename]: The RVC '{}' was renamed '{}'.".format(
										self.alias,new_name)]
		print("RVC '{}' renamed '{}'".format(self.alias,new_name))
	# round: RVC=round(RVC,n)
	def round(self,n):
		if (self.category() not in ['float','int']):
			raise Exception("The RVC must be numerical in order to apply round")
		elif (type(n) not in [int]):
			raise TypeError("The parameter 'n' must be an integer")
		elif (n<0):
			raise TypeError("The parameter 'n' must be positive")
		else:
			self.transformations+=[("round({},{})".format("{}",n),self.ctype(),self.category())]
			self.parent.rvd_history+=["{"+time.strftime("%c")+"} "+"[Round{}]: 'round' was apply on the RVC '{}'.".format(n,self.alias)]
	# sign: RVC=sign(RVC)
	def sign(self):
		if (self.category() not in ['float','int']):
			raise Exception("The RVC must be numerical in order to apply sign")
		else:
			self.transformations+=[("sign({})","float","float")]
			self.parent.rvd_history+=["{"+time.strftime("%c")+"} "+"[Sign]: The RVC '{}' was transformed with the function 'sign'.".format(self.alias)]
	# sin: RVC=sin(RVC)
	def sin(self):
		if (self.category() not in ['float','int']):
			raise Exception("The RVC must be numerical in order to apply sin")
		else:
			self.transformations+=[("sin({})","float","float")]
			self.parent.rvd_history+=["{"+time.strftime("%c")+"} "+"[Sin]: The RVC '{}' was transformed with the function 'sin'.".format(self.alias)]
	# sinh: RVC=sinh(RVC)
	def sinh(self):
		if (self.category() not in ['float','int']):
			raise Exception("The RVC must be numerical in order to apply sinh")
		else:
			self.transformations+=[("sinh({})","float","float")]
			self.parent.rvd_history+=["{"+time.strftime("%c")+"} "+"[Sinh]: The RVC '{}' was transformed with the function 'sinh'.".format(self.alias)]
	# sqrt: RVC=sqrt(RVC)
	def sqrt(self):
		if (self.category() not in ['float','int']):
			raise Exception("The RVC must be numerical in order to apply sqrt")
		else:
			self.transformations+=[("sqrt({})","float","float")]
			self.parent.rvd_history+=["{"+time.strftime("%c")+"} "+"[Sqrt]: The RVC '{}' was transformed with the function 'sqrt'.".format(self.alias)]
	# Return the RVC stddev
	def std(self):
		query="select stddev({}) from {}".format(
			self.alias,self.parent._table_transf_())
		self.parent._display_query_(query,title="Compute the stddev of "+self.alias)
		start_time = time.time()
		self.parent.cursor.execute(query)
		self.parent._display_time_(elapsed_time=time.time()-start_time)
		query_result=self.parent.cursor.fetchone()
		return query_result[0]
	# sub: RVC=RVC-x
	def sub(self,x):
		if (self.category() not in ['float','int']):
			raise Exception("The RVC must be numerical in order to apply sub")
		elif (type(x) not in [int,float]):
			raise TypeError("The parameter 'x' must be numerical")
		elif (x<0):
			raise TypeError("The parameter 'x' must be positive")
		else:
			if (type(x)==float):
				self.transformations+=[("{}-{}".format("{}",x),"float","float")]
			else:
				self.transformations+=[("{}-{}".format("{}",x),self.ctype(),self.category())]
			self.parent.rvd_history+=["{"+time.strftime("%c")+"} "+"[Sub]: {} was substracted from the RVC '{}'.".format(x,self.alias)]
	# tan: RVC=tan(RVC)
	def tan(self):
		if (self.category() not in ['float','int']):
			raise Exception("The RVC must be numerical in order to apply tan")
		else:
			self.transformations+=[("tan({})","float","float")]
			self.parent.rvd_history+=["{"+time.strftime("%c")+"} "+"[Tan]: The RVC '{}' was transformed with the function 'tan'.".format(self.alias)]
	# tanh: RVC=tanh(RVC)
	def tanh(self):
		if (self.category() not in ['float','int']):
			raise Exception("The RVC must be numerical in order to apply tanh")
		else:
			self.transformations+=[("tanh({})","float","float")]
			self.parent.rvd_history+=["{"+time.strftime("%c")+"} "+"[Tanh]: The RVC '{}' was transformed with the function 'tanh'.".format(self.alias)]
	# Undo the imputation
	def undo_impute(self):
		if (len(self.transformations)>1):
			self.parent.rvd_history+=["{"+time.strftime("%c")+"} "+"[Undo Impute]: Last Imputation of RVC '{}' was canceled.".format(
										self.alias)]
			del self.transformations[-1]
		else:
			print("/!\\ Warning: Nothing to undo")
	# Generates the RVC count per category
	def value_counts(self,max_cardinality=30):
		return self.describe(mode="categorical",max_cardinality=max_cardinality)









