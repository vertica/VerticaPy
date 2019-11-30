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
#
# AUTHOR: BADR OUALI
#
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
# Vertica-ML-Python allows user to create Virtual Dataframe. vDataframes simplify   #
# data exploration,   data cleaning   and   machine   learning   in    Vertica.     #
# It is an object which keeps in it all the actions that the user wants to achieve  # 
# and execute them when they are needed.    										#
#																					#
# The purpose is to bring the logic to the data and not the opposite                #
#####################################################################################
#
# Libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as plt_colors
from mpl_toolkits.mplot3d import Axes3D
import math
from vertica_ml_python.utilities import isnotebook
from vertica_ml_python.utilities import tablesample
from vertica_ml_python.utilities import to_tablesample
from random import shuffle
##
#   /$$$$$$$  /$$        /$$$$$$  /$$$$$$$$
#  | $$__  $$| $$       /$$__  $$|__  $$__/
#  | $$  \ $$| $$      | $$  \ $$   | $$   
#  | $$$$$$$/| $$      | $$  | $$   | $$   
#  | $$____/ | $$      | $$  | $$   | $$   
#  | $$      | $$      | $$  | $$   | $$   
#  | $$      | $$$$$$$$|  $$$$$$/   | $$   
#  |__/      |________/ \______/    |__/   
##
#
#
def bar(vdf,
		method: str = "density",
		of = None,
		max_cardinality: int = 6,
		bins: int = 0,
		h: float = 0,
		color: str = '#214579'):
	x, y, z, h, is_categorical = compute_plot_variables(vdf, method = method, of = of, max_cardinality = max_cardinality, bins = bins, h = h)
	plt.figure(figsize = (8,10))
	plt.rcParams['axes.facecolor'] = '#F5F5F5'
	plt.barh(x, y, h, color = color, alpha = 0.86)
	plt.ylabel(vdf.alias)
	plt.gca().xaxis.grid()
	plt.gca().set_axisbelow(True)
	if (is_categorical):
		if (vdf.category() == "text"):
			new_z = []
			for item in z:
				new_z += [item[0:47] + "..."] if (len(str(item)) > 50) else [item]
		else:
			new_z = z
		plt.yticks(x,new_z)
		plt.subplots_adjust(left = max(0.1, min(len(max([str(item) for item in z], key = len)), 20) / 80.0))
	if (method == "density"):
		plt.xlabel('Density')
		plt.title('Distribution of {}'.format(vdf.alias))
	elif ((method in ["avg","min","max","sum"]) and (of != None)):
		aggregate="{}({})".format(method.upper(), of)
		plt.ylabel(aggregate)
		plt.title('{} group by {}'.format(aggregate, vdf.alias))
	else:
		plt.xlabel('Frequency')
		plt.title('Count by {}'.format(vdf.alias))
	plt.show()
#
def bar2D(vdf,
		  columns: list,
		  method: str = "density",
		  of: str = "",
		  max_cardinality: tuple = (6,6),
		  h: tuple = (None, None),
		  limit_distinct_elements: int = 200,
		  stacked: bool = False,
		  fully_stacked: bool = False):
	colors = gen_colors()
	all_columns = vdf.pivot_table(columns, method = method, of = of, h = h, max_cardinality = max_cardinality, show = False, limit_distinct_elements = limit_distinct_elements).values
	all_columns = [[column] + all_columns[column] for column in all_columns]
	plt.figure(figsize = (8,10))
	plt.rcParams['axes.facecolor'] = '#F5F5F5'
	n = len(all_columns)
	m = len(all_columns[0])
	n_groups = m - 1
	index = np.arange(n_groups)
	bar_width = 0.5
	if not(fully_stacked):
		for i in range(1, n):
			current_column = all_columns[i][1:m]
			for idx,item in enumerate(current_column):
				try:
					current_column[idx] = float(item)
				except:
					current_column[idx] = 0
			current_label = str(all_columns[i][0])
			if (stacked):
				if (i == 1):
					last_column = [0 for item in all_columns[i][1:m]]
				else:
					for idx, item in enumerate(all_columns[i - 1][1:m]):
						try:
							last_column[idx] += float(item)
						except:
							last_column[idx] += 0
				plt.barh(index, current_column, bar_width, alpha = 0.86, color = colors[i-1], label = current_label, left = last_column)
			else:
				plt.barh(index + (i - 1) * bar_width / (n - 1), current_column, bar_width / (n - 1),alpha=0.86, color=colors[i-1], label = current_label)
		if (stacked):
			plt.yticks(index, all_columns[0][1:m])
		else:
			plt.yticks(index + bar_width / 2 - bar_width / 2 / (n - 1),all_columns[0][1:m])
		plt.subplots_adjust(left = max(0.3, len(max([str(item) for item in all_columns[0][1:m]], key=len)) / 140.0))
		plt.ylabel(columns[0])
		if (method == "mean"):
			method = "avg"
		if (method == "density"):
			plt.xlabel('Density')
			plt.title('Distribution of {} group by {}'.format(columns[0], columns[1]))
		elif ((method in ["avg","min","max","sum"]) and (of != None)):
			plt.xlabel("{}({})".format(method, of))
			plt.title('{}({}) of {} group by {}'.format(method, of, columns[0], columns[1]))
		else:
			plt.xlabel('Frequency')
			plt.title('Count by {} group by {}'.format(columns[0], columns[1]))
	else:
		total=[0 for item in range(1, m)]
		for i in range(1,n):
			for j in range(1, m):
				if not(type(all_columns[i][j]) in [str]):
					total[j-1] += float(all_columns[i][j])
		for i in range(1,n):
			for j in range(1,m):
				if not(type(all_columns[i][j]) in [str]):
					all_columns[i][j] = float(all_columns[i][j]) / total[j - 1]
		for i in range(1,n):
			current_column = all_columns[i][1:m]
			for idx,item in enumerate(current_column):
				try:
					current_column[idx] = float(item)
				except:
					current_column[idx] = 0
			current_label = str(all_columns[i][0])
			if (i == 1):
				last_column = [0 for item in all_columns[i][1:m]]
			else:
				for idx,item in enumerate(all_columns[i - 1][1:m]):
					try:
						last_column[idx] += float(item)
					except:
						last_column[idx] += 0
			plt.barh(index, current_column, bar_width, alpha=0.86, color = colors[i-1], label = current_label, left = last_column)
		plt.yticks(index,all_columns[0][1:m])
		plt.subplots_adjust(left = max(0.3, len(max([str(item) for item in all_columns[0][1:m]], key = len)) / 140.0))
		plt.ylabel(columns[0])
		plt.xlabel('Density per category')
		plt.title('Distribution per category of {} group by {}'.format(columns[0], columns[1]))
	plt.legend(title = columns[1])
	plt.gca().set_axisbelow(True)
	plt.gca().xaxis.grid()
	plt.show()
#
def boxplot(vdf, 
			by: str = "", 
			h: float = 0, 
			max_cardinality: int = 8, 
			cat_priority: list = []):
	plt.figure(figsize = (10,8))
	# SINGLE BOXPLOT
	if (by == ""):
		if not(vdf.isnum()):
			raise TypeError("The column must be numerical in order to draw a boxplot")
		summarize = vdf.summarize_numcol().values["value"]
		for i in range(0,2):
			del summarize[0]
		plt.rcParams['axes.facecolor'] = '#F5F5F5'
		plt.xlabel(vdf.alias)
		box = plt.boxplot(summarize, notch = False, sym = '', whis = np.inf, vert = False, widths = 0.7, labels = [""], patch_artist = True)
		for median in box['medians']:
			median.set(color = 'black', linewidth = 1, )
		for patch in box['boxes']:
			patch.set_facecolor("#214579")
		plt.gca().xaxis.grid()
		plt.gca().set_axisbelow(True)
		plt.title('BoxPlot of {}'.format(vdf.alias))
		plt.show()
	# MULTI BOXPLOT
	else:
		try:
			by = '"' + by.replace('"', '') + '"'
			if (vdf.alias == by):
				raise NameError("The column and the groupby can not be the same")
			elif (by not in vdf.parent.get_columns()):
				raise NameError("The column " + by + " doesn't exist")
			count = vdf.parent.shape()[0]
			cardinality = vdf.parent[by].nunique()
			is_numeric = vdf.isnum()
			is_categorical = (cardinality <= max_cardinality) or not(is_numeric)
			table = vdf.parent.genSQL()
			if not(is_categorical):
				enum_trans = vdf.parent[by].to_enum(h = h, return_enum_trans = True)[0].replace("{}", by) + " AS " + by
				enum_trans += " , " + vdf.alias
				table = "(SELECT " + enum_trans + " FROM " + table + ") enum_table"
				query = "SELECT COUNT(DISTINCT {}) FROM {}".format(by, table)
				vdf.executeSQL(query = query, title="Compute the cardinality of the enum feature " + by)
				cardinality = vdf.parent.cursor.fetchone()[0]
			if not(cat_priority):
				query = "SELECT {} FROM {} WHERE {} IS NOT NULL GROUP BY {} ORDER BY COUNT(*) DESC LIMIT {}".format(by, table, vdf.alias, by, max_cardinality)
				query_result = vdf.executeSQL(query = query, title = "Compute the categories of the feature " + by).fetchall()
				cat_priority = [item for sublist in query_result for item in sublist]
			with_summarize=False
			query = []
			lp = "(" if (len(cat_priority) == 1) else ""
			rp = ")" if (len(cat_priority) == 1) else ""
			for idx, category in enumerate(cat_priority):
				tmp_query = "SELECT MIN({}) AS min, APPROXIMATE_PERCENTILE ({} USING PARAMETERS percentile = 0.25) AS Q1, APPROXIMATE_PERCENTILE ({}".format(vdf.alias, vdf.alias, vdf.alias)
				tmp_query += "USING PARAMETERS percentile = 0.5) AS Median, APPROXIMATE_PERCENTILE ({} USING PARAMETERS percentile = 0.75) AS Q3, MAX".format(vdf.alias)
				tmp_query += "({}) AS max, '{}' FROM {}".format(vdf.alias, '{}', table)
				tmp_query = tmp_query.format("None") if (category in ('None', None)) else tmp_query.format(category)
				tmp_query += " WHERE {} IS NULL".format(by) if (category in ('None', None)) else " WHERE {} = '{}'".format(by, category)
				query += [lp + tmp_query + rp]
			query = " UNION ALL ".join(query)
			vdf.executeSQL(query = query, title = "Compute all the descriptive statistics for each category to draw the box plot")
			query_result = vdf.parent.cursor.fetchall()
			cat_priority = [item[-1] for item in query_result]
			result = [[float(item[i]) for i in range(0,5)] for item in query_result]
			result.reverse()
			cat_priority.reverse()
			if (vdf.parent[by].category() == "text"):
				labels = []
				for item in cat_priority:
					labels += [item[0:47] + "..."] if (len(str(item)) > 50) else [item]
			else:
				labels=cat_priority
			plt.rcParams['axes.facecolor'] = '#F8F8F8'
			plt.ylabel(vdf.alias)
			plt.xlabel(by)
			plt.xticks(rotation = 90)
			plt.gca().yaxis.grid()
			other_labels = []
			other_result = []
			all_idx = []
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
								other_labels += [labels[idx]]
								other_result += [result[idx]]
								all_idx += [idx]
				for idx in all_idx:
					del labels[idx]
					del result[idx]
			if not(is_categorical):
				sorted_boxplot = sorted([[float(labels[i][1:-1].split(';')[0]), labels[i], result[i]] for i in range(len(labels))])
				labels, result = [item[1] for item in sorted_boxplot] + other_labels, [item[2] for item in sorted_boxplot] + other_result
			else:
				sorted_boxplot = sorted([(labels[i], result[i]) for i in range(len(labels))])
				labels, result = [item[0] for item in sorted_boxplot], [item[1] for item in sorted_boxplot]
			box = plt.boxplot(result, notch = False, sym = '', whis = np.inf, widths = 0.5, labels = labels, patch_artist = True)
			plt.title('BoxPlot of {} group by {}'.format(vdf.alias, by))
			plt.subplots_adjust(bottom = max(0.3, len(max([str(item) for item in labels], key = len)) / 90.0))
			colors = gen_colors()
			for median in box['medians']:
				median.set(color = 'black', linewidth = 1,)
			for patch,color in zip(box['boxes'], colors):
				patch.set_facecolor(color)
			plt.show()
		except:
			print("/!\\ Warning: An error occured during the BoxPlot creation")
			raise
#
def boxplot2D(vdf, columns: list = []):
	if not(columns):
		columns = vdf.numcol()
	for column in columns:
		if column not in vdf.get_columns() and ('"' + column + '"' not in vdf.get_columns()):
			print("/!\\ Warning: Column '{}' is not in the vDataframe.\nIt will be ignored.".format(column))
			columns.remove(column)
		elif (column not in vdf.numcol()) and ('"' + column + '"' not in vdf.numcol()):
			print("/!\\ Warning: Column '{}' is not numerical.\nIt will be ignored.".format(column))
			columns.remove(column)
	if not(columns):
		print("/!\\ Warning: No numerical columns found to draw the multi boxplot")
		raise
	# SINGLE BOXPLOT	
	if (len(columns) == 1):
		vdf[columns[0]].boxplot()
	# MULTI BOXPLOT
	else:
		try:
			summarize = vdf.describe(columns = columns).transpose()
			result = [summarize.values[column][3:8] for column in summarize.values]
			columns = [column for column in summarize.values]
			del columns[0]
			del result[0]
			plt.figure(figsize = (10,8))
			plt.rcParams['axes.facecolor'] = '#F8F8F8'
			plt.xticks(rotation = 90)
			box = plt.boxplot(result, notch = False, sym = '', whis = np.inf, widths = 0.5, labels = columns, patch_artist = True)
			plt.title('Multi BoxPlot of the vDataframe')
			plt.subplots_adjust(bottom = max(0.3, len(max([str(item) for item in columns], key = len)) / 90.0))
			colors = gen_colors()
			for median in box['medians']:
				median.set(color = 'black', linewidth = 1,)
			for patch,color in zip(box['boxes'], colors):
				patch.set_facecolor(color)
			plt.show()
		except:
			print("/!\\ Warning: An error occured during the BoxPlot creation")
			raise
#
def cmatrix(matrix,
			columns_x,
			columns_y,
			n: int,
			m: int,
			vmax: float,
			vmin: float,
			cmap:str = 'PRGn',
			title: str = "",
			colorbar: str = "",
			x_label: str = "",
			y_label: str = "",
			with_numbers: bool = True,
			mround: int = 3):
		matrix_array = np.ndarray(shape = (n, m), dtype = float)
		for i in range(n):
			for j in range(m):
				try:
					matrix_array[i][j] = matrix[j + 1][i + 1]
				except:
					matrix_array[i][j] = None
		plt.figure(figsize = (8,8))
		plt.title(title)
		plt.imshow(matrix_array, cmap = cmap, interpolation = 'nearest', vmax = vmax, vmin = vmin)
		plt.colorbar().set_label(colorbar)
		plt.gca().set_xlabel(x_label)
		plt.gca().set_ylabel(y_label)
		plt.gca().set_yticks([i for i in range(0, n)])
		plt.gca().set_xticks([i for i in range(0, m)])
		plt.yticks(rotation = 0)
		plt.xticks(rotation = 90)
		plt.subplots_adjust(bottom = max(0.2, len(max([str(item) for item in columns_y], key = len)) / 90.0))	
		plt.gca().set_xticklabels(columns_y)
		plt.gca().set_yticklabels(columns_x)
		x_positions = np.linspace(start = 0,stop = m, num = m, endpoint = False)
		y_positions = np.linspace(start = 0,stop = n, num = n, endpoint = False)
		if (with_numbers):
			for y_index, y in enumerate(y_positions):
			    for x_index, x in enumerate(x_positions):
			        label=round(matrix_array[y_index, x_index], mround)
			        plt.gca().text(x, y, label, color = 'black', ha = 'center', va = 'center')
		plt.show()
#
def compute_plot_variables(vdf, 
			   	   		   method: str = "density", 
			   	   		   of: str = "", 
			       		   max_cardinality: int = 6, 
			       		   bins: int = 0, 
			       		   h: float = 0, 
			       		   pie: bool = False):
	# aggregation used for the bins height
	if (method == "mean"):
		method = "avg"
	if ((method in ["avg","min","max","sum"]) and (of)):
		if (of in vdf.parent.get_columns()) or ('"' + of.replace('"', '') + '"' in vdf.parent.get_columns()):
			aggregate = "{}({})".format(method.upper(), '"' + of.replace('"', '') + '"')
			others_aggregate = method
		else:
			raise NameError("The column '" + of + "' doesn't exist")
	elif (method in ["density", "count"]):
		aggregate = "count(*)"
		others_aggregate = "sum"
	else:
		raise ValueError("The parameter 'method' must be in avg|mean|min|max|sum")
	# depending on the cardinality, the type, the vColumn can be treated as categorical or not
	cardinality, count, is_numeric, is_date, is_categorical = vdf.nunique(), vdf.parent.shape()[0], vdf.isnum(), (vdf.category()=="date"), False
	rotation = 0 if ((is_numeric) and (cardinality > max_cardinality)) else 90
	# case when categorical
	if ((((cardinality <= max_cardinality) or not(is_numeric)) or pie) and not(is_date)):
		if ((is_numeric) and not(pie)):
			query = "SELECT {}, {} FROM {} WHERE {} IS NOT NULL GROUP BY {} ORDER BY {} ASC LIMIT {}".format(
				vdf.alias, aggregate, vdf.parent.genSQL(), vdf.alias, vdf.alias, vdf.alias, max_cardinality)
		else:
			table = vdf.parent.genSQL()
			if ((pie) and (is_numeric)):
				enum_trans = vdf.to_enum(h = h, return_enum_trans = True)[0].replace("{}", vdf.alias) + " AS " + vdf.alias
				if (of):
					enum_trans += " , " + of
				table = "(SELECT " + enum_trans + " FROM " + table + ") enum_table"
			query = "(SELECT {} || '', {} FROM {} GROUP BY {} ORDER BY 2 DESC LIMIT {})".format(
				vdf.alias, aggregate, table, vdf.alias, max_cardinality)
			if (cardinality > max_cardinality):
				query += (" UNION (SELECT 'Others', {}(count) FROM (SELECT {} AS count FROM {} " + 
					"GROUP BY {} ORDER BY {} DESC OFFSET {}) y LIMIT 1) ORDER BY 2 DESC")
				query = query.format(others_aggregate, aggregate, table, vdf.alias, aggregate, max_cardinality)
		vdf.executeSQL(query, title = "Compute the histogram heights")
		query_result = vdf.parent.cursor.fetchall()
		if (query_result[-1][1] == None):
			del query_result[-1]
		z = [item[0] for item in query_result]
		y = [item[1] / float(count) if item[1] != None else 0 for item in query_result] if (method == "density") else [item[1] if item[1] != None else 0 for item in query_result]
		x = [0.4 * i + 0.2 for i in range(0, len(y))]
		h = 0.39
		is_categorical = True
	# case when date
	elif (is_date):
		if ((h <= 0) and (bins <= 0)):
			h = vdf.numh()
		elif (bins > 0):
			query = "SELECT DATEDIFF('second', MIN(" + vdf.alias + "), MAX(" + vdf.alias  + ")) FROM "
			vdf.executeSQL(query = query, title = "Compute the histogram interval")
			query_result = vdf.parent.cursor.fetchone()
			h = float(query_result[0]) / bins
		min_date = vdf.min()
		converted_date = "DATEDIFF('second', '" + str(min_date) + "', "+vdf.alias + ")"
		query = "SELECT FLOOR({} / {}) * {}, {} FROM {} WHERE {} IS NOT NULL GROUP BY 1 ORDER BY 1"
		query = query.format(converted_date, h, h, aggregate, vdf.parent.genSQL(), vdf.alias)
		vdf.executeSQL(query = query, title = "Compute the histogram heights")
		query_result = vdf.parent.cursor.fetchall()
		x = [float(item[0]) for item in query_result]
		y = [item[1] / float(count) for item in query_result] if (method == "density") else [item[1] for item in query_result]
		query = ""
		for idx, item in enumerate(query_result):
			query += " UNION (SELECT TIMESTAMPADD('second' , " + str(math.floor(h * idx)) + ", '" + str(min_date) + "'::timestamp))"
		query = query[7:-1] + ")"
		h = 0.94 * h
		vdf.parent.cursor.execute(query)
		query_result = vdf.parent.cursor.fetchall()
		z = [item[0] for item in query_result]
		z.sort()
		is_categorical = True
	# case when numerical
	else:
		if ((h <= 0) and (bins <= 0)):
			h = vdf.numh()
		elif (bins > 0):
			h = float(vdf.max() - vdf.min()) / bins
		if (vdf.ctype == "int"):
			h = max(1.0,h)
		query = "SELECT FLOOR({} / {}) * {}, {} FROM {} WHERE {} IS NOT NULL GROUP BY 1 ORDER BY 1"
		query = query.format(vdf.alias, h, h, aggregate, vdf.parent.genSQL(), vdf.alias)
		vdf.executeSQL(query = query, title = "Compute the histogram heights")
		query_result = vdf.parent.cursor.fetchall()
		y = [item[1] / float(count) for item in query_result] if (method == "density") else [item[1] for item in query_result]
		x = [float(item[0]) + h/2 for item in query_result]
		h = 0.94 * h
		z = None
	return [x, y, z, h, is_categorical]
#
def density(vdf,
			a = None,
			kernel: str = "gaussian",
			smooth: int = 200,
			color: str = '#214579'):
	if (kernel == "gaussian"):
		def fkernel(x):
			return math.exp( - 1 / 2 * ((x) ** 2)) / (math.sqrt(2 * math.pi))
	elif (kernel == "logistic"):
		def fkernel(x):
			return 1 / (2 + math.exp(x) + math.exp( - x))
	elif (kernel == "sigmoid"):
		def fkernel(x):
			return 2 / (math.pi * (math.exp(x) + math.exp( - x)))
	elif (kernel == "silverman"):
		def fkernel(x):
			return math.exp( - 1 / math.sqrt(2) * abs(x)) / (2) * math.sin(abs(x) / math.sqrt(2) + math.pi / 4)
	else:
		raise ValueError("The parameter 'kernel' must be in gaussian|logistic|sigmoid|silverman")
	if (a == None):
		a = 1.06 * vdf.std() / vdf.count() ** (1.0 / 5.0)
	if not(vdf.isnum()):
		raise TypeError("Cannot draw a density plot for non-numerical columns")
	x, y, z, h, is_categorical = compute_plot_variables(vdf, method = "count", max_cardinality = 1) 
	x = [item - h / 2 / 0.94 for item in x]
	N = sum(y)
	y_smooth = []
	x_smooth = [(max(x) - min(x)) * i / smooth+min(x) for i in range(0, smooth + 1)]
	n = len(y)
	for x0_smooth in x_smooth:
		K = sum([y[i] * fkernel(((x0_smooth - x[i]) / a) ** 2) / (a * N) for i in range(0, len(x))])
		y_smooth += [K]
	plt.figure(figsize = (10,8))
	plt.rcParams['axes.facecolor'] = '#F5F5F5'
	plt.plot(x_smooth, y_smooth, color = "#222222")
	plt.xlim(min(x), max(x))
	plt.ylim(0, max(y_smooth) * 1.1)
	plt.grid()
	plt.gca().set_axisbelow(True)
	plt.fill_between(x_smooth, y_smooth, facecolor = color, alpha = 0.7)
	plt.ylabel("density")
	plt.title('Distribution of {} ({} kernel)'.format(vdf.alias, kernel))
	plt.show()
# 
def gen_colors():
	colors = ['#214579', '#FFCC01', 'dimgrey', 'firebrick', 'darkolivegreen', 'black', 'gold', 'tan', 'pink', 'darksalmon', 'lightskyblue', 'lightgreen', 'palevioletred', 'coral']
	all_colors = [item for item in plt_colors.cnames]
	shuffle(all_colors)
	for c in all_colors:
		if c not in colors:
			colors += [c]
	return colors*10
#
def hexbin(vdf,
		   columns: list,
		   method: str = "count",
		   of: str = "",
		   cmap: str = 'Blues',
		   gridsize: int = 10,
		   color: str = "white"):
	if (len(columns) != 2):
		raise ValueError("The parameter 'columns' must be exactly of size 2 to draw the hexbin")
	if (method == "mean"):
		method = "avg"
	if ((method in ["avg", "min", "max", "sum"]) and (of) and ((of in vdf.get_columns()) or ('"' + of.replace('"', '') + '"' in vdf.get_columns()))):
		aggregate = "{}({})".format(method, of)
		others_aggregate = method
		if (method == "avg"):
			reduce_C_function = np.mean
		elif (method == "min"):
			reduce_C_function = min
		elif (method == "max"):
			reduce_C_function = max
		elif (method=="sum"): 
			reduce_C_function = sum
	else:
		aggregate = "count(*)"
		reduce_C_function = sum
	count = vdf.shape()[0]
	if (method == "density"):
		over="/"+str(float(count))
	else:
		over=""
	query="SELECT {}, {}, {}{} FROM {} GROUP BY {}, {}".format(columns[0], columns[1], aggregate, over, vdf.genSQL(), columns[0], columns[1])
	query_result=vdf.executeSQL(query = query, title = "Group all the elements for the Hexbin Plot").fetchall()
	column1, column2, column3 = [], [], []
	for item in query_result:
		if ((item[0] != None) and (item[1] != None) and (item[2] != None)):
			column1 += [float(item[0])] * 2
			column2 += [float(item[1])] * 2
			if (reduce_C_function in [min, max, np.mean]):
				column3 += [float(item[2])] * 2
			else:
				column3 += [float(item[2])/2] * 2
	plt.figure(figsize = (10,8))
	plt.rcParams['axes.facecolor'] = 'white'
	plt.title('Hexbin of {} vs {}'.format(columns[0], columns[1]))
	plt.ylabel(columns[1])
	plt.xlabel(columns[0])
	plt.hexbin(column1, column2, C=column3, reduce_C_function = reduce_C_function, gridsize = gridsize, color = color, cmap = cmap, mincnt = 1)
	if (method == "density"):
		plt.colorbar().set_label(method)
	else:
		plt.colorbar().set_label(aggregate)
	plt.show()
#
def hist(vdf,
		 method: str = "density",
		 of = None,
		 max_cardinality: int = 6,
		 bins: int = 0,
		 h: float = 0,
		 color: str = '#214579'):
	x, y, z, h, is_categorical = compute_plot_variables(vdf, method, of, max_cardinality, bins, h)
	is_numeric = vdf.isnum()
	rotation = 0 if ((is_numeric) and not(is_categorical)) else 90
	plt.figure(figsize = (10,8))
	plt.rcParams['axes.facecolor'] = '#F5F5F5'
	plt.bar(x, y, h, color = color, alpha = 0.86)
	plt.xlabel(vdf.alias)
	plt.gca().set_axisbelow(True)
	plt.gca().yaxis.grid()
	if (is_categorical):
		if not(is_numeric):
			new_z = []
			for item in z:
				new_z += [item[0:47] + "..."] if (len(str(item))>50) else [item]
		else:
			new_z = z
		plt.xticks(x, new_z, rotation = rotation)
		plt.subplots_adjust(bottom = max(0.3, len(max([str(item) for item in z], key = len)) / 140.0))
	if (method == "density"):
		plt.ylabel('Density')
		plt.title('Distribution of {}'.format(vdf.alias))
	elif ((method in ["avg", "min", "max", "sum", "mean"]) and (of != None)):
		aggregate = "{}({})".format(method, of)
		plt.ylabel(aggregate)
		plt.title('{} group by {}'.format(aggregate, vdf.alias))
	else:
		plt.ylabel('Frequency')
		plt.title('Count by {}'.format(vdf.alias))
	plt.show()
#
def hist2D(vdf,
		   columns: list,
		   method = "density",
		   of: str = "",
		   max_cardinality: tuple = (6, 6),
		   h: tuple = (None, None),
		   limit_distinct_elements: int = 200,
		   stacked: bool = False):
	colors = gen_colors()
	all_columns = vdf.pivot_table(columns, method = method, of = of, h = h, max_cardinality = max_cardinality, show = False, limit_distinct_elements = limit_distinct_elements).values
	all_columns = [[column] + all_columns[column] for column in all_columns]
	plt.figure(figsize = (10,8))
	plt.rcParams['axes.facecolor'] = '#F5F5F5'
	n, m = len(all_columns), len(all_columns[0])
	n_groups = m-1
	index = np.arange(n_groups)
	bar_width = 0.5
	for i in range(1, n):
		current_column = all_columns[i][1:m]
		for idx, item in enumerate(current_column):
			try:
				current_column[idx] = float(item)
			except:
				current_column[idx] = 0
		current_label = str(all_columns[i][0])
		if (stacked):
			if (i == 1):
				last_column = [0 for item in all_columns[i][1:m]]
			else:
				for idx,item in enumerate(all_columns[i - 1][1:m]):
					try:
						last_column[idx] += float(item)
					except:
						last_column[idx] += 0
			plt.bar(index, current_column, bar_width, alpha = 0.86, color = colors[i - 1], label = current_label, bottom = last_column)
		else:
			plt.bar(index + (i - 1) * bar_width / (n - 1), current_column, bar_width / (n - 1), alpha=0.86, color = colors[i - 1], label = current_label)
	if (stacked):
		plt.xticks(index, all_columns[0][1:m], rotation=90)
	else:
		plt.xticks(index + bar_width / 2 - bar_width / 2 / (n - 1), all_columns[0][1:m], rotation = 90)
	plt.subplots_adjust(bottom = max(0.3, len(max([str(item) for item in all_columns[0][1:m]],key = len)) / 140.0))
	plt.xlabel(columns[0])
	if (method.lower() == "mean"):
		method = "avg"
	if (method == "density"):
		plt.ylabel('Density')
		plt.title('Distribution of {} group by {}'.format(columns[0], columns[1]))
	elif ((method in ["avg","min","max","sum"]) and (of != None)):
		plt.ylabel("{}({})".format(method, of))
		plt.title('{}({}) of {} group by {}'.format(method, of, columns[0], columns[1]))
	else:
		plt.ylabel('Frequency')
		plt.title('Count by {} group by {}'.format(columns[0], columns[1]))
	plt.legend(title = columns[1])
	plt.gca().set_axisbelow(True)
	plt.gca().yaxis.grid()
	plt.show()
#
def pie(vdf,
		method: str = "density",
		of = None,
		max_cardinality: int = 6,
		h: float = 0,
		donut: bool = False):
	colors = ['#214579', '#FFCC01'] * 100
	count = vdf.parent.shape()[0]
	cardinality = vdf.nunique()
	is_numeric = vdf.isnum()
	is_categorical = (cardinality <= max_cardinality) or not(is_numeric)
	x, y, z, h, is_categorical = compute_plot_variables(vdf, max_cardinality = max_cardinality, method = method, of = of, pie = True)
	z.reverse()
	y.reverse()
	explode = [0 for i in y]
	explode[np.argmax(y)] = 0.13
	current_explode = 0.15
	total_count = sum(y)
	for idx,item in enumerate(y):
		if ((item < 0.05) or ((item > 1) and (item / float(total_count) < 0.05))):
			current_explode = min(0.9, current_explode * 1.4) 
			explode[idx] = current_explode
	if (method == "density"):
		autopct = '%1.1f%%'
	else:
		def make_autopct(values, category):
		    def my_autopct(pct):
		        total = sum(values)
		        val = pct * total / 100.0
		        if (category == "int"):
		        	val = int(round(val))
		        	return '{v:d}'.format(v = val)
		        else:
		        	return '{v:f}'.format(v = val)
		    return my_autopct
		if ((method in ["sum", "count"]) or ((method in ["min", "max"]) and (vdf.parent[of].category == "int"))):
			category = "int"
		else:
			category = None
		autopct = make_autopct(y,category)
	plt.figure(figsize = (10,8))
	if (donut):
		explode = None
		centre_circle = plt.Circle((0,0), 0.72, color='#666666', fc='white', linewidth=1.25)
		fig = plt.gcf()
		fig.gca().add_artist(centre_circle)
	plt.pie(y, labels = z, autopct = autopct, colors = colors, shadow = True, startangle = 290, explode = explode)
	plt.subplots_adjust(bottom = 0.2)
	if (method == "density"):
		plt.title('Distribution of {}'.format(vdf.alias))
	elif ((method in ["avg","min","max","sum"]) and (of != None)):
		aggregate = "{}({})".format(method,of)
		plt.title('{} group by {}'.format(aggregate,vdf.alias))
	else:
		plt.title('Count by {}'.format(vdf.alias))
	plt.show()
# 
def multiple_hist(vdf,
				  columns: list,
				  method: str = "density",
				  of: str = "",
				  h: float = 0):
	colors = gen_colors()
	if (len(columns) > 5):
		raise Exception("The number of column must be <= 5 to use 'multiple_hist' method")
	else:
		plt.figure(figsize = (10,8))
		plt.rcParams['axes.facecolor'] = '#F5F5F5'
		alpha, all_columns, all_h = 1, [], []
		if (h <= 0):
			for idx,column in enumerate(columns):
				all_h += [vdf[column].numh()]
			h = min(all_h)
		for idx, column in enumerate(columns):
			if (vdf[column].isnum()):
				[x, y, z, h, is_categorical] = compute_plot_variables(vdf[column], method = method, of = of, max_cardinality = 1, h = h)
				h = h / 0.94
				plt.bar(x, y, h, color = colors[idx], alpha = alpha, label = column)
				alpha -= 0.2
				all_columns += [columns[idx]]
			else:
				print("/!\\ Warning: {} is not numerical. Its histogram will not be draw.".format(column))
		plt.xlabel(", ".join(all_columns))
		plt.gca().set_axisbelow(True)
		plt.gca().yaxis.grid()
		if (method == "density"):
			plt.ylabel('Density')
		elif ((method in ["avg", "min", "max", "sum", "mean"]) and (of)):
			plt.ylabel(method + "(" + of + ")")
		else:
			plt.ylabel('Frequency')
		plt.title("Multiple Histograms")
		plt.legend(title = "columns")
		plt.show()
#
def multi_ts_plot(vdf, 
				  order_by: str,
				  columns: list = [], 
				  order_by_start: str = "",
				  order_by_end: str = ""):
	if (len(columns) == 1):
		return vdf[columns[0]].plot(ts = order_by, start_date = order_by_start, end_date = order_by_end, area = False)
	if not(columns):
		columns = vdf.numcol()
	for column in columns:
		if column not in vdf.get_columns() and ('"' + column + '"' not in vdf.get_columns()):
			print("/!\\ Warning: Column '{}' is not in the vDataframe.\nIt will be ignored.".format(column))
			columns.remove(column)
		elif (column not in vdf.numcol()) and ('"' + column + '"' not in vdf.numcol()):
			print("/!\\ Warning: Column '{}' is not numerical.\nIt will be ignored.".format(column))
			columns.remove(column)
	if not(columns):
		print("/!\\ Warning: No numerical columns found to draw the multi plot")
		raise
	columns = ['"' + column.replace('"', '') + '"' for column in columns]
	order_by = '"' + order_by.replace('"', '') + '"'
	colors = gen_colors()
	query = "SELECT {}, {} FROM {} WHERE {} IS NOT NULL".format(order_by, ", ".join(columns), vdf.genSQL(), order_by)
	query += " AND {} > '{}'".format(order_by, order_by_start) if (order_by_start) else ""
	query += " AND {} < '{}'".format(order_by, order_by_end) if (order_by_end) else ""
	query_result = vdf.executeSQL(query = query, title = "Select the needed points to draw the curves").fetchall()
	order_by_values = [item[0] for item in query_result]
	alpha = 0.3
	plt.figure(figsize = (10,8))
	plt.gca().grid()
	for i in range(0, len(columns)):
		plt.plot(order_by_values, [item[i + 1] for item in query_result], color = colors[i], label = columns[i])
	plt.rcParams['axes.facecolor'] = '#FCFCFC'
	plt.title('Multi Plot of the vDataframe')
	plt.xticks(rotation = 90)
	plt.subplots_adjust(bottom = 0.24)
	plt.xlabel(order_by)
	plt.legend(title = "columns")
	plt.show()
#
def pivot_table(vdf,
				columns,
				method="count",
				of: str = "",
				h: tuple = (None, None),
				max_cardinality: tuple = (20, 20),
				show: bool = True,
				cmap: str = 'Blues',
				limit_distinct_elements: int = 1000,
				with_numbers: bool = True):
	# aggregation used for the bins height
	if (method == "mean"):
		method = "avg"
	if ((method.lower() in ["avg", "min", "max", "sum"]) and (of)):
		if ((of.replace('"', '') in vdf.get_columns()) or ('"' + of.replace('"', '') + '"' in vdf.get_columns())):
			aggregate = "{}({})".format(method.upper(), '"' + of.replace('"', '') + '"')
			others_aggregate = method
		else:
			raise NameError("the column '" + of + "' doesn't exist")
	elif (method.lower() in ["density", "count"]):
		aggregate = "COUNT(*)"
		others_aggregate = "sum"
	else:
		raise ValueError("The parameter 'method' must be in count|density|avg|mean|min|max|sum")
	columns = ['"' + column.replace('"', '') + '"' for column in columns]
	all_columns = []
	is_column_date = [False, False]
	timestampadd = ["", ""]
	for idx, column in enumerate(columns):
		is_numeric = vdf[column].isnum()
		is_date = vdf[column].isdate()
		if (is_numeric):
			interval = round(vdf[column].numh(), 2) if (h[idx] == None) else h[idx]
			if (vdf[column].category() == "int"):
				floor_end = "-1"
				interval = int(max(math.floor(interval),1))
			else:
				floor_end = ""
			expr = "'[' || FLOOR({} / {}) * {} || ';' || (FLOOR({} / {}) * {} + {}{}) || ']'".format(column, interval, interval, column, interval, interval, interval, floor_end)
			all_columns += [expr] if (interval > 1) or (vdf[column].category() == "float") else ["FLOOR({}) || ''".format(column)]
			order_by = "ORDER BY MIN(FLOOR({} / {}) * {}) ASC".format(column, interval, interval)
		elif (is_date):
			interval = vdf[column].numh() if (h[idx] == None) else max(math.floor(h[idx]),1)
			min_date = vdf[column].min()
			all_columns += ["FLOOR(DATEDIFF('second', '"+str(min_date)+"', "+column+") / "+str(interval)+") * "+str(interval)]
			is_column_date[idx] = True
			timestampadd[idx] = "TIMESTAMPADD('second', "+columns[idx]+"::int, '"+str(min_date)+"'::timestamp)"
			order_by = "ORDER BY 1 ASC"
		else:
			all_columns += [column]
			order_by = "ORDER BY 1 ASC"
	if (len(columns) == 1):
		query = "SELECT {} AS {}, {} FROM {} WHERE {} IS NOT NULL GROUP BY 1 {} LIMIT {}".format(all_columns[-1], columns[0], aggregate, vdf.genSQL(), columns[0], order_by, limit_distinct_elements)
		return to_tablesample(query, vdf.cursor, name = aggregate)
	alias = ", " + '"' + of.replace('"', '') + '"' + " AS " + '"' + of.replace('"', '') + '"' if of else ""
	aggr = ", " + of if (of.replace('"', '') != '') else ""
	subtable = "(SELECT " + all_columns[0] + " AS " + columns[0] + ", " + all_columns[1] + " AS " + columns[1] + alias + " FROM " + vdf.genSQL() + ") pivot_table"
	if (is_column_date[0] and not(is_column_date[1])):
		subtable = "(SELECT " + timestampadd[0] + " AS " + columns[0] + ", " + columns[1] + aggr + " FROM " + subtable + ") pivot_table_date"
	elif (is_column_date[1] and not(is_column_date[0])):
		subtable = ("(SELECT " + columns[0] + ", " + timestampadd[1] + " AS " + columns[1] + aggr + " FROM " + subtable + ") pivot_table_date")
	elif (is_column_date[1] and is_column_date[0]):
		subtable = "(SELECT " + timestampadd[0] + " AS " + columns[0] + ", " + timestampadd[1] + " AS " + columns[1] + aggr + " FROM " + subtable + ") pivot_table_date"
	is_finished = limit_distinct_elements
	limit_distinct_elements = " LIMIT " + str(limit_distinct_elements)
	over = "/" + str(vdf.shape()[0]) if (method=="density") else ""
	query="SELECT {}, {}, {}{} FROM {} WHERE {} IS NOT NULL AND {} IS NOT NULL GROUP BY {}, {} ORDER BY {}, {} ASC".format(columns[0], columns[1], aggregate, over, subtable, columns[0], columns[1], columns[0], columns[1], columns[0], columns[1]) + limit_distinct_elements
	vdf.executeSQL(query = query, title = "Group the features to compute the pivot table")
	query_result = vdf.executeSQL(query = query, title = "Group the features to compute the pivot table").fetchall()
	# Column0 sorted categories
	all_column0_categories = list(set([str(item[0]) for item in query_result]))
	all_column0_categories.sort()
	try:
		try:
			order = []
			for item in all_column0_categories:
				order += [float(item.split(";")[0].split('[')[1])]
		except:
			order = [float(item) for item in all_column0_categories]
		all_column0_categories = [x for _, x in sorted(zip(order, all_column0_categories))]
	except:
		pass
	# Column1 sorted categories
	all_column1_categories = list(set([str(item[1]) for item in query_result])) 
	all_column1_categories.sort()
	try:
		try:
			order = []
			for item in all_column1_categories:
				order += [float(item.split(";")[0].split('[')[1])]
		except:
			order = [float(item) for item in all_column1_categories]
		all_column1_categories = [x for _, x in sorted(zip(order, all_column1_categories))]
	except:
		pass
	all_columns = [['' for item in all_column0_categories] for item in all_column1_categories]
	is_finished = (is_finished >= len(all_column0_categories) * len(all_column1_categories))
	for item in query_result:
		j = all_column0_categories.index(str(item[0]))
		i = all_column1_categories.index(str(item[1]))
		all_columns[i][j] = item[2]
	all_columns = [[all_column1_categories[i]] + all_columns[i] for i in range(0, len(all_columns))]
	all_columns = [[columns[0] + "/" + columns[1]] + all_column0_categories] + all_columns
	if (show):
		all_count = [item[2] for item in query_result]
		cmatrix(all_columns, all_column0_categories, all_column1_categories, len(all_column0_categories), len(all_column1_categories), vmax = max(all_count), vmin = min(all_count), cmap = cmap, title = "Pivot Table of " + columns[0] + " vs " + columns[1], colorbar=aggregate, x_label=columns[1], y_label=columns[0], with_numbers=with_numbers)
	values = {all_columns[0][0] : all_columns[0][1:len(all_columns[0])]}
	del(all_columns[0])
	for column in all_columns:
		values[column[0]] = column[1:len(column)]
	return tablesample(values = values, name = "Pivot Table of " + columns[0] + " vs " + columns[1], table_info = False)
#
def scatter_matrix(vdf, columns: list = []):
	for column in columns:
		if (column not in vdf.get_columns()) and ('"' + column.replace('"', '') + '"' not in vdf.get_columns()):
			raise NameError("The column '{}' doesn't exist".format(column))
	if not(columns):
		columns = vdf.numcol()
	elif (len(columns) == 1):	
		return vdf[columns[0]].hist()
	n = len(columns)
	fig, axes = plt.subplots(nrows = n, ncols = n, figsize = (11,11))
	query = "SELECT " + ",".join(columns) + ", random() AS rand FROM {} ".format(vdf.genSQL(tablesample = 50)) + "ORDER BY rand LIMIT 1000"
	all_scatter_points = vdf.executeSQL(query = query, title = "Select random points for the scatter plot").fetchall()
	all_scatter_columns = []
	all_h = []
	for idx, column in enumerate(columns):
		all_h += [vdf[column].numh()]
	h = min(all_h)
	for i in range(n):
		all_scatter_columns += [[item[i] for item in all_scatter_points]]
	for i in range(n):
		x = columns[i]
		axes[-1][i].set_xlabel(x, rotation = 90)
		axes[i][0].set_ylabel(x, rotation = 0)
		axes[i][0].yaxis.get_label().set_ha('right')
		for j in range(n):
			axes[i][j].get_xaxis().set_ticks([])
			axes[i][j].get_yaxis().set_ticks([])
			axes[i][j].set_facecolor("#F0F0F0")
			y = columns[j]
			if (x == y):
				x0, y0, z0, h0, is_categorical = compute_plot_variables(vdf[x], method = "density", h = h, max_cardinality = 1)
				axes[i, j].bar(x0, y0, h0 / 0.94, color = '#FFCC01')
			else:
				axes[i, j].scatter(all_scatter_columns[j], all_scatter_columns[i], color = '#214579', s = 4, marker = 'o')
	fig.suptitle('Scatter Plot Matrix of {}'.format(vdf.input_relation))
	plt.show()
#
def scatter2D(vdf,
			  columns: list,
			  max_cardinality: int = 3,
			  cat_priority: list = [],
			  with_others: bool = True,
			  max_nb_points: int = 1000):
	colors = gen_colors()
	markers = ["^", "o", "+", "*", "h", "x", "D", "1"] * 10
	columns = ['"' + column.replace('"', '') + '"' for column in columns]
	for column in columns:
		if (column not in vdf.get_columns()):
			raise NameError("The column '{}' doesn't exist".format(column))
	if not(vdf[columns[0]].isnum()) or not(vdf[columns[1]].isnum()):
		raise TypeError("The two first columns of the parameter 'columns' must be numerical")
	if (len(columns) == 2):
		tablesample = max_nb_points / vdf.shape()[0]
		query = "SELECT {}, {} FROM {} WHERE {} IS NOT NULL AND {} IS NOT NULL LIMIT {}".format(columns[0], columns[1], vdf.genSQL(tablesample), columns[0], columns[1], max_nb_points)
		query_result = vdf.executeSQL(query = query, title = "Select random points for the scatter plot").fetchall()
		column1, column2 = [item[0] for item in query_result], [item[1] for item in query_result]
		plt.figure(figsize = (10,8))
		plt.gca().grid()
		plt.gca().set_axisbelow(True)
		plt.title('Scatter Plot of {} vs {}'.format(columns[0], columns[1]))
		plt.ylabel(columns[1])
		plt.xlabel(columns[0])
		plt.scatter(column1, column2, color = colors[0], s = 14)
		plt.show()
	else:
		column_groupby = columns[2]
		count = vdf.shape()[0]
		if (cat_priority):
			query_result = cat_priority
		else:
			query = "SELECT {} FROM {} WHERE {} IS NOT NULL GROUP BY {} ORDER BY COUNT(*) DESC LIMIT {}".format(
					column_groupby, vdf.genSQL(), column_groupby, column_groupby, max_cardinality)
			query_result = vdf.executeSQL(query = query, title = "Select all the category of the column").fetchall()
			query_result = [item for sublist in query_result for item in sublist]
		all_columns, all_scatter, all_categories = [query_result], [], query_result
		fig = plt.figure(figsize = (10,8))
		ax = plt
		others = []
		groupby_cardinality = vdf[column_groupby].nunique()
		count = vdf.shape()[0]
		tablesample = 10 if (count>10000) else 90
		for idx, category in enumerate(all_categories):
			if ((max_cardinality < groupby_cardinality) or (len(cat_priority) < groupby_cardinality)):
				others += ["{} != '{}'".format(column_groupby, category)]
			query = "SELECT {},{} FROM {} WHERE {} = '{}' AND {} IS NOT NULL AND {} IS NOT NULL LIMIT {}"
			query = query.format(columns[0], columns[1], vdf.genSQL(tablesample), columns[2], category, columns[0], columns[1], int(max_nb_points / len(all_categories))) 
			vdf.executeSQL(query = query, title = "Select random points for the scatter plot (category = '"+str(category)+"')")
			query_result = vdf.cursor.fetchall()
			column1, column2 = [float(item[0]) for item in query_result], [float(item[1]) for item in query_result]
			all_columns += [[column1, column2]]
			all_scatter += [ax.scatter(column1, column2, alpha=0.8, marker = markers[idx], color = colors[idx])]
		if (with_others and idx + 1 < groupby_cardinality):
			all_categories += ["others"]
			query = "SELECT {}, {} FROM {} WHERE {} AND {} IS NOT NULL AND {} IS NOT NULL LIMIT {}"
			query = query.format(columns[0], columns[1], vdf.genSQL(tablesample), " AND ".join(others), columns[0], columns[1], int(max_nb_points / len(all_categories)))
			query_result = vdf.executeSQL(query = query, title = "Select random points for the scatter plot (category='others')").fetchall()
			column1, column2 = [float(item[0]) for item in query_result], [float(item[1]) for item in query_result]
			all_columns += [[column1, column2]]
			all_scatter += [ax.scatter(column1, column2, alpha = 0.8, marker = markers[idx + 1], color = colors[idx + 1])]
		for idx, item in enumerate(all_categories):
			if (len(str(item)) > 20):
				all_categories[idx] = str(item)[0:20]+"..."
		plt.gca().grid()
		plt.gca().set_axisbelow(True)
		plt.title('Scatter Plot of {} vs {}'.format(columns[0], columns[1]))
		ax.xlabel(columns[0])
		ax.ylabel(columns[1])
		ax.legend(all_scatter, all_categories, title=column_groupby)
		plt.show()
#
def scatter3D(vdf,
			  columns: list,
			  max_cardinality: int = 3,
			  cat_priority: list = [],
			  with_others: bool = True,
			  max_nb_points: int = 1000):
	columns = ['"' + column.replace('"', '') + '"' for column in columns]
	colors = gen_colors()
	markers = ["^", "o", "+", "*", "h", "x", "D", "1"] * 10
	if ((len(columns) < 3) or (len(columns) > 4)):
		raise Exception("3D Scatter plot can only be done with at least two columns and maximum with four columns")
	else:
		for column in columns:
			if (column not in vdf.get_columns()):
				raise NameError("The column '{}' doesn't exist".format(column))
		for i in range(3):
			if not(vdf[columns[i]].isnum()):
				raise TypeError("The three first columns of the parameter 'columns' must be numerical")
		if (len(columns) == 3):
			tablesample = max_nb_points / vdf.shape()[0]
			query = "SELECT {}, {}, {} FROM {} WHERE {} IS NOT NULL AND {} IS NOT NULL AND {} IS NOT NULL LIMIT {}".format(
						columns[0], columns[1], columns[2], vdf.genSQL(tablesample), columns[0], columns[1], columns[2], max_nb_points)
			query_result = vdf.executeSQL(query = query, title = "Select random points for the scatter plot").fetchall()
			column1, column2, column3 = [float(item[0]) for item in query_result], [float(item[1]) for item in query_result], [float(item[2]) for item in query_result]
			fig = plt.figure(figsize = (10,8))
			ax = fig.add_subplot(111, projection = '3d')
			plt.title('Scatter Plot of {} vs {} vs {}'.format(columns[0], columns[1], columns[2]))
			ax.scatter(column1, column2, column3, color = colors[0])
			ax.set_xlabel(columns[0])
			ax.set_ylabel(columns[1])
			ax.set_zlabel(columns[2])
			ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
			ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
			ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
			plt.show()
		else:
			column_groupby = columns[3]
			count = vdf.shape()[0]
			if (cat_priority):
				query_result = cat_priority
			else:
				query = "SELECT {} FROM {} WHERE {} IS NOT NULL GROUP BY {} ORDER BY COUNT(*) DESC LIMIT {}".format(
						column_groupby, vdf.genSQL(), column_groupby, column_groupby, max_cardinality)
				query_result = vdf.executeSQL(query = query, title = "Select all the category of the column " + column_groupby).fetchall()
				query_result = [item for sublist in query_result for item in sublist]
			all_columns, all_scatter, all_categories = [query_result], [], query_result
			fig = plt.figure(figsize = (10,8))
			ax = fig.add_subplot(111,projection='3d')
			others = []
			groupby_cardinality = vdf[column_groupby].nunique()
			tablesample = 10 if (count>10000) else 90
			for idx,category in enumerate(all_categories):
				if ((max_cardinality < groupby_cardinality) or (len(cat_priority) < groupby_cardinality)):
					others += ["{} != '{}'".format(column_groupby, category)]
				query = "SELECT {}, {}, {} FROM {} WHERE {} = '{}' AND {} IS NOT NULL AND {} IS NOT NULL AND {} IS NOT NULL limit {}"
				query = query.format(columns[0], columns[1], columns[2], vdf.genSQL(tablesample), columns[3], category, columns[0], columns[1], columns[2], int(max_nb_points / len(all_categories))) 
				query_result = vdf.executeSQL(query = query, title = "Select random points for the scatter plot (category = '"+str(category)+"')").fetchall()
				column1, column2, column3 = [float(item[0]) for item in query_result], [float(item[1]) for item in query_result], [float(item[2]) for item in query_result]
				all_columns += [[column1, column2, column3]]
				all_scatter += [ax.scatter(column1, column2, column3, alpha=0.8, marker = markers[idx], color = colors[idx])]
			if (with_others and idx + 1 < groupby_cardinality):
				all_categories += ["others"]
				query = "SELECT {}, {}, {} FROM {} WHERE {} AND {} IS NOT NULL AND {} IS NOT NULL AND {} IS NOT NULL LIMIT {}"
				query = query.format(columns[0], columns[1], columns[2], vdf.genSQL(tablesample), " and ".join(others), columns[0], columns[1], columns[2], int(max_nb_points / len(all_categories)))
				query_result = vdf.executeSQL(query = query, title = "Select random points for the scatter plot (category='others')").fetchall()
				column1, column2 = [float(item[0]) for item in query_result], [float(item[1]) for item in query_result]
				all_columns += [[column1, column2]]
				all_scatter += [ax.scatter(column1, column2, alpha = 0.8, marker = markers[idx + 1], color = colors[idx + 1])]
			for idx, item in enumerate(all_categories):
				if (len(str(item)) > 20):
					all_categories[idx] = str(item)[0:20] + "..."
			plt.title('Scatter Plot of {} vs {} vs {}'.format(columns[0], columns[1], columns[2]))
			ax.set_xlabel(columns[0])
			ax.set_ylabel(columns[1])
			ax.set_zlabel(columns[2])
			ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
			ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
			ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
			ax.legend(all_scatter, all_categories, scatterpoints = 1, title = column_groupby)
			plt.show()
#
def ts_plot(vdf, 
			order_by: str, 
			by: str = "",
			order_by_start: str = "",
			order_by_end: str = "",
			color: str = '#214579', 
			area: bool = False):
	order_by = '"' + order_by.replace('"', '') + '"'
	if not(by):
		query = "SELECT {}, {} FROM {} WHERE {} IS NOT NULL AND {} IS NOT NULL".format(order_by, vdf.alias, vdf.parent.genSQL(), order_by, vdf.alias)
		query += " AND {} > '{}'".format(order_by, order_by_start) if (order_by_start) else ""
		query += " AND {} < '{}'".format(order_by, order_by_end) if (order_by_end) else ""
		query_result = vdf.executeSQL(query = query, title = "Select the needed points to draw the curve").fetchall()
		order_by_values = [item[0] for item in query_result]
		column_values = [float(item[1]) for item in query_result]
		plt.figure(figsize = (10,8))
		plt.rcParams['axes.facecolor'] = '#FCFCFC'
		plt.plot(order_by_values, column_values, color = color)
		if (area):
			area_label = "Area "
			plt.fill_between(order_by_values, column_values, facecolor = color)
		else:
			area_label = ""
		plt.title(area_label + 'Plot of {} vs {}'.format(vdf.alias, order_by))
		plt.xticks(rotation = 90)
		plt.subplots_adjust(bottom = 0.24)
		plt.xlabel(order_by)
		plt.ylabel(vdf.alias)
		plt.gca().grid()
		plt.show()
	else:
		colors = gen_colors()
		by = '"' + by.replace('"', '') + '"'
		cat = vdf.parent[by].distinct()
		all_data = []
		for column in cat:
			query = "SELECT {}, {} FROM {} WHERE {} IS NOT NULL AND {} IS NOT NULL".format(order_by, vdf.alias, vdf.parent.genSQL(), order_by, vdf.alias)
			query += " AND {} > '{}'".format(order_by, order_by_start) if (order_by_start) else ""
			query += " AND {} < '{}'".format(order_by, order_by_end) if (order_by_end) else ""
			query += " AND {} = '{}'".format(by, column) 
			query_result = vdf.executeSQL(query = query, title = "Select the needed category points to draw the curve").fetchall()
			all_data += [[[item[0] for item in query_result], [float(item[1]) for item in query_result], column]]
		plt.figure(figsize = (10,8))
		plt.rcParams['axes.facecolor'] = '#FCFCFC'
		for idx, elem in enumerate(all_data):
			plt.plot(elem[0], elem[1], color = colors[idx], label = elem[2])
		plt.title('Plot of {} vs {}'.format(vdf.alias, order_by))
		plt.xticks(rotation = 90)
		plt.subplots_adjust(bottom = 0.24)
		plt.xlabel(order_by)
		plt.ylabel(vdf.alias)
		plt.gca().grid()
		plt.legend(title = vdf.alias)
		plt.show()
