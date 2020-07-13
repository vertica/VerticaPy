# (c) Copyright [2018-2020] Micro Focus or one of its affiliates. 
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
# |_     |~) _  _| _  /~\    _ |.
# |_)\/  |_)(_|(_||   \_/|_|(_|||
#    /                           
#              ____________       ______
#             / __        `\     /     /
#            |  \/         /    /     /
#            |______      /    /     /
#                   |____/    /     /
#          _____________     /     /
#          \           /    /     /
#           \         /    /     /
#            \_______/    /     /
#             ______     /     /
#             \    /    /     /
#              \  /    /     /
#               \/    /     /
#                    /     /
#                   /     /
#                   \    /
#                    \  /
#                     \/
#                    _
# \  / _  __|_. _ _ |_)
#  \/ (/_|  | |(_(_|| \/
#                     /  
# VerticaPy allows user to create vDataFrames (Virtual Dataframes). 
# vDataFrames simplify data exploration, data cleaning and MACHINE LEARNING     
# in VERTICA. It is an object which keeps in it all the actions that the user 
# wants to achieve and execute them when they are needed.    										
#																					
# The purpose is to bring the logic to the data and not the opposite !
#
# 
# Modules
#
# Standard Python Modules
from random import shuffle
import math, statistics
# Other Python Modules
import matplotlib.pyplot as plt
import matplotlib.colors as plt_colors
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
# VerticaPy Modules
from verticapy.utilities import *
from verticapy.toolbox import *
#
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
# Functions used by vDataFrames to draw graphics which are not useful independantly.
#
#---#
def autocorr_plot(x: list, 
				  y: list, 
				  color = '#263133',
				  title = ""):
	plt.figure(figsize = (12, 9)) if isnotebook() else plt.figure(figsize = (10, 6))
	plt.rcParams['axes.facecolor'] = '#FCFCFC'
	plt.plot(x, y, color = color)
	plt.title(title)
	plt.xticks(rotation = 90)
	plt.subplots_adjust(bottom = 0.24)
	plt.xlabel("lag")
	plt.ylabel("Autocorrelation")
	plt.gca().grid()
	plt.show()
#---#
def bar(vdf,
		method: str = "density",
		of = None,
		max_cardinality: int = 6,
		bins: int = 0,
		h: float = 0,
		color: str = '#263133'):
	x, y, z, h, is_categorical = compute_plot_variables(vdf, method = method, of = of, max_cardinality = max_cardinality, bins = bins, h = h)
	plt.figure(figsize = (14, min(len(x), 600))) if isnotebook() else plt.figure(figsize = (10, 6))
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
	elif ((method in ["avg", "min", "max", "sum"] or '%' in method) and (of != None)):
		aggregate = "{}({})".format(method.upper(), of)
		plt.ylabel(aggregate)
		plt.title('{} group by {}'.format(aggregate, vdf.alias))
	else:
		plt.xlabel('Frequency')
		plt.title('Count by {}'.format(vdf.alias))
	plt.show()
#---#
def bar2D(vdf,
		  columns: list,
		  method: str = "density",
		  of: str = "",
		  max_cardinality: tuple = (6, 6),
		  h: tuple = (None, None),
		  stacked: bool = False,
		  fully_stacked: bool = False):
	colors = gen_colors()
	all_columns = vdf.pivot_table(columns, method = method, of = of, h = h, max_cardinality = max_cardinality, show = False).values
	all_columns = [[column] + all_columns[column] for column in all_columns]
	n = len(all_columns)
	m = len(all_columns[0])
	n_groups = m - 1
	bar_width = 0.5
	plt.figure(figsize = (14, min(m * 3, 600))) if isnotebook() else plt.figure(figsize = (10, 6))
	plt.rcParams['axes.facecolor'] = '#F5F5F5'
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
				plt.barh([elem for elem in range(n_groups)], current_column, bar_width, alpha = 0.86, color = colors[(i - 1) % len(colors)], label = current_label, left = last_column)
			else:
				plt.barh([elem + (i - 1) * bar_width / (n - 1) for elem in range(n_groups)], current_column, bar_width / (n - 1), alpha = 0.86, color = colors[(i - 1) % len(colors)], label = current_label)
		if (stacked):
			plt.yticks([elem for elem in range(n_groups)], all_columns[0][1:m])
		else:
			plt.yticks([elem + bar_width / 2 - bar_width / 2 / (n - 1) for elem in range(n_groups)], all_columns[0][1:m])
		plt.subplots_adjust(left = max(0.3, len(max([str(item) for item in all_columns[0][1:m]], key=len)) / 140.0))
		plt.ylabel(columns[0])
		if (method == "mean"):
			method = "avg"
		if (method == "density"):
			plt.xlabel('Density')
			plt.title('Distribution of {} group by {}'.format(columns[0], columns[1]))
		elif ((method in ["avg", "min", "max", "sum"]  or '%' in method) and (of != None)):
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
					total[j-1] += float(all_columns[i][j] if (all_columns[i][j] != None) else 0)
		for i in range(1,n):
			for j in range(1, m):
				if not(type(all_columns[i][j]) in [str]):
					if (total[j - 1] != 0):
						all_columns[i][j] = float(all_columns[i][j] if (all_columns[i][j] != None) else 0) / total[j - 1]
					else:
						all_columns[i][j] = 0
		for i in range(1, n):
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
			plt.barh([elem for elem in range(n_groups)], current_column, bar_width, alpha = 0.86, color = colors[(i - 1) % len(colors)], label = current_label, left = last_column)
		plt.yticks([elem for elem in range(n_groups)], all_columns[0][1:m])
		plt.subplots_adjust(left = max(0.3, len(max([str(item) for item in all_columns[0][1:m]], key = len)) / 140.0))
		plt.ylabel(columns[0])
		plt.xlabel('Density per category')
		plt.title('Distribution per category of {} group by {}'.format(columns[0], columns[1]))
	plt.legend(title = columns[1], loc = 'center left', bbox_to_anchor = [1, 0.5])
	plt.gca().set_axisbelow(True)
	plt.gca().xaxis.grid()
	plt.show()
#---#
def boxplot(vdf, 
			by: str = "", 
			h: float = 0, 
			max_cardinality: int = 8, 
			cat_priority: list = []):
	# SINGLE BOXPLOT
	if (by == ""):
		plt.figure(figsize = (12, 8)) if isnotebook() else plt.figure(figsize = (10, 6))
		if not(vdf.isnum()):
			raise TypeError("The column must be numerical in order to draw a boxplot")
		summarize = vdf.parent.describe(method = "numerical", columns = [vdf.alias], unique = False).transpose().values[vdf.alias.replace('"', '')]
		for i in range(0,2):
			del summarize[0]
		plt.rcParams['axes.facecolor'] = '#F5F5F5'
		plt.xlabel(vdf.alias)
		box = plt.boxplot(summarize, notch = False, sym = '', whis = float('Inf'), vert = False, widths = 0.7, labels = [""], patch_artist = True)
		for median in box['medians']:
			median.set(color = 'black', linewidth = 1, )
		for patch in box['boxes']:
			patch.set_facecolor("#263133")
		plt.gca().xaxis.grid()
		plt.gca().set_axisbelow(True)
		plt.title('BoxPlot of {}'.format(vdf.alias))
		plt.show()
	# MULTI BOXPLOT
	else:
		try:
			if (vdf.alias == by):
				raise NameError("The column and the groupby can not be the same")
			elif (by not in vdf.parent.get_columns()):
				raise NameError("The column " + by + " doesn't exist")
			count = vdf.parent.shape()[0]
			is_numeric = vdf.parent[by].isnum()
			is_categorical = (vdf.parent[by].nunique(True) <= max_cardinality) or not(is_numeric)
			table = vdf.parent.__genSQL__()
			if not(is_categorical):
				enum_trans = vdf.parent[by].discretize(h = h, return_enum_trans = True)[0].replace("{}", by) + " AS " + by
				enum_trans += ", {}".format(vdf.alias)
				table = "(SELECT {} FROM {}) enum_table".format(enum_trans, table)
			if not(cat_priority):
				query = "SELECT {} FROM {} WHERE {} IS NOT NULL GROUP BY {} ORDER BY COUNT(*) DESC LIMIT {}".format(by, table, vdf.alias, by, max_cardinality)
				query_result = vdf.__executeSQL__(query = query, title = "Compute the categories of {}".format(by)).fetchall()
				cat_priority = [item for sublist in query_result for item in sublist]
			with_summarize = False
			query = []
			lp = "(" if (len(cat_priority) == 1) else ""
			rp = ")" if (len(cat_priority) == 1) else ""
			for idx, category in enumerate(cat_priority):
				tmp_query = "SELECT MIN({}) AS min, APPROXIMATE_PERCENTILE ({} USING PARAMETERS percentile = 0.25) AS Q1, APPROXIMATE_PERCENTILE ({}".format(vdf.alias, vdf.alias, vdf.alias)
				tmp_query += "USING PARAMETERS percentile = 0.5) AS Median, APPROXIMATE_PERCENTILE ({} USING PARAMETERS percentile = 0.75) AS Q3, MAX".format(vdf.alias)
				tmp_query += "({}) AS max, '{}' FROM vdf_table".format(vdf.alias, '{}')
				tmp_query = tmp_query.format("None") if (category in ('None', None)) else tmp_query.format(category)
				tmp_query += " WHERE {} IS NULL".format(by) if (category in ('None', None)) else " WHERE {} = '{}'".format(by, str(category).replace("'", "''"))
				query += [lp + tmp_query + rp]
			query = "WITH vdf_table AS (SELECT * FROM {}) {}".format(table, " UNION ALL ".join(query))
			try:
				vdf.__executeSQL__(query = query, title = "Compute all the descriptive statistics for each category to draw the box plot")
				query_result = vdf.parent._VERTICAPY_VARIABLES_["cursor"].fetchall()
			except:
				query_result = []
				for idx, category in enumerate(cat_priority):
					tmp_query = "SELECT MIN({}) AS min, APPROXIMATE_PERCENTILE ({} USING PARAMETERS percentile = 0.25) AS Q1, APPROXIMATE_PERCENTILE ({}".format(vdf.alias, vdf.alias, vdf.alias)
					tmp_query += "USING PARAMETERS percentile = 0.5) AS Median, APPROXIMATE_PERCENTILE ({} USING PARAMETERS percentile = 0.75) AS Q3, MAX".format(vdf.alias)
					tmp_query += "({}) AS max, '{}' FROM {}".format(vdf.alias, '{}', vdf.parent.__genSQL__())
					tmp_query = tmp_query.format("None") if (category in ('None', None)) else tmp_query.format(str(category).replace("'", "''"))
					tmp_query += " WHERE {} IS NULL".format(by) if (category in ('None', None)) else " WHERE {} = '{}'".format(by, str(category).replace("'", "''"))
					vdf.__executeSQL__(query = tmp_query, title = "Compute all the descriptive statistics for each category to draw the box plot, one at a time")
					query_result += [vdf.parent._VERTICAPY_VARIABLES_["cursor"].fetchone()]
			cat_priority = [item[-1] for item in query_result]
			result = [[float(item[i]) for i in range(0,5)] for item in query_result]
			result.reverse()
			cat_priority.reverse()
			if (vdf.parent[by].category() == "text"):
				labels = []
				for item in cat_priority:
					labels += [item[0:47] + "..."] if (len(str(item)) > 50) else [item]
			else:
				labels = cat_priority
			plt.figure(figsize = (14, 8)) if isnotebook() else plt.figure(figsize = (10, 6))
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
			box = plt.boxplot(result, notch = False, sym = '', whis = float('Inf'), widths = 0.5, labels = labels, patch_artist = True)
			plt.title('BoxPlot of {} group by {}'.format(vdf.alias, by))
			plt.subplots_adjust(bottom = max(0.3, len(max([str(item) for item in labels], key = len)) / 90.0))
			colors = gen_colors()
			for median in box['medians']:
				median.set(color = 'black', linewidth = 1,)
			for patch,color in zip(box['boxes'], colors):
				patch.set_facecolor(color)
			plt.show()
		except Exception as e:
			raise Exception("{}\nAn error occured during the BoxPlot creation.".format(e))
#---#
def boxplot2D(vdf, 
			  columns: list = []):
	if not(columns):
		columns = vdf.numcol()
	for column in columns:
		if (column not in vdf.numcol()):
			print("\u26A0 Warning: The Virtual Column {} is not numerical.\nIt will be ignored.".format(column))
			columns.remove(column)
	if not(columns):
		print("\u26A0 Warning: No numerical columns found to draw the multi boxplot")
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
			plt.figure(figsize = (14, 8)) if isnotebook() else plt.figure(figsize = (10, 6))
			plt.rcParams['axes.facecolor'] = '#F8F8F8'
			plt.xticks(rotation = 90)
			box = plt.boxplot(result, notch = False, sym = '', whis = float('Inf'), widths = 0.5, labels = columns, patch_artist = True)
			plt.title('Multi BoxPlot of the vDataFrame')
			plt.subplots_adjust(bottom = max(0.3, len(max([str(item) for item in columns], key = len)) / 90.0))
			colors = gen_colors()
			for median in box['medians']:
				median.set(color = 'black', linewidth = 1,)
			for patch,color in zip(box['boxes'], colors):
				patch.set_facecolor(color)
			plt.show()
		except Exception as e:
			raise Exception("{}\nAn error occured during the BoxPlot creation.".format(e))
#---#
def bubble(vdf,
		   columns: list,
		   catcol: str = "",
		   max_nb_points: int = 1000,
		   bbox: list = [],
		   img: str = ""):
	colors = gen_colors()
	if not(catcol):
		tablesample = max_nb_points / vdf.shape()[0]
		query = "SELECT {}, {}, {} FROM {} WHERE __verticapy_split__ < {} AND {} IS NOT NULL AND {} IS NOT NULL AND {} IS NOT NULL LIMIT {}".format(columns[0], columns[1], columns[2], vdf.__genSQL__(True), tablesample, columns[0], columns[1], columns[2], max_nb_points)
		query_result = vdf.__executeSQL__(query = query, title = "Select random points to draw the scatter plot").fetchall()
		max_size = max([float(item[2]) for item in query_result])
		min_size = min([float(item[2]) for item in query_result])
		column1, column2, size = [float(item[0]) for item in query_result], [float(item[1]) for item in query_result], [1000 * (float(item[2]) - min_size) / max((max_size - min_size), 1e-50) for item in query_result]
		plt.figure(figsize = (14, 10)) if isnotebook() else plt.figure(figsize = (10, 6))
		if (bbox):
			plt.xlim(bbox[0], bbox[1])
			plt.ylim(bbox[2], bbox[3])
		if (img):
			im = plt.imread(img)
			if not(bbox):
				bbox = (min(column1), max(column1), min(column2), max(column2))
				plt.xlim(bbox[0], bbox[1])
				plt.ylim(bbox[2], bbox[3])
			plt.imshow(im, extent = bbox)
		plt.gca().grid()
		plt.gca().set_axisbelow(True)
		plt.title('Bubble Plot of {} vs {}'.format(columns[0], columns[1]))
		plt.ylabel(columns[1])
		plt.xlabel(columns[0])
		scatter = plt.scatter(column1, column2, color = colors[0], s = size, alpha = 0.5)
		plt.gca().grid()
		plt.gca().set_axisbelow(True)
		kw = dict(prop = "sizes", num = 6, color = colors[0], alpha = 0.6, func = lambda s: (s * (max_size - min_size) + min_size) / 1000)
		plt.legend(*scatter.legend_elements(**kw), bbox_to_anchor = [1, 0.5], loc="center left", title = columns[2])
	else:
		count = vdf.shape()[0]
		all_categories = vdf[catcol].distinct()
		fig = plt.figure(figsize = (14, 10)) if isnotebook() else plt.figure(figsize = (10, 6))
		ax = plt
		if (bbox):
			plt.xlim(bbox[0], bbox[1])
			plt.ylim(bbox[2], bbox[3])
		if (img):
			im = plt.imread(img)
			if not(bbox):
				aggr = vdf.agg(columns = [columns[0], columns[1]], func = ["min", "max"])
				bbox = (aggr.values["min"][0], aggr.values["max"][0], aggr.values["min"][1], aggr.values["max"][1])
				plt.xlim(bbox[0], bbox[1])
				plt.ylim(bbox[2], bbox[3])
			plt.imshow(im, extent = bbox)
		others = []
		groupby_cardinality = vdf[catcol].nunique(True)
		count = vdf.shape()[0]
		tablesample = 0.1 if (count > 10000) else 0.9
		all_columns, all_scatter = [], []
		max_size, min_size = float(vdf[columns[2]].max()), float(vdf[columns[2]].min())
		for idx, category in enumerate(all_categories):
			query = "SELECT {}, {}, {} FROM {} WHERE  __verticapy_split__ < {} AND {} = '{}' AND {} IS NOT NULL AND {} IS NOT NULL AND {} IS NOT NULL LIMIT {}"
			query = query.format(columns[0], columns[1], columns[2], vdf.__genSQL__(True), tablesample, catcol, str(category).replace("'", "''"), columns[0], columns[1], columns[2], int(max_nb_points / len(all_categories))) 
			vdf.__executeSQL__(query = query, title = "Select random points to draw the bubble plot (category = '{}')".format(str(category)))
			query_result = vdf._VERTICAPY_VARIABLES_["cursor"].fetchall()
			column1, column2, size = [float(item[0]) for item in query_result], [float(item[1]) for item in query_result], [1000 * (float(item[2]) - min_size) / max((max_size - min_size), 1e-50) for item in query_result]
			all_columns += [[column1, column2, size]]
			all_scatter += [ax.scatter(column1, column2, s = size, alpha = 0.8, color = colors[idx % len(colors)])]
		for idx, item in enumerate(all_categories):
			if (len(str(item)) > 20):
				all_categories[idx] = str(item)[0:20] + "..."
		plt.gca().grid()
		plt.gca().set_axisbelow(True)
		kw = dict(prop = "sizes", num = 6, color = colors[0], alpha = 0.6, func = lambda s: (s * (max_size - min_size) + min_size) / 1000)
		leg1 = ax.legend(*all_scatter[0].legend_elements(**kw), bbox_to_anchor = [1, 0.5], loc="center left", title = columns[2])
		plt.title('Scatter Plot of {} vs {}'.format(columns[0], columns[1]))
		ax.xlabel(columns[0])
		ax.ylabel(columns[1])
		leg2 = ax.legend(all_scatter, all_categories, title = catcol, loc = 'center right', bbox_to_anchor = [-0.06, 0.5])
		fig.add_artist(leg1)
	plt.show()
#---#
def cmatrix(matrix,
			columns_x,
			columns_y,
			n: int,
			m: int,
			vmax: float,
			vmin: float,
			cmap: str = 'PRGn',
			title: str = "",
			colorbar: str = "",
			x_label: str = "",
			y_label: str = "",
			with_numbers: bool = True,
			mround: int = 3):
		matrix_array = [[round(float(matrix[i][j]), mround) if (matrix[i][j] != None and matrix[i][j] != '') else float('nan') for i in range(1, m + 1)] for j in range(1, n + 1)]
		plt.figure(figsize = (min(m * 1.4, 500), min(n * 1.4, 500))) if isnotebook() else plt.figure(figsize = (min(int(m/1.3) + 2, 500), min(int(n/1.3) + 1, 500)))
		plt.title(title)
		if ((vmax == 1) and vmin in [0, -1]):
			plt.imshow(matrix_array, cmap = cmap, interpolation = 'nearest', vmax = vmax, vmin = vmin)
		else:
			plt.imshow(matrix_array, cmap = cmap, interpolation = 'nearest')
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
		if (with_numbers):
			for y_index in range(n):
			    for x_index in range(m):
			        label = matrix_array[y_index][x_index]
			        plt.gca().text(x_index, y_index, label, color = 'black', ha = 'center', va = 'center')
		plt.show()
#---#
def compute_plot_variables(vdf, 
			   	   		   method: str = "density", 
			   	   		   of: str = "", 
			       		   max_cardinality: int = 6, 
			       		   bins: int = 0, 
			       		   h: float = 0, 
			       		   pie: bool = False):
	if (method == "median"): method = "50%" 
	elif (method == "mean"): method = "avg" 
	if (((method in ["avg", "min", "max", "sum"]) or (method and method[-1] == "%")) and (of)):
		if (method in ["avg", "min", "max", "sum"]):
			aggregate = "{}({})".format(method.upper(), str_column(of))
			others_aggregate = method
		elif (method and method[-1] == "%"):
			aggregate = "APPROXIMATE_PERCENTILE({} USING PARAMETERS percentile = {})".format(str_column(of), float(method[0:-1]) / 100)
			others_aggregate = method
		else:
			raise NameError("The column '" + of + "' doesn't exist")
	elif (method in ["density", "count"]):
		aggregate = "count(*)"
		others_aggregate = "sum"
	else:
		raise ValueError("The parameter 'method' must be in [avg|mean|min|max|sum|median|q%]")
	# depending on the cardinality, the type, the vColumn can be treated as categorical or not
	cardinality, count, is_numeric, is_date, is_categorical = vdf.nunique(True), vdf.parent.shape()[0], vdf.isnum(), (vdf.category()=="date"), False
	rotation = 0 if ((is_numeric) and (cardinality > max_cardinality)) else 90
	# case when categorical
	if ((((cardinality <= max_cardinality) or not(is_numeric)) or pie) and not(is_date)):
		if ((is_numeric) and not(pie)):
			query = "SELECT {}, {} FROM {} WHERE {} IS NOT NULL GROUP BY {} ORDER BY {} ASC LIMIT {}".format(
				vdf.alias, aggregate, vdf.parent.__genSQL__(), vdf.alias, vdf.alias, vdf.alias, max_cardinality)
		else:
			table = vdf.parent.__genSQL__()
			if ((pie) and (is_numeric)):
				enum_trans = vdf.discretize(h = h, return_enum_trans = True)[0].replace("{}", vdf.alias) + " AS " + vdf.alias
				if (of):
					enum_trans += " , " + of
				table = "(SELECT {} FROM {}) enum_table".format(enum_trans, table)
			query = "(SELECT {} AS {}, {} FROM {} GROUP BY {} ORDER BY 2 DESC LIMIT {})".format(convert_special_type(vdf.category(), True, vdf.alias), vdf.alias, aggregate, table, convert_special_type(vdf.category(), True, vdf.alias), max_cardinality)
			if (cardinality > max_cardinality):
				query += (" UNION (SELECT 'Others', {}(count) FROM (SELECT {} AS count FROM {} " + 
					"GROUP BY {} ORDER BY {} DESC OFFSET {}) y LIMIT 1) ORDER BY 2 DESC")
				query = query.format(others_aggregate, aggregate, table, vdf.alias, aggregate, max_cardinality)
		vdf.__executeSQL__(query, title = "Compute the histogram heights")
		query_result = vdf.parent._VERTICAPY_VARIABLES_["cursor"].fetchall()
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
			query = "SELECT DATEDIFF('second', MIN({}), MAX({})) FROM ".format(vdf.alias, vdf.alias)
			vdf.__executeSQL__(query = query, title = "Compute the histogram interval")
			query_result = vdf.parent._VERTICAPY_VARIABLES_["cursor"].fetchone()
			h = float(query_result[0]) / bins
		min_date = vdf.min()
		converted_date = "DATEDIFF('second', '{}', {})".format(min_date, vdf.alias)
		query = "SELECT FLOOR({} / {}) * {}, {} FROM {} WHERE {} IS NOT NULL GROUP BY 1 ORDER BY 1".format(converted_date, h, h, aggregate, vdf.parent.__genSQL__(), vdf.alias)
		vdf.__executeSQL__(query = query, title = "Compute the histogram heights")
		query_result = vdf.parent._VERTICAPY_VARIABLES_["cursor"].fetchall()
		x = [float(item[0]) for item in query_result]
		y = [item[1] / float(count) for item in query_result] if (method == "density") else [item[1] for item in query_result]
		query = ""
		for idx, item in enumerate(query_result):
			query += " UNION (SELECT TIMESTAMPADD('second' , {}, '{}'::timestamp))".format(math.floor(h * idx), min_date)
		query = query[7:-1] + ")"
		h = 0.94 * h
		vdf.parent._VERTICAPY_VARIABLES_["cursor"].execute(query)
		query_result = vdf.parent._VERTICAPY_VARIABLES_["cursor"].fetchall()
		z = [item[0] for item in query_result]
		z.sort()
		is_categorical = True
	# case when numerical
	else:
		if ((h <= 0) and (bins <= 0)):
			h = vdf.numh()
		elif (bins > 0):
			h = float(vdf.max() - vdf.min()) / bins
		if (vdf.ctype == "int") or (h == 0):
			h = max(1.0, h)
		query = "SELECT FLOOR({} / {}) * {}, {} FROM {} WHERE {} IS NOT NULL GROUP BY 1 ORDER BY 1"
		query = query.format(vdf.alias, h, h, aggregate, vdf.parent.__genSQL__(), vdf.alias)
		vdf.__executeSQL__(query = query, title = "Compute the histogram heights")
		query_result = vdf.parent._VERTICAPY_VARIABLES_["cursor"].fetchall()
		y = [item[1] / float(count) for item in query_result] if (method == "density") else [item[1] for item in query_result]
		x = [float(item[0]) + h/2 for item in query_result]
		h = 0.94 * h
		z = None
	return [x, y, z, h, is_categorical]
#---#
def density(vdf,
			a = None,
			kernel: str = "gaussian",
			smooth: int = 200,
			color: str = '#263133'):
	if (kernel == "gaussian"):
		def fkernel(x):
			try:
				return math.exp( - 1 / 2 * ((x) ** 2)) / (math.sqrt(2 * math.pi))
			except:
				return 0
	elif (kernel == "logistic"):
		def fkernel(x):
			try:
				return 1 / (2 + math.exp(x) + math.exp( - x))
			except:
				return 0
	elif (kernel == "sigmoid"):
		def fkernel(x):
			try:
				return 2 / (math.pi * (math.exp(x) + math.exp( - x)))
			except:
				return 0
	elif (kernel == "silverman"):
		def fkernel(x):
			return math.exp( - 1 / math.sqrt(2) * abs(x)) / (2) * math.sin(abs(x) / math.sqrt(2) + math.pi / 4)
	else:
		raise ValueError("The parameter 'kernel' must be in [gaussian|logistic|sigmoid|silverman]")
	if (a == None):
		a = max(1.06 * vdf.std() / vdf.count() ** (1.0 / 5.0), 1e-5)
	if not(vdf.isnum()):
		raise TypeError("Cannot draw a density plot for non-numerical columns")
	x, y, z, h, is_categorical = compute_plot_variables(vdf, method = "count", max_cardinality = 1) 
	x = [item - h / 2 / 0.94 for item in x]
	N = sum(y)
	y_smooth = []
	x_smooth = [(max(x) - min(x)) * i / smooth + min(x) for i in range(0, smooth + 1)]
	n = len(y)
	for x0_smooth in x_smooth:
		K = sum([y[i] * fkernel(((x0_smooth - x[i]) / a) ** 2) / (a * N) for i in range(0, len(x))])
		y_smooth += [K]
	plt.figure(figsize = (12, 10)) if isnotebook() else plt.figure(figsize = (10, 6))
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
#---#
def gen_cmap():
	cm1 = LinearSegmentedColormap.from_list("vml", ["#FFFFFF", "#263133"], N = 1000)
	cm2 = LinearSegmentedColormap.from_list("vml", ["#FE5016", "#FFFFFF", "#263133"], N = 1000)
	return (cm1, cm2)
#---#
def gen_colors():
	colors = ['#263133',
			  '#FE5016',
	          '#0073E7',
	          '#19A26B',
	          '#FCDB1F',
	          '#000000',
	          '#2A6A74',
	          '#861889',
	          '#00B4E0',
	          '#90EE90',
	          '#FF7F50',
	          '#B03A89']
	all_colors = [item for item in plt_colors.cnames]
	shuffle(all_colors)
	for c in all_colors:
		if c not in colors:
			colors += [c]
	return colors
#---#
def gmaps_geo_plot_cmap(vdf,
				   	 	lat: str,
				   	 	lon: str,
				   	 	column_map: str,
				   	 	api_key: str,
				   	 	max_nb_points: int = 500):
	import gmaps
	gmaps.configure(api_key = api_key)
	fig = gmaps.figure(mouse_handling = "NONE", display_toolbar = False)
	query = "SELECT {}, {}, {}".format(lat, lon, column_map)
	where = " WHERE {} IS NOT NULL AND {} IS NOT NULL AND {} IS NOT NULL".format(lat, lon, column_map)
	query += " FROM {}{} LIMIT {}".format(vdf.__genSQL__(), where, max_nb_points)
	query_result = vdf.__executeSQL__(query = query, title = "Compute the different elements of the Geo Plot").fetchall()
	lat_lon = [(float(elem[0]), float(elem[1])) for elem in query_result]
	column_map = [float(elem[2]) for elem in query_result]
	heatmap_layer = gmaps.heatmap_layer(lat_lon, 
										weights = column_map,
										max_intensity = 30, 
										point_radius = 3.0)
	fig.add_layer(heatmap_layer)
	return fig
#---#
def gmaps_geo_plot_regular(vdf,
		   	 	     	   lat: str,
		   	 	     	   lon: str,
		   	 	     	   api_key: str,
		   	 	     	   catcol: str = "",
		   	 	     	   max_nb_points: int = 500):
	import gmaps
	gmaps.configure(api_key = api_key)
	fig = gmaps.figure(mouse_handling = "NONE", display_toolbar = False)
	if (catcol):
		cat = vdf[catcol].distinct()
		colors = gen_colors()
		max_nb_points = int(max_nb_points / len(cat)) 
		for idx, elem in enumerate(cat):
			query = "SELECT {}, {}".format(lat, lon)
			where = " WHERE {} IS NOT NULL AND {} IS NOT NULL AND {} = '{}'".format(lat, lon, catcol, str(elem).replace("'", "''"))
			query += " FROM {}{} LIMIT {}".format(vdf.__genSQL__(), where, max_nb_points)
			query_result = vdf.__executeSQL__(query = query, title = "Compute the different elements of the Geo Plot").fetchall()
			lat_lon = [(float(elem[0]), float(elem[1])) for elem in query_result]
			layer = gmaps.symbol_layer(lat_lon, 
									   fill_color = colors[idx % len(colors)],
									   stroke_color = colors[idx % len(colors)], 
									   scale = 2)
			fig.add_layer(layer)
	else:
		query = "SELECT {}, {}".format(lat, lon)
		where = " WHERE {} IS NOT NULL AND {} IS NOT NULL".format(lat, lon)
		query += " FROM {}{} LIMIT {}".format(vdf.__genSQL__(), where, max_nb_points)
		query_result = vdf.__executeSQL__(query = query, title = "Compute the different elements of the Geo Plot").fetchall()
		lat_lon = [(float(elem[0]), float(elem[1])) for elem in query_result]
		layer = gmaps.symbol_layer(lat_lon, 
								   fill_color = '#263133',
								   stroke_color = '#263133', 
								   scale = 2)
		fig.add_layer(layer)
	return fig
#---#
def hexbin(vdf,
		   columns: list,
		   method: str = "count",
		   of: str = "",
		   cmap: str = 'Blues',
		   gridsize: int = 10,
		   color: str = "white",
		   bbox: list = [],
		   img: str = ""):
	if (len(columns) != 2):
		raise ValueError("The parameter 'columns' must be exactly of size 2 to draw the hexbin")
	if (method == "mean"):
		method = "avg"
	if ((method in ["avg", "min", "max", "sum"]) and (of) and ((of in vdf.get_columns()) or (str_column(of) in vdf.get_columns()))):
		aggregate = "{}({})".format(method, of)
		others_aggregate = method
		if (method == "avg"):
			reduce_C_function = statistics.mean
		elif (method == "min"):
			reduce_C_function = min
		elif (method == "max"):
			reduce_C_function = max
		elif (method=="sum"): 
			reduce_C_function = sum
	elif (method in ("count", "density")):
		aggregate = "count(*)"
		reduce_C_function = sum
	else:
		raise ValueError("The parameter 'method' must be in [avg|mean|min|max|sum|median|q%]")
	count = vdf.shape()[0]
	if (method == "density"):
		over = "/" + str(float(count))
	else:
		over=""
	query = "SELECT {}, {}, {}{} FROM {} GROUP BY {}, {}".format(columns[0], columns[1], aggregate, over, vdf.__genSQL__(), columns[0], columns[1])
	query_result = vdf.__executeSQL__(query = query, title = "Group all the elements for the Hexbin Plot").fetchall()
	column1, column2, column3 = [], [], []
	for item in query_result:
		if ((item[0] != None) and (item[1] != None) and (item[2] != None)):
			column1 += [float(item[0])] * 2
			column2 += [float(item[1])] * 2
			if (reduce_C_function in [min, max, statistics.mean]):
				column3 += [float(item[2])] * 2
			else:
				column3 += [float(item[2])/2] * 2
	plt.figure(figsize = (14, 10)) if isnotebook() else plt.figure(figsize = (10, 6))
	if (bbox):
		plt.xlim(bbox[0], bbox[1])
		plt.ylim(bbox[2], bbox[3])
	if (img):
		im = plt.imread(img)
		if not(bbox):
			bbox = (min(column1), max(column1), min(column2), max(column2))
			plt.xlim(bbox[0], bbox[1])
			plt.ylim(bbox[2], bbox[3])
		plt.imshow(im, extent = bbox)
	plt.rcParams['axes.facecolor'] = 'white'
	plt.title('Hexbin of {} vs {}'.format(columns[0], columns[1]))
	plt.ylabel(columns[1])
	plt.xlabel(columns[0])
	plt.hexbin(column1, column2, C = column3, reduce_C_function = reduce_C_function, gridsize = gridsize, color = color, cmap = cmap, mincnt = 1)
	if (method == "density"):
		plt.colorbar().set_label(method)
	else:
		plt.colorbar().set_label(aggregate)
	plt.show()
#---#
def hist(vdf,
		 method: str = "density",
		 of = None,
		 max_cardinality: int = 6,
		 bins: int = 0,
		 h: float = 0,
		 color: str = '#263133'):
	x, y, z, h, is_categorical = compute_plot_variables(vdf, method, of, max_cardinality, bins, h)
	is_numeric = vdf.isnum()
	rotation = 0 if ((is_numeric) and not(is_categorical)) else 90
	plt.figure(figsize = (min(len(x), 600), 8)) if isnotebook() else plt.figure(figsize = (10, 6))
	plt.rcParams['axes.facecolor'] = '#F5F5F5'
	plt.bar(x, y, h, color = color, alpha = 0.86)
	plt.xlabel(vdf.alias)
	plt.gca().set_axisbelow(True)
	plt.gca().yaxis.grid()
	if (is_categorical):
		if not(is_numeric):
			new_z = []
			for item in z:
				new_z += [item[0:47] + "..."] if (len(str(item)) > 50) else [item]
		else:
			new_z = z
		plt.xticks(x, new_z, rotation = rotation)
		plt.subplots_adjust(bottom = max(0.3, len(max([str(item) for item in z], key = len)) / 140.0))
	if (method == "density"):
		plt.ylabel('Density')
		plt.title('Distribution of {}'.format(vdf.alias))
	elif ((method in ["avg", "min", "max", "sum", "mean"] or ('%' in method)) and (of != None)):
		aggregate = "{}({})".format(method, of)
		plt.ylabel(aggregate)
		plt.title('{} group by {}'.format(aggregate, vdf.alias))
	else:
		plt.ylabel('Frequency')
		plt.title('Count by {}'.format(vdf.alias))
	plt.show()
#---#
def hist2D(vdf,
		   columns: list,
		   method = "density",
		   of: str = "",
		   max_cardinality: tuple = (6, 6),
		   h: tuple = (None, None),
		   stacked: bool = False):
	colors = gen_colors()
	all_columns = vdf.pivot_table(columns, method = method, of = of, h = h, max_cardinality = max_cardinality, show = False).values
	all_columns = [[column] + all_columns[column] for column in all_columns]
	n, m = len(all_columns), len(all_columns[0])
	n_groups = m - 1
	bar_width = 0.5
	plt.figure(figsize = (min(600, 3 * m), 11)) if isnotebook() else plt.figure(figsize = (10, 6))
	plt.rcParams['axes.facecolor'] = '#F5F5F5'
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
				for idx, item in enumerate(all_columns[i - 1][1:m]):
					try:
						last_column[idx] += float(item)
					except:
						last_column[idx] += 0
			plt.bar([elem for elem in range(n_groups)], current_column, bar_width, alpha = 0.86, color = colors[(i - 1) % len(colors)], label = current_label, bottom = last_column)
		else:
			plt.bar([elem + (i - 1) * bar_width / (n - 1) for elem in range(n_groups)], current_column, bar_width / (n - 1), alpha=0.86, color = colors[(i - 1) % len(colors)], label = current_label)
	if (stacked):
		plt.xticks([elem for elem in range(n_groups)], all_columns[0][1:m], rotation = 90)
	else:
		plt.xticks([elem + bar_width / 2 - bar_width / 2 / (n - 1) for elem in range(n_groups)], all_columns[0][1:m], rotation = 90)
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
	plt.legend(title = columns[1], loc = 'center left', bbox_to_anchor = [1, 0.5])
	plt.gca().set_axisbelow(True)
	plt.gca().yaxis.grid()
	plt.show()
#---#
def hcharts(vdf,
		    columns: list,
		    chart_type: str = "bar",
		    method: str = "density",
		    color: str = '#263133',
		    x_axis: str = "",
		    x_unit: str = "",
		    y_axis: str = "",
		    y_unit: str = "",
		    title: str = "",
		    subtitle: str = "",
		    lineWidth: float = 2.0):
	from highcharts import Highchart
	chart = Highchart()
	chart.set_options('chart', {'inverted': True})
	options = {
	    'title': {
	        'text': title
	    },
	    'subtitle': {
	        'text': subtitle
	    },
	    'xAxis': {
	        'reversed': False,
	        'title': {
	            'enabled': True,
	            'text': x_axis
	        },
	        'labels': {
	            'formatter': 'function () {\
	                return this.value + "' + x_unit + '";\
	            }'
	        },
	        'maxPadding': 0.05,
	        'showLastLabel': True
	    },
	    'yAxis': {
	        'title': {
	            'text': y_axis
	        },
	        'labels': {
	            'formatter': "function () {\
	                return this.value + '" + y_unit + "';\
	            }"
	        },
	        'lineWidth': lineWidth
	    },
	    'legend': {
	        'enabled': False
	    },
	    'tooltip': {
	        'headerFormat': '<b>{series.name}</b><br/>',
	        'pointFormat': '{point.x} ' + x_unit + ': {point.y}' + y_unit
	    }
	}
	if (chart_type == "bar"):
		x, y, z, h, is_categorical = compute_plot_variables(vdf[columns[0]], 
											   	   		    method = "density", 
											   	   		    of = "", 
											       		    max_cardinality = 6,
											       		    h = 10,
											       		    pie = False)
		if (is_categorical):
			options['xAxis']["categories"] = z
			data = [[z[i], y[i]] for i in range(len(y))]
	chart.set_dict_options(options)
	chart.add_data_set(data, 
					   chart_type,
					   'Test',
					   color = color,
					   marker = {'enabled': True, 
					   			 'fillColor': 'blue'})
	if isnotebook():
		chart.save_file()
		from IPython.display import IFrame
		IFrame(src='./Chart.html', width = 600, height = 300)
	return chart
#---#
def multiple_hist(vdf,
				  columns: list,
				  method: str = "density",
				  of: str = "",
				  h: float = 0):
	colors = gen_colors()
	if (len(columns) > 5):
		raise Exception("The number of column must be <= 5 to use 'multiple_hist' method")
	else:
		plt.figure(figsize = (12, 7)) if isnotebook() else plt.figure(figsize = (12, 6))
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
				plt.bar(x, y, h, color = colors[idx % len(colors)], alpha = alpha, label = column)
				alpha -= 0.2
				all_columns += [columns[idx]]
			else:
				print("\u26A0 Warning: {} is not numerical. Its histogram will not be draw.".format(column))
		plt.xlabel(", ".join(all_columns))
		plt.gca().set_axisbelow(True)
		plt.gca().yaxis.grid()
		if (method == "density"):
			plt.ylabel('Density')
		elif ((method in ["avg", "min", "max", "sum", "mean"] or ('%' in method)) and (of)):
			plt.ylabel(method + "(" + of + ")")
		else:
			plt.ylabel('Frequency')
		plt.title("Multiple Histograms")
		plt.legend(title = "columns", loc = 'center left', bbox_to_anchor = [1, 0.5])
		plt.show()
#---#
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
		if not(vdf[column].isnum()):
			print("\u26A0 Warning: The Virtual Column {} is not numerical.\nIt will be ignored.".format(column))
			columns.remove(column)
	if not(columns):
		raise Exception("No numerical columns found to draw the multi TS plot")
	colors = gen_colors()
	query = "SELECT {}, {} FROM {} WHERE {} IS NOT NULL".format(order_by, ", ".join(columns), vdf.__genSQL__(), order_by)
	query += " AND {} > '{}'".format(order_by, order_by_start) if (order_by_start) else ""
	query += " AND {} < '{}'".format(order_by, order_by_end) if (order_by_end) else ""
	query += " AND " + " AND ".join(["{} IS NOT NULL".format(column) for column in columns])
	query += " ORDER BY {}".format(order_by)
	vdf.__executeSQL__(query = query, title = "Select the needed points to draw the curves")
	query_result = vdf._VERTICAPY_VARIABLES_["cursor"].fetchall()
	order_by_values = [item[0] for item in query_result]
	try:
		if (type(order_by_values[0]) == str):
			from dateutil.parser import parse
			order_by_values = [parse(elem) for elem in order_by_values]
	except:
		pass
	alpha = 0.3
	plt.figure(figsize = (14, 10)) if isnotebook() else plt.figure(figsize = (10, 6))
	plt.gca().grid()
	for i in range(0, len(columns)):
		plt.plot(order_by_values, [item[i + 1] for item in query_result], color = colors[i], label = columns[i])
	plt.rcParams['axes.facecolor'] = '#FCFCFC'
	plt.title('Multi Plot of the vDataFrame')
	plt.xticks(rotation = 90)
	plt.subplots_adjust(bottom = 0.24)
	plt.xlabel(order_by)
	plt.legend(title = "columns", loc = 'center left', bbox_to_anchor = [1, 0.5])
	plt.show()
#---#
def pie(vdf,
		method: str = "density",
		of = None,
		max_cardinality: int = 6,
		h: float = 0,
		donut: bool = False):
	colors = gen_colors() * 100
	x, y, z, h, is_categorical = compute_plot_variables(vdf, max_cardinality = max_cardinality, method = method, of = of, pie = True)
	z.reverse()
	y.reverse()
	explode = [0 for i in y]
	explode[max(zip(y, range(len(y))))[1]] = 0.13
	current_explode = 0.15
	total_count = sum(y)
	for idx,item in enumerate(y):
		if ((item < 0.05) or ((item > 1) and (float(item) / float(total_count) < 0.05))):
			current_explode = min(0.9, current_explode * 1.4) 
			explode[idx] = current_explode
	if (method == "density"):
		autopct = '%1.1f%%'
	else:
		def make_autopct(values, category):
		    def my_autopct(pct):
		        total = sum(values)
		        val = float(pct) * float(total) / 100.0
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
	plt.figure(figsize = (12, 10)) if isnotebook() else plt.figure(figsize = (10, 6))
	if (donut):
		explode = None
		centre_circle = plt.Circle((0,0), 0.72, color = '#666666', fc = 'white', linewidth = 1.25)
		fig = plt.gcf()
		fig.gca().add_artist(centre_circle)
	plt.pie(y, labels = z, autopct = autopct, colors = colors, shadow = True, startangle = 290, explode = explode)
	plt.subplots_adjust(bottom = 0.2)
	if (method == "density"):
		plt.title('Distribution of {}'.format(vdf.alias))
	elif ((method in ["avg", "min", "max", "sum"] or '%' in method) and (of != None)):
		aggregate = "{}({})".format(method,of)
		plt.title('{} group by {}'.format(aggregate,vdf.alias))
	else:
		plt.title('Count by {}'.format(vdf.alias))
	plt.show()
#---#
def pivot_table(vdf,
				columns,
				method: str = "count",
				of: str = "",
				h: tuple = (None, None),
				max_cardinality: tuple = (20, 20),
				show: bool = True,
				cmap: str = 'Blues',
				with_numbers: bool = True):
	if (method == "median"): method = "50%"
	elif (method == "mean"): method = "avg" 
	if ((method.lower() in ["avg", "min", "max", "sum"]) and (of)):
		aggregate = "{}({})".format(method.upper(), str_column(of))
		others_aggregate = method
	elif (method and method[-1] == "%"):
		aggregate = "APPROXIMATE_PERCENTILE({} USING PARAMETERS percentile = {})".format(str_column(of), float(method[0:-1]) / 100)
	elif (method.lower() in ["density", "count"]):
		aggregate = "COUNT(*)"
		others_aggregate = "sum"
	else:
		raise ValueError("The parameter 'method' must be in [count|density|avg|mean|min|max|sum|q%]")
	columns = [str_column(column) for column in columns]
	all_columns = []
	is_column_date = [False, False]
	timestampadd = ["", ""]
	for idx, column in enumerate(columns):
		is_numeric = vdf[column].isnum() and (vdf[column].nunique(True) > 2)
		is_date = vdf[column].isdate()
		where = []
		if (is_numeric):
			if (h[idx] == None):
				interval = vdf[column].numh()
				if (interval > 0.01):
					interval = round(interval, 2)
				elif (interval > 0.0001):
					interval = round(interval, 4)
				elif (interval > 0.000001):
					interval = round(interval, 6)
				interval = int(max(math.floor(interval), 1)) if (vdf[column].category() == "int") else interval
			else:
				interval = h[idx]
			if (vdf[column].category() == "int"):
				floor_end = "-1"
				interval = int(max(math.floor(interval),1))
			else:
				floor_end = ""
			expr = "'[' || FLOOR({} / {}) * {} || ';' || (FLOOR({} / {}) * {} + {}{}) || ']'".format(column, interval, interval, column, interval, interval, interval, floor_end)
			all_columns += [expr] if (interval > 1) or (vdf[column].category() == "float") else ["FLOOR({}) || ''".format(column)]
			order_by = "ORDER BY MIN(FLOOR({} / {}) * {}) ASC".format(column, interval, interval)
			where += ["{} IS NOT NULL".format(column)]
		elif (is_date):
			interval = vdf[column].numh() if (h[idx] == None) else max(math.floor(h[idx]),1)
			min_date = vdf[column].min()
			all_columns += ["FLOOR(DATEDIFF('second', '"+str(min_date)+"', "+column+") / "+str(interval)+") * "+str(interval)]
			is_column_date[idx] = True
			timestampadd[idx] = "TIMESTAMPADD('second', "+columns[idx]+"::int, '"+str(min_date)+"'::timestamp)"
			order_by = "ORDER BY 1 ASC"
			where += ["{} IS NOT NULL".format(column)]
		else:
			all_columns += [column]
			order_by = "ORDER BY 1 ASC"
			distinct = vdf[column].topk(max_cardinality[idx]).values["index"]
			if (len(distinct) < max_cardinality[idx]):
				where += ["({} IN ({}))".format(convert_special_type(vdf[column].category(), False, column), ", ".join(["'{}'".format(str(elem).replace("'", "''")) for elem in distinct]))]
			else:
				where += ["({} IS NOT NULL)".format(column)]
	where = " WHERE {}".format(" AND ".join(where))
	if (len(columns) == 1):
		query = "SELECT {} AS {}, {} FROM {}{} GROUP BY 1 {}".format(convert_special_type(vdf[columns[0]].category(), True, all_columns[-1]), columns[0], aggregate, vdf.__genSQL__(), where, order_by)
		return to_tablesample(query, vdf._VERTICAPY_VARIABLES_["cursor"], name = aggregate)
	alias = ", " + str_column(of) + " AS " + str_column(of) if of else ""
	aggr = ", " + of if (of) else ""
	subtable = "(SELECT {} AS {}, {} AS {}{} FROM {}{}) pivot_table".format(all_columns[0], columns[0], all_columns[1], columns[1], alias, vdf.__genSQL__(), where)
	if (is_column_date[0] and not(is_column_date[1])):
		subtable = "(SELECT {} AS {}, {}{} FROM {}{}) pivot_table_date".format(timestampadd[0], columns[0], columns[1], aggr, subtable, where)
	elif (is_column_date[1] and not(is_column_date[0])):
		subtable = "(SELECT {}, {} AS {}{} FROM {}{}) pivot_table_date".format(columns[0], timestampadd[1], columns[1], aggr, subtable, where)
	elif (is_column_date[1] and is_column_date[0]):
		subtable = "(SELECT {} AS {}, {} AS {}{} FROM {}{}) pivot_table_date".format(timestampadd[0], columns[0], timestampadd[1], columns[1], aggr, subtable, where)
	over = "/" + str(vdf.shape()[0]) if (method=="density") else ""
	cast = []
	for column in columns:
		cast += [convert_special_type(vdf[column].category(), True, column)]
	query = "SELECT {} AS {}, {} AS {}, {}{} FROM {} WHERE {} IS NOT NULL AND {} IS NOT NULL GROUP BY {}, {} ORDER BY {}, {} ASC".format(cast[0], columns[0], cast[1], columns[1], aggregate, over, subtable, columns[0], columns[1], columns[0], columns[1], columns[0], columns[1])
	vdf.__executeSQL__(query = query, title = "Group the features to compute the pivot table")
	query_result = vdf.__executeSQL__(query = query, title = "Group the features to compute the pivot table").fetchall()
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
	for item in query_result:
		j, i = all_column0_categories.index(str(item[0])), all_column1_categories.index(str(item[1]))
		all_columns[i][j] = item[2]
	all_columns = [[all_column1_categories[i]] + all_columns[i] for i in range(0, len(all_columns))]
	all_columns = [[columns[0] + "/" + columns[1]] + all_column0_categories] + all_columns
	if (show):
		all_count = [item[2] for item in query_result]
		cmatrix(all_columns, all_column0_categories, all_column1_categories, len(all_column0_categories), len(all_column1_categories), vmax = max(all_count), vmin = min(all_count), cmap = cmap, title = "Pivot Table of " + columns[0] + " vs " + columns[1], colorbar = aggregate, x_label = columns[1], y_label = columns[0], with_numbers = with_numbers)
	values = {all_columns[0][0] : all_columns[0][1:len(all_columns[0])]}
	del(all_columns[0])
	for column in all_columns:
		values[column[0]] = column[1:len(column)]
	return tablesample(values = values, name = "Pivot Table of {} vs {}".format(columns[0], columns[1]), table_info = False)
#---#
def scatter_matrix(vdf, 
				   columns: list = []):
	for column in columns:
		if (column not in vdf.get_columns()) and (str_column(column) not in vdf.get_columns()):
			raise NameError("The Virtual Column {} doesn't exist".format(column))
	if not(columns):
		columns = vdf.numcol()
	elif (len(columns) == 1):	
		return vdf[columns[0]].hist()
	n = len(columns)
	fig, axes = plt.subplots(nrows = n, ncols = n, figsize = (min(1.5 * n, 500), min(1.5 * n, 500))) if isnotebook() else plt.subplots(nrows = n, ncols = n, figsize = (min(int(n / 1.1, 500)), min(int(n / 1.1, 500))))
	query = "SELECT {}, RANDOM() AS rand FROM {} WHERE __verticapy_split__ < 0.5 ORDER BY rand LIMIT 1000".format(", ".join(columns), vdf.__genSQL__(True))
	all_scatter_points = vdf.__executeSQL__(query = query, title = "Select random points to draw the scatter plot").fetchall()
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
				axes[i, j].bar(x0, y0, h0 / 0.94, color = '#FE5016')
			else:
				axes[i, j].scatter(all_scatter_columns[j], all_scatter_columns[i], color = '#263133', s = 4, marker = 'o')
	fig.suptitle('Scatter Plot Matrix of {}'.format(vdf._VERTICAPY_VARIABLES_["input_relation"]))
	plt.show()
#---#
def scatter2D(vdf,
			  columns: list,
			  max_cardinality: int = 6,
			  cat_priority: list = [],
			  with_others: bool = True,
			  max_nb_points: int = 100000,
			  bbox: list = [],
			  img: str = ""):
	colors = gen_colors()
	markers = ["^", "o", "+", "*", "h", "x", "D", "1"] * 10
	columns = [str_column(column) for column in columns]
	if (bbox) and len(bbox) != 4:
		print("\u26A0 Warning: Parameter 'bbox' must be a list of 4 numerics containing the 'xlim' and 'ylim'.\nIt was ignored.")
		bbox = []
	for column in columns:
		if (column not in vdf.get_columns()):
			raise NameError("The Virtual Column {} doesn't exist".format(column))
	if not(vdf[columns[0]].isnum()) or not(vdf[columns[1]].isnum()):
		raise TypeError("The two first columns of the parameter 'columns' must be numerical")
	if (len(columns) == 2):
		tablesample = max_nb_points / vdf.shape()[0]
		query = "SELECT {}, {} FROM {} WHERE __verticapy_split__ < {} AND {} IS NOT NULL AND {} IS NOT NULL LIMIT {}".format(columns[0], columns[1], vdf.__genSQL__(True), tablesample, columns[0], columns[1], max_nb_points)
		query_result = vdf.__executeSQL__(query = query, title = "Select random points to draw the scatter plot").fetchall()
		column1, column2 = [item[0] for item in query_result], [item[1] for item in query_result]
		plt.figure(figsize = (14, 10)) if isnotebook() else plt.figure(figsize = (10, 6))
		if (bbox):
			plt.xlim(bbox[0], bbox[1])
			plt.ylim(bbox[2], bbox[3])
		if (img):
			im = plt.imread(img)
			if not(bbox):
				bbox = (min(column1), max(column1), min(column2), max(column2))
				plt.xlim(bbox[0], bbox[1])
				plt.ylim(bbox[2], bbox[3])
			plt.imshow(im, extent = bbox)
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
					column_groupby, vdf.__genSQL__(), column_groupby, column_groupby, max_cardinality)
			query_result = vdf.__executeSQL__(query = query, title = "Compute {} categories".format(column_groupby)).fetchall()
			query_result = [item for sublist in query_result for item in sublist]
		all_columns, all_scatter, all_categories = [query_result], [], query_result
		fig = plt.figure(figsize = (14, 10)) if isnotebook() else plt.figure(figsize = (10, 6))
		if (bbox):
			plt.xlim(bbox[0], bbox[1])
			plt.ylim(bbox[2], bbox[3])
		if (img):
			im = plt.imread(img)
			if not(bbox):
				aggr = vdf.agg(columns = [columns[0], columns[1]], func = ["min", "max"])
				bbox = (aggr.values["min"][0], aggr.values["max"][0], aggr.values["min"][1], aggr.values["max"][1])
				plt.xlim(bbox[0], bbox[1])
				plt.ylim(bbox[2], bbox[3])
			plt.imshow(im, extent = bbox)
		ax = plt
		others = []
		groupby_cardinality = vdf[column_groupby].nunique(True)
		count = vdf.shape()[0]
		tablesample = 0.1 if (count > 10000) else 0.9
		for idx, category in enumerate(all_categories):
			if ((max_cardinality < groupby_cardinality) or (len(cat_priority) < groupby_cardinality)):
				others += ["{} != '{}'".format(column_groupby, str(category).replace("'", "''"))]
			query = "SELECT {}, {} FROM {} WHERE  __verticapy_split__ < {} AND {} = '{}' AND {} IS NOT NULL AND {} IS NOT NULL LIMIT {}"
			query = query.format(columns[0], columns[1], vdf.__genSQL__(True), tablesample, columns[2], str(category).replace("'", "''"), columns[0], columns[1], int(max_nb_points / len(all_categories))) 
			vdf.__executeSQL__(query = query, title = "Select random points to draw the scatter plot (category = '{}')".format(str(category)))
			query_result = vdf._VERTICAPY_VARIABLES_["cursor"].fetchall()
			column1, column2 = [float(item[0]) for item in query_result], [float(item[1]) for item in query_result]
			all_columns += [[column1, column2]]
			all_scatter += [ax.scatter(column1, column2, alpha=0.8, marker = markers[idx], color = colors[idx % len(colors)])]
		if (with_others and idx + 1 < groupby_cardinality):
			all_categories += ["others"]
			query = "SELECT {}, {} FROM {} WHERE {} AND {} IS NOT NULL AND {} IS NOT NULL AND __verticapy_split__ < {} LIMIT {}"
			query = query.format(columns[0], columns[1], vdf.__genSQL__(True), " AND ".join(others), columns[0], columns[1], tablesample, int(max_nb_points / len(all_categories)))
			query_result = vdf.__executeSQL__(query = query, title = "Select random points to draw the scatter plot (category = 'others')").fetchall()
			column1, column2 = [float(item[0]) for item in query_result], [float(item[1]) for item in query_result]
			all_columns += [[column1, column2]]
			all_scatter += [ax.scatter(column1, column2, alpha = 0.8, marker = markers[idx + 1], color = colors[(idx + 1) % len(colors)])]
		for idx, item in enumerate(all_categories):
			if (len(str(item)) > 20):
				all_categories[idx] = str(item)[0:20] + "..."
		plt.gca().grid()
		plt.gca().set_axisbelow(True)
		plt.title('Scatter Plot of {} vs {}'.format(columns[0], columns[1]))
		ax.xlabel(columns[0])
		ax.ylabel(columns[1])
		ax.legend(all_scatter, all_categories, title = column_groupby, loc = 'center left', bbox_to_anchor = [1, 0.5])
		plt.show()
#---#
def scatter3D(vdf,
			  columns: list,
			  max_cardinality: int = 3,
			  cat_priority: list = [],
			  with_others: bool = True,
			  max_nb_points: int = 1000):
	columns = [str_column(column) for column in columns]
	colors = gen_colors()
	markers = ["^", "o", "+", "*", "h", "x", "D", "1"] * 10
	if ((len(columns) < 3) or (len(columns) > 4)):
		raise Exception("3D Scatter plot can only be done with at least two columns and maximum with four columns")
	else:
		for column in columns:
			if (column not in vdf.get_columns()):
				raise NameError("The Virtual Column {} doesn't exist".format(column))
		for i in range(3):
			if not(vdf[columns[i]].isnum()):
				raise TypeError("The three first columns of the parameter 'columns' must be numerical")
		if (len(columns) == 3):
			tablesample = max_nb_points / vdf.shape()[0]
			query = "SELECT {}, {}, {} FROM {} WHERE __verticapy_split__ < {} AND {} IS NOT NULL AND {} IS NOT NULL AND {} IS NOT NULL LIMIT {}".format(
						columns[0], columns[1], columns[2], vdf.__genSQL__(True), tablesample, columns[0], columns[1], columns[2], max_nb_points)
			query_result = vdf.__executeSQL__(query = query, title = "Select random points to draw the scatter plot").fetchall()
			column1, column2, column3 = [float(item[0]) for item in query_result], [float(item[1]) for item in query_result], [float(item[2]) for item in query_result]
			fig = plt.figure(figsize = (14, 12)) if isnotebook() else plt.figure(figsize = (10, 6))
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
						column_groupby, vdf.__genSQL__(), column_groupby, column_groupby, max_cardinality)
				query_result = vdf.__executeSQL__(query = query, title = "Compute the vcolumn {} distinct categories".format(column_groupby)).fetchall()
				query_result = [item for sublist in query_result for item in sublist]
			all_columns, all_scatter, all_categories = [query_result], [], query_result
			fig = plt.figure(figsize = (14, 12)) if isnotebook() else plt.figure(figsize = (10, 6))
			ax = fig.add_subplot(111, projection = '3d')
			others = []
			groupby_cardinality = vdf[column_groupby].nunique(True)
			tablesample = 10 if (count > 10000) else 90
			for idx,category in enumerate(all_categories):
				if ((max_cardinality < groupby_cardinality) or (len(cat_priority) < groupby_cardinality)):
					others += ["{} != '{}'".format(column_groupby, str(category).replace("'", "''"))]
				query = "SELECT {}, {}, {} FROM {} WHERE __verticapy_split__ < {} AND {} = '{}' AND {} IS NOT NULL AND {} IS NOT NULL AND {} IS NOT NULL limit {}"
				query = query.format(columns[0], columns[1], columns[2], vdf.__genSQL__(True), tablesample, columns[3], str(category).replace("'", "''"), columns[0], columns[1], columns[2], int(max_nb_points / len(all_categories))) 
				query_result = vdf.__executeSQL__(query = query, title = "Select random points to draw the scatter plot (category = '{}')".format(category)).fetchall()
				column1, column2, column3 = [float(item[0]) for item in query_result], [float(item[1]) for item in query_result], [float(item[2]) for item in query_result]
				all_columns += [[column1, column2, column3]]
				all_scatter += [ax.scatter(column1, column2, column3, alpha=0.8, marker = markers[idx], color = colors[idx % len(colors)])]
			if (with_others and idx + 1 < groupby_cardinality):
				all_categories += ["others"]
				query = "SELECT {}, {}, {} FROM {} WHERE {} AND {} IS NOT NULL AND {} IS NOT NULL AND {} IS NOT NULL AND __verticapy_split__ < {} LIMIT {}"
				query = query.format(columns[0], columns[1], columns[2], vdf.__genSQL__(True), " AND ".join(others), columns[0], columns[1], columns[2], tablesample, int(max_nb_points / len(all_categories)))
				query_result = vdf.__executeSQL__(query = query, title = "Select random points to draw the scatter plot (category = 'others')").fetchall()
				column1, column2 = [float(item[0]) for item in query_result], [float(item[1]) for item in query_result]
				all_columns += [[column1, column2]]
				all_scatter += [ax.scatter(column1, column2, alpha = 0.8, marker = markers[idx + 1], color = colors[(idx + 1) % len(colors)])]
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
			ax.legend(all_scatter, all_categories, scatterpoints = 1, title = column_groupby, loc = 'center left', bbox_to_anchor = [1, 0.5])
			plt.show()
#---#
def ts_plot(vdf, 
			order_by: str, 
			by: str = "",
			order_by_start: str = "",
			order_by_end: str = "",
			color: str = '#263133', 
			area: bool = False):
	if not(by):
		query = "SELECT {}, {} FROM {} WHERE {} IS NOT NULL AND {} IS NOT NULL".format(order_by, vdf.alias, vdf.parent.__genSQL__(), order_by, vdf.alias)
		query += " AND {} > '{}'".format(order_by, order_by_start) if (order_by_start) else ""
		query += " AND {} < '{}'".format(order_by, order_by_end) if (order_by_end) else ""
		query += " ORDER BY {}, {}".format(order_by, vdf.alias)
		query_result = vdf.__executeSQL__(query = query, title = "Select points to draw the curve").fetchall()
		order_by_values = [item[0] for item in query_result]
		try:
			if (type(order_by_values[0]) == str):
				from dateutil.parser import parse
				order_by_values = [parse(elem) for elem in order_by_values]
		except:
			pass
		column_values = [float(item[1]) for item in query_result]
		plt.figure(figsize = (12, 9)) if isnotebook() else plt.figure(figsize = (10, 6))
		plt.rcParams['axes.facecolor'] = '#FCFCFC'
		plt.plot(order_by_values, column_values, color = color)
		if (area):
			area_label = "Area "
			plt.fill_between(order_by_values, column_values, facecolor = color)
		else:
			area_label = ""
		plt.title('{}Plot of {} vs {}'.format(area_label, vdf.alias, order_by))
		plt.xticks(rotation = 90)
		plt.subplots_adjust(bottom = 0.24)
		plt.xlabel(order_by)
		plt.ylabel(vdf.alias)
		plt.gca().grid()
		plt.show()
	else:
		colors = gen_colors()
		by = str_column(by)
		cat = vdf.parent[by].distinct()
		all_data = []
		for column in cat:
			query = "SELECT {}, {} FROM {} WHERE {} IS NOT NULL AND {} IS NOT NULL".format(order_by, vdf.alias, vdf.parent.__genSQL__(), order_by, vdf.alias)
			query += " AND {} > '{}'".format(order_by, order_by_start) if (order_by_start) else ""
			query += " AND {} < '{}'".format(order_by, order_by_end) if (order_by_end) else ""
			query += " AND {} = '{}'".format(by, str(column).replace("'", "''")) 
			query += " ORDER BY {}, {}".format(order_by, vdf.alias)
			query_result = vdf.__executeSQL__(query = query, title = "Select points to draw the curve").fetchall()
			all_data += [[[item[0] for item in query_result], [float(item[1]) for item in query_result], column]]
			try:
				if (type(all_data[-1][0][0]) == str):
					from dateutil.parser import parse
					all_data[-1][0] = [parse(elem) for elem in all_data[-1][0]]
			except:
				pass
		plt.figure(figsize = (12, 9)) if isnotebook() else plt.figure(figsize = (10, 6))
		plt.rcParams['axes.facecolor'] = '#FCFCFC'
		for idx, elem in enumerate(all_data):
			plt.plot(elem[0], elem[1], color = colors[idx % len(colors)], label = elem[2])
		plt.title('Plot of {} vs {}'.format(vdf.alias, order_by))
		plt.xticks(rotation = 90)
		plt.subplots_adjust(bottom = 0.24)
		plt.xlabel(order_by)
		plt.ylabel(vdf.alias)
		plt.gca().grid()
		plt.legend(title = by, loc = 'center left', bbox_to_anchor = [1, 0.5])
		plt.show()