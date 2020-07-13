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
import random, os, math, shutil, re
# VerticaPy Modules
from verticapy.utilities import *
#
#
# Functions used to simplify the coding.
#
#---#
def arange(start: float, stop: float, step: float):
	check_types([
		("start", start, [int, float], False),
		("stop", stop, [int, float], False),
		("step", step, [int, float], False)])
	if (step < 0):
		raise Exception("Parameter 'step' must be greater than 0")
	L_final = []
	tmp = start
	while (tmp < stop):
		L_final += [tmp]
		tmp += step
	return (L_final)
#---#
def category_from_type(ctype: str = ""):
	check_types([("ctype", ctype, [str], False)])
	ctype = ctype.lower()
	if (ctype != ""):
		if (ctype[0:4] == "date") or (ctype[0:4] == "time") or (ctype == "smalldatetime") or (ctype[0:8] == "interval"):
			return "date"
		elif ((ctype[0:3] == "int") or (ctype[0:4] == "bool")  or (ctype in ("tinyint", "smallint", "bigint"))):
			return "int"
		elif ((ctype[0:3] == "num") or (ctype[0:5] == "float") or (ctype[0:7] == "decimal") or (ctype == "money") or (ctype[0:6] == "double") or (ctype[0:4] == "real")):
			return "float"
		elif (("binary" in ctype) or ("byte" in ctype) or (ctype == "raw")):
			return "binary"
		elif (ctype[0:3] == "geo"):
			return "spatial"
		elif ("uuid" in ctype):
			return "uuid"
		else:
			return "text"
	else:
		return "undefined"
#---#
def check_cursor(cursor):
	if 'cursor' not in (str(type(cursor))).lower():
		raise TypeError("Parameter 'cursor' must be a DataBase cursor, found '{}'\nYou can find how to set up your own cursor using the vHelp function of the utilities module (option number 1).".format(type(cursor)))
#---#
def check_types(types_list: list = [], 
				vdf: list = []):
	if (vdf):
		if 'vdataframe' not in (str(type(vdf[1]))).lower():
			print("\u26A0 Warning: Parameter '{}' must be a Virtual DataFrame, found '{}'".format(vdf[0], type(vdf[1])))
	for elem in types_list:
		if (elem[3]):
			if (type(elem[1]) != str):
				print("\u26A0 Warning: Parameter '{}' must be of type {}, found type {}".format(elem[0], str, type(elem[1])))
			if (elem[1].lower() not in elem[2]):
				print("\u26A0 Warning: Parameter '{}' must be in [{}], found '{}'".format(elem[0], "|".join(elem[2]), elem[1]))
		else:
			convert_success = False
			if (int in elem[2]):
				try:
					int(elem[1])
					convert_success = True
				except:
					pass
			if (float in elem[2]):
				try:
					float(elem[1])
					convert_success = True
				except:
					pass
			if (list in elem[2]):
				try:
					list(elem[1])
					convert_success = True
				except:
					pass
			if (type(elem[1]) not in elem[2]) and not(convert_success):
				if (len(elem[2]) == 1):
					print("\u26A0 Warning: Parameter '{}' must be of type {}, found type {}".format(elem[0], elem[2][0], type(elem[1])))
				else:
					print("\u26A0 Warning: Parameter '{}' type must be one of the following {}, found type {}".format(elem[0], elem[2], type(elem[1])))
#---#
def column_check_ambiguous(column: str, 
						   columns: list):
	column = column.replace('"', '').lower()
	for col in columns:
		col = col.replace('"', '').lower()
		if (column == col):
			return True
	return False
#---#
def columns_check(columns: list, 
				  vdf, 
				  columns_nb = None):
	vdf_columns = vdf.get_columns()
	if (columns_nb != None and len(columns) not in columns_nb):
		raise Exception("The number of Virtual Columns expected is {}, found {}".format("|".join([str(elem) for elem in columns_nb]), len(columns)))
	for column in columns:
		if not(column_check_ambiguous(column, vdf_columns)):
			try:
				e = ""
				nearestcol = nearest_column(vdf_columns, column)
				if (nearestcol[1] < 5): e = "\nDid you mean {} ?".format(nearestcol[0])
			except:
				e = ""
			raise ValueError("The Virtual Column '{}' doesn't exist{}".format(column.lower().replace('"', ''), e))
#---#
def convert_special_type(category: str, 
						 convert_date: bool = True, 
						 column: str = '{}'):
	if (category == "binary"):
		return 'TO_HEX({})'.format(column)
	elif (category == "spatial"):
		return 'ST_AsText({})'.format(column)
	elif (category == "date") and (convert_date):
		return '{}::varchar'.format(column)
	else:
		return column
#---#
def data_to_columns(data: list, n: int):
	columns = [[]] * n
	for elem in data:
		for i in range(n):
			try:
				columns[i] = columns[i] + [float(elem[i])]
			except:
				columns[i] = columns[i] + [elem[i]]
	return (columns)
#---#
def gen_name(L: list):
	return "_".join([''.join(ch for ch in str(elem).lower() if ch.isalnum() or ch == "_") for elem in L])
#---#
def get_narrow_tablesample(t, use_number_as_category: bool = False):
	result = []
	t = t.values
	if (use_number_as_category):
		categories_alpha = t["index"]
		categories_beta = [elem for elem in t]
		del categories_beta[0]
		bijection_categories = {}
		for idx, elem in enumerate(categories_alpha):
			bijection_categories[elem] = idx
		for idx, elem in enumerate(categories_beta):
			bijection_categories[elem] = idx
	for elem in t:
		if elem != "index":
			for idx, val_tmp in enumerate(t[elem]):
				try:
					val = float(val_tmp)
				except:
					val = val_tmp
				if not (use_number_as_category):
					result += [[elem, t["index"][idx], val]]
				else:
					result += [[bijection_categories[elem], bijection_categories[t["index"][idx]], val]]
	if (use_number_as_category):
		return result, categories_alpha, categories_beta
	else:
		return result
#---#
def indentSQL(query: str):
	query = query.replace("SELECT", "\n   SELECT\n    ").replace("FROM", "\n   FROM\n").replace(",", ",\n    ")
	query = query.replace("VERTICAPY_SUBTABLE", "\nVERTICAPY_SUBTABLE")
	n = len(query)
	return_l = []
	j = 1
	while (j < n - 9):
		if (query[j] == "(" and (query[j - 1].isalnum() or query[j - 5:j] == "OVER ") and query[j + 1:j + 7] != "SELECT"):
			k = 1
			while (k > 0 and j < n - 9):
				j += 1
				if (query[j] == "\n"):
					return_l += [j]
				elif (query[j] == ")"):
					k -= 1
				elif (query[j] == "("):
					k += 1
		else:
			j += 1
	query_print = ""
	i = 0 if query[0] != "\n" else 1
	while (return_l):
		j = return_l[0]
		query_print += query[i:j]
		if (query[j] != "\n"):
			query_print += query[j]
		else:
			i = j + 1
			while (query[i] == " " and i < n - 9):
				i += 1 
			query_print += " "
		del(return_l[0])
	query_print += query[i:n]
	return (query_print)
#---#
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
#---#
def levenshtein(s: str, t: str):
    rows = len(s) + 1
    cols = len(t) + 1
    dist = [[0 for x in range(cols)] for x in range(rows)]
    for i in range(1, rows):
        dist[i][0] = i
    for i in range(1, cols):
        dist[0][i] = i
    for col in range(1, cols):
        for row in range(1, rows):
            if s[row - 1] == t[col - 1]:
                cost = 0
            else:
                cost = 1
            dist[row][col] = min(dist[row-1][col] + 1, dist[row][col-1] + 1, dist[row-1][col-1] + cost)
    return dist[row][col]
#---#
def nearest_column(columns: list, column: str):
	col = column.replace('"', '').lower()
	result = (columns[0], levenshtein(col, columns[0].replace('"', '').lower()))
	if (len(columns) == 1):
		return result
	for elem in columns:
		if (elem != result[0]):
			current_col = elem.replace('"', '').lower()
			d = levenshtein(current_col, col)
			if result[1] > d:
				result = (elem, d)
	return result
#---#
def order_discretized_classes(categories):
	try:
		try:
			order = []
			for item in categories:
				order += [float(item.split(";")[0].split('[')[1])]
		except:
			order = []
			for item in all_subcategories:
				order += [float(item)]
		order = [x for _, x in sorted(zip(order, categories))]
	except:
		return categories
	return order
#---#
def print_query(query: str, title: str = ""):
	screen_columns = shutil.get_terminal_size().columns
	query_print = indentSQL(query)
	if (isnotebook()):
		from IPython.core.display import HTML, display
		display(HTML("<h4 style = 'color : #444444; text-decoration : underline;'>{}</h4>".format(title)))
		query_print = query_print.replace('\n',' <br>').replace('  ',' &emsp; ')
		display(HTML(query_print))
		display(HTML("<div style = 'border : 1px dashed black; width : 100%'></div>"))
	else:
		print("$ {} $\n".format(title))
		print(query_print)
		print("-" * int(screen_columns) + "\n")
#---#
def print_time(elapsed_time: float):
	screen_columns = shutil.get_terminal_size().columns
	if (isnotebook()):
		from IPython.core.display import HTML,display
		display(HTML("<div><b>Elapsed Time : </b> {}</div>".format(elapsed_time)))
		display(HTML("<div style = 'border : 1px dashed black; width : 100%'></div>"))
	else:
		print("Elapsed Time: {}".format(elapsed_time))
		print("-" * int(screen_columns) + "\n")
#---#
def print_table(data_columns, 
				is_finished: bool = True, 
				offset: int = 0, 
				repeat_first_column: bool = False, 
				first_element: str = ""):
	data_columns_rep = [] + data_columns
	if (repeat_first_column):
		del data_columns_rep[0]
		columns_ljust_val = min(len(max([str(item) for item in data_columns[0]], key = len)) + 4, 40)
	else:
		columns_ljust_val = len(str(len(data_columns[0]))) + 2
	screen_columns = shutil.get_terminal_size().columns
	formatted_text = ""
	rjust_val = []
	for idx in range(0,len(data_columns_rep)):
		rjust_val += [min(len(max([str(item) for item in data_columns_rep[idx]], key = len)) + 2, 40)]
	total_column_len = len(data_columns_rep[0])
	while (rjust_val != []):
		columns_to_print = [data_columns_rep[0]]
		columns_rjust_val = [rjust_val[0]]
		max_screen_size = int(screen_columns) - 14 - int(rjust_val[0])
		del data_columns_rep[0]
		del rjust_val[0]
		while ((max_screen_size > 0) and (rjust_val != [])):
			columns_to_print += [data_columns_rep[0]]
			columns_rjust_val += [rjust_val[0]]
			max_screen_size = max_screen_size - 7 - int(rjust_val[0])
			del data_columns_rep[0]
			del rjust_val[0]
		if (repeat_first_column):
			columns_to_print = [data_columns[0]] + columns_to_print
		else:
			columns_to_print=[[i - 1 + offset for i in range(0,total_column_len)]] + columns_to_print
		columns_to_print[0][0] = first_element
		columns_rjust_val = [columns_ljust_val]+columns_rjust_val
		column_count = len(columns_to_print)
		for i in range(0,total_column_len):
			for k in range(0,column_count):
				val = columns_to_print[k][i]
				if len(str(val)) > 40:
					val = str(val)[0:37] + "..."
				if (k == 0):
					formatted_text += str(val).ljust(columns_rjust_val[k])
				else:
					formatted_text += str(val).rjust(columns_rjust_val[k])+"  "
			if ((rjust_val != [])):
				formatted_text += " \\\\"
			formatted_text += "\n"	
		if (not(is_finished) and (i == total_column_len-1)):
			for k in range(0,column_count):
				if (k==0):
					formatted_text += "...".ljust(columns_rjust_val[k])
				else:
					formatted_text += "...".rjust(columns_rjust_val[k])+"  "
			if (rjust_val != []):
				formatted_text += " \\\\"
			formatted_text += "\n"
	try:	
		if (isnotebook()):
			from IPython.core.display import HTML, display
			if not(repeat_first_column):
				data_columns=[[""] + list(range(0 + offset, len(data_columns[0]) - 1 + offset))] + data_columns
			m = len(data_columns)
			n = len(data_columns[0])
			html_table = "<table style=\"border-collapse: collapse; border: 2px solid white\">"
			for i in range(n):
				html_table += "<tr style=\"{border: 1px solid white;}\">"
				for j in range(m):
					if (j == 0):
						html_table += "<td style=\"border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#263133;color:white\"><b>" + str(data_columns[j][i]) + "</b></td>"
					elif (i == 0):
						html_table += "<td style=\"font-size:1.02em;background-color:#263133;color:white\"><b>" + str(data_columns[j][i]) + "</b></td>"
					else:
						html_table += "<td style=\"border: 1px solid white;\">" + str(data_columns[j][i]) + "</td>"
				html_table += "</tr>"
			if not(is_finished):
				html_table += "<tr>"
				for j in range(m):
					if (j == 0):
						html_table += "<td style=\"border-top: 1px solid white;background-color:#263133;color:white\"></td>"
					else:
						html_table += "<td style=\"border: 1px solid white;\">...</td>"
				html_table += "</tr>"
			html_table += "</table>"
			display(HTML(html_table))
			return "<object>  "
		else:
			return formatted_text
	except:
		return formatted_text
#---#
def schema_relation(relation: str):
	quote_nb = relation.count('"')
	if (quote_nb not in (0, 2, 4)):
		raise ValueError("The format of the input relation is incorrect.")
	if (quote_nb == 4):
		schema_input_relation = relation.split('"')[1], relation.split('"')[3]
	elif (quote_nb == 4):
		schema_input_relation = relation.split('"')[1], relation.split('"')[2][1:] if (relation.split('"')[0] == '') else relation.split('"')[0][0:-1], relation.split('"')[1]
	else:
		schema_input_relation = relation.split(".")
	if (len(schema_input_relation) == 1):
		schema, relation = "public", relation 
	else: 
		schema, relation = schema_input_relation[0], schema_input_relation[1]
	return (schema, relation)
#---#
def sort_str(columns, vdf):
	if not(columns):
		return ""
	if (type(columns) == dict):
		order_by = []
		for elem in columns:
			column_name = vdf_columns_names([elem], vdf)[0]
			if (columns[elem].lower() not in ("asc", "desc")):
				print("\u26A0 Warning: Method of {} must be in (asc, desc), found '{}'\nThis column was ignored.".format(column_name, columns[elem].lower()))
			else:
				order_by += ["{} {}".format(column_name, columns[elem].upper())]
	else:
		order_by = [elem for elem in columns]
	return (" ORDER BY {}".format(", ".join(order_by)))
#---#
def str_column(column: str):
	return ('"{}"'.format(column.replace('"', '')))
#---#
def str_function(key: str):
	if key in ("median", "med", "approximate_median"):
		key = "50%"
	elif key == "100%":
		key = "max"
	elif key == "0%":
		key = "min"
	elif key == "approximate_count_distinct":
		key = "approx_unique"
	elif key == "approximate_count_distinct":
		key = "approx_unique"
	elif key == "ema":
		key = "exponential_moving_average"
	elif key == "mean":
		key = "avg"
	elif key in ("stddev", "stdev"):
		key = "std"
	elif key == "product":
		key = "prod"
	elif key == "variance":
		key = "var"
	elif key == "kurt":
		key = "kurtosis"
	elif key == "skew":
		key = "skewness"
	elif key in ("top1", "mode"):
		key = "top"
	elif key == "top1_percent":
		key = "top_percent"
	elif ('%' == key[-1]):
		if (float(key[0:-1]) == int(float(key[0:-1]))):
			key = "{}%".format(int(float(key[0:-1])))
	elif (key == "row"):
		key = "row_number"
	elif (key == "first"):
		key = "first_value"
	elif (key == "last"):
		key = "last_value"
	elif (key == "next"):
		key = "lead"
	elif (key in ("prev", "previous")):
		key = "lag"
	return key
#---#
def vdf_columns_names(columns: list, 
					  vdf):
	check_types([("columns", columns, [list], False)], 
				 vdf = ["vdf", vdf])
	vdf_columns = vdf.get_columns()
	columns_names = []
	for column in columns:
		for vdf_column in vdf_columns:
			if (str_column(column).lower() == str_column(vdf_column).lower()):
				columns_names += [str_column(vdf_column)]
	return (columns_names)