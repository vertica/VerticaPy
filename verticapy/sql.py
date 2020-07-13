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
##
#  _____  _____ _      ___  ___  ___  _____ _____ _____ 
# /  ___||  _  | |     |  \/  | / _ \|  __ \_   _/  __ \
# \ `--. | | | | |     | .  . |/ /_\ \ |  \/ | | | /  \/
#  `--. \| | | | |     | |\/| ||  _  | | __  | | | |    
# /\__/ /\ \/' / |____ | |  | || | | | |_\ \_| |_| \__/\
# \____/  \_/\_\_____/ \_|  |_/\_| |_/\____/\___/ \____/
#
##
#
#---#
def sql(line, cell = ""):
	from verticapy.connections.connect import read_auto_connect
	from verticapy.utilities import vdf_from_relation
	from verticapy.hchart import hchartSQL
	conn = read_auto_connect()
	cursor = conn.cursor()
	if (not(cell) and (line)):
		line = line.replace(";", "")
		try:
			return vdf_from_relation("({}) x".format(line), cursor = cursor)
		except:
			cursor.execute(line)
			conn.close()
			return "SUCCESS"
	elif (not(line) and (cell)):
		queries = cell.split(";")
		try:
			queries.remove("")
		except:
			pass
		n = len(queries)
		for i in range(n):
			if i == (n - 1):
				try:
					return vdf_from_relation("({}) x".format(queries[i]), cursor = cursor)
				except:
					cursor.execute(queries[i])
					conn.close()
					return "SUCCESS"
			else:
				cursor.execute(queries[i])
	else:
		queries = cell.split(";")
		try:
			queries.remove("")
		except:
			pass
		queries = queries[-1]
		chart = hchartSQL(queries, cursor, line)
		conn.close()
		return chart
#---#
def load_ipython_extension(ipython):
    ipython.register_magic_function(sql, 'cell')
    ipython.register_magic_function(sql, 'line')
