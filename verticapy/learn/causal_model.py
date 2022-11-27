# import
from verticapy import vDataFrame, vDataFrameSQL
from verticapy.toolbox import check_types
import verticapy as vp
import itertools
from tqdm.auto import tqdm
from time import time
from typing import Union

import warnings

from ..stats.tools import conditional_chi_square, is_dir_path
import numpy as np
from verticapy import plot


# ---#
class CausalModel:
	"""
-------------------------------------------------
Main Class for Causal Models.

Parameters
----------
input_relation: [str, vDataframe]
	Name of original relation
	"""	
 
	# ---#
	def __init__(self, input_relation : Union[str, vDataFrame]):
		check_types([("input_relation", input_relation, [str, vDataFrame])])
		vdf = input_relation
		if isinstance(vdf, str):
			vdf = vDataFrameSQL(relation=vdf)

		# self.vdf is the main vDataframe used within the class and is needed when
		# an user wants to discretize the data using reformat instead of themselves.
		# self.input_relation is an unmodified relation to the vDataframe and 
		# is needed in case a user wants to start the reformat from scratch.
		self.input_relation = vdf
		self.vdf = vdf.copy()
			
		self.skeleton = None
		self.sepsets = None
		self.pdag : dict = None


	# ---#
	def reformat(self, input_columns:list = [], 
					   exclude_columns:list = [], 
					   max_unique: int = 5, 
					   method:str = ""): 
		"""
		-----------------------------------------------------------
		Reformats the input vDataframe to optimize Causal Discovery.
		
		Parameters
		----------
		input_columns: list, optional
			Columns relevant to the Causal Network
		exclude_columns: list, optional
			Columns not relevant to the Causal Network
		max_unique: int, optional
			The maximum number of unqiue elements in any column.
		method: str, optional
			If max_unique is specified, the method is used to 
			discretize the numeric elements in the columns.

		Returns
		-------
		None
		"""
		
		temp = self.input_relation.copy()
		if isinstance(input_columns, str):
			input_columns = [input_columns]
		if isinstance(exclude_columns, str):
			exclude_columns = [exclude_columns]
		check_types(
			[
				("input_columns", input_columns, [list]),
				("exclude_columns", exclude_columns, [list]),
				("max_unique", max_unique, [int]),
				("method", method, [str])
			]
		)
		if (max_unique != None and max_unique > 20):
			warning_message = (
            	"having a max_unique '{0}' over 20 may cause performance issues"
        	).format(max_unique)
			warnings.warn(warning_message, Warning)

		if input_columns != []:
			temp.are_namecols_in(input_columns)
		if exclude_columns != []:
			temp.are_namecols_in(exclude_columns)
		if method != "":
			if method not in ["auto", "same_freq", "same_width", "smart", "topk"]:
				raise ParameterError(f"{method} is not a valid method for vDataFrame.discretize.")

		if input_columns != []:
			remove_cols = set(temp._VERTICAPY_VARIABLES_['columns']) - set(input_columns)
			temp.drop(remove_cols)

		if exclude_columns != []:
			remove_cols = set(exclude_columns) - set(temp._VERTICAPY_VARIABLES_['columns'])	
			temp.drop(remove_cols)

		temp.dropna()
		for col in temp._VERTICAPY_VARIABLES_['columns']:
			if temp[col].isnum():
				if max_unique > 0:
					temp[col].discretize(
						method=method,
						nbins=max_unique,
					)
		self.vdf = temp		

	# ---#
	def drop(self):
		self.input_relation.drop()
		self.vdf.drop()


	# ---#
	def visualize_skeleton(self):
		if self.skeleton != None:
			return CausalModel.visualize(self.skeleton, directed=False)

	# ---#
	def visualize_pdag(self):
		if self.pdag != None:
			return CausalModel.visualize(self.pdag, directed=True)

	# ---#
	@staticmethod
	def visualize(adj_dict, directed=True):
		nodes = list(adj_dict)
		directed_edges = set([(x, y) for x in nodes for y in adj_dict[x]])
		try:
			import networkx as nx
			import matplotlib.pyplot as plt
			from pyvis.network import Network

			if directed:
				G = nx.DiGraph()
			else:
				G = nx.Graph()
			G.add_nodes_from(nodes)
			G.add_edges_from(directed_edges)

			try:
				shell = get_ipython().__class__.__name__
				if shell == 'ZMQInteractiveShell':
					notebook = True
				else:
					notebook = False
			except NameError:
				notebook = False
			
			nt = Network('500px', '500px', notebook=notebook, directed = directed)
			nt.from_nx(G)
			nt.save_graph('nx.html')
			
			from IPython.display import IFrame
			return IFrame(src='./nx.html', width=700, height=700)
			
		except ModuleNotFoundError:
			warnings.warn("Could not produce graph because one or more of the following modules are missing: [networkx, matplotlib, pyvis]")
					


class PC(CausalModel):
	# ---#
	def __init__(self, input_relation):
		super().__init__(input_relation)

	# ---#
	def build_skeleton(self, method='stable', max_cond_vars=5, significance_level = 0.01, show_progress=True):
		"""
	----------------------------------------------------------------------------------------
	Generate the skeleton for the causal network.
	
	Parameters
	----------
	method: str, optional
		stable: Produces the same result despite the ordering of variables.
		orig: Follows the original pc algorithm.

	max_cond_vars: int, optional
		non-zero value that determines how many conditional variables are used, representing
		the expected number of parents any node has. The higher the number the longer the test
		will take to run, but the more precise the result will be.

	significance_level: float, optional
		p-value that determines independence.

	show_progress: bool, optional
		The value to control if the progress bar is displayed.

	Returns
	--------
	skeleton_edges: dict
		undirected edges in an adjacency dictionary
	sepsets: dict
		maps frozenset edge pairs to there corresponding seperation set
		"""

		columns = [col[1:-1] for col in self.vdf._VERTICAPY_VARIABLES_['columns']]
		all_pairs = set(itertools.combinations(columns, 2))
		max_cond_vars = min(len(columns), max_cond_vars+1) 
		sepsets = {} 
		skeleton_edges = {x: set(columns) - set([x]) for x in columns}
		
		cit = conditional_chi_square # Conditional Independence Test
		show_progress = show_progress and vp.options["tqdm"] # Verticapy Override option
		if show_progress:
			progress_bar = tqdm(total=max_cond_vars)
			progress_bar.set_description("Using n conditional variables: 0")

		if method not in ['stable', 'orig']:
			raise AttributeError("Choose from valid methods")

		# Discover the skeleton
		for cond_set_size in range(max_cond_vars):
			if method == 'stable':
				neighbors = {x: skeleton_edges[x].copy() for x in columns}
			if method == 'orig':
				neighbors = skeleton_edges
			
			deleted_edges = set()
			for (x,y) in all_pairs:                
				for z in itertools.chain(
										 itertools.combinations(neighbors[x]-set([y]), cond_set_size),
										 itertools.combinations(neighbors[y]-set([x]), cond_set_size)): # set of all conditionals
					z = list(z)
					if cit(self.vdf, x, y, z, alpha = significance_level):
						deleted_edges.add((x,y))
						skeleton_edges[x].remove(y)
						skeleton_edges[y].remove(x)
						sepsets[frozenset((x,y))] = z
						break
			all_pairs = all_pairs - deleted_edges
			
			if show_progress:
				progress_bar.update(1)
				progress_bar.set_description(f"Using n conditional variables: {cond_set_size + 1}")
		
		if show_progress:
			progress_bar.close()

		self.skeleton = skeleton_edges
		self.sepsets = sepsets

		return (skeleton_edges, sepsets)

	# ---#
	def build_pdag(self):
		"""
		Builds the partially directed acyclic graph

		Returns
		-------
		pdag: dict
			directed egdes in an adjacency dictionary
		"""
		skeleton_edges = self.skeleton
		sepsets = self.sepsets   
		
		# Create partially directed graph
		columns = skeleton_edges.keys()	
		pdag = {x: skeleton_edges[x].copy() for x in columns}
		
		# 1 Orientate the triples X - Z - Y into X -> Z <- Y  
		node_pairs = set(itertools.permutations(columns, 2))

		for (x,y) in node_pairs:
			if y not in skeleton_edges[x]:
				for z in set(skeleton_edges[x] & skeleton_edges[y]):
					#if z in pdag[x] and z in pdag[y] and x in pdag[z] and y in pdag[z]:
					if z not in sepsets[frozenset((x,y))]:
						if x in pdag[z]:
							pdag[z].remove(x)
						if y in pdag[z]:
							pdag[z].remove(y)
		
		edges_to_be_oriented = True
		while edges_to_be_oriented:
			remaining_edges = sum([len(pdag[x]) for x in columns])
		# 2 Orientate non-colliders X -> Z - Y into X -> Z -> Y
			for (x,z) in node_pairs:
				if z in pdag[x] and x not in pdag[z]:
					for y in pdag[z]:
						if z in pdag[y] and y not in skeleton_edges[x]:
							pdag[y].remove(z)             
		# 3 Orientate non cycles X-Y with a directed path from X to Y orienate X->Y
			for (x,y) in node_pairs:
				if x in pdag[y] and y in pdag[x]:
					if (is_dir_path(pdag, x, y)):
						pdag[y].remove(x)   
		# 4 For each X - Z - Y with X -> W, Y -> W and Z-W orienate Z->W
			for (z,w) in node_pairs:
				if w in pdag[z] and z in pdag[w]:
					for (x,y) in itertools.combinations(pdag[z], 2):
						if z in pdag[x] and z in pdag[y] and w in pdag[x] and w in pdag[y] and x not in pdag[w] and y not in pdag[w]:
							pdag[w].remove(z)
			edges_to_be_oriented = remaining_edges > sum([len(pdag[x]) for x in columns])
		
		self.pdag = pdag	
		return pdag	

	# ---#
	def fit(self, method='stable', max_cond_vars=5, significance_level = 0.01, show_progress=True):
		"""
	----------------------------------------------------------------------------------------
	Generate the skeleton and pdag for the causal network.
	
	Parameters
	----------
	method: str, optional
		stable: Produces the same result despite the ordering of variables.
		orig: Follows the original pc algorithm.

	max_cond_vars: int, optional
		non-zero value that determines how many conditional variables are used, representing
		the expected number of parents any node has. The higher the number the longer the test
		will take to run, but the more precise the result will be.

	significance_level: float, optional
		p-value that determines independence.

	show_progress: bool, optional
		The value to control if the progress bar is displayed.

	Returns
	-------
	pdag: dict
		directed egdes in an adjacency dictionary

		"""
		self.build_skeleton(method, max_cond_vars, significance_level, show_progress)
		return self.build_pdag()
