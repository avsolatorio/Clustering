from collections import defaultdict
from suffix import SuffixArray
from edistance import *
import numpy as np

	
'''
	
	Algorithm: double_cluster(data)
	
	A recursive double-clustering algorithm that is dependent on
	common substring length and pairwise Levenshtein (edit) distances.
	
	The first step of this algorithm is grabbing clusters of data
	(see cluster method) using the current min_common substring length
	threshold. After which, the algorithm checks if subclusters are
	expandable. In this case, the parameter min_common increments
	by step amount. 
	
	Parameters:

	data
		- string [...]
		- a list of string data to cluster
		
	min_common
		- float (0, INF)
		- the minimum common substring length that
		  would be considered to be 'similar'
	
	step
		- float (0, INF)
		- value for which min_common would increment
		  when a next level of clustering is needed
	
	eps
		- float (0, 1]
		- allowed median edit ratio for a cluster
		  to be considered final
	
	leaf_size
		- int [1, INF)
		- the maximum leaf-cluster size
		- threshold for when to calculate for
		  brute force pairwise edit distances
	
	
	algorithm
		- 'lattice' or 'dense'
		- lattice: uses strict integer-based common
			substring length clustering
		- dense: uses density-based common
			substring length clustering
	
	heirarchy
		- boolean
		- display clusters in a heirarchy or not
		
		
'''
def double_cluster(data, min_common = 3, step = 1, eps = 0.4, leaf_size = 60, algorithm = 'lattice', heirarchy = True):
	
	# immediately instantiate into suffix array for later
	if isinstance(data, SuffixArray):
		suffix_array = data
		data = suffix_array.data
	else:
		# base case: check first if singleton
		if len(data) == 1: return data
		suffix_array = SuffixArray(data)
		
	# draft initial clusters
	if algorithm == 'lattice':
		draft = lattice_spanning(suffix_array, min_common)
		
	elif algorithm == 'dense':
		draft = dense_spanning(suffix_array, min_common)
	
	
	if len(draft) == 1:
		# special case
		# no valid clustering found using current parameters
		# reuse suffix array to avoid recomputation
		return double_cluster(suffix_array, min_common + step, step, eps, leaf_size, algorithm, heirarchy)
	
	
	final_clustering = []
		
	for subcluster in draft:
		# check first if subcluster is expandable or not
		if len(subcluster) <= leaf_size and edit_radius(subcluster, eps) <= eps:
			# subcluster is ok, no need for expansion
			final_clustering.append(subcluster)
			
		else:
			# subcluster can still be expanded
			expanded = double_cluster(subcluster, min_common + step, step, eps, leaf_size, algorithm, heirarchy)
			if heirarchy: final_clustering.append(expanded)
			else: final_clustering.extend(expanded)
			
	# keep big clusters in front
	return sorted(final_clustering, key=len, reverse=True)
	
	
# filters noise based on expected feature count
def dense_spanning(data, min_common = 8):
	
	if isinstance(data, SuffixArray):
		suffix_array = data
		data = suffix_array.data
	else:
		suffix_array = SuffixArray(data)
	
	graph = suffix_array.similarity_graph()
	eps = 2 ** min_common
	graph = filter(lambda x: x[1] >= eps, graph)
	ind = spanning_tree(graph, len(data))
	return [map(lambda i: data[i], row) for row in ind]
	

def lattice_spanning(data, min_common = 10):
	
	if isinstance(data, SuffixArray):
		suffix_array = data
		data = suffix_array.data
	else:
		suffix_array = SuffixArray(data)

	f = range(len(data))
	def find(x):
		if x == f[x]:
			return x
		f[x] = find(f[x])
		return f[x]
	for conn in suffix_array.connectivity(min_common):
		for i in xrange(conn.start + 1, conn.stop):
			a = suffix_array.G[suffix_array[i-1]]
			b = suffix_array.G[suffix_array[i]]
			f[find(a)] = find(b)
	
	m = [[] for x in xrange(len(data))]
	for i in xrange(len(f)):
		m[find(i)].append(data[i])
	all = filter(lambda x: len(x) != 0, m)
	return all
				
# O(nlogn) clustering
def spanning_forest(data, n_clusters = 2):
	
	graph = data.similarity_graph() if isinstance(data, SuffixArray) else SuffixArray(data).similarity_graph()
	ind = spanning_tree(graph, len(data), n_clusters)
	return [map(lambda i: data[i], row) for row in ind]

def spanning_tree(graph, n, n_clusters = 1):

	needed = n - n_clusters
	
	if needed <= 0: return [[i] for i in xrange(n)]
	edges = sorted(graph, key=lambda x:x[1], reverse=True)
	
	f = range(n)
	
	def find(x):
		if x == f[x]: return x
		f[x] = find(f[x])
		return f[x]
		
	for e in edges:
		a = find(e[0][0])
		b = find(e[0][1])
		if a != b:
			f[a] = b
			needed -= 1
			if needed == 0:
				break
	
	# collect clusters
	clusters = [[] for i in f]
	for i in xrange(n):
		clusters[find(i)].append(i)
		
	return filter(lambda x: len(x) > 0, clusters)


# for debugging purposes
depth = 0
prev = 0
	
def label_heirarchy(tree, verbose = True):

	global depth
	global prev
	depth += 1
	
	if not isinstance(tree, list):
		
		if verbose: print ('-' * depth) + '+ %s' % (tree)
		depth -= 1
		
		return (tree, None)
		
	subtree = map(label_heirarchy, tree)
	label, child = zip(*subtree)
	
	dist = [np.average([edit_ratio(a, b) for b in label]) for a in label]
	id = min(range(len(dist)), key=lambda x: dist[x])
	if verbose: print ('-' * depth) + '+ %s' % (label[id])
	depth -= 1
	
	
	return (label[id], list(zip(label, child)))
	
	