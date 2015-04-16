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
def double_cluster(data, min_common = 3, step = 1, eps = 0.4, leaf_size = 60, algorithm = 'lattice', heirarchy = True, **kwargs):
	
	# get initial clusters
	if algorithm == 'lattice':
		cl = lattice_spanning(data, min_common, **kwargs)
		
	elif algorithm == 'dense':
		cl = dense_spanning(data, min_common, **kwargs)
		
	ans = []
	for c in cl:
		if len(c) > leaf_size or edit_radius(c, eps) > eps:
			# recluster
			recluster = double_cluster(c, min_common + step, step, eps, leaf_size, algorithm, heirarchy, **kwargs)
			if heirarchy: ans.append(recluster)
			else: ans.extend(recluster)
		else:
			ans.append(c)
	# keep big clusters in front
	ans.sort(key=len, reverse=True)
	# normalize bridges
	if heirarchy:
		while len(ans) == 1 and isinstance(ans[0], list):
			ans = ans[0]
		# for i in xrange(len(ans)):
			# item = ans[i]
			# while len(item) == 1 and isinstance(item, list):
				# ans[i] = item[0]
				# item = ans[i]
		
	
	return ans
	
	
# filters noise based on expected feature count
def dense_spanning(data, min_common = 8, **kwargs):

	sa = data if isinstance(data, SuffixArray) else SuffixArray(data)
	graph = sa.similarity_graph()
	eps = 2 ** min_common
	graph = filter(lambda x: x[1] >= eps, graph)
	ind = spanning_tree(graph, len(data), **kwargs)
	return [map(lambda i: data[i], row) for row in ind]
	

def lattice_spanning(data, min_common = 10, **kwargs):

	sa = data if isinstance(data, SuffixArray) else SuffixArray(data)
	f = range(len(data))
	def find(x):
		if x == f[x]:
			return x
		f[x] = find(f[x])
		return f[x]
	for conn in sa.connectivity(min_common):
		for i in xrange(conn.start + 1, conn.stop):
			a = sa.G[sa[i-1]]
			b = sa.G[sa[i]]
			f[find(a)] = find(b)
	
	m = [[] for x in xrange(len(data))]
	for i in xrange(len(f)):
		m[find(i)].append(data[i])
	all = filter(lambda x: len(x) != 0, m)
	return all
				
# O(nlogn) clustering
def spanning_forest(data, n_clusters = 2, **kwargs):
	
	graph = data.similarity_graph() if isinstance(data, SuffixArray) else SuffixArray(data).similarity_graph()
	ind = spanning_tree(graph, len(data), n_clusters, **kwargs)
	return [map(lambda i: data[i], row) for row in ind]

def spanning_tree(graph, n, n_clusters = 1, **kwargs):

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
		
	return sorted(filter(lambda x: len(x) > 0, clusters), key=len, reverse=True)



depth = 0
prev = 0
	
def label_heirarchy(tree):

	global depth
	global prev
	depth += 1
	
	if not isinstance(tree, list):
		
		print ('-' * depth) + '+ %s' % (tree)
		depth -= 1
		
		return (tree, 1, tree)
		
	
	h = map(label_heirarchy, tree)
	label, count, child = zip(*h)
	
	dist = [np.average([edit_distance(a, b) for b in label]) for a in label]
	id = min(range(len(dist)), key=lambda x: dist[x])
	print ('-' * depth) + '+ %s' % (label[id])
	depth -= 1
	
	
	return (label[id], sum(count), list(zip(label, child)))
	
	