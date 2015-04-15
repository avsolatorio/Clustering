from collections import defaultdict
import numpy as np
from sarray import suffix_array

# filters noise based on expected feature count
def dense_spanning(data, eps = 8, **kwargs):
	
	graph = suffix_array(data).similarity_matrix().items()
	eps2 = 2 ** eps
	graph = filter(lambda x: x[1] > eps2, graph)
	ind = spanning_tree(graph, len(data), **kwargs)
	return [map(lambda i: data[i], row) for row in ind]
	
# O(nlogn) clustering
def spanning_clusters(data, n_clusters = 2, **kwargs):
	
	graph = suffix_array(data).similarity_matrix().items()
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
		
def cluster(data, min_common = 10, **kwargs):
	sa = suffix_array(data)
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
		

def edit_distance(a, b):	
	
	if len(a) == 0 or len(b) == 0:
		return max(len(a), len(b))
	
	dp = [[0] * (len(b)+1) for x in xrange(len(a) + 1)]
	
	for i in xrange(len(a) + 1):
		dp[i][0] = i
	for j in xrange(len(b) + 1):
		dp[0][j] = j
	
	for i in xrange(1, len(a) + 1):
		for j in xrange(1, len(b) + 1):
			dp[i][j] = min(
				dp[i - 1][j] + 1,
				dp[i][j - 1] + 1,
				dp[i - 1][j - 1] + 1
			)
			if a[i - 1] == b[j - 1]:
				dp[i][j] = min(dp[i][j], dp[i - 1][j - 1])
			if i > 1 and j > 1 and a[i - 2] == b[j - 1] and a[i - 1] == b[j - 2]:
				dp[i][j] = min(dp[i][j], dp[i - 2][j - 2] + 1)
	
	return float(dp[len(a)][len(b)]) / max(len(a), len(b))

def edit_radius(data):
	n = len(data)
	if n <= 1:
		return 0
	dist = [edit_distance(data[i], data[j]) for i in xrange(n) for j in xrange(i+1, n)]
	
	return np.median(dist)
	
def double_cluster(data, min_common = 3, step = 1, eps = 0.4, leaf_size = 100, heirarchy = True, **kwargs):
	cl = cluster(data, min_common, **kwargs)
	ans = []
	for c in cl:
		if len(c) > leaf_size or edit_radius(c) > eps:
			# recluster
			recluster = double_cluster(c, min_common + step, step, eps, leaf_size, **kwargs)
			if heirarchy: ans.append(recluster)
			else: ans.extend(recluster)
		else:
			ans.append(c)
	ans.sort(key=len, reverse=True)
	# normalize bridges
	if heirarchy:
		while len(ans) == 1 and isinstance(ans[0], list):
			ans = ans[0]
	return ans
	
if __name__ == '__main__':
	sa = suffix_array(['hello', 'world'])
	for i in sa:
		print i, sa.G[i], sa.lcp[i], sa.string[i:]