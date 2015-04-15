from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

class suffix_array(list):

	def __init__(self, strings, key = None, no_spaces = False):
		
		if key is not None:
			copy = strings
			list.__init__(self, list(copy).__getitem__(key))
			self.lcp = copy.lcp
			self.S = copy.S
			self.G = copy.G
			self.string = copy.string
			return
		
		# build long 'string'
		S = []
		G = []
		n = len(strings)
		for i, s in enumerate(strings):
			if no_spaces: s = s.replace(' ', '')
			S.extend(map(ord, s))
			S.append(n-i-1)
			G.extend([i] * (len(s) + 1))
		
		# build main suffix array
		N = len(S)
		sa = range(N)
		pos = S[:]
		gap = 1
		
		def check(boolean):
			return -1 if boolean else 1
		
		def suffix_cmp(i, j):
			if pos[i] != pos[j]:
				return check(pos[i] < pos[j])
			i += gap
			j += gap
			return cmp(pos[i], pos[j]) if (i < N and j < N) else cmp(j, i)
		
		tmp = [0] * N
		
		while True:
			sa.sort(cmp = suffix_cmp)
			for i in xrange(N-1):
				tmp[i+1] = tmp[i] + int(suffix_cmp(sa[i], sa[i+1]) < 0)
			for i in xrange(N):
				pos[sa[i]] = tmp[i]
			if tmp[N-1] == N-1: break
			gap <<= 1
		
		# build lowest common prefix
		lcp = [0] * N
		k = 0
		for i in xrange(N):
			if pos[i] != 0:
				j = sa[pos[i] - 1]
				while S[i+k] == S[j+k]:
					k += 1
				lcp[i] = k
				if k > 0: k -= 1
		
		
		# assign attributes
		list.__init__(self, sa)
		self.data = strings
		self.lcp = lcp
		self.S = S
		self.G = G
		self.string = '$'.join(strings) + '$'
	
	# slicer
	def __getitem__(self, key):
		# print key
		if isinstance(key, slice):
			return suffix_array(self, key)
		return list.__getitem__(self, key)
		# return None
		
	def __getslice__(self, i, j):
		return self.__getitem__(slice(i, j))
		
	def connectivity(self, eps):
		prev = None
		x,y = 0,-1
		res = []
		for i, id in enumerate(self):
			
			if self.lcp[id] >= eps:
				y = i
			elif y >= x:
				res.append(slice(x-1, y+1))
				x,y = i,-1
			else:
				x = i+1
			
		if y != -1:
			res.append(slice(x, y+1))
		return res
		
	def similarity_matrix(self):
		n = len(self.data)
		m = defaultdict(int)
		prev = None
		for i in self:
			if self.lcp[i] > 0:
				a = self.G[i]
				b = self.G[prev]
				if a > b: a, b = b, a
				c = (1 << self.lcp[i])
				m[a, b] += c
			prev = i
		return m
	
	def plot(self, eps = 4):
		eps = 2 ** eps
		
		G = nx.Graph()
		data = self.data
		sm = self.similarity_matrix()
		emap = defaultdict(list)
		for edge in sm.items():
			a = data[edge[0][0]]
			b = data[edge[0][1]]
			w = edge[1]
			emap[w].append((a, b))
			G.add_edge(a, b, weight = w)
		
		pos = nx.spring_layout(G)
		
		# nodes
		nx.draw_networkx_nodes(G,pos,node_size=1)
		
		from math import log
		
		# edges
		for e in emap.items():
			print 'plotting %d (%d items)' % (e[0], len(e[1]))
			nx.draw_networkx_edges(G,pos,edgelist=e[1],width=1,alpha=log(e[0]) / 32)
		
		plt.axis('off')
		plt.show()		
	
	def dbscan(self):
		raise 'incomplete function suffix_array.dbscan'
		"""
		cluster based on density
		minimize number of clusters using
		variance * clusters using ternary search
		"""
		
		graph = self.similarity_matrix().items()
		n = len(self.data)			
		
		forests = n
		edges = sorted(graph, key=lambda x:x[1], reverse=True)
		
		f = range(n)
		
		def find(x):
			if x == f[x]: return x
			f[x] = find(f[x])
			return f[x]
			
		def variance(data):
			if len(data) <= 1: return 0
			mean = float(sum(data)) / len(data)
			var = sum(map(lambda x: (x - mean) ** 2, data)) / (len(data) - 1)
			return var
		
		def value(cluster):
			score = len(cluster) * (variance(cluster) + 1)
			print forests, score
			return score
		
		state = [[d] for d in self.data]
		mnstate, mnvalue = state[:], value(filter(lambda x: x > 0, map(len, state)))
			
		for e in edges:
			a = find(e[0][0])
			b = find(e[0][1])
			if a != b:
				f[a] = b
				state[b].extend(state[a])
				state[a] = []
				v = value(filter(lambda x: x > 0, map(len, state)))
				if v < mnvalue:
					mnstate, mnvalue = state[:], v
					print 'UPDATED'
				forests -= 1
				if forests <= 1:
					break
		
		# collect clusters
		return sorted(filter(lambda x: len(x) > 0, mnstate), key=len, reverse=True)
			
		