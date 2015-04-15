# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn import metrics
# from pylev import levenshtein

import time
import re
# import numpy as np

from collections import defaultdict

_transform = {
	'tfidf': TfidfVectorizer,
	'count': CountVectorizer
}

def homogenized(strings):
    lengths = [len(s) for s in strings]
    n = max(lengths)
    for s in strings:
        k = len(s)
        yield [k] + [ord(c) for c in s] + [0] * (n - k)

def dehomogenized(points):
	lis = [''.join(map(chr, p[1:p[0] + 1])) for p in points]

def edit_distance(a, b):
	
	
	
	if len(a) == 0 or len(b) == 0:
		return max(len(a), len(b))
	
	if abs(len(a) - len(b)) > eps:
		return eps
	
	dp = [[0] * len(b) for x in xrange(len(a) + 1)]
	
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
	
	return dp[len(a)][len(b)] - abs(len(a) - len(b))

	
def text_to_vector(list_of_terms, vectorizer = 'tfidf', **kwargs):
	if vectorizer == 'homogenize':
		return np.array(list(homogenized(list_of_terms)))
		
	tf_vec = _transform[vectorizer](analyzer = 'char', ngram_range = (2, 4))
	return tf_vec.fit_transform(list_of_terms).toarray()

def get_cluster_name(cluster):
	votes = defaultdict(int)
	for name in cluster:
		try:
			name = return_or_recommend_proper_term(name)
		except:
			pass
		votes[name] += 1
	if len(votes) == 0:
		return cluster[0]
	cluster_name = max(votes.items(), key = lambda x: x[1])[0]
	
	return cluster_name

# returns list of clusters
def identify_clusters(data, vectorizer = 'tfidf', scanner = DBSCAN, debug = True, **kwargs):
	
	dp = {}
	
	def lev_metric(i, j):
		i, j = min(i, j), max(i, j)
		if (i, j) in dp:
			return dp[i, j]
		ans = edit_distance(data[i], data[j])
		dp[i, j] = ans
		return ans
	
	start_time = time.time()
	
	V = [[i] for i in xrange(len(data))]
	
	if debug: print ('Performing %s clustering...' % str(scanner))
	
	cluster_data = scanner(metric = lev_metric, **kwargs).fit(V)
	labels = cluster_data.labels_

	L = set(labels)
	n_clusters = len(L) - int(-1 in L)
	n_noise = list(labels).count(-1)
	
	if debug: print ('Clustering done!')
	if debug: print ('Time: %0.3f' % (time.time() - start_time))
	if debug: print ('Clusters: %d' % n_clusters)
	if debug: print ('Noise: %d' % n_noise)
	
	clusters = [[] for x in xrange(len(L) + int(-1 not in L))]
	
	
	for i, label in enumerate(labels):
		clusters[label].append(data[i])
	
	return clusters
	