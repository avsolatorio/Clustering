from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as similarity
from collections import defaultdict
# affinity can be cosine or euclidian

# create likelihood matrix
# vectorizer can be tfdif or count
# data should be a 2D list of attributes
# assume data is clean

def map_to_cluster(data, clusters, vectorizer = 'tfidf', **kwargs):
	
	if vectorizer == 'tfidf': vectorizer = TfidfVectorizer
	else: vectorizer = CountVectorizer
	
	vectorizer = vectorizer(analyzer = 'char', ngram_range = (2, 4))
	
	flat = set()
	
	def deep_find(item):
		if isinstance(item, list):
			for i in item: deep_find(i)
		else:
			flat.add(item)
			
	deep_find(data)
	flat = list(flat)
	
	mat = vectorizer.fit_transform(flat + clusters)
	mapper = {item: mat[i:i+1] for i, item in enumerate(flat)}
	
	def dist(d, y):
		y += len(flat)
		return similarity(mapper[d], mat[y:y+1])
		
	dp = {}
	
	def deep_map(item):
		if isinstance(item, list):
			return [deep_map(i) for i in item]
		else:
			if item not in dp:
				dp[item] = clusters[max(range(len(clusters)), key = lambda j: dist(item, j))]
			return dp[item]
	
	return deep_map(data)


# education_data is a 2D list of attributes
	
def likelihood_matrix(education_clusters, industry_clusters, education_data, industry_data, **kwargs):
	
	assert(len(education_data) == len(industry_data))
	
	edu = map_to_cluster(data = education_data, clusters = education_clusters, **kwargs)
	indu = map_to_cluster(data = industry_data, clusters = industry_clusters, **kwargs)
	
	likelihood = defaultdict(int)
	
	# return zip(edu, indu)
	
	for education, industry in zip(edu, indu):
		flat = []
		def flatten(item):
			if isinstance(item, list):
				for i in item:
					flatten(i)
			else:
				flat.append(item)
		flatten(education)
		for e in flat:
			likelihood[e, industry] += 1
	
	return likelihood
	# return likelihood
'''
print likelihood_matrix(
	education_clusters = ['Business', 'Marketing', 'Technology'],
	industry_clusters = ['Stocks', 'Information'],
	education_data = [['Business Ad', 'Marketin'], 'Info Technology', 'Marketing', 'Computer Tech'],
	industry_data = ['Stock', 'Information', 'Stocks', 'tocks']
)
'''