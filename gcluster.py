from suffix import Trie

def gravitational_clustering(data):

	print 'Creating mapper...'
	
	mapper = {entry: index for index, entry in enumerate(data)}
	clusters =[[entry] for entry in data]
	parent = range(len(data))
	
	n_clusters = len(clusters)
	
	def find_parent(id):
		if id == parent[id]:
			return id
		parent[id] = find_parent(parent[id])
		return parent[id]
	
	def merge(i, j):
		i = find_parent(i)
		j = find_parent(j)
		if i != j:
			clusters[i].extend(clusters[j])
			clusters[j] = None
			parent[j] = i
			return True
			n_clusters -= 1
		return False
	
	print 'Creating trie...'
	
	trie = Trie(data)
	nearest = map(trie.nearest_neighbor, data)
	
	print 'Done creating trie.'
	
	import time
	st = time.time()
	
	# ignore first entry
	for node in nearest:
		print node.next()
		
	print 'Ignored first entry %0.3f' % (time.time() - st) 
	
	while n_clusters > 1000:
		print 'CLUSTERS: %d' % n_clusters
		st = time.time()
		for i, node in enumerate(nearest):
			try: node = node.next()[0]
			except: continue
			j = mapper[node]
			merge(i, j)
			print 'Merged %0.3f' % (time.time() - st)
		print 'Promised %0.03f' % (time.time() - st)
		# normalize
		clusters = map(lambda x: [x] if x != None and len(x) > 1 else x, clusters)
		print n_clusters
	
	return clusters