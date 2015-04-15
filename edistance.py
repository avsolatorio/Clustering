import numpy as np

def edit_distance(a, b, eps = 1):
	'''
		Implements k-bounded Damerau-Levenshtein (edit) distance
		via epsilon (percentage of edits). Counts inserting,
		deleting, changing, and swapping of characters.
	'''
	
	n = len(a)
	m = len(b)
	
	if n==0 or m==0: return max(n,m)
	
	# make sure second string is shorter to save space
	if n < m:
		a,b = b,a
		n,m = m,n
	
	cur, prev = range(m+1), None
	k = eps * max(n,m)
	
	for i in xrange(1, n+1):
		cur, prev, prev2 = [i], cur, prev
		for j in xrange(1, m+1):
			insert = cur[j-1] + 1
			delete = prev[j] + 1
			change = prev[j-1] + int(a[i-1] != b[j-1])
			swap = prev[j-2] + 1 if prev2 and a[i-2:i] == b[j-2:j] else change
			cur.append(min(insert, delete, change, swap))
		if min(cur) > k: return 1
	return cur[m]

def edit_ratio(a, b, eps = 1):
	return float(edit_distance(a,b)) / max(len(a), len(b))
	
def edit_radius(data, eps = 1):
	'''
		Returns the median pairwise edit ratio.
	'''
	n = len(data)
	if n <= 1: return 0
	dist = [edit_ratio(data[i], data[j], eps) for i in xrange(n) for j in xrange(i+1, n)]
	return np.median(dist)

# class Trie
# By Steve Hanov, 2011. Released to the public domain
# Adopted from http://stevehanov.ca/blog/index.php?id=114

# The Trie data structure keeps a set of words, organized with one node for
# each letter. Each node has a branch for each letter that may follow it in the
# set of words.

class Trie(object):

	def __init__(self, data = None):
		self.word = None
		self.children = {}
		if data != None:
			if isinstance(data, Trie):
				# make a copy of trie
				self.word = data.word
				self.children = {letter: Trie(trie) for letter, trie in data.children.items()}
			else:
				for entry in data:
					self.insert(entry)
	
	def insert(self, word):
	
		node = self
		for letter in map(ord, word):
			if letter not in node.children:
				node.children[letter] = Trie()
			node = node.children[letter]
		
		node.word = word
			

	# The search function returns a list of all words that are less than the given
	# maximum distance from the target word
	def search(self, word, k = 3):
		word = map(ord, word)
		# build first row
		currentRow = range(len(word) + 1)

		results = []

		# recursively search each branch of the trie
		for letter in self.children:
			self.searchRecursive(self.children[letter], letter, word, currentRow, 
				results, k)

		return results

	
	# This recursive helper is used by the search function above. It assumes that
	# the previousRow has been filled in already.
	def searchRecursive(self, node, letter, word, previousRow, results, k):

		columns = len(word) + 1
		currentRow = [previousRow[0] + 1]

		# Build one row for the letter, with a column for each letter in the target
		# word, plus one for the empty string at column 0
		for column in xrange(1, columns):

			insertCost = currentRow[column - 1] + 1
			deleteCost = previousRow[column] + 1

			if word[column - 1] != letter:
				replaceCost = previousRow[column - 1] + 1
			else:                
				replaceCost = previousRow[column - 1]

			currentRow.append(min(insertCost, deleteCost, replaceCost))

		# if the last entry in the row indicates the optimal cost is less than the
		# maximum cost, and there is a word in this trie node, then add it.
		if currentRow[-1] <= k and node.word != None:
			results.append((node.word, currentRow[-1]))

		# if any entries in the row are less than the maximum cost, then 
		# recursively search each branch of the trie
		if min(currentRow) <= k:
			for letter in node.children:
				self.searchRecursive(node.children[letter], letter, word, currentRow, 
					results, k)