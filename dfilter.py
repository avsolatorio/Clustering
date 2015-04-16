# dictionary filter
from main import split
from suffix import Trie

def respell(data, dictionary):
	trie = Trie()
	split_data = split(data)
	
	for row in split_data:
		for word in row:
			if word in dictionary:
				trie.insert(word)
				
	for row in split_data:
		# correct single words
		for word in row:
			if word not in dictionary:
				corrected = trie.nearest_neighbor(word).next()
		