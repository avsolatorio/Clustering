def full_split(data, characters = '/'):
	import re
	return [re.split(' [' + characters + '] ', s) for s in data]
	

def flat(d):
	res = []
	def flatter(data):
		if isinstance(data, list):
			for d in data: flatter(d)
		else:
			res.append(data)
	flatter(d)
	return res