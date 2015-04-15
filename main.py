import re
from cfilter import clean_course, clean_industry


cleaners = {
	'course': clean_course,
	'industry': clean_industry
}

def clean(data, style):
	if style not in cleaners:
		raise "style must be: " + str(cleaners.keys())
	return map(cleaners[style], data)

def read(filename):
	ans = None
	with open(filename, 'r') as f:
		lines = f.readlines()
		ans = [str(s.strip()) for s in lines]
	return ans

def write(data, filename = 'out.txt'):
	f = open(filename, 'w')
	for d in data:
		f.write(d)
		f.write('\n')
	print 'Printed to:', filename
	f.close()

def split(data, ch = '&/'):
	return [map(str.strip, re.split('['+ch+']', s)) for s in data]
	
def flat(data):
	res = []
	for d in data:
		if isinstance(d, list):
			res.extend(flat(d))
		else:
			res.append(d)
	return res

def smash(data, ch = '&/'):
	return list(set(flat(split(data, ch))))