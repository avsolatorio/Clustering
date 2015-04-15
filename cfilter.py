# HOW TO USE COURSE FILTER
#
# 1) import
# 2) read raw data
# 3) get clean data
#
#	import cf
#	data = cf.read(filename)
#	clean = cf.clean(data)


abbreviations = [
	("IT", "INFORMATION TECHNOLOGY", ""),
	("MBA", "BUSINESS ADMINISTRATION", ""),
	("BA", "BUSINESS ADMINISTRATION", ""),
	("ED", "EDUCATION"),
	("ECE", "ELECTRICAL COMPUTER ENGINEERING", ""),
	# # ("CE", "COMPUTER ENGINEERING", ""),
	# # ("CE", "CIVIL ENGINEERING", ""),
	("CS", "COMPUTER SCIENCE", ""),
	("HRM", "HOTEL RESTAURANT MANAGEMENT", ""),
	("HRTM", "HOTEL RESTAURANT TOURISM MANAGEMENT", ""),
	("ICT", "INFORMATION TECHNOLOGY / COMMUNICATIONS TECHNOLOGY", ""),
	("CTM", "COMMUNICATIONS TECHNOLOY / MANAGEMENT", ""),
	("COM SCI", "COMPUTER SCIENCE"),
	# # ("COMSCI", "COMPUTER SCIENCE"),
	# # ("COMPSCI", "COMPUTER SCIENCE"),
	("COMP", "COMPUTER"),
	("SCI", "SCIENCE"),
	("COM", "COMMUNICATIONS"),
	("COMM", "COMMUNICATIONS"),
	("MED", "MEDICINE"),
	("CA", "CULINARY ARTS"),
	("BIO", "BIOLOGY"),
	("CHEM", "CHEMISTRY"),
	("MATH", "MATHEMATICS"),
	("TECH", "TECHNOLOGY"),
	("MGT", "MANAGEMENT"),
	("LIT", "LITERATURE"),
	("PSYCH", "PSYCHOLOGY"),
	("ENGG", "ENGGINEERING"),
	("ENTREP", "ENTREPRENEURSHIP")
]

prefixes = [
	# layer 1
	# ex: BSM AMF
	"BSC ", "BSM ", "BS M ", "BS C ", "BFA ", "B FA",
	"B SC ", "B S M ", "B S C ", "B F A",
	"BE ", "B E ",
	# layer 2
	# ex: BSECE
	"BA ", "BS", "AB ",
	"B A ", "B S", "A B ",
	# layer 3
	# ex: BSMS CS, MS MA
	"MS ", "M S ",
	"ASS ", "AS ", "A S ",
	"MA ", "M A ",
	 "M ",
	# layer 4
	"BACHELOR OF ENINEERING ",
	"BACHELlOR OF ENINEERING ",
	"BACHELOR ENINEERING ",
	"BACHELLOR ENINEERING ",
	# layer 5
	"BACHELOR SCIENCE ",
	"BACHELLOR SCIENCE ",
	"BACHELOR ARTS ",
	"BACHELLOR ARTS ",
	"DIPLOMA ",
	"COURSE ",
	"BACHELORS ",
	"BACHELLORS ",
	"BACHELOR ",
	"BACHELLOR ",
	"GRADUATE ",
	"MINOR ",
	"MAJOR ",
	"DEGREE ",
	"CAREER ",
	"CERTIFICATE ",
	"MAJ ",
	"MAJORING ",
	"TRACK ",
	"SPECIALIZATION ",
	"SPECIALIZING ",
	"ASSOCIATED ",
	"ASSOCIATE ",
	"MASTERS ",
	"MASTER ",
	"DOCTOR ",
	"DOCTORATE ",
	"COLLEGE ",
	"IN SCIENCES ",
	"IN SCIENCE ",
	"IN ARTS ",
	"IN FINE ARTS ",
	"IN ART ",
	"IN FINE ART ",
	"OF SCIENCES ",
	"OF SCIENCE ",
	"OF ARTS ",
	"OF FINE ARTS ",
	"OF ART ",
	"OF FINE ART ",
	"WITH "
]

import re

# recursive replace
def replace_all(fr, to, s):
	while True:
		_ = s.replace(fr, to)
		if len(_) == len(s): break
		s = _
	return s

def remove_special_characters(s):
	s = s.strip()
	s = s.replace("'", "")
	s = s.replace("&", " & ")
	s = s.replace(" AND ", " & ")
	s = s.replace("/", " / ")
	s = re.sub(r"[^a-zA-Z0-9&/]", " ", s)
	return s.strip()

def remove_and(s):
	return s.replace('&', '')
	
to_trim = ["&", "/", "IN", "OF"]
	
def remove_contiguous(s):
	s = s.strip()
	# remove contiguous whitespace
	s = re.sub(r"\s\s+", " ", s)
	s = replace_all("/ &", "/", s)
	s = replace_all("& /", "/", s)
	
	modified = True
	while modified:
		modified = False
		for word in to_trim:
			if s.startswith(word + " "):
				s = s[len(word) + 1:]
				modified = True
				
	modified = True
	while modified:
		modified = False
		for word in to_trim:
			if s.endswith(" " + word):
				s = s[:-len(word) - 1]
				modified = True
	return s

upper = str.upper
	
def remove_year(s):
	return re.sub(r"[0-9].*Y.*[\S]]*[RS|R]", "", s)

def remove_duplicate_words(s):
	sp = s.split(" ")
	ws = set()
	sb = []
	
	for word in sp:
		if word == "/": ws.clear()
		if word != "&" and word in ws: continue
		if len(sb) != 0: sb.append(" ")
		ws.add(word)
		sb.append(word)
	
	s = ''.join(sb)
	s = replace_all("& &", "&", s)
	s = replace_all("/ /", "/", s)
	
	s.replace("/ /", "/")
	return s


	
def remove_abbreviations(s):
	
	s = " " + s + " "
	
	for abb in abbreviations:
		a = abb[0]
		b = " " + abb[1] + " "
		s = s.replace(" " + a + " ", b)
		
		# try with spaces in between the abbreviations
		if len(abb) == 3:
			sb = []
			for i in xrange(len(a)):
				if i > 0: sb.append(" ")
				sb.append(a[i])
			a = "".join(sb)
			s = s.replace(" " + a + " ", b)
		
	return s.strip()
	

	
def remove_prefix_bachelor(s):

	s = s + " "
	
	for r in prefixes:	
		if s.startswith(r):
			s = s[len(r):].strip() + " "
			
	s = s.strip()
	
	if s.startswith("IN ") or s.startswith("OF "):
		return s[3:]
	else:
		return s

def separate_majors(s):

	s = " " + s + " "
	
	for r in prefixes:
		s = s.replace(" " + r, " / ")
	
	s = remove_contiguous(s)
	s = s.replace("/ IN ", "/ ")
	s = s.replace("/ OF ", "/ ")
	if s.startswith("IN ") or s.startswith("OF "):
		s = s[3:]
		
	return s

course_filters = [
	upper,
	remove_special_characters,
	remove_and,
	remove_contiguous,
	remove_year,
	remove_prefix_bachelor,
	remove_abbreviations,
	separate_majors,
	remove_abbreviations,
	remove_duplicate_words,
	remove_contiguous
]

industry_filters = [
	upper,
	remove_special_characters,
	remove_contiguous,
	remove_abbreviations,
	remove_duplicate_words,
	remove_contiguous
]

# cleans a single course
def clean_course(course):
	for f in course_filters: course = f(str(course))
	return course

# cleans a single industry
def clean_industry(industry):
	for f in industry_filters: industry = f(str(industry))
	return industry
		