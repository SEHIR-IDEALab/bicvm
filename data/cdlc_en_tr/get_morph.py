import pipeline_caller
import codecs
import sys
from os import listdir
from os.path import isfile, join
import os


caller = pipeline_caller.PipelineCaller()
caller.token = 'xKktd3CBONoiwvB6okbu61W5gCoOMsTh'
caller.processing_type = 'whole'
caller.tool = 'pipelineNoisy'


classes = ['art','arts','biology','business','creativity','culture','design','economics','education','entertainment','health','politics','science','technology']

for cls in classes:
	mypath = 'turkish/'+cls+'/negative/'
	mypathnew = 'turkish_morph/'+cls+'/'
	
	try:
	    os.stat(mypathnew)
	except:
	    os.mkdir(mypathnew)
	mypathnew = 'turkish_morph/'+cls+'/negative/'
	try:
	    os.stat(mypathnew)
	except:
	    os.mkdir(mypathnew)

	allfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
	for fname in allfiles:
		inputf = codecs.open(mypath+fname, 'r','utf-8') 

		outputf = codecs.open(mypathnew+fname,'w','utf-8')

		data = inputf.read()
		caller.text = data
		while True:
			try:
				result = caller.call()
				break 
			except:
				pass

		#print(result)
		words = result.split('\n')

		sent = []
		till = -1
		for i in range(len(words)):
			parts = words[i].split('\t')
			if len(parts) > 5:
				if i == till:
					parts[2] = keep
					#print(parts[2])
					till = -1
				#print(i)
				if parts[1] == '_':
					keep = parts[2]
					till = int(parts[6])-1
					#print(till)
					continue
	
				#print(parts)	 
				if parts[5] != '_':
					morphs = parts[5].split('|')
					if parts[2] == '_' or parts[1] == parts[2]:
						word = parts[1]+' '+' '.join(morphs)
					else:
						word = parts[1]+' '+parts[2]+' '+' '.join(morphs)
				else:
					if parts[2] == '_' or parts[1] == parts[2]:
						word = parts[1]+' '+parts[4]
					else:
						word = parts[1]+' '+parts[2]+' '+parts[4]

				sent.append(word)
		outputf.write('\t'.join(sent)+'\n')
