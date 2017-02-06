import os
import nltk
import codecs


classes = os.listdir('data_tok/english')
print classes

for cls in classes:
	in_path = 'data/turkish/'+cls+'/positive/'
	out_path = 'data_tok/turkish/'+cls+'/positive/'

	for filename in os.listdir(in_path):
		inpf = codecs.open(in_path+filename, 'r','utf-8')
		outf = codecs.open(out_path+filename,'w','utf-8')
		data = inpf.read()
		tokens = nltk.word_tokenize(data)
		outf.write('\t'.join(tokens))
