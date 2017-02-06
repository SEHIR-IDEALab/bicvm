import os
import nltk
import codecs

in_path = 'data/english/art/positive/'
out_path = 'data_tok/english/art/positive/'

for filename in os.listdir(in_path):
	inpf = codecs.open(in_path+filename, 'r','utf-8')
	outf = codecs.open(out_path+filename,'w','utf-8')
	data = inpf.read()
	tokens = nltk.word_tokenize(data)
	outf.write('\t'.join(tokens))
