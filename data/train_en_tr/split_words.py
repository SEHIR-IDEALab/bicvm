import nltk
import codecs

inpf1 = codecs.open('turkish.10000.morph', 'r','utf-8')
outf1 = codecs.open('turkish.10000.words','w','utf-8')
max_len = 100

data1 = inpf1.read()
sents1 = data1.split('\n')


no_of_sent = len(sents1)
for i in range(no_of_sent):
	tokens1 = sents1[i].split(' ')
	outf1.write('\t'.join(tokens1)+'\n')

