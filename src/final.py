import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers.embeddings import Embedding,Layer
from keras.layers import Flatten, Activation, Merge, Reshape
from keras.preprocessing.text import Tokenizer,  text_to_word_sequence
from keras.preprocessing.sequence import skipgrams, make_sampling_table
import matplotlib.pyplot as plt
from keras.layers import Convolution1D, MaxPooling1D
# from keras.utils.vis_utils import model_to_dot
from IPython.display import Image
import numpy as np
np.random.seed(13)
from keras.models import Sequential
from keras.layers import Embedding, Merge, Reshape, Activation, Flatten, Input, merge, Dense
from keras.layers.core import Lambda
from keras.utils import np_utils
from keras.utils.data_utils import get_file
# from keras.utils.vis_utils import model_to_dot, plot_model
from keras.preprocessing.text import Tokenizer#, base_filter
from keras.preprocessing.sequence import skipgrams, pad_sequences
from keras import backend as K
from keras.models import Model

from gensim.models.doc2vec import Word2Vec
from IPython.display import SVG, display
import pandas as pd

file=open("../data/hineng.txt")
data = file.readlines()
# print len(data)
def char_ngram_generator(text, n1=1, n2=6):
    z = []
    text2 = text
    for k in range(n1,n2):
        z.append([text2[i:i+k] for i in range(len(text2)-k+1)])
    z = [ngram for ngrams in z for ngram in ngrams]
    z.append(text)
    return z
just={'en':[0,0,0,0,0,1],"hi":[0,0,0,0,1,0],"univ":[0,0,0,1,0,0],"acro":[0,0,1,0,0,0],"ne":[0,1,0,0,0,0],"mixed":[1,0,0,0,0,0]}
vocab=[]
langdict ={}
langid={}
for line in data:
	x = line.split('\t')
	if(len(x)==3):
		if x[0] not in vocab:
			vocab.append(x[0])
			langdict[x[0]] = just[x[1]]
		else:
			langdict[x[0]] = [langdict[x[0]][i]|just[x[1]][i] for i in range(6)] 
top_words=len(vocab)
max_words=500
revvocab= [i+4 for i,x in enumerate(vocab)]
# print vocab

train_datax=[i for i,x in enumerate(vocab[:top_words])]
train_datay=[langdict[i] for i in vocab[:top_words]]
test_datax=[i for i,x in enumerate(vocab[:500])]
test_datay=[langdict[i] for i in vocab[:500]]

# vocab = train_data
totaldata=[]
for line in data:
	x=line.split('\t')
	if(len(x)==3):
		y=char_ngram_generator(x[0])
		totaldata.append(y)
w2vmodel = Word2Vec(totaldata,min_count=1)
vectdict = {}
for i in totaldata:
	newlist=[j for j in w2vmodel[i[0]]]
	for j in i[1:]:
		for k in range(len(w2vmodel[j])):
			newlist[k]+=w2vmodel[j][k]
	vectdict[i[-1]]=newlist
train_datax=[vectdict[x] for i,x in enumerate(vocab[:top_words])]
train_datay=[langdict[i] for i in vocab[:top_words]]
test_datax=[vectdict[x] for i,x in enumerate(vocab[:500])]
test_datay=[langdict[i] for i in vocab[:500]]


print train_datax[0]
# create the model
model = Sequential()
input_shape = (100,)
model_input = Input(shape=input_shape)
# z = Embedding(len(vocab), 300, input_length=1)(model_input)
# z = Flatten()(z)
z = Dense(128,activation='relu')(model_input)
z = Dense(250, activation='relu')(z)
model_output = Dense(6, activation='softmax')(z)
model = Model(model_input, model_output)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# print(model.summary())
# print len(test_datax)
# exit(0)
train_datax = np.array(train_datax)
train_datay = np.array(train_datay)
test_datax = np.array(test_datax)
test_datay = np.array(test_datay)
model.fit(train_datax,train_datay,validation_data=(test_datax,test_datay), epochs=10 ,batch_size=128 , verbose=2)