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
# v = len(vocab)+1
# ngrams2Idx = {}
# ngrams_list = []
# vocab_ngrams = {}
# vocabdict={x:i for i,x in enumerate(vocab)}
# for i in vocab:
# 	ngrams_list.append(char_ngram_generator(i))
# 	vocab_ngrams[i] = char_ngram_generator(i)
# ngrams_vocab = [ngram for ngrams in ngrams_list for ngram in ngrams]
# ngrams2Idx = dict((c, i ) for i, c in enumerate(ngrams_vocab))
# ngrams2Idx.update(vocabdict)
# words_and_ngrams_vocab = len(ngrams2Idx)
# print words_and_ngrams_vocab

# # print vocab_ngrams.items()
# new_dict = {}
# for k,v in vocab_ngrams.items():
#     new_dict[ngrams2Idx[k]] = [ngrams2Idx[j] for j in v]

# print new_dict.keys()


# create the model
model = Sequential()
input_shape = (1,)
model_input = Input(shape=input_shape)
z = Embedding(len(vocab), 300, input_length=1)(model_input)
z = Flatten()(z)
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
model.fit(train_datax,train_datay,validation_data=(test_datax,test_datay), epochs=3 ,batch_size=128 , verbose=2)
print model.predict(np.array([1,2,3]))