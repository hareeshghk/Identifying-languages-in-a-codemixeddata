import numpy as np
import re
file=open("../data/hineng.txt")
data = file.readlines()
print len(data)
train_data=data[:17109]
test_data = data[17109:]
result_data=[]
tag={}
english={}
hindi={}
other={}
langcount=0
just={'en':0,"hi":1,"univ":2,"acro":3,"ne":4,"mixed":5}
just_rev ={0:"en",1:"hi",2:"univ",3:"acro",4:"ne",5:"mixed"}
for line in train_data:
	x =line.split('\t')
	if(len(x)==3):
		if(x[1]=="en"):
			langcount+=1
		elif(x[1]=="hi"):
			langcount-=1
		if x[0] not in tag:
			tag[x[0]]=[0,0,0,0,0,0]
		tag[x[0]][just[x[1]]]+=1
		if(x[1]=="en"):
			if(x[0] not in english):
				english[x[0]]=0
			english[x[0]]+=1
		elif(x[1]=="hi"):
			if(x[0] not in hindi):
				hindi[x[0]]=0
			hindi[x[0]]+=1
		else:
			if(x[0] not in other):
				other[x[0]]=0
			other[x[0]]+=1
# for i in tag.keys():
	# print i,tag[i]
english_keys = english.keys()
hindi_keys = hindi.keys()
other_keys = other.keys()
keys = tag.keys()
for line in test_data:
	x = line.split('\t')
	if(len(x)==3):
		if(x[0] in keys):
			result_data.append([x[0],just_rev[np.argmax(tag[x[0]])]])
		else:
			flag=0
			if(x[0].isalpha()==0):
				flag=1
			if(x[0][0]=="#"):
				flag=1
			url = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', x[0])
			if(len(url)!=0):
				flag=1
			##repitions left
			if(flag):
				result_data.append([x[0],"univ"])
			else:
				if(langcount<0):
					result_data.append([x[0],"hi"])
				else:
					result_data.append([x[0],"en"])

print len(result_data)
k=0
error=0
for line in test_data:
	x = line.split("\t")
	if(len(x)==3):
		if(x[1]!=result_data[k][1]):
			error+=1
		k+=1
print "\nerror percentage : " ,error*100.0/len(result_data) 
print "Accuracy : ",100-(error*100.0/len(result_data))