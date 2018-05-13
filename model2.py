import re
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as plt
from numpy import linalg as LA
import os
from openpyxl import load_workbook
import xlrd
import csv
import math
from scipy import stats



garr=[]
gmap=[]


#### Worker Data

for name in os.listdir("Data2/"):
	f=open("Data2/"+name)
	arr = f.read().split('\n')
	for i in range(7, len(arr)):
		ar2=arr[i].split(',')
		garr.append(ar2)

farr=np.zeros((641,1))
count=0

statelist=[]
for i in xrange(len(garr)):
	ar2=[]
	val=garr[i]
	if len(val)==len(garr[0]):
		ar1=val[3].split('-')
		s1=val[5].rstrip()
		s2=val[6].rstrip()
		s3=val[4].rstrip()
		s4=ar1[0].rstrip()

		if s4=="State" and len(ar1)>1:
			v1=ar1[1].rstrip()
			v2=v1.lstrip()
			if v2 not in statelist:
				statelist.append(v2)
		if s4=="District" and s3=="Total"    and s2=="Total":
			count+=1
			if s1=="Marginal workers seeking / available for work":
				farr[int(val[2]),0]=1+float(val[7])
			if s1=="Non-workers seeking / available for work":
				farr[int(val[2]),0]+=1+float(val[7])



#####Literacy rate
garr=[]
gmap=[]
for name in os.listdir("C-11_csv"):
	f=open("C-11_csv/"+name)
	arr = f.read().split('\n')
	for i in range(7, len(arr)):
		ar2=arr[i].split(',')
		garr.append(ar2)

# print len(garr)
Literacy=np.zeros((641,1))
count=0

statelist=[]
for i in xrange(len(garr)):
	ar2=[]
	val=garr[i]
	if len(val)==len(garr[0]):
		ar1=val[3].split('-')
		s1=val[5].rstrip()
		s3=val[4].rstrip()
		s4=ar1[0].rstrip()

		if s4=="State" and len(ar1)>1:
			v1=ar1[1].rstrip()
			v2=v1.lstrip()
			if v2 not in statelist:
				statelist.append(v2)
		if s4=="District" and s3=="Total"  and s1=="All ages":
			count+=1
			Literacy[int(val[2]),0]=(1+float(val[9])+float(val[15])+float(val[18])+float(val[21])+float(val[24])+float(val[27])+float(val[30])+float(val[33])+float(val[36])+float(val[39])+float(val[42]))
			#/(1+float(val[6]))





#####Census and Nightlight data

f=open('Dataset/census_data.txt')
g=open('Dataset/vectorsToDriveExample_2012.csv')
nl=g.read().split('\n')
arr = f.read().split('\n')


censusdata=[]

arr1=arr[0].split('\t')
census_fmap=[]
for j in xrange(len(arr1)):
	if j==6 or j==8 or j==9 or j==10 or j==13 or j==14:
		census_fmap.append(arr1[j])
census_fmap.append('Marginal Worker/Non-workers')
census_fmap.append('Literacy rate')
census_fmap.append('Urban/Rural Household')
nighlight_fmap=nl[0].split(',')

fmap=[]
district_name=[" " for i in xrange(750)]
fmap=census_fmap
Y=[]
Ymap=[]
for i in range(4,len(nighlight_fmap)):
	Ymap.append(nighlight_fmap[i])


censusdata=np.ones((641,9))

for i in range(1, len(arr)):
	val=arr[i].split('\t')
	if val[3]=='DISTRICT' and val[5]=='Total':
		a1=[]
		y1=[]
		s2=int(val[1].rstrip())
		district_name[s2]=val[4]
		# a1.append(s2)

		for j in range(6,len(val)):
			if j==6 or j==8 or j==9 or j==10 or j==13 or j==14:
				s=val[j]
				if j>=6:
					s = s.replace(',', '')
					a1.append(float(s)+1.0)
				else:
					a1.append(s)
		a1.append(farr[s2,0])
		# a1.append(farr[s2,1])
		a1.append(Literacy[s2,0])
		a1.append(1)
		censusdata[s2]=a1
	if val[3]=='DISTRICT' and val[5]=='Rural':
		s2=int(val[1].rstrip())
		s=val[9]
		s = s.replace(',', '')
		# censusdata[s2,8]/=float(s)+1.0

	if val[3]=='DISTRICT' and val[5]=='Urban':
		s2=int(val[1].rstrip())
		s=val[9]
		s = s.replace(',', '')
		censusdata[s2,8]*=(float(s)+1.0)



Y=np.ones((641,4))
ind=-1
for k in range(1,len(nl)):
	y1=[]
	dist_val=nl[k]
	dist_ar=re.split(r',', dist_val)
	if len(dist_ar)==9:
		s2=int(dist_ar[6])
		ind=k
		for l in range(3,len(dist_ar)-1):
			if l!=5:
				y1.append(float(dist_ar[l+1]))
		Y[s2]=y1
	


print ('Number of district ',len(censusdata))

np.savetxt("data.txt", censusdata, delimiter=" ", newline = "\n", fmt="%s")
lm = linear_model.LinearRegression()


censusdata=np.array(censusdata)		# census data
Y=np.array(Y)			# Nightlight data
np.savetxt('y.txt',Y)

print ('attributes for censusdata',fmap)		 	#featuremap for census data
print (censusdata[0,:])		
# print ('Y attributes ',Ymap)			#featuremap for nightlight data
# print (Y[1,:])

## Dataset for classification

###Normalized census data
mean=np.mean(censusdata,axis=0)
print ('mean: ',mean)
vari=np.max(censusdata,axis=0)
print ('variance: ',vari)
data1=(censusdata)/vari
m2=np.mean(Y,axis=0)
v2=np.max(Y,axis=0)
data2=(Y)/v2


y=Y[0:,3]			# Stable Lights
X=censusdata[0:,:]	#Population
# X=np.reshape(X,(X.size,1))

#X=np.concatenate((censusdata[:,2:3],X),axis=1)
X=np.array(X)
y=np.array(y)
X[X==0]=1
y[y==0]=1
for i in range(0,641):
	for j in xrange(1):
		if not np.isfinite(X[i,j]):
			print 'cenus i ',i,' ','j ',district_name[i]
	for j in xrange(4):
		if not np.isfinite(Y[i,j]):
			print 'NL i ', i,' ','j ',district_name[i]



varo=np.var(y)*len(Y)

X0=np.log(X)
y0=np.log(y)
a1=y0.argsort()[-10:]
X11=[]
y1=[]


for i in xrange(len(X0)):
	if i not in a1:
		X11.append(X0[i])
		y1.append(y0[i])
# X0=np.array(X11)
# y0=np.array(y1)

model = lm.fit(X0,y0)
predictions = lm.predict(X0)
# print predictions
print('slope classifier ',lm.coef_)
print ('intercept classifier ',lm.intercept_)
x_min=np.min(X0)-1
x_max=np.max(X0)+1
ar1=[ x_min*lm.coef_[0]+lm.intercept_ , x_max*lm.coef_[0]+lm.intercept_ ]
plt.plot([x_min,x_max],ar1)
error=LA.norm((predictions)-y0)
var=np.var(y0)*len(y0)
r2=1-float(error**2)/float(var)
print('r2',r2)
sse = np.sum((lm.predict(X0) - y0) ** 2, axis=0) / float(X0.shape[0] - X0.shape[1])
print sse
se = np.array([np.sqrt(np.diagonal(sse * np.linalg.inv(np.dot(X0.T, X0)))) ])
t = lm.coef_ / se
p = 2 * (1 - stats.t.cdf(np.abs(t), y0.shape[0] - X0.shape[1]))
print 'p',p



# ssreg=LA.norm(np.exp(predictions)-np.exp(y0))
# r2o=1-float(ssreg**2)/float(varo)
# print ('r20rig',r2o)



misclassified=(predictions-y0)
pos=misclassified.argsort()[-10:][::-1]
neg=misclassified.argsort()[:10][::-1]


pos_dist=[district_name[i] for i in pos]
neg_dist=[district_name[i] for i in neg]
print ('neg_dist', neg_dist)
print('pos_dist', pos_dist)


snow_covered=[248,245,32,27,33,34,26,23,24,31,25,3,4,12,8,19,1,10,16,14,241,242,243,244,289,290,291,292,64,60, 63,56,59,65,61,57,62,58,66,327]
Dmetro=[474,572,603,92,93,94,95,96,97,98,99,100,536,342,519,518,521, 94]

X1=[0 for i in xrange(len(y))]
for i in snow_covered:
	X1[i]=1 


X1=np.array(X1)
X1=np.reshape(X1,(X1.size,1))
# X0=np.reshape(X0,(X0.size,1))
X2=np.concatenate((X0,X1),axis=1)


#model2
lm2= linear_model.LinearRegression()

model2=lm2.fit(X2,y0)
ypredict2=lm2.predict(X2)
ssreg2=LA.norm((ypredict2)-y0)
var2=np.var(np.exp(X2))*len(Y)
print ('var2',var2,'len y', len(Y))
r22=1-float(ssreg2**2)/float(var)
print 'r22',r22
print 'coeff', lm2.coef_
print 'inter', lm2.intercept_
misclassified=((ypredict2)-y0)
pos=misclassified.argsort()[-10:][::-1]
neg=misclassified.argsort()[:10][::-1]


pos_dist=[district_name[i] for i in pos]
neg_dist=[district_name[i] for i in neg]
print ('neg_dist', neg_dist)
print('pos_dist', pos_dist)


#model3

lm3 = linear_model.LinearRegression()

X31=np.zeros(len(y))
for i in Dmetro:
	X31[i]=1

X31=np.reshape(X31,(X31.size,1))
X3=np.concatenate((X2,X31),axis=1)
model3=lm3.fit(X3,y0)
ypredict3=lm3.predict(X3)
ssreg3=LA.norm((ypredict3)-y0)
var3=np.var(np.exp(X3))*len(Y)
print ('var2',var2,'len y', len(Y))
r3=1-float(ssreg3**2)/float(var)
print 'r3',r3
print 'coeff', lm3.coef_
print 'inter', lm3.intercept_
misclassified3=((ypredict3)-y0)
pos3=misclassified3.argsort()[-10:][::-1]
neg3=misclassified3.argsort()[:10][::-1]


pos_dist3=[district_name[i] for i in pos3]
neg_dist3=[district_name[i] for i in neg3]
print ('neg_dist', neg_dist3 , 'id', neg3)
print('pos_dist', pos_dist3, 'id', pos3)

# plt.scatter(X0,y0)
# plt.show()