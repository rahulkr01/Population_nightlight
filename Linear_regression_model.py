import re
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as plt
from numpy import linalg as LA


f=open('Dataset/census_data.txt')
g=open('Dataset/Nightlight_data.csv')
nl=g.read().split('\n')
arr = f.read().split('\n')


censusdata=[]

census_fmap=arr[0].split('\t')
nighlight_fmap=nl[0].split(',')

fmap=[]
district_name=[" " for i in xrange(750)]
fmap.append('District code/census_code')
for i in range(6,len(census_fmap)):
	fmap.append(census_fmap[i])
Y=[]
Ymap=[]
for i in range(4,len(nighlight_fmap)):
	Ymap.append(nighlight_fmap[i])


for i in range(1, len(arr)):
	val=arr[i].split('\t')
	if val[3]=='DISTRICT' and val[5]=='Total':
		a1=[]
		y1=[]
		s=int(val[1].rstrip())
		district_name[s]=val[4]
		a1.append(s)
		ind=-1
		for k in range(1,len(nl)):
			dist_val=nl[k]
			dist_ar=re.split(r',', dist_val)
			if int(dist_ar[6])==s:
				ind=k
				for l in range(3,len(dist_ar)-1):
					y1.append(float(dist_ar[l+1]))
				break 
		if ind==-1:
			print 'i', i, 'dist ',val[4]
		for j in range(6,len(val)):
			s=val[j]
			if j>=6:
				s = s.replace(',', '')
				a1.append(float(s))
			else:
				a1.append(s)


		censusdata.append(a1)
		Y.append(y1)

print ('Number of district ',len(censusdata))

np.savetxt("data.txt", censusdata, delimiter=" ", newline = "\n", fmt="%s")
lm = linear_model.LinearRegression()


censusdata=np.array(censusdata)		# census data
Y=np.array(Y)			# Nightlight data

print ('attributes for censusdata',fmap)		 	#featuremap for census data
print (censusdata[0,:])		
print ('Y attributes ',Ymap)			#featuremap for nightlight data
print (Y[0,:])

## Dataset for classification

y=Y[:,len(Y[0])-1]			# Stable Lights
X=censusdata[:,len(censusdata[0])-1]	#Population
X=np.array(X)
X=np.reshape(X,(X.size,1))
y=np.array(y)
X[X==0]=1
y[y==0]=1

varo=np.var(y)*len(Y)

X0=np.log(X)
y0=np.log(y)
model = lm.fit(X0,y0)
predictions = lm.predict(X0)
# print predictions

print('slope classifier ',lm.coef_)
print ('intercept classifier ',lm.intercept_)
x_min=np.min(X0)-1
x_max=np.max(X0)+1
ar1=[ x_min*lm.coef_[0]+lm.intercept_ , x_max*lm.coef_[0]+lm.intercept_ ]
plt.plot([x_min,x_max],ar1)
error=LA.norm(predictions-y0)
var=np.var(y0)*len(X)
r2=1-float(error**2)/float(var)
print('r2',r2)

ssreg=LA.norm(np.exp(predictions)-np.exp(y0))
r2o=1-float(ssreg**2)/float(varo)
print('error ',error)
print ('r20rig',r2o)


misclassified=(predictions-y0)
pos=misclassified.argsort()[-10:][::-1]
neg=misclassified.argsort()[:10][::-1]


pos_dist=[district_name[i] for i in pos]
neg_dist=[district_name[i] for i in neg]
print ('neg_dist', neg_dist)
print('pos_dist', pos_dist)


snow_covered=[248,245,32,27,33,34,26,23,24,31,25,3,4,12,8,19,1,10,16,14,241,242,243,244,289,290,291,292,64,60, 63,56,59,65,61,57,62,58,66,327]
Dmetro=[474,572,603,92,93,94,95,96,97,98,99,100,536,342,519,518,521, 94]

X1=[0 for i in xrange(len(Y))]
for i in xrange(len(Y)):
	if X[i,0] in snow_covered:
		X1[i]=1 

X1=np.array(X1)
X1=np.reshape(X1,(X1.size,1))
X0=np.reshape(X0,(X0.size,1))
X2=np.concatenate((X0,X1),axis=1)


#model2
lm2= linear_model.LinearRegression()

model2=lm2.fit(X2,y0)
ypredict2=lm2.predict(X2)
ssreg2=LA.norm(np.exp(ypredict2)-y)
var2=np.var(np.exp(X2))*len(Y)
print ('var2',var2,'len y', len(Y))
r22=1-float(ssreg2**2)/float(varo)
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

X31=np.zeros(len(Y))
for i in xrange(len(Y)):
	if X[i,0] in Dmetro:
		X31[i]=1

X31=np.reshape(X31,(X31.size,1))
X3=np.concatenate((X2,X31),axis=1)
model3=lm3.fit(X3,y0)
ypredict3=lm3.predict(X3)
ssreg3=LA.norm(np.exp(ypredict3)-y)
var3=np.var(np.exp(X3))*len(Y)
print ('var2',var2,'len y', len(Y))
r3=1-float(ssreg3**2)/float(varo)
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

plt.scatter(X0,y0)
plt.show()