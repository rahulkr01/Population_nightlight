import os
from openpyxl import load_workbook
import xlrd
import csv
import numpy as np

garr=[]
gmap=[]




for name in os.listdir("Data2/"):
	f=open("Data2/"+name)
	arr = f.read().split('\n')
	for i in range(7, len(arr)):
		ar2=arr[i].split(',')
		garr.append(ar2)

print len(garr)
farr=np.zeros((641,2))
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
				farr[int(val[2]),0]=float(val[7])
			if s1=="Non-workers seeking / available for work":
				farr[int(val[2]),1]=float(val[7])



print count
print len(farr)
print statelist




