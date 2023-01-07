import os
import csv
basepath = '/mnt/58a96951-a863-483e-b6bd-01c265d94667/single-layer-multikernel-cnn/Marathi-cnn-model/data'
out_file =  open('CUETNLP_subtask-1f-test_run-5.txt', 'a+')

from numpy import array
from sklearn.decomposition import TruncatedSVD
# define array
A = array([
	[1,2,3,4,5,6,7,8,9,10],
	[11,12,13,14,15,16,17,18,19,20],
	[21,22,23,24,25,26,27,28,29,30]])
print(A)
# svd
svd = TruncatedSVD(n_components=3)
svd.fit(A)
result = svd.transform(A)
print(result)