import sys
import pandas as pd
import sklearn
from sklearn.feature_selection import r_regression
import matplotlib.pyplot as plt 


#import matplotlib.pyplot as plt
path = '/home/shared/kunchaya/data/must-c-v1/en-de/result/'
dataset = sys.argv[1]

#bleu, chrf, ter score
bleu_metric = pd.read_csv(path + dataset+ '/bleu.txt', names = ["score"])
chrf_metric = pd.read_csv(path + dataset+ '/chrf.txt', sep = "=" , names = ["setup","score"])
ter_metric = pd.read_csv(path + dataset+ '/ter.txt',sep = "=" , names = ["setup","score"])

#extract coment score
comet_metric = pd.read_csv(path + dataset+ '/comet.txt', sep = ' ', header = None, names=["filename", "line", "score"])
comet_metric.drop(comet_metric.tail(1).index,inplace=True) #drop last 1 row

#geting decoder confidence score
decoder_confi = pd.read_csv(path + dataset+'/decoder_confi.txt', header = None)

#print(auto_metric)
# print(decoder_confi)
# print(auto_metric.iloc[:,-1:])

#calculating pearson colloreation of bleu, chrf, ter and comet
#bleu, chrf, ter
bleu_p = r_regression(decoder_confi, bleu_metric.values.ravel())
chrf_p = r_regression(decoder_confi, chrf_metric.iloc[:,-1:].values.ravel())
ter_p = r_regression(decoder_confi, ter_metric.iloc[:,-1:].values.ravel())

#comet
comet_p = r_regression(decoder_confi, comet_metric.iloc[:,-1:].values.ravel()) 

print(bleu_p)
print(chrf_p)
print(ter_p) #ter is negative bacause its measure negative correlation
print(comet_p)


pearson = [float(bleu_p), float(chrf_p), float(ter_p), float(comet_p)]
index = ['BLEU', 'Chrf', 'Ter',
         'COMET' ]
df = pd.DataFrame({'Pearson Correlation':pearson}, index = index)
#df['Pearson Correlation'] = df['Pearson Correlation'].astype(float) #changes data type to float
#print(df.dtypes)
ax = df.plot.bar(y='Pearson Correlation', rot=0)
plt.show() #the graph does not show in command line
