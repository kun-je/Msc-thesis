import sys
import pandas as pd


dataset = sys.argv[1]
path_result = "/home/shared/kunchaya/data/must-c-v1/en-de/result/"+dataset+"/" #ref file
path_split_audio = "/home/shared/kunchaya/data/must-c-v1/en-de/result/audio/"+dataset

wav_list_path = pd.read_table(path_result+dataset+"_align.wav_list", header=None, names = ["wav_path"], quoting=3) 



wav_path = wav_list_path["wav_path"].to_list()

# print(wav_path)

file = open(path_result+'manifest.tsv', 'a')
file.write(path_split_audio+"\n")

for i in wav_path:
     file.write(str(i)+"\t"+str(777)+"\n")
