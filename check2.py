import sys
import pandas as pd

#getting source transcript from reference result and reference.de

# Get the filename from command line arguments
#path_result = sys.argv[1] #path to the destinate file(root directory of result for each set) /home/shared/kunchaya/data/must-c-v1/en-de/result/tst-COMMON/
#path_data = sys.argv[2] #root path to data ie: /home/shared/kunchaya/data/must-c-v1/en-de/data/dev/txt/dev(.en)


# Open the file for reading
# with open(path_result+"ref.txt", 'r') as file:
#     # Read the entire file contents
#     ref_result=file.readlines()
#     # line_num = ref_result.index(ref_result[0])
#     # print(line_num)
     
ref_result = pd.read_csv('/home/shared/kunchaya/data/must-c-v1/en-de/'+ 'train_st.tsv', sep = " " , names = ["id",	"audio"	,"n_frames",	"tgt_text"	,"speaker"])
#ref_result = ref_result.iloc[: , 3:-1]
print(ref_result)

# with open(path_data+".de", 'r') as file:
#     # Read the entire file contents
#     ref_de=file.readlines()

# with open(path_data+".en", 'r') as file:
#     # Read the entire file contents
#     transcript=file.readlines()


# file = open(path_result+"transcript.txt", "w") 
# index_list = []
# #compare reference result with de
# for result in ref_result:
#       for de in ref_de:
#          not_special = de.replace("‌", "")
#          if result == not_special: #if equal
#             #write source transcript to new file
#             #print(ref_de.index(de))
#             # index_list.append(ref_result.index(result))
#             # index_list.sort()

#             file.write(transcript[ref_de.index(de)])
#             break

    

    