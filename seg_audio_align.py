import sys
import pandas as pd

#getting source transcript from reference result and reference.de

# Get the filename from command line arguments
dataset = sys.argv[1]
path_result = "/home/shared/kunchaya/data/must-c-v1/en-de/result/"+dataset+"/" #ref file
path_split_audio = "/home/shared/kunchaya/data/must-c-v1/en-de/result/audio/"+dataset+"/"+dataset+".wav_list"
tsv_path  = "/home/shared/kunchaya/data/must-c-v1/en-de/"+dataset+"_st.tsv"

def make_tsv_dataframe(tsv_path):
    tsv_df = pd.read_table(tsv_path, quoting=3)
    return tsv_df

#return id of wav_file path and wav_file path
def get_utt_id_for_wav_path(path_split_audio):
    df_path = pd.read_table(path_split_audio, header=None, names = ["wav_path"], quoting=3) 
    df = pd.read_table(path_split_audio, header=None, names = ["id"], quoting=3)
    len_file = len("/home/shared/kunchaya/data/must-c-v1/en-de/result/audio/"+dataset+"/")
    df = df["id"].str[len_file:-4]
    df_path_with_id = pd.concat([df, df_path], axis=1)
    return df_path_with_id

#return new df1 that column[name1] contain rows in df2[name2]
def drop_rows_not_in_column(df1, df2, name1, name2):
    return df1[df1[name1].isin(df2[name2])]




if __name__ == "__main__":

    audio_df = get_utt_id_for_wav_path(path_split_audio)
    # print(audio_df)

    tsv_df = make_tsv_dataframe(tsv_path)
    # print(tsv_df)

    #if audio_df id is not in tsv_df id then drop
    audio_drop = drop_rows_not_in_column(audio_df, tsv_df, "id", "id")

    #add row
    audio_drop = audio_drop.assign(rows=range(len(audio_drop)))
    # print(audio_drop)


    #algin audio_drop with result ref
    result = pd.read_csv(path_result+"T_arr.txt", sep="\t", header=None, names = ["rows", "tgt"], quoting=3)
    result["rows"]= result["rows"].str[2:]
    result["rows"] = pd.to_numeric(result["rows"])
    # print(result)

    # print(result["rows"])
    # print(audio_drop["rows"])

    audio_as_result = drop_rows_not_in_column(audio_drop, result, "rows", "rows")
    print(audio_as_result)



    #write in new audio path
    # print(audio_as_result["wav_path"].to_csv(path_result+dataset+'_align.wav_list', sep="\n"))

    wav_path = audio_as_result["wav_path"].to_list()

    file = open(path_result+dataset+'_align.wav_list', 'a')
    for i in wav_path:
        file.write(str(i)+"\n")
