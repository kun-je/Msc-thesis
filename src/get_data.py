import pandas as pd
from sklearn.metrics import mean_squared_error

#global variable
path = '/home/shared/kunchaya/data/must-c-v1/en-de/result/'



#read entire sentence
def read_hyp(dataset):
    with open(path + dataset+ '/hyp.txt') as file_hyp:
        lines = []
        for line in file_hyp:
            lines.append(line[:-1]) #get rid of '\n' character
        df = pd.DataFrame(lines) 
        df = df.rename(columns = { 0 :'hyp'})
    return df

#read score
def get_metric(dataset, metric_name):
    if metric_name == 'bleu':
        return pd.read_csv(path + dataset+ '/bleu.txt', names = ["bleu score"])
    elif metric_name == 'chrf':
        return pd.read_csv(path + dataset+ '/chrf.txt', sep = "=" , names = ["setup","chrf score"])
    elif metric_name == 'comet':
        metric = pd.read_csv(path + dataset+ '/comet.txt', sep = ' ', header = None, names=["filename", "line", "comet score"])
        metric =  metric.loc[:, "comet score"] #get only comet column
        if dataset != "train-tiny" :
            metric.drop(metric.tail(1).index,inplace=True) #drop last 1 row
        return metric
        


#making label for classfication task for 0-100 score
def make_classi_label(score_df):
    bins = [0, 11, 21, 31, 41, 51, 61, 71, 81, 91, 100]
    names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    score_df['scoreClass'] = pd.cut(score_df.iloc[:, 1], bins, labels=names)
    
    #drop regression column
    score_df=score_df.drop(score_df.columns[1], axis=1)
    return score_df


def get_bleu(dataset):
    bleu_metric = get_metric(dataset, 'bleu')

    hyp = read_hyp(dataset) #get hypothesis
    hyp_df = pd.concat([hyp, bleu_metric], axis=1)
    # hyp_df["bleu score"] =hyp_df["bleu score"].div(100) #make bleu score 0-1
    return hyp_df

def get_comet(dataset):
    comet_metric = get_metric(dataset, 'comet')
    
    hyp = read_hyp(dataset)

    hyp_df = pd.concat([hyp, comet_metric], axis=1)
    hyp_df["comet score"] =hyp_df["comet score"].mul(100) #make it score 0-100
    return hyp_df

def check_mse():
    y_true = [0.106013, 0.367206, 0.110429]
    y_pred = [0.23586144, 0.22312337, 0.2375267]
    return mean_squared_error(y_true, y_pred)




if __name__ == "__main__":
    dataset = 'train-tiny'
    hyp_df = get_bleu(dataset)
    print(hyp_df)
    #print(hyp_df.loc[[488]])
    # dev_df = get_bleu('dev').loc[2,:].to_frame()
    # print(dev_df)
    # print(isinstance(dev_df, pd.DataFrame))
    print(get_comet(dataset))

    #print(make_classi_label(get_comet(dataset)))


    #display entire row
    # pd.set_option('display.max_columns',None)
    # pd.set_option('display.max_rows', None)
    # pd.set_option('display.width',None)
    # pd.set_option('display.max_colwidth', None)

    # df = read_hyp(dataset)

    # print(df.loc[504])
    
