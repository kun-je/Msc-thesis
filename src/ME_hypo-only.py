import torch
from simpletransformers.classification import ClassificationModel
# from transformers import BertModel
import pandas as pd
import logging
from get_data import get_bleu, get_comet
from store_model import save_model
import sklearn
import os
import wandb
import tarfile
import sys

import numpy as np
os.environ["TOKENIZERS_PARALLELISM"] = "false" #to avoid error

#basic logging
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("ME_hyp-only")
transformers_logger.setLevel(logging.WARNING)

#using GPU 
cuda_available = torch.cuda.is_available() 

#get parameter from command line
DATASET = sys.argv[1]
LEARNING_RATE = 1e-3
BATCH_SIZE = 5

#getting data for training, evaluating and predicting
# train_df = get_bleu('train-tiny')
# dev_df = get_bleu('dev')
# test_df = get_bleu('tst-COMMON').loc[:,"hyp"] #get only hyp column

train_df = get_comet(DATASET)
dev_df = get_comet('dev')
test_df = get_comet('tst-COMMON').loc[:,"hyp"] #get only hyp column
# print(train_df)
test_df = test_df.tolist() #make tst set become list[str]

# define hyperparameter
train_args ={"reprocess_input_data": True,
             "overwrite_output_dir": True,
             "fp16":False,
             "num_train_epochs": 4,
             "max_seq_length" : 512,
             "evaluate_during_training" : True,
             "eval_batch_size" : BATCH_SIZE,
             "train_batch_size" : BATCH_SIZE,
             "use_multiprocessing_for_evaluation": False, 
             "process_count": 1,
             "regression": True,
             "use_early_stopping" : True,
             "early_stopping_metric" : "mse",
            #  "loss_type" : 'custom',
             "use_multiprocessing": False, #it will have infinite loop in eval and predict if true(although with false it will run a long time)
             'wandb_project': 'ME_hypothesis',
             "train_custom_parameters_only" : True,
             "custom_parameter_groups" : [
                {
                    "params": ["classifier.weight"],
                    "lr": LEARNING_RATE,
                },
                {
                    "params": ["classifier.bias"],
                    "lr": LEARNING_RATE,
                    # "weight_decay": 0.0,
                },
            ]             
            }


# Create a ClassificationModel
model = ClassificationModel(
    "bert", "bert-base-german-cased",
    num_labels=1,
    args=train_args,
    use_cuda=cuda_available
)

model.train_model(train_df, eval_df = dev_df, mse = sklearn.metrics.mean_squared_error)


result, model_outputs, wrong_predictions = model.eval_model(dev_df, mse = sklearn.metrics.mean_squared_error)
print(result)
print(model_outputs)


save_model('../result/regression/tar/','german-simplebert-regression-'+DATASET+'_'+str(LEARNING_RATE)+'_'+str(BATCH_SIZE))





