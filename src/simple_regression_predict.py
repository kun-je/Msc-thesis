import torch
from simpletransformers.classification import ClassificationModel
# from transformers import BertModel
import pandas as pd
import logging
from get_data import get_bleu, get_comet
from store_model import unpack_model
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

#get parameter from command line
DATASET = sys.argv[1]
LEARNING_RATE = 1e-3
BATCH_SIZE = 5
save_path = '../result/regression/'


unpack_model(save_path+'tar/german-simplebert-regression-'+DATASET+'_'+str(LEARNING_RATE)+'_'+str(BATCH_SIZE))



test_df = get_comet('tst-COMMON').loc[:,"hyp"] #get only hyp column
test_df = test_df.tolist() #make tst set become list[str]


# define hyperparameter


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
            #  'wandb_project': 'ME_hypothesis',
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
    "bert", "outputs/",
    num_labels=1,
    args=train_args,
)


# # Make predictions with the model
predictions, raw_outputs = model.predict(test_df)
np.savetxt(save_path+'ME_comet-1e-3.txt', predictions)
df = pd.DataFrame(predictions, columns = ['ME_comet'])
print(df.round(3))
print(df.dtypes)


