import torch
from simpletransformers.classification import ClassificationModel
from sklearn.metrics import accuracy_score

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


unpack_model('german-simplebert-classification-'+DATASET+'_'+str(LEARNING_RATE))


# define hyperparameter



# define hyperparameter
train_args ={"reprocess_input_data": True,
             "overwrite_output_dir": True,
             "fp16":False,
             "num_train_epochs": 1,
            #  "max_seq_length" : 512,
             "evaluate_during_training" : True,
             "eval_batch_size" : 3,
             "train_batch_size" : 3,
             "use_multiprocessing_for_evaluation": False, 
             "process_count": 1,
            #  "regression": True,
             "use_early_stopping" : True,
            #  "loss_type" : 'custom',
             "use_multiprocessing": False, #it will have infinite loop in eval and predict if true(although with false it will run a long time)
             'wandb_project': 'simple_classification',
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
    num_labels=10,
    args=train_args,
)

names = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100']

test_df = get_comet('tst-COMMON').loc[:,"hyp"] #get only hyp column
test_df = test_df.tolist() #make tst set become list[str]

# # Make predictions with the model
predictions, raw_outputs = model.predict(test_df)
print(names[predictions[0]])

np.savetxt('ME-COMET-classification_'+DATASET+'_'+str(LEARNING_RATE)+'_3.txt', predictions)
df = pd.DataFrame(predictions, columns = ['class'])
print(df)


