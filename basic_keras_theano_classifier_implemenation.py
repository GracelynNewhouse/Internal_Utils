
# coding: utf-8

# In[2]:

get_ipython().magic('matplotlib inline')

##########
## IMPORTS
##########

# load required libraries
import pandas as pd
import numpy as np
import pickle
import ast
import matplotlib
import matplotlib.pyplot as plt
import sys
import csv
import gc
import os
import operator
import time
from __future__ import print_function


# In[3]:

# neural network libraries
from keras.models import Model
from keras.layers.core import Dense, Activation, Dropout, Merge, Masking
from keras.layers.recurrent import LSTM
from keras.layers import Input, merge
from keras.models import model_from_json
from keras.preprocessing import sequence
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from keras.optimizers import Adam, Nadam


# In[4]:

from keras import __version__ as keras_version
try:
    from keras.utils.visualize_util import plot
    plot_import_success = True
except:
    plot_import_success = False


# In[ ]:

# seed the generator before we import other modules with numpy
np.random.seed(1)
sys.setrecursionlimit(100000)

##############
## MCM IMPORTS
##############
if 'preproc_utils' in locals():
    del preproc_utils
import InternalUtils.Preproc_Utlis as preproc_utils


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[7]:

#### ## DATA DECLARATIONS
####################

# this is the cell to be shared between the evaluator notebook and the model training notebook

model_name = "SearchEventNoPrevSearch"
iteration = 3
model_description = ("Search-event based version of Austin with NO context like previous search"
                   + "max_len_srch_arg of 40 characters "
                   + "and max_known_chars of 72")

models_folder = "K:\\FoundProjects\\Baustin\\Models\\{}\\{}".format(model_name, iteration)
#models_folder = "/root/search_suggestions/models/prev_srch/{0}/".format(model_name)
try:
    os.makedirs(models_folder)
except:
    print("ERROR: you are about to overwrite previous training iteration: CHANGE ITERATION")


# In[8]:

search_directory = r"K:\FoundProjects\Baustin\VisitData\TRAIN_DATA"
#search_directory = "/root/search_suggestions/data/MegaIntGraphExtracts2/With_OP_and_Match_Appended/"
#search_file_pattern = "megaIntGraph2_201609[01][158]"
search_file_pattern = "search"
part_pn_pickle_path = r"K:\FoundProjects\ProductFamilyV3\part_info.pkl"
session_id_search_col_name = "id_to_shuffle"
search_entry_text_col_name = 'main_search_li'
last_object_col_name = 'pf_target'

#context_name = 'prev_srch_event_set' 

columns_to_keep = [search_entry_text_col_name]
    
filter_rule = "keep_all"


# In[9]:

#####################
## TRAINING DECLARATIONS
#####################

# this is the cell that will require human intervention

# data transformation parameters
max_len_srch_arg = 40
max_known_chars = 72
train_row_limit = int(1000000)
vldt_row_limit = int(train_row_limit/10)

# model structure hyper-parameters
char_embed_output_dim = 150
lstm_layer_width = 512
lstm_dropout_W = 0.05
lstm_dropout_U = 0.1
dropout_layer_val = 0.4

write_out_pre_split_rows = False

# collection of lines to write out to a README
out = []
out.append("model_name:{0}".format(model_name))
out.append("model_description:{0}".format(model_description))
pwd = get_ipython().magic('pwd')
out.append("working directory of ipynb notebook: {0}".format(pwd))


# In[10]:

preproc_utils.print_append('reading in search data', out)
raw_rows = preproc_utils.ingest_data(search_directory, search_file_pattern,
                                          columns_to_keep, filter_rule, out)

raw_rows.tail()


# In[ ]:

raw_rows.iloc


# In[ ]:

print(raw_rows["successful_parts"][:5])
print(raw_rows["main_search_li"][:5])
      


# In[19]:

class Part(object):
    def __init__(self,nbr,id_code,spec_codes, specs, leaf, term_names, term_ids, legacy_pfs):
        self.nbr = nbr
        self.id = id_code
        self.spec_codes = {key:val for key,val in spec_codes.items() if key != 'part_id'}
        self.specs = specs
        self.leaf = leaf
        self.term_names = term_names
        self.term_ids = term_ids
        self.legacy_pfs = legacy_pfs
        
part_obj_dict = pickle.load(open(part_pn_pickle_path,'rb'))


# In[ ]:

new_inps=[]
targs=[]
ids = []

n = 100
import ast
counter = 10000
unk_count = 0
for idx, part_list in enumerate(raw_rows["successful_parts"]):
    try:
        pfs = []
        for pn in ast.literal_eval(part_list):
            try:
                pfs.extend(list(part_obj_dict[pn.upper()].legacy_pfs))
            except:
                pfs.append("UNKNOWN")
                unk_count +=1
        for pf in list(set(pfs)):
            try:
                ids.append(counter)
                new_inps.append(raw_rows[search_entry_text_col_name].values[idx])
                targs.append(pf)
                counter +=1
            except:
                print("counter:",counter)
                print(raw_rows[search_entry_text_col_name].values[idx])
    except:
        print(idx, part_list)


# In[29]:

pre_split_rows = pd.DataFrame({session_id_search_col_name : ids,
                              search_entry_text_col_name : new_inps,
                               last_object_col_name :targs})
print(len(pre_split_rows))
pre_split_rows.head(100)


# In[ ]:

chars=list(set(c for c in ''.join(new_inps)))
print(chars)

char_to_int = {char : i for char, i in enumerate(chars) }
int_to_char= {i: char for char, i in enumerate(chars)}
print(char_to_int)
print(int_to_char)

pf_to_int ={char : i for char, i in enumerate(list(set(targs)))}
int_to_pf= {i: char for char, i in enumerate(list(set(targs)))}
print(pf_to_int)
print(int_to_pf)


# In[ ]:

if write_out_pre_split_rows:
    pre_split_rows.to_csv(search_directory + "prev_search_" + search_file_pattern
                          + "_" + filter_rule + ".csv")

preproc_utils.print_append('splitting into training, validation, and test data (not used)', out)
train_rows, vldt_rows, test_rows = preproc_utils.split_rows_into_sets(pre_split_rows, 
                                                     id_col_name=session_id_search_col_name)

preproc_utils.print_append('train_rows shape:{0}\nvldt_rows shape:{1}'.format(train_rows.shape, vldt_rows.shape), out)

preproc_utils.print_append('shuffling and limiting training and validation rows', out)
train_rows = preproc_utils.randomize_and_sample(train_rows, train_row_limit)
vldt_rows = preproc_utils.randomize_and_sample(vldt_rows, vldt_row_limit)

preproc_utils.print_append('train_rows shape:{0}\nvldt_rows shape:{1}'.format(train_rows.shape, vldt_rows.shape), out)


# In[ ]:

train_rows.head(10)


# In[ ]:

# for visual inspection
print(train_rows[1:10])
print(vldt_rows[1:10])
print(test_rows[1:10])


# In[ ]:

# get train data by column
train_char_data = train_rows[search_entry_text_col_name].values 
train_targ_data = train_rows[last_object_col_name].values

# get vldt data by column
vldt_char_data = vldt_rows[search_entry_text_col_name].values
vldt_targ_data = vldt_rows[last_object_col_name].values


print (train_char_data)
print (train_targ_data)

print (vldt_char_data)
print (vldt_targ_data)

print (train_char_data.shape)
print (train_targ_data.shape)

print (vldt_char_data.shape)
print (vldt_targ_data.shape)


# In[ ]:

# get context data column
#train_ctxt_data = train_rows[context_name].values
#vldt_ctxt_data = vldt_rows[context_name].values

#print (train_ctxt_data[1:10])
#print (vldt_ctxt_data[1:10])
#print (len(train_ctxt_data))
#print (len(vldt_ctxt_data))


# In[ ]:

# for visual inspection
print(train_char_data[1:20])
#print(train_ctxt_data[1:20])


# In[ ]:

# rank by frequency
def calc_chars_freq(char_data):
    character_cnts = {}
    for example in char_data:
        example_characters = list(str(example))
        for character in example_characters:
            if character in character_cnts:
                character_cnts[character] += 1
            else:
                character_cnts[character] = 1
    return character_cnts

# cap number of known values by their frequency
def cap_data_by_count(count_dict, max_known):
    count_tuples = sorted(count_dict.items(), key=operator.itemgetter(1), reverse=True)
    count_tuples = count_tuples[:max_known]
    count_list = [k for (k,v) in count_tuples]
    count_list.append('UNKNOWN')
    return count_list


# In[ ]:

# create character-index dicts

ranked_chars = calc_chars_freq(train_char_data)
character_list = cap_data_by_count(ranked_chars, max_known_chars)
character_list.append('\n')

customer_characters_indices = dict((c, i + 1) for i, c in enumerate(character_list))
indices_customer_characters = dict((i + 1, c) for i, c in enumerate(character_list))


# every seen output becomes a label: no fancy freq counting 
# and sorting and capping

new_list = list(train_targ_data)
new_list.append("UNKNOWN")
new_list.extend(list(vldt_targ_data))
labels = list(set(new_list))
labels.sort()
#labels.append('UNKNOWN')

label_indices = dict((c, i) for i, c in enumerate(labels))
print(label_indices)
indices_label = dict((i, c) for i, c in enumerate(labels))
print(indices_label)


# In[ ]:

# vectorize the the train and validation data

x_char_train = preproc_utils.vectorize_char_data(train_char_data, customer_characters_indices, max_len_srch_arg)
# here we're using the same vectorizing function for the context since the context is a search argument
#x_ctxt_train = preproc_utils.vectorize_char_data(train_ctxt_data, customer_characters_indices, max_len_srch_arg)
y_targ_train = preproc_utils.vectorize_targ_data(train_targ_data, label_indices)

print(x_char_train[1:10])
#print(x_ctxt_train[1:10])
print(y_obj_train[1:10])

x_char_vldt = preproc_utils.vectorize_char_data(vldt_char_data, customer_characters_indices, max_len_srch_arg)
# here we're using the same vectorizing function for the context since the context is a search argument
#x_ctxt_vldt = preproc_utils.vectorize_char_data(vldt_ctxt_data, customer_characters_indices, max_len_srch_arg)
y_targ_vldt = preproc_utils.vectorize_targ_data(vldt_targ_data, label_indices)

print(x_char_vldt[1:5])
#print(x_ctxt_vldt[1:5])
print(y_targ_vldt[1:5])

#x_char_ctxt_train = np.concatenate((x_ctxt_train, x_char_train), axis=1)
#x_char_ctxt_vldt = np.concatenate((x_ctxt_vldt, x_char_vldt), axis=1)
x_char_ctxt_train = x_char_train
x_char_ctxt_vldt =  x_char_vldt


# In[ ]:

# for visual inspection
print(x_char_train.shape)
#print(x_ctxt_train.shape)
print(y_obj_train.shape)
print(x_char_vldt.shape)
#print(x_ctxt_vldt.shape)
print(y_obj_vldt.shape)
#print(x_char_ctxt_train.shape)
#print(x_char_ctxt_vldt.shape)


# In[ ]:

# save off data and with optional
# read-only reload for performance
def save_off_array(arr, name, reload_ind):
    file = model_name + "_" + name + ".npy"
    np.save(file=file, arr=arr)
    if reload_ind:
        del name
        return np.load(file, mmap_mode='r')
    
# save_off_array(x_char_train, "x_char_train", False)
# save_off_array(x_ctxt_train, "x_ctxt_train", False)
# y_obj_train = save_off_array(y_obj_train, "y_obj_train", True)

# save_off_array(x_char_vldt, "x_char_vldt", False)
# save_off_array(x_ctxt_vldt, "x_ctxt_vldt", False)
# y_obj_vldt = save_off_array(y_obj_vldt, "y_obj_vldt", True)


# In[ ]:

def get_model():

    # the context (previous search arg) and the search event arg get concatenated
    # before being fed in, hence no merge layer and a max_len_srch_arg * 2
    main_input = Input(shape=(max_len_srch_arg,), name='main_input', dtype='int32')
    
    # max_known_char + UNKNOWN + newline + 0s as padding
    # makes max_known_char + 3 the input dimension of the embedding matrix

    embedding_layer = Embedding(input_dim=max_known_chars+3, output_dim=char_embed_output_dim, 
                  input_length=max_len_srch_arg, mask_zero=True)

    emb_main = embedding_layer(main_input)
    
    # a LSTM will transform the vector sequence into a single vector,
    # containing information about the entire sequence.

    lstm_layer = LSTM(lstm_layer_width, return_sequences=False, unroll=True
                      , dropout_W=lstm_dropout_W, dropout_U=lstm_dropout_U)

    lstm_out_main = lstm_layer(emb_main)
    
    dropout_out = Dropout(dropout_layer_val)(lstm_out_main)
       
    main_output = Dense(len(labels), activation='softmax', name='main_out')(dropout_out)                

    model = Model(input=main_input, output=main_output)

    opt = Nadam(lr=.001)
    model.compile(loss='categorical_crossentropy', optimizer=opt, 
                  metrics=["accuracy"])

    return model


# In[ ]:

gc.collect()

# save off model hyperparameters
model = get_model()
json_string = model.to_json()
with open(models_folder + '\\model_out.json', 'w') as f:
    for l in  json_string.split(","):
        f.write(l)

# save off dictionaries and some training parameters
tokenizing_dict = {'label_indices': label_indices, 'indices_label': indices_label
                    , 'customer_characters_indices': customer_characters_indices
                   , 'indices_customer_characters': indices_customer_characters
                   , 'customer_context_indices': {}
                   , 'indices_customer_context': {}
                   , 'max_len_srch_arg': max_len_srch_arg
                   , 'part_pn_pickle_path': part_pn_pickle_path}

with open(models_folder + "\\model_details.json", 'w') as f:
    for k, v in  tokenizing_dict.items():
        f.write(k  + "  "+ str(v)+ "\n")

# save off a diagram of the model
if plot_import_success:
    plot(model, models_folder + model_name + "_diagram.png", show_shapes=True)
    


# In[ ]:

# save off README

out.append("\nstarted training at {0}".format(time.strftime("%y/%m/%d:%H:%M:%S")))
out.append("\nversions of imported modules\n")
out.append("pandas:{0}".format(pd.__version__))
out.append("numpy:{0}".format(np.__version__))
out.append("keras:{0}".format(keras_version))
out.append("\n")
out.append("models_folder:{0}".format(models_folder))
out.append("search_directory:{0}".format(search_directory))
out.append("search_file_pattern:{0}".format(search_file_pattern))
out.append("\n")
out.append("session_id_search_col_name:{0}".format(session_id_search_col_name))
out.append("search_entry_text_col_name:{0}".format(search_entry_text_col_name))
out.append("last_object_col_name:{0}".format(last_object_col_name))
#out.append("context_name:{0}".format(context_name)) 
out.append("columns_to_keep:{0}".format(columns_to_keep))
out.append("filter_rule:{0}".format(filter_rule))   
out.append("\n")
out.append("max_len_srch_arg:{0}".format(max_len_srch_arg))
out.append("max_known_chars:{0}".format(max_known_chars))
out.append("train_row_limit:{0}".format(train_row_limit))
out.append("vldt_row_limit:{0}".format(vldt_row_limit))
out.append("\n")
out.append("char_embed_output_dim:{0}".format(char_embed_output_dim))
out.append("lstm_layer_width:{0}".format(lstm_layer_width))
out.append("lstm_dropout_W:{0}".format(lstm_dropout_W))
out.append("lstm_dropout_U:{0}".format(lstm_dropout_U))
out.append("dropout_layer_val:{0}".format(dropout_layer_val))
out.append("\n")

final_out = '\n'.join(out)

readme_path = models_folder + "\\README_" + model_name + ".txt"
with open(readme_path, "w") as f:
    f.writelines(final_out)


# In[ ]:

print(model.summary()) 


# In[ ]:

# train the model!

stopper = EarlyStopping(monitor='val_loss', patience=5)
checkpoint = ModelCheckpoint(models_folder+"\\"+model_name+ '.hdf5', monitor='val_loss', verbose=0, save_best_only=True, mode='auto')

start_time = time.time()
training_history = model.fit(x=x_char_ctxt_train, y=y_obj_train, batch_size = 1000, nb_epoch=5, verbose = 1
                             , validation_data=(x_char_ctxt_vldt, y_obj_vldt)
                             , callbacks=[stopper,checkpoint])
run_time = time.time() - start_time

curr_main_out_loss = min(training_history.history["val_loss"])
curr_main_out_acc = max(training_history.history["val_acc"])
message_list = ['I just finished training!']
message_list.append('Best Val Loss = {0}'.format(curr_main_out_loss))
message_list.append('Best Val Accuracy = {0}'.format(curr_main_out_acc))
message_list.append('Training Time = {0}'.format(run_time))
for message in message_list:
    print(message)


# In[ ]:

# append training history to README
with open(readme_path, "a") as f:
    f.write("\nfinished training at {0}\n".format(time.strftime("%y/%m/%d:%H:%M:%S")))
    f.write("loss --- acc --- val_loss --- val_acc")
    for epoch in range(len(training_history.history["val_loss"])):
        f.write("\n{0} {1} {2} {3}".format(training_history.history["loss"][epoch],
                                           training_history.history["acc"][epoch],
                                           training_history.history["val_loss"][epoch],
                                           training_history.history["val_acc"][epoch]))


# In[ ]:

# push message to Slack with model stats
message_list.append('\nDescription: {0}'.format(model_description))
preproc_utils.push_slack_training_update(model_name, message_list, False)


# In[ ]:

# save the model summary
# this hack having to do with stdout was 
# the only way I could figure out how to save this
saveout = sys.stdout
with open('{0}_summary.txt'.format(model_name),'w') as f:
    sys.stdout = f
    print(model.summary())
    sys.stdout = saveout


# In[ ]:

# save off the entire history of this notebook's execution
get_ipython().magic('notebook -e execution_history.py')


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



