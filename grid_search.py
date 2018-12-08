#
# Helper Methods
#

import numpy as np

def get_dataset_info_string(dataset):
    """Ridiculous way to print out dataset info"""
    
    # count number of samples/documents in dataset
    num_docs = lambda i: list(zip(*np.unique(dataset.target, return_counts=True)))[i][1]
    
    # ordering of output
    display_column_order = ['Target', 'Target Name', 'Documents']
    
    # uses target as index
    column_param_funcs = {
        'Target' : lambda i: i,
        'Target Name' : lambda i: dataset.target_names[i],
        'Documents' : lambda i: num_docs(i)
    }
    
    column_names = list(column_param_funcs.keys())
    column_headers_dict = {column_name:column_name for column_name in column_names}
    column_values = zip(*[[v(i) for v in column_param_funcs.values()] for i in range(len(dataset.target_names))])

    # useful dictionaries 
    info_dict = [{k:v(i) for k,v in column_param_funcs.items()} for i in range(len(dataset.target_names))]
    merged_values_by_column = dict(zip(column_names, column_values))    
    
    # get maximum length string for each column name in dataset
    get_max_str_len = lambda list: max([len(str(i)) for i in list])
    max_header_len = {k: max(len(k),get_max_str_len(v)) for k,v in merged_values_by_column.items()}
    ordered_max_header_len = [(column_name, max_header_len[column_name]) for column_name in display_column_order] 
    
    # format output
    template = '|'.join(["{%s:%d}" % (column_name, max_len) for column_name, max_len in ordered_max_header_len])
    
    # create header
    header = template.format(**column_headers_dict)
    bar = '-'*(sum([o[1] for o in ordered_max_header_len]) + len(ordered_max_header_len))

    # add category info to display string
    description = dataset.DESCR
    if dataset.DESCR is None:
        description = "None"
    data_set_info_string = 'Dataset Description: \n' + dataset.DESCR + '\n' + bar + '\n' + header + '\n' + bar + '\n'
    for rec in info_dict: 
          data_set_info_string += template.format(**rec) + '\n'
    data_set_info_string += bar
    
    # add total number of documents to string
    total_documents = dataset.target.shape[0]
    data_set_info_string += "\nTotal Documents:\t" + str(total_documents)

            
    return data_set_info_string    

#
# Create Glove Embedding Matrix from Gloves Pretrained Models
#    and format for specific Word Index
#

def format_glove_embedding_matrix(dimensions, word_index):
    """ 
        returns embedding_matrix corresponding to word_index columns
        
        embdedding_index 
            format: {key : word, value : word vector}
            
        Note: unfound words in word_index will be zero vectors
    """
    
    # the embedding dimensions should match the file you load glove from
    assert dimensions in [50, 100, 200, 300]
    
    GLOVE_EMBEDDINGS_FILE_TEMPLATE = 'data/glove/glove.6B.%sd.txt'
    glove_file = GLOVE_EMBEDDINGS_FILE_TEMPLATE % dimensions
    
    #
    # create embeddings index
    #
    
    # format: {key : word, value : word vector} 
    embeddings_index = {}
    
    # load glove embeddings file and fill index
    f = open(glove_file)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    #
    # build embedding matrix coressponding to given word_index
    #    Note: words not found in embedding index will be all-zeros.
    
    embedding_matrix = np.zeros((len(word_index) + 1, dimensions))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    # return index and matrix
    return embedding_matrix

#   
#   Convert Documents to Sequences, One-Hot Encode Targets
#
#        NOTE: KerasClassifier already performs One-Hot Encoding transformation
#

import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def encode_data(tokenizer, bunch, one_hot_encode=True, max_document_len=500):

    # tokenize, build vocab and generate document sequences
    sequences = tokenizer.texts_to_sequences(bunch.data)

    # pad and clip
    X = pad_sequences(sequences, maxlen=max_document_len)
    
    y =  bunch.target
    if one_hot_encode:
        y = keras.utils.to_categorical(bunch.target)

    return X, y

#
#    KerasClassifier Create Model Function -- Wrapper for scikitlearn
#

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, GRU, SimpleRNN, Embedding

#from keras.constraints import maxnorm
#from keras.layers import Dropout
# kernel_initializer='uniform', kernel_constraint=maxnorm(weight_constraint)

def create_model(rnn_type=None, embedding_dimensions=None , num_outputs=None, 
                 state_size=None, dropout=None, recurrent_dropout=None, 
                 activation=None, loss=None, optimizer=None, metrics=None):
    
        # build model
        model = Sequential()
        
        # build recurent layer
        if rnn_type == 'LSTM':
            model.add(LSTM(embedding_dimensions, dropout=dropout, recurrent_dropout=recurrent_dropout ))
        else:
            model.add(GRU(embedding_dimensions, dropout=dropout, recurrent_dropout=recurrent_dropout ))
        
        # build single dense layer
        model.add(Dense(num_outputs, activation=activation))
        
        # compile model
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        
        return model

#
# Displaying And Sumarizing Results history,  src: https://stackoverflow.com/questions/37161563/how-to-graph-grid-scores-from-gridsearchcv
#
    
from matplotlib import pyplot as plt

def plot_grid_search(cv_results, selected_scorer, grid_param_1, grid_param_2, name_param_1, name_param_2):
    
    # Get Test Scores Mean and std for each grid search
    mean_test_score_string = 'mean_test_%s' %  selected_scorer 
    std_test_score_string = 'std_test_%s'  %  selected_scorer 
    
    scores_mean = cv_results[mean_test_score_string]
    scores_mean = np.array(scores_mean).reshape(len(grid_param_2),len(grid_param_1))

    scores_sd = cv_results[std_test_score_string]
    scores_sd = np.array(scores_sd).reshape(len(grid_param_2),len(grid_param_1))

    # Plot Grid search scores
    _, ax = plt.subplots(1,1)

    # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
    for idx, val in enumerate(grid_param_2):
        ax.plot(grid_param_1, scores_mean[idx,:], '-o', label= name_param_2 + ': ' + str(val))

    ax.set_title("Grid Search Scores", fontsize=20, fontweight='bold')
    ax.set_xlabel(name_param_1, fontsize=16)
    ax.set_ylabel('CV Average Score', fontsize=16)
    ax.legend(loc="best", fontsize=15)
    ax.grid('on')
    
#
# Report Best Scores from Paramters Search, src: scikitlearn
#

def report_top_candidates( cv_results, selected_scorer = None, n=3 ):
    
    rank_test_score_string = 'rank_test_score'
    mean_test_score_string = 'mean_test_score'
    std_test_score_string  = 'std_test_score'
    
    if selected_scorer is not None:
        rank_test_score_string = 'rank_test_%s'   %  selected_scorer 
        mean_test_score_string = 'mean_test_%s'   %  selected_scorer 
        std_test_score_string  = 'std_test_%s'    %  selected_scorer 
    
    for i in range( 1, n+1 ):
        
        candidates = np.flatnonzero( cv_results[rank_test_score_string] == i )
        for candidate in candidates:
            
            print("Model with rank: {0}".format(i))
            
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  cv_results[mean_test_score_string][candidate],
                  cv_results[std_test_score_string][candidate]))
            
            print("Parameters: {0}\n".format(cv_results['params'][candidate]))
            
    return candidates[0]

#
# plot, src: https://medium.com/@sabber/classifying-yelp-review-comments-using-cnn-lstm-and-pre-trained-glove-word-embeddings-part-3-53fcea9a17fa
#

def plot_history(history):
    
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]
    
    if len(loss_list) == 0:
        print('Loss is missing in history')
        return 
    
    epochs = range(1,len(history.history[loss_list[0]]) + 1)
    
    #
    # Loss
    #
    
    plt.figure(1)
    for l in loss_list:
        plt.plot(epochs, history.history[l], 'b', label='Training loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], 'g', label='Validation loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    #
    # Accuracy
    #
    
    plt.figure(2)
    for l in acc_list:
        plt.plot(epochs, history.history[l], 'b', label='Training accuracy (' + str(format(history.history[l][-1],'.5f'))+')')
    for l in val_acc_list:    
        plt.plot(epochs, history.history[l], 'g', label='Validation accuracy (' + str(format(history.history[l][-1],'.5f'))+')')

    #
    # Plot Settings
    #
    
    plt.title('Accuracy')
    
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    
    plt.legend()
    
    #
    # Display Plot
    #
    plt.show()


    #
#
#   GloveEmbedder 
#
#
#       -- Used for Embedding the Train/Test Dataset before 
#          performing Hyper-Paramater Tuning / Grid Search.
#
#       -- Used for Writing and Loading Tokenizer
#
#          members:
#
#                -Embedding Matrix,
#                -Tokenizer,
# 
#

from keras.preprocessing.text import Tokenizer
from keras import layers
from keras.models import Sequential

import warnings
warnings.filterwarnings(action='once')

class GloveEmbedder():

  def __init__(self, embedding_dimensions=50):
    assert embedding_dimensions in [50,100,200,300]
    self.embedding_dimensions = embedding_dimensions

  def fit(self, tokenizer=None):
        
    assert tokenizer is not None
    self.tokenizer = tokenizer
    
    # build model
    self.model = Sequential()
    
    # load embeddings weights
    self.embeddings_matrix = format_glove_embedding_matrix(self.embedding_dimensions, self.tokenizer.word_index)
    
    # Define single layer here.     
    self.embeddings_layer = layers.Embedding(len(self.tokenizer.word_index) + 1, self.embedding_dimensions,
                    weights=[self.embeddings_matrix],
                    trainable=False, 
                    name='embeddings_layer')
    
    # add to model
    self.model.add(self.embeddings_layer)
    
    # these parameters do not matter
    self.model.compile(optimizer='rmsprop',loss='categorical_crossentropy', metrics=['accuracy'])

  def compute_output_shape(self, input_shape):
    shape = tf.TensorShape(input_shape).as_list()
    shape[-1] = self.embedding_dimensions
    return tf.TensorShape(shape)

  def get_embedding_matrix(self):
        return self.model.layers[0].get_weights()[0]

  def embed(self, inputs ):
    return self.model.predict(inputs)

# Class for embedding dataset and storing embedding weights
# [TODO]: impliment functions bellow
# Modify to perform dataset clipping and padding as well, Fit function should take X and y data only
# Modify init to include max article length parameter as well
# save weights
# save tokenizer
# load weights
# load tokenizer
# move format glove embeddings function to this class

#
#
#
#
#              Grid Search / Random Search / Hyper Parameter Tuning 
#
#                   selecting model parameters 
#
#

from scipy.stats import randint as sp_randint

#
#
#           Test/Search Settings -- apply to each model tested
#
#

# - Single Dense Layer Variable Activation
# - Shared Glove Embeddings

DATA_SETS_PATH = 'data/data_sets'
DATA_SET_NAME = 'bbcsport'
CATEGORIES = ['athletics', 'cricket', 'football', 'rugby', 'tennis']

MAXIMUM_ARTICLE_LENGTH = 500
EMBEDDING_DIMENSIONS = 50 # Possible: 50, 100, 200, 300

TEST_SIZE = 0.2
RANDOM_SEED = 42
VERBOSE = 2

EPOCHS = 5
BATCH_SIZE = 128

#
# Grid Search Settings
#

GRID_N_FOLDS = 10 # Number of folds for Stratified KFold
GRID_N_JOBS = 1
GRID_IID = False # If True, the data is assumed to be identically distributed across the folds,
            #  and the loss minimized is the total loss per sample, 
            #  and not the mean loss across the folds.
        
GRID_RNN_TYPE = ['GRU', 'LSTM'] # Possible : 'GRU', 'LSTM'
GRID_STATE_SIZE = [60] # [5*i for i in range(20)]
GRID_DROPOUT = np.linspace(0.2, 0.3, 2)
GRID_RECURRENT_DROPOUT = np.linspace(0.2, 0.3, 2)
GRID_ACTIVATION = ['sigmoid'] #['relu', 'tanh', 'sigmoid'] #'softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
GRID_LOSS = ['categorical_crossentropy']
GRID_OPTIMIZER = ['rmsprop'] #, 'SGD',  'Adam'] # 'RMSprop','Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
GRID_METRICS = [['accuracy']]
    
#
# Randomized Search Settings
#

RAND_N_JOBS = 4  # Parallel tasks
RAND_N_FOLDS = 2 # Number of folds for Stratified KFold
RAND_IID = False # If True, the data is assumed to be identically distributed across the folds,
            #  and the loss minimized is the total loss per sample, 
            #  and not the mean loss across the folds.
RAND_N_ITER = 10 # Number of parameter settings that are sampled.  (default 128)
    
RAND_RNN_TYPE = GRID_RNN_TYPE
RAND_STATE_SIZE = sp_randint(20, 30)
RAND_DROPOUT = np.random.uniform(0.1,0.5,9)
RAND_RECURRENT_DROPOUT = np.random.uniform(0.1,0.5,9)
RAND_ACTIVATION = GRID_ACTIVATION
RAND_LOSS = GRID_LOSS
RAND_OPTIMIZER = GRID_OPTIMIZER
RAND_METRICS = GRID_METRICS
    
#
# Scorer Settings
#

SELECTED_SCORER = 'f1_macro' #'Accuracy'
REFIT = SELECTED_SCORER

MODEL_SAVE_DIR = "data/model_files/"

# Note: You may want to tune these parameters as well 
#weight_constraint = [1, 2, 3, 4, 5]
#init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
#learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
#momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
#neurons = [1, 5, 10, 15, 20, 25, 30]

#
# Grid Search / Random Search / Hyper Parameter Tuning and Cross Validation (Cont.)
#

#
#
#           Preparing Dataset -- 
#               - Perform Embedding step before Grid Search
#               - Attach Embedding Layer to best performing model
#
#


#
# Import Dataset
#

import os
from sklearn.datasets import load_files

container_path = os.path.join(DATA_SETS_PATH, DATA_SET_NAME)
bunch = load_files(container_path=container_path, description=DATA_SET_NAME, categories=CATEGORIES, decode_error='ignore', encoding='utf-8')


#
#   Convert Documents to Sequences, but do not One-Hot Encode Targets
#   NOTE: for scikit learn gridSearchCV as KerasClassifier already 
#         performs this transformation


import keras
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split


#
# build tokenizer
#

tokenizer_master = Tokenizer()
tokenizer_master.fit_on_texts(bunch.data)

#
# Format Dataset for RNN 
#
#     Note: Do Not One-Hot Encode Targets, KerasClassifier already performs this transformation
#

X_enc, y_enc = encode_data(tokenizer_master, bunch, one_hot_encode=False, max_document_len=MAXIMUM_ARTICLE_LENGTH)

#
# Embed Train/Test Inputs from Dataset Before Performing Grid Search for Best Model Parameters 
#

# Class for embedding dataset and storing embedding weights and tokenizer
glove_embedder = GloveEmbedder( embedding_dimensions=EMBEDDING_DIMENSIONS )
glove_embedder.fit( tokenizer_master )

# Embed Dataset
X_embedded = glove_embedder.embed( X_enc )


#
# Split dataset it into train / test subsets 
#

# train / test data
X_train_embedded, X_test_embedded, y_train, y_test = train_test_split(X_embedded, y_enc, test_size=TEST_SIZE,
                                                            stratify=bunch.target, 
                                                            random_state=RANDOM_SEED)


#
# Grid Search / Random Search / Hyper Parameter Tuning and Cross Validation (Cont.)
#

#
#
#           Setup Classifier Wrapper, Scorers, and Parameters Grid
#               
#               
#
#

#
# Setup Classifier Wrapper
#

from keras.wrappers.scikit_learn import KerasClassifier

# Keras Classifier Wrapper
clf = KerasClassifier(build_fn=create_model,
                        epochs=EPOCHS, 
                        batch_size=BATCH_SIZE,
                        verbose=VERBOSE )

#
# Setup Scorers
#

from sklearn.metrics import make_scorer
from sklearn import metrics

#
# Custom Scorer Example: scoring = { 'custom': make_scorer(my_scorer) }
#

def my_scorer(y_true, y_pred):
    return -np.sum(np.abs(y_true-y_pred))

def f1_macro(y_true, y_pred):
    return metrics.f1_score( y_true, y_pred, average='macro' )

#metrics.f1_score( y_label_test[i], yhat, average='macro' )

#['accuracy', 'adjusted_mutual_info_score', 'adjusted_rand_score', 
# 'average_precision', 'completeness_score', 'explained_variance', 
# 'f1', 'f1_macro', 'f1_micro', 'f1_samples', 'f1_weighted', 
# 'fowlkes_mallows_score', 'homogeneity_score', 'mutual_info_score', 
# 'neg_log_loss', 'neg_mean_absolute_error', 'neg_mean_squared_error', 
# 'neg_mean_squared_log_error', 'neg_median_absolute_error', 
# 'normalized_mutual_info_score', 'precision', 'precision_macro', 
# 'precision_micro', 'precision_samples', 'precision_weighted', 'r2', 
# 'recall', 'recall_macro', 'recall_micro', 'recall_samples', 
# 'recall_weighted', 'roc_auc', 'v_measure_score']

scoring = {
           'adjusted_mutual_info_score':make_scorer(metrics.adjusted_mutual_info_score),
           'adjusted_rand_score': make_scorer(metrics.adjusted_rand_score),
           'Accuracy': make_scorer(metrics.accuracy_score),
           'f1_macro': make_scorer(f1_macro)
        }

#
# Setup Random Search Paramater Distributions to Sample
#

param_dist = {
    
    # const
    'embedding_dimensions' : [EMBEDDING_DIMENSIONS],
    'num_outputs' : [len(bunch.target_names)],
    
    # variable
    'rnn_type' : RAND_RNN_TYPE,
    'state_size' : RAND_STATE_SIZE, 
    'dropout' : RAND_DROPOUT,
    'recurrent_dropout' : RAND_RECURRENT_DROPOUT, 
    'activation' : RAND_ACTIVATION,
    'loss' : RAND_LOSS,
    'optimizer' : RAND_OPTIMIZER,
    'metrics' : RAND_METRICS
}


#
# Setup Parameters Grid
#

param_grid = {
    
    # const
    'embedding_dimensions' : [EMBEDDING_DIMENSIONS],
    'num_outputs' : [len(bunch.target_names)],
    
    # variable
    'rnn_type' : GRID_RNN_TYPE,
    'state_size' : GRID_STATE_SIZE, 
    'dropout' : GRID_DROPOUT, 
    'recurrent_dropout' : GRID_RECURRENT_DROPOUT,
    'activation' : GRID_ACTIVATION,
    'loss' : GRID_LOSS,
    'optimizer' : GRID_OPTIMIZER,
    'metrics' : GRID_METRICS
}

#
# Setup CV object -- Stratified Kfold
#


#KFold(n_splits=N_FOLDS, random_state=None, shuffle=False)
#N_FOLDS,
#n_iter=N_ITER,
#iid=IID,

#
# Grid Search / Random Search / Hyper Parameter Tuning and Cross Validation (Cont.)
#

#
#
#        Perform Random Search 
#               and store Cross Validation results
#               
#               
#

from time import time
from sklearn.model_selection import RandomizedSearchCV

random_search = RandomizedSearchCV(clf, 
                                   param_distributions=param_dist,
                                   n_iter=RAND_N_ITER, 
                                   cv=RAND_N_FOLDS,
                                   n_jobs=RAND_N_JOBS,
                                   iid=RAND_IID,
                                   verbose=10)

# run randomized search
start = time()
random_search.fit(X_train_embedded,y_train)

#
# display report
#

print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), RAND_N_ITER))
report_top_candidates( random_search.cv_results_ )


#
# Grid Search / Random Search / Hyper Parameter Tuning and Cross Validation (Cont.)
#

#
#
#        Perform Grid Search 
#               and store Cross Validation results
#               
#               
#

from time import time
from sklearn.model_selection import GridSearchCV

grid_search = GridSearchCV(clf,
                  param_grid=param_grid,
                  scoring=scoring,
                  cv=GRID_N_FOLDS,
                  iid=GRID_IID,
                  refit=REFIT, 
                  return_train_score=True,
                  verbose=10)

start = time()
grid_search.fit(X_train_embedded,y_train)
grid_time_elapsed = time() - start

#
# store results
#

cv_results = grid_search.cv_results_


#
# Display Quick Grid Search Report
#

print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (grid_time_elapsed, len(cv_results['params'])))

report_top_candidates(cv_results, selected_scorer=SELECTED_SCORER)

#
# Store Best Estimator
#

best_score, best_params = grid_search.best_score_, grid_search.best_params_
optimised_rnn = grid_search.best_estimator_

#
# Grid Search / Random Search /  Hyper Parameter Tuning and Cross Validation (Cont.)
#

#
# Display Results
#


# Dropout and RNN Type
PARAM_1 = ('rnn_type', 'RNN Type')
PARAM_2 = ('dropout', 'Dropout')

plot_grid_search(cv_results, SELECTED_SCORER, param_grid[PARAM_1[0]], param_grid[PARAM_2[0]], PARAM_1[1], PARAM_2[1])

#
# Grid Search / Random Search /  Hyper Parameter Tuning and Cross Validation (Cont.)
#

#
# Display Results
#

# Reccurent Dropout and RNN Type
PARAM_1 = ('rnn_type', 'RNN Type')
PARAM_2 = ('recurrent_dropout', 'Reccurent Dropout')

plot_grid_search(cv_results, SELECTED_SCORER, param_grid[PARAM_1[0]], param_grid[PARAM_2[0]], PARAM_1[1], PARAM_2[1])


# src:  src: scikitlearn

metric = 'mean_test_%s' %  SELECTED_SCORER 

plt.figure(figsize=(10, 10))
plt.title("Grid Search Comparing Multiple Scorer's",
          fontsize=17)

plt.xlabel(metric)
plt.ylabel("Score")

cmap = plt.get_cmap("tab10")
scoring_colors = [cmap(i) for i in range(len(scoring))]
ax = plt.gca()


X_axis = np.array( cv_results[metric].data, dtype=float )
for scorer, color in zip(sorted(scoring), scoring_colors):

    for sample, style in (('train', '--'), ('test', '-')):
        
        sample_score_mean = cv_results['mean_%s_%s' % (sample, scorer)]
        sample_score_std = cv_results['std_%s_%s' % (sample, scorer)]
        
        ax.fill_between(X_axis, sample_score_mean - sample_score_std,
                        sample_score_mean + sample_score_std,
                        alpha=0.9 if sample == 'test' else 0.1, color=color)
        
        ax.plot(X_axis, sample_score_mean, style, color=color,
                alpha=0.9 if sample == 'test' else 0.7,
                label="%s (%s)" % (scorer, sample))

    best_index = np.nonzero( cv_results['rank_test_%s' % scorer] == 1 )[0][0]
    best_score = cv_results['mean_test_%s' % scorer][best_index]
 
    # Plot a dotted vertical line at the best score for that scorer marked by x
    ax.plot([X_axis[best_index], ] * 2, [0, best_score],
            linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)

    # Annotate the best score for that scorer
    ax.annotate("%0.2f" % best_score,
                (X_axis[best_index], best_score + 0.005))

plt.legend(loc="best")
plt.show()

#
# Grid Search / Random Search /  Hyper Parameter Tuning and Cross Validation (Cont.)
#

#
# Summarize Results
#

print("Best: %f using %s" % (grid_search.best_score_, grid_search.best_params_))

mean_test_string = 'mean_test_%s' % SELECTED_SCORER
means = grid_search.cv_results_[mean_test_string]

std_test_string = 'std_test_%s' % SELECTED_SCORER
stds = grid_search.cv_results_[std_test_string]

params = grid_search.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

#
#
#  Build Two Layer RNN and attach Embedding Layer
#    - Parameters Chosen Based on Search Results
#

from keras.models import Input
from keras.models import Model

NUM_CLASSES = len(bunch.target_names)
tokernizer = glove_embedder.tokenizer
embedding_matrix = glove_embedder.get_embedding_matrix()
#glove_embedder.compute_output_shape()

#
# print best parameters
#

print(best_params)

#
# define two layer model
#

input_holder = Input(shape=(MAXIMUM_ARTICLE_LENGTH,))

x = Embedding(len(tokernizer.word_index) + 1,                      # input dimensions
                EMBEDDING_DIMENSIONS,                              # output dimensions
                input_length=MAXIMUM_ARTICLE_LENGTH)(input_holder) # number of words in each sequence

x = LSTM(best_params['state_size'], dropout=best_params['dropout'], recurrent_dropout=best_params['recurrent_dropout'],return_sequences=True)(x)
x = LSTM(best_params['state_size'], dropout=best_params['dropout'], recurrent_dropout=best_params['recurrent_dropout'])(x)

x = Dense(NUM_CLASSES, activation='softmax')(x)


two_layer_rnn_model = Model(inputs=input_holder, outputs=x)
two_layer_rnn_model.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])


#
# Train Two Layer Model
#

#
# train / test data
# 

X_train, X_test, y_train, y_test = train_test_split(X_enc, y_enc, test_size=TEST_SIZE,
                                                            stratify=bunch.target, 
                                                            random_state=RANDOM_SEED)

history = two_layer_rnn_model.fit(X_train, y_train_ohe, epochs=10, batch_size=250)  # starts training

# 
# Display Plot History
#

plot_history(history)
