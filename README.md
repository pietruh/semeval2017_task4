# semeval2017_task4
My approach to solve task 4 : "Sentiment Analysis in Twitter" from semeval2017 workshop on Semantic Evaluation. 

## What is in this repo
This repo contains all code needed for training and evaluation of the Semeval2017 task 4A.

### Short description of files 
train_model.py - Training a model using keras/tensorflow. 
test_model.py - Testing model on a specified test set, evaluation/prediction.
models.py - Model definition. Currently only one model is described. 2-layer biLSTM with fully connected layer, featuring a lot of dropout layers, batch normalization, and additionally gaussian noise on the inputs. 
config.py - Definition of a configuration dictionary that is imported almost in every other file in this repo. This provides information and control over algorithms.
data_generator.py - Definition of a data generator class, that uses word synonimization during calling model.fit(). It dynamically creates augmented examples on a batch that is used for training.
data_generator_offline - This script generates new data using techniques of data augmentation 'before' fitting the model.
text_processor.py - Based on twitter data, this preprocessor can handle complicated tweets, and turn them into useful sequences.
logger_to_file.py - Logger that saves std output into a .dat file that can be read in a notepad.
nlp_utilities.py - Utilities functions for setting up environment, data processing pipeline, saving results to file and losses
definition
LICENCE - Licence for using this code.

## Features = What is really interesting
#### Data augmentation with DataGenerator
Custom written data generator, that implements some simple word operations that I found useful in the training process. It can change words in the twitter message into synonyms and hypernyms(words that describe more general concept. e.g. printer->machine). Generation of an additional tweets can help inject more human-level understanding of the real life into the messages. However this still needs a lot of polishing. it was done in two manner, either online, during the fitting, or before, as a standalone tweet generator.

#### TextProcessor
Based on ekphrasis, it is very interesting library for parsing tweets.

#### Model
2 layers bidirectional LSTM model that has embedding layer, barch normalization, gaussian noise, dropout before setting recurrent layers with biLSTM, with fully connected dense layer with softmax activation fucntion on the output. It was set to optimize metrics that I called 'macro_averaged_recall_tf_soft' that calculates recall using probabilities instead of binary encoding. The recall function is therefore differentiable as well as the whole model so it can be used by gradient based optimizers. The real task metrics function was implemented as 'macro_averaged_recall_tf_onehot' and has argmax(nondifferentiable) operation. 

#### Score
The score on macro averaged recall reached 0.6614 after perfoming argmax operation to one-hot encode model output. The accuracy of the classification reached 0.6188

## What is missing
datasets - due to github's limitation they are not available here. test and train data can be easily found on the semeval website (). Twitter glove embedding can be found here https://mega.nz/#!u4hFAJpK!UeZ5ERYod-SwrekW-qsPSsl-GYwLFQkh06lPTR7K93I, tokenizer will be uploaded soon, as well as pretrained model.h5.
fully featured installer - installer that will install all prerequisites and download all needed datasets automagically.
I put a lot of effort into keeping some documentation standard, but there may be something missing or commented not quite clear. Please do not hesitate to contact me if you spot something like this.

## How to use that
Modify config.py wth your settings and either train or test model. Data generator in the offline mode can be used as a standalone software.
