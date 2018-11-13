"""Exercise 2"""

import argparse
import keras.backend as K
# K is just another name for the keras backend: tensorflow (or theaso,
# if you are using a different backend).
import pandas
import pickle
import datetime

import keras
from keras.layers import Embedding, Average, Lambda, Dense, Dropout
from keras.models import Sequential
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from utils import FilteredFastText
from sklearn.metrics import accuracy_score ,roc_auc_score, f1_score, confusion_matrix



def read_args():
    parser = argparse.ArgumentParser(description='Exercise 2')
    # Here you have some examples of classifier parameters. You can add
    # more arguments or change these if you need to.
    parser.add_argument('--num_units', nargs='+', default=[100], type=int,
                        help='Number of hidden units of each hidden layer.')
    parser.add_argument('--dropout', nargs='+', default=[0.5], type=float,
                        help='Dropout ratio for every layer.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Number of instances in each batch.')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs')
    parser.add_argument('--experiment_name', type=str,
                        #default=str('NumUnits-',args.num_units,'_Dropout-',args.dropout,'_BatchSize-',args.batch_size,'_Epochs-',args.epochs),
                        default=datetime.datetime.now().strftime('%Y-%m-%d_%H%M'),
                        help='Name of the experiment, used in the filename'
                             'where the results are stored.')
    parser.add_argument('--embeddings_filename', type=str,
                        default='word_vectors',
                        help='Name of the file with the embeddings.')
    args = parser.parse_args()

    assert len(args.num_units) == len(args.dropout)
    return args


def load_dataset():
    dataset = load_files('dataset/txt_sentoken', shuffle=False)

    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data, dataset.target, test_size=0.25, random_state=42)

    print('Training samples {}, test_samples {}'.format(
        len(X_train), len(X_test)))

    return X_train, X_test, y_train, y_test


def transform_input(instances, mapping):
    """Replaces the words in instances with their index in mapping.

    Args:
        instances: a list of text instances.
        mapping: an dictionary from words to indices.

    Returns:
        A matrix with shape (n_instances, max_text_length)."""
    word_indices = []
    for instance in instances:
        word_indices.append([mapping[word.decode('utf-8')]
                             for word in instance.split()])
    # Check consistency
    assert len(instances[0].split()) == len(word_indices[0])

    # Pad the sequences to obtain a matrix instead of a list of lists.
    from keras.preprocessing.sequence import pad_sequences
    return pad_sequences(word_indices)


def main():
    args = read_args()
    X_train, X_test, y_train, y_test_original = load_dataset()

    # TODO 1: Convert the labels to categorical
    # ...
    y_train =  keras.utils.to_categorical(y_train,2)
    y_test =  keras.utils.to_categorical(y_test_original,2)

    # Load the filtered FastText word vectors, using only the vocabulary in
    # the movie reviews dataset
    with open(args.embeddings_filename, 'rb') as model_file:
        filtered_fasttext = pickle.load(model_file)
    
    # The next thing to do is to choose how we are going to represent our
    # training matrix. Each review must be translated into a single vector.
    # This means we have to combine, somehow, the word vectors of each
    # word in the review. Some options are:
    #  - Take the average of all vectors.
    #  - Take the minimum and maximum value of each feature.
    # All these operations are vectorial and easier to compute using a GPU.
    # Then, it is better to put them inside the Keras model.

    # The Embedding layer will be quite handy in solving this problem for us.
    # To use this layer, the input to the network has to be the indices of the
    # words on the embedding matrix.
    X_train = transform_input(X_train, filtered_fasttext.word2index)

    word_indices = []
    for instance in X_train:
        word_indices.append([word for word in instance])
    # Check consistency
    assert len(X_train[0]) == len(word_indices[0])

    # The input is ready, start the model
    model = Sequential()
    model.add(Embedding(
        filtered_fasttext.wv.shape[0],  # Vocabulary size
        filtered_fasttext.wv.shape[1],  # Embedding size
        weights=[filtered_fasttext.wv],  # Word vectors
        trainable=False  # This indicates the word vectors must not be changed
                         # during training.
    ))
    # The output here has shape
    #     (batch_size (?), words_in_reviews (?), embedding_size)
    # To use a Dense layer, the input must have only 2 dimensions. We need to
    # create a single representation for each document, combining the word
    # embeddings of the words in the intance.
    # For this, we have to use a Tensorflow (K) operation directly.
    # The operation we need to do is to take the average of the embeddings
    # on the second dimension. We wrap this operation on a Lambda
    # layer to include it into the model.
    lamda = 'max'
    # model.add(Lambda(lambda xin: K.mean(xin, axis=1), name='embedding_average'))
    # model.add(Lambda(lambda xin: K.min(xin, axis=1), name='embedding_min'))
    model.add(Lambda(lambda xin: K.max(xin, axis=1), name='embedding_max'))
    # Now the output shape is (batch_size (?), embedding_size)

    # TODO 2: Finish the Keras model
    # Add all the layers
    # ...
    model.add(Dense(args.num_units[0],
                input_shape=(filtered_fasttext.wv.shape[1],),
                activation='relu'))
    model.add(Dropout(args.dropout[0]))
    for i in range(1,len(args.num_units)):
        model.add(Dense(args.num_units[i],activation='relu'))
        model.add(Dropout(args.dropout[i]))
    model.add(Dense(2,activation='softmax'))

    # model.compile(...)
    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adagrad(lr=0.001, decay=0.0001),
                  metrics=['accuracy'])


    # TODO 3: Fit the model
    model.fit(X_train, y_train,
            batch_size=args.batch_size,epochs=args.epochs,
            validation_split=0.2,
            verbose=1)

    # TODO 4: Evaluate the model, calculating the metrics.
    # Option 1: Use the model.evaluate() method. For this, the model must be
    # already compiled with the metrics.
    # performance = model.evaluate(transform_input(X_test), y_test)

    # Option 2: Use the model.predict() method and calculate the metrics using
    # sklearn. We recommend this, because you can store the predictions if
    # you need more analysis later. Also, if you calculate the metrics on a
    # notebook, then you can compare multiple classifiers.
    # predictions = ...
    # performance = ...

    X_test = transform_input(X_test, filtered_fasttext.word2index)
    predictions =  model.predict_classes(X_test)
    performances = {'accuracy' : accuracy_score(y_test_original,predictions),
                   'auc' : roc_auc_score(y_test_original,
                           model.predict(X_test)[:,1]),
                   'f1' : f1_score(y_test_original,predictions),
                   'matriz' : confusion_matrix(y_test_original,predictions)}

    # TODO 5: Save the results.
    # ...
    print(X_train.shape)
    print(performances)
    # One way to store the predictions:
    results = pandas.DataFrame(y_test_original, columns=['true_label'])
    results.loc[:, 'predicted'] = predictions
    results.to_csv('prediction_of_{}.csv'.format(args.experiment_name),
                   index=False)

    results_metrics = pandas.read_csv('results2.csv').set_index('experiment_name')
    results_metrics.loc[args.experiment_name] = [lamda, args.num_units, args.dropout,
    args.batch_size, args.epochs, performances['accuracy'],
    performances['auc'], performances['f1']]
    results_metrics.to_csv('results2.csv')

    print(results_metrics)



if __name__ == '__main__':
    main()
