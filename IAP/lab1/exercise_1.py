# EJERCICIO 1

import argparse
import pandas

import keras
from keras.layers import Dense, Dropout

from keras.models import Sequential
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score ,roc_auc_score, f1_score, confusion_matrix
import datetime

def read_args():
    parser = argparse.ArgumentParser(description='Exercise 1')
    # Parámetros de clasificación
    parser.add_argument('--num_units', nargs='+', default=[100], type=int,
                        help='Number of hidden units of each hidden layer.')
    parser.add_argument('--dropout', nargs='+', default=[0.5], type=float,
                        help='Dropout ratio for every layer.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Number of instances in each batch.')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs')

    # --- NOTAS
    # Cambiar acá el default los parámetros utilizados para entrenar

    parser.add_argument('--experiment_name', type=str,
            #default=str('NumUnits-',args.num_units,'_Dropout-',args.dropout,'_BatchSize-',args.batch_size,'_Epochs-',args.epochs),
            default=datetime.datetime.now().strftime('%Y-%m-%d_%H%M'),
                        help='Name of the experiment, used in the filename'
                             'where the results are stored.')
    args = parser.parse_args()

    assert len(args.num_units) == len(args.dropout)
    return args


def load_dataset():
    dataset = load_files('dataset/txt_sentoken', shuffle=False)

    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data, dataset.target, test_size=0.25, random_state=42)

    print('Training samples {}, test_samples {}'.format(
        len(X_train), len(X_test)))



    # TODO 1: Apply the Tfidf vectorizer to create input matrix
    # ....
    vect =  TfidfVectorizer(stop_words='english')
    vect.fit(X_train)
    X_train =  vect.transform(X_train)
    X_test = vect.transform(X_test)
    return X_train, X_test, y_train, y_test


def main():
    args = read_args()
    X_train, X_test, y_train, y_test_original = load_dataset()
    # TODO 2: Convert the labels to categorical
    # ...
    input_size = X_train.shape[1]
    y_train =  keras.utils.to_categorical(y_train,2)
    y_test =  keras.utils.to_categorical(y_test_original,2)
    # TODO 3: Build the Keras model
    model = Sequential()
    # Add all the layers
    model.add(Dense(args.num_units[0],
                input_shape=(input_size,),activation='relu'))
    model.add(Dropout(args.dropout[0]))
    for i in range(1,len(args.num_units)):
        model.add(Dense(args.num_units[i],activation='relu'))
        model.add(Dropout(args.dropout[i]))
    model.add(Dense(2,activation='softmax'))

    # model.compile(...)
    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adagrad(lr=0.001, decay=0.0001),
                  metrics=['accuracy'])

    # TODO 4: Fit the model
    # hitory = model.fit(batch_size=??, ...)i
    model.fit(X_train, y_train,
              batch_size=args.batch_size,epochs=args.epochs,
              validation_split=0.1,
              verbose=1)

    # TODO 5: Evaluate the model, calculating the metrics.
    # Option 1: Use the model.evaluate() method. For this, the model must be
    # already compiled with the metrics.
    # performance = model.evaluate(X_test, y_test)

    # Option 2: Use the model.predict() method and calculate the metrics using
    # sklearn. We recommend this, because you can store the predictions if
    # you need more analysis later. Also, if you calculate the metrics on a
    # notebook, then you can compare multiple classifiers.
    # predictions = ...
    # performance = ...
    predictions =  model.predict_classes(X_test)
    performances = {'accuracy' : accuracy_score(y_test_original,predictions),
                   'auc' : roc_auc_score(y_test_original,
                           model.predict(X_test)[:,1]),
                   'f1' : f1_score(y_test_original,predictions),
                   'matriz' : confusion_matrix(y_test_original,predictions)}
    # TODO 6: Save the results.
    # ...
    print(X_train.shape)
    print(performances)
    # One way to store the predictions:
    results = pandas.DataFrame(y_test_original, columns=['true_label'])
    results.loc[:, 'predicted'] = predictions
    results.to_csv('prediction_of_{}.csv'.format(args.experiment_name),
                   index=False)

    results_metrics = pandas.read_csv('results.csv').set_index('experiment_name')
    results_metrics.loc[args.experiment_name] = [args.num_units, args.dropout,
    args.batch_size, args.epochs, performances['accuracy'],
    performances['auc'], performances['f1']]
    results_metrics.to_csv('results.csv')

    print(results_metrics)

if __name__ == '__main__':
    main()