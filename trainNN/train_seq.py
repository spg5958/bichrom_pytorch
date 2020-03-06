from __future__ import division
import numpy as np

from sklearn.metrics import average_precision_score as auprc
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv1D, MaxPooling1D, BatchNormalization
from keras.callbacks import EarlyStopping
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint

# local imports
import iterutils as iu


def data_generator(filename, batchsize, seqlen):
    X = iu.train_generator(filename + ".seq", batchsize, seqlen, 'seq', 'repeat')
    y = iu.train_generator(filename + ".labels", batchsize, seqlen, 'labels', 'repeat')
    while True:
        yield X.next(), y.next()


class PrecisionRecall(Callback):
    def on_train_begin(self, logs=None):
        self.val_auprc = []
        self.train_auprc = []

    def on_epoch_end(self, epoch, logs=None):
        """ monitor PR """
        x_val, y_val = self.validation_data[0], self.validation_data[1]
        predictions = self.model.predict(x_val)
        aupr = auprc(y_val, predictions)
        self.val_auprc.append(aupr)


def build_model(hyperparameters, seq_length):
    """ Define a Keras graph model with sequence as input """
    batch_size, n_layers, n_filters, filter_size, (pooling_size, pooling_strides), dropout, layer_size = hyperparameters
    # Defining the model
    seq_input = Input(shape=(seq_length, 4,), name='seq')
    xs = Conv1D(filter_size, n_filters, activation='relu')(seq_input)
    xs = BatchNormalization()(xs)
    xs = MaxPooling1D(padding="same", strides=pooling_strides, pool_size=pooling_size)(xs)
    xs = Flatten()(xs)
    # adding a specified number of dense layers
    for idx in range(n_layers):
        xs = Dense(layer_size, activation='relu')(xs)
        xs = Dropout(dropout)(xs)
    # sigmoid output
    result = Dense(1, activation='sigmoid')(xs)
    model = Model(inputs=seq_input, outputs=result)
    return model


def save_metrics(hist_object, pr_history, records_path):
    loss = hist_object.history['loss']
    val_loss = hist_object.history['val_loss']
    val_pr = pr_history.val_auprc

    # Saving the training metrics
    np.savetxt(records_path + '.trainingLoss.txt', loss, fmt='%1.2f')
    np.savetxt(records_path + '.valLoss.txt', val_loss, fmt='%1.2f')
    np.savetxt(records_path + '.valPRC.txt', val_pr, fmt='%1.2f')
    return loss, val_loss, val_pr


# NOTE: ADDING A RECORDS PATH HERE!
def train(model, train_path, val_path, steps_per_epoch, batch_size, records_path):
    """
    Train the NN using Adam
    inputs: NN architecture
    outputs: trained model
    """
    model.compile(loss='binary_crossentropy', optimizer='adam')
    # Calculate the number of steps per epoch:
    train_generator = data_generator(train_path, batch_size, seqlen=500)
    val_generator = data_generator(val_path, 500000, seqlen=500)

    validation_data = val_generator.next()
    precision_recall_history = PrecisionRecall()
    # adding check-pointing
    checkpointer = ModelCheckpoint(records_path + '.{epoch:02d}.hdf5', verbose=1, save_best_only=False)
    # Parameters for early stopping
    earlystop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
    # Training the model..
    hist = model.fit_generator(epochs=10, steps_per_epoch=steps_per_epoch, generator=train_generator,
                               validation_data=validation_data,
                               callbacks=[precision_recall_history, checkpointer, earlystop])
    save_metrics(hist, precision_recall_history, records_path=records_path)
    return model


def build_and_train_net(hyperparams, train_path, val_path, batch_size, records_path):

    # Calculate size of training set
    training_set_size = len(np.loadtxt(train_path + '.labels'))
    # Calculate the steps per epoch
    steps = training_set_size/batch_size
    model = build_model(hyperparameters=hyperparams, seq_length=500)

    # This saves all iterations of the model. But I should do an early stopping really.
    train(model, train_path=train_path, val_path=val_path, steps_per_epoch=steps,
          batch_size=batch_size, records_path=records_path)