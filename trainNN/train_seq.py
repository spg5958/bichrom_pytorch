import h5py
import numpy as np

from sklearn.metrics import average_precision_score as auprc
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Dense, Dropout, Input, LSTM
# from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization
# from tensorflow.keras.callbacks import Callback
# from tensorflow.keras.callbacks import ModelCheckpoint
# from tensorflow.keras.optimizers import Adam

import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# import tensorflow as tf

# local imports
import iterutils

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

def TFdataset(path, batchsize, dataflag):

    TFdataset_batched = iterutils.train_TFRecord_dataset(path, batchsize, dataflag)

#     print(next(iter(TFdataset_batched)))
    return TFdataset_batched

# class PrecisionRecall(Callback):

#     def __init__(self, val_data):
#         super().__init__()
#         self.validation_data = val_data
#         self.labels = np.concatenate([i for i in self.validation_data.map(lambda x,y: y, num_parallel_calls=tf.data.AUTOTUNE)])

#     def on_train_begin(self, logs=None):
#         self.val_auprc = []
#         self.train_auprc = []

#     def on_epoch_end(self, epoch, logs=None):
#         """ monitor PR """
#         predictions = np.concatenate([i for i in self.validation_data.map(lambda x,y: self.model(x, training=False),
#                                                                          num_parallel_calls=tf.data.AUTOTUNE)])
#         aupr = auprc(self.labels, predictions)
#         self.val_auprc.append(aupr)

class bichrom_seq(nn.Module):
    def __init__(self,params):
        super().__init__()
        self.params=params
        self.conv1d=nn.Conv1d(4, params.n_filters, params.filter_size)
        self.relu=nn.ReLU()
        self.batchNorm1d=nn.BatchNorm1d(params.n_filters)
        self.maxPool1d=nn.MaxPool1d(params.pooling_size,params.pooling_stride)
        self.lstm=nn.LSTM(params.n_filters, params.lstm_out, batch_first=True)
        self.model_dense_repeat = nn.Sequential()
        self.model_dense_repeat.append(nn.Linear(params.lstm_out, params.dense_layer_size))
        for idx in range(params.dense_layers):
            self.model_dense_repeat.append(nn.Linear(params.dense_layer_size, params.dense_layer_size))
            self.model_dense_repeat.append(nn.ReLU())
            self.model_dense_repeat.append(nn.Dropout(0.5))
        self.linear=nn.Linear(params.dense_layer_size, 1)
        self.sigmoid=nn.Sigmoid()

    def forward(self,x):
        xs=self.conv1d(x)
        xs=self.relu(xs)
        xs=self.batchNorm1d(xs)
#         print(xs.shape)
        lin=xs.shape[-1]
        pad=((lin-1)*self.params.pooling_stride+1+(self.params.pooling_size-1)-lin)//2
        xs=F.pad(xs, (pad, pad), mode='constant', value=0)
        xs=self.maxPool1d(xs)
#         print(xs.shape)
        xs=torch.permute(xs, (0, 2, 1))
        xs=self.lstm(xs)[0][:, -1, :]
        xs=self.model_dense_repeat(xs)
        xs=self.linear(xs)
        xs=self.sigmoid(xs)
        return xs

def build_model(params, seq_length):
    return bichrom_seq(params)

def save_metrics(hist_object, pr_history, records_path):
    loss = hist_object.history['loss']
    val_loss = hist_object.history['val_loss']
    val_pr = pr_history.val_auprc

    # Saving the training metrics
    np.savetxt(records_path + 'trainingLoss.txt', loss, fmt='%1.4f')
    np.savetxt(records_path + 'valLoss.txt', val_loss, fmt='%1.4f')
    np.savetxt(records_path + 'valPRC.txt', val_pr, fmt='%1.4f')
    return loss, val_pr


# NOTE: ADDING A RECORDS PATH HERE!
def train(model, train_path, val_path, batch_size, records_path):
    """
    Train the Keras graph model
    Parameters:
        model (keras Model): The Model defined in build_model
        train_path (str): Path to training data
        val_path (str): Path to validation data
        steps_per_epoch (int): Len(training_data)/batch_size
        batch_size (int): Size of mini-batches used during training
        records_path (str): Path + prefix to output directory
    Returns:
        loss (ndarray): An array with the validation loss at each epoch
    """
    
#     adam = Adam(learning_rate=0.001)
#     model.compile(loss='binary_crossentropy', optimizer=adam)

    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    train_dataset = TFdataset(train_path, batch_size, "seqonly")
    val_dataset = TFdataset(val_path, batch_size, "seqonly")

#     precision_recall_history = PrecisionRecall(val_dataset)
    # adding check-pointing
#     checkpointer = ModelCheckpoint(records_path + 'model_epoch{epoch}.hdf5',
#                                    verbose=1, save_best_only=False)
    # defining parameters for early stopping
    # earlystop = EarlyStopping(monitor='val_loss', mode='min', verbose=1,
    #                           patience=5)
    # training the model..
#     hist = model.fit(train_dataset, epochs=15,
#                         validation_data = val_dataset,
#                         callbacks=[precision_recall_history,
#                                     checkpointer])

#     loss, val_pr = save_metrics(hist, precision_recall_history,
#                                 records_path=records_path)

    
    def train_one_epoch(epoch_index):
        running_loss = 0.
        last_loss = 0.

        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for i, data in enumerate(train_dataset):
            # Every data instance is an input + label pair
            seq,chrom,target,labels = data
            
            print(seq.shape,chrom.shape)
            print(f"Total Parameters = {sum(p.numel() for p in model.parameters())}")
            print(f"Total Trainable Parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
            
            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            outputs = model(seq)
            labels=labels.to(torch.float32)
            # Compute the loss and its gradients
#             print(outputs.dtype, labels.dtype)
            loss = loss_fn(outputs, labels)
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            if i % 1000 == 999:
                last_loss = running_loss / 1000 # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = epoch_index * len(training_loader) + i + 1
#                 tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.

        return last_loss
    
    # Initializing in a separate cell so we can easily add more epochs to the same run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#     writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
    epoch_number = 0

    EPOCHS = 2

    best_vloss = 1_000_000.

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        print(model)
        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(epoch_number)

        # We don't need gradients on to do reporting
        model.train(False)

        running_vloss = 0.0#torch.tensor(0.0)
        avg_vloss=0.0#torch.tensor(0.0)
        for i, vdata in enumerate(val_dataset):
            vseq,vchrom,vtarget,vlabels = vdata
            voutputs = model(vseq)
            vlabels=vlabels.to(torch.float32)
            
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += float(vloss)

            avg_vloss = running_vloss / (i + 1)
            print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
#         writer.add_scalars('Training vs. Validation Loss',
#                         { 'Training' : avg_loss, 'Validation' : avg_vloss },
#                         epoch_number + 1)
#         writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = records_path+'model_epoch{}.hdf5'.format(epoch_number+1)
            torch.save(model.state_dict(), model_path)

        epoch_number += 1
    
    return avg_vloss, running_vloss


def build_and_train_net(hyperparams, train_path, val_path, batch_size,
                        records_path, seq_len):

    model = build_model(params=hyperparams, seq_length=seq_len)

    loss, val_pr = train(model, train_path=train_path, val_path=val_path,
                        batch_size=batch_size, records_path=records_path)

    return loss, val_pr

if __name__ == '__main__':
    records_path="train_out/seqnet/"
    from train import Params
    model=build_model(params=Params(), seq_length=500)
    for l,p in zip(model.state_dict().keys(), model.parameters()):
        print(l,p.requires_grad)
    torch.save(model.state_dict(), records_path+'model_epoch1.hdf5')
