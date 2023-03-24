import h5py
import numpy as np

from sklearn.metrics import average_precision_score as auprc

import os

# local imports
import iterutils

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


def TFdataset(path, batchsize, dataflag):

    TFdataset_batched = iterutils.train_TFRecord_dataset(path, batchsize, dataflag)

    return TFdataset_batched


class bichrom_seq(nn.Module):
    def __init__(self,params):
        super().__init__()
        self.params=params
        self.conv1d=nn.Conv1d(4, params.n_filters, params.filter_size)
        self.relu=nn.ReLU() 
        self.batchNorm1d=nn.BatchNorm1d(params.n_filters)
        self.maxPool1d=nn.MaxPool1d(params.pooling_size,params.pooling_stride, ceil_mode=True)
        self.lstm=nn.LSTM(params.n_filters, 32, batch_first=True)
        self.tanh=nn.Tanh()
        self.model_dense_repeat = nn.Sequential()
        self.model_dense_repeat.append(nn.Linear(params.lstm_out, params.dense_layer_size))
        self.model_dense_repeat.append(nn.ReLU())
        self.model_dense_repeat.append(nn.Dropout(0.5))
        for idx in range(params.dense_layers-1):
            self.model_dense_repeat.append(nn.Linear(params.dense_layer_size, params.dense_layer_size))
            self.model_dense_repeat.append(nn.ReLU())
            self.model_dense_repeat.append(nn.Dropout(0.5))        
        self.linear=nn.Linear(params.dense_layer_size, 1)
        self.sigmoid=nn.Sigmoid()
                          
    def forward(self,x):
        xs=self.conv1d(x)
        xs=self.relu(xs)
        xs=self.batchNorm1d(xs)
        xs=self.maxPool1d(xs)
        xs=torch.permute(xs, (0, 2, 1))
        xs=self.lstm(xs)[0][:, -1, :]
        xs=self.tanh(xs)
        xs=self.model_dense_repeat(xs)
        xs=self.linear(xs)
        xs=self.sigmoid(xs)
        return xs


def build_model(params, seq_length):
    return bichrom_seq(params)


def save_metrics(hist_object, pr_history, records_path):
    loss = hist_object['loss']
    val_loss = hist_object['val_loss']
    val_pr = pr_history["val_auprc"]

    # Saving the training metrics
    np.savetxt(records_path + 'trainingLoss.txt', loss, fmt='%1.4f')
    np.savetxt(records_path + 'valLoss.txt', val_loss, fmt='%1.4f')
    np.savetxt(records_path + 'valPRC.txt', val_pr, fmt='%1.4f')
    return loss, val_pr
    
    
# NOTE: ADDING A RECORDS PATH HERE!
def train(model, train_path, val_path, batch_size, records_path):
    
    # GPU
    device = torch.device("mps")
    
    # transfer model to GPU
    model.to(device)
    
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

    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    train_dataset = TFdataset(train_path, batch_size, "seqonly")
    val_dataset = TFdataset(val_path, batch_size, "seqonly")

    
    def train_one_epoch(epoch_index):
        running_loss = 0.
        batch_avg_vloss = 0.

        print(f"Total Parameters = {sum(p.numel() for p in model.parameters())}")
        print(f"Total Trainable Parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
            
        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for i, data in enumerate(train_dataset):
            # Every data instance is an input + label pair
            seq,chrom,target,labels = data
            
            # transfer data to GPU
            seq, labels = seq.to(device), labels.to(device)
            
            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            outputs = model(seq)
            labels=labels.to(torch.float32)
            # Compute the loss and its gradients
            loss = loss_fn(outputs, labels)
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            batch_avg_vloss = running_loss / (i + 1) # loss per batch
            print('  batch {} loss: {}'.format(i + 1, batch_avg_vloss))

        return batch_avg_vloss
    
    epoch = 0

    EPOCHS = 2

    best_vloss = 1_000_000.

    hist={"loss":[],"val_loss":[]}
    precision_recall_history={"val_auprc":[]}
    
    for epoch in range(EPOCHS):
        
        print('EPOCH {}:'.format(epoch + 1))

        print(model)
        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(epoch)

        # We don't need gradients on to do reporting
        model.train(False)

        running_vloss = 0.0#torch.tensor(0.0)
        avg_vloss=0.0#torch.tensor(0.0)
        val_predictions=[]
        val_labels=[]
        for i, vdata in enumerate(val_dataset):
            vseq,vchrom,vtarget,vlabels = vdata
            
            # transfer datat GPU
            vseq, vlabels = vseq.to(device), vlabels.to(device)
            
            voutputs = model(vseq)
            vlabels=vlabels.to(torch.float32)
            
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += float(vloss)

            avg_vloss = running_vloss / (i + 1)
            print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
            val_predictions.append(voutputs.detach().numpy())
            val_labels.append(vlabels)
            
        torch.save(model.state_dict(), records_path+'model_epoch{}.hdf5'.format(epoch+1))    
        hist["loss"].append(avg_loss)
        hist["val_loss"].append(avg_vloss)
        predictions=np.concatenate(val_predictions)
        labels=np.concatenate(val_labels)
        aupr = auprc(labels, predictions)
        precision_recall_history["val_auprc"].append(aupr)
        
        epoch += 1
    
    loss, val_pr = save_metrics(hist, precision_recall_history, records_path=records_path)
    
    return loss, val_pr


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
