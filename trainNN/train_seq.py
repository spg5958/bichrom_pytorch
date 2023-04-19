import h5py
import numpy as np

from sklearn.metrics import average_precision_score

import os

# local imports
import iterutils

import torch
from torch import nn
import torch.nn.functional as F
from datetime import datetime


def TFdataset(path, batchsize, dataflag, seed):
    
    TFdataset_batched = iterutils.train_TFRecord_dataset(path, batchsize, dataflag, seed=seed)

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
        self.model_dense_repeat.append(nn.Linear(32, params.dense_layer_size))
        self.model_dense_repeat.append(nn.ReLU())
        self.model_dense_repeat.append(nn.Dropout(params.dropout))
        for idx in range(params.dense_layers-1):
            self.model_dense_repeat.append(nn.Linear(params.dense_layer_size, params.dense_layer_size))
            self.model_dense_repeat.append(nn.ReLU())
            self.model_dense_repeat.append(nn.Dropout(params.dropout))        
        self.linear=nn.Linear(params.dense_layer_size, 1)
        self.sigmoid=nn.Sigmoid()
                   
        # initialization
        torch.nn.init.xavier_uniform_(self.conv1d.weight)
        torch.nn.init.constant_(self.conv1d.bias,0)
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            if 'weigth_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param,0)
        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias_ih" in n,  names):
                bias = getattr(self.lstm, name)
                n = bias.size(0)
                start, end = n//4, n//2
                bias.data[start:end].fill_(1.)
        for layer in self.model_dense_repeat:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.constant_(layer.bias,0)
        torch.nn.init.xavier_uniform_(self.linear.weight)
        torch.nn.init.constant_(self.linear.bias,0)
             
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
    
    
def train(model, train_path, val_path, batch_size, records_path, epochs, seed):
   
    # GPU
    device = iterutils.getDevice()
    
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
    
    model.train(False)
    w0=model.model_dense_repeat[0].weight.clone().detach().cpu().numpy()
    
    train_dataset = TFdataset(train_path, batch_size, "seqonly", seed)
    val_dataset = TFdataset(val_path, batch_size, "seqonly", seed)
    
    def train_one_epoch(epoch_index):
        running_loss = 0.
        batch_avg_vloss = 0.

        for i, data in enumerate(train_dataset):
            seq,chrom,target,labels = data
            
            # transfer data to GPU
            seq, labels = seq.to(device), labels.to(device)
            
            optimizer.zero_grad()

            outputs = model(seq)
            labels=labels.to(torch.float32)
            loss = loss_fn(outputs, labels)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            batch_avg_vloss = running_loss / (i + 1) # loss per batch
            print('SEQ - EPOCH {}: batch {} loss: {}\r'.format(epoch_index+1, i + 1, batch_avg_vloss), end="\r")

        return batch_avg_vloss
 
    print(f"Epochs = {epochs}")
    EPOCHS = epochs

    best_vloss = 1_000_000.

    hist={"loss":[],"val_loss":[]}
    precision_recall_history={"val_auprc":[]}
    
    for epoch in range(EPOCHS):
        
        print('\nEPOCH {}:'.format(epoch + 1))

        model.train(True)
        avg_loss = train_one_epoch(epoch)

        model.train(False)

        wi=model.model_dense_repeat[0].weight.clone().detach().cpu().numpy()
        dw=wi-w0
        print()
        print(np.linalg.norm(dw))
        
        running_vloss = 0.0
        avg_vloss=0.0
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
            print('SEQ - EPOCH {}: LOSS train {} valid {}'.format(epoch + 1, avg_loss, avg_vloss),end="\r")
            val_predictions.append(voutputs.cpu().detach().numpy())
            val_labels.append(vlabels.cpu().detach().numpy())
            
        torch.save(model.state_dict(), records_path+'model_epoch{}.torch'.format(epoch+1))    
        hist["loss"].append(avg_loss)
        hist["val_loss"].append(avg_vloss)
        predictions=np.concatenate(val_predictions)
        labels=np.concatenate(val_labels)
        
        aupr = average_precision_score(labels, predictions)

        precision_recall_history["val_auprc"].append(aupr)
        
        epoch += 1
    
    loss, val_pr = save_metrics(hist, precision_recall_history, records_path=records_path)
    
    return loss, val_pr


def build_and_train_net(hyperparams, train_path, val_path, batch_size,
                        records_path, seq_len, epochs, seed):

    iterutils.setRandomSeed(seed)

    model = build_model(params=hyperparams, seq_length=seq_len)

    loss, val_pr = train(model, train_path=train_path, val_path=val_path,
                        batch_size=batch_size, records_path=records_path, epochs=epochs, seed=seed)

    return loss, val_pr

if __name__ == '__main__':
    records_path="train_out/seqnet/"
    from train import Params
    model=build_model(params=Params(), seq_length=500)
    torch.save(model.state_dict(), records_path+'model_epoch1.torch')
