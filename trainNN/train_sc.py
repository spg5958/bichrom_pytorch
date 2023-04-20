import h5py
import numpy as np
import pandas as pd

from sklearn.metrics import average_precision_score

import iterutils

import torch
from torch import nn
from train_seq import bichrom_seq,build_model
from datetime import datetime

from functools import partial


def transforms(x,bin_size):
    return x.reshape((x.shape[0],bin_size,-1)).mean(axis=1).flatten()
    
    
def TFdataset(path, batchsize, dataflag, bin_size, seed):
        
    transform_frozen = partial(transforms, bin_size = bin_size)
    
    TFdataset_batched = iterutils.train_TFRecord_dataset(path, batchsize, dataflag, transforms={"chrom": transform_frozen}, seed=seed)
    
    return TFdataset_batched


class _reshape(nn.Module):
    def forward(self, x, shape):
        return torch.reshape(x,(x.shape[0],)+shape)
                
        
class bichrom_chrom(nn.Module):
    def __init__(self, no_of_chromatin_tracks, seq_len, bin_size):
        super().__init__()
        self.no_of_chromatin_tracks = no_of_chromatin_tracks
        self.seq_len = seq_len
        self.bin_size = bin_size
        self._reshape=_reshape()
        self.conv1d=nn.Conv1d(no_of_chromatin_tracks, 15, 1, padding="valid")
        self.relu=nn.ReLU()
        self.lstm=nn.LSTM(15, 5, batch_first=True)
        self.relu1=nn.ReLU()
        self.linear=nn.Linear(5, 1)
        self.tanh=nn.Tanh()
        
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
        torch.nn.init.xavier_uniform_(self.linear.weight)
        torch.nn.init.constant_(self.linear.bias,0)
        
    def forward(self,x):
        xc=self._reshape(x, (self.no_of_chromatin_tracks, int(self.seq_len/self.bin_size)))
        xc=self.conv1d(xc)
        xc=self.relu(xc)
        xc=torch.permute(xc, (0, 2, 1))
        xc=self.lstm(xc)[0][:, -1, :]
        xc=self.relu1(xc)
        xc=self.linear(xc)
        xc=self.tanh(xc)
        return xc

    
class bimodal_network(nn.Module):
    def __init__(self,base_model, no_of_chromatin_tracks,params, seq_len, bin_size):
        super().__init__()
        self.params=params
        self.base_model=base_model
        self.activation = {}
        def get_activation(name):
            def hook(seq_model, _input, output):
                self.activation[name] = output.detach()
            return hook
        self.base_model.model_dense_repeat[7].register_forward_hook(get_activation('dense_2'))
        self.linear=nn.Linear(self.params.dense_layer_size, 1)
        self.tanh=nn.Tanh()
        self.model=bichrom_chrom(no_of_chromatin_tracks,seq_len,bin_size)
        self.linear1=nn.Linear(2, 1)
        self.sigmoid=nn.Sigmoid()
        
        # initialization
        torch.nn.init.xavier_uniform_(self.linear.weight)
        torch.nn.init.constant_(self.linear.bias,0)
        torch.nn.init.xavier_uniform_(self.linear1.weight)
        torch.nn.init.constant_(self.linear1.bias,0)
               
    def forward(self,seq_input,chromatin_input):
        self.base_model(seq_input)
        curr_tensor=self.activation['dense_2']
        xs=self.linear(curr_tensor)
        xs=self.tanh(xs)
        xc=self.model(chromatin_input)
        xsc=torch.cat((xs, xc), dim=1)
        xsc=self.linear1(xsc)
        result=self.sigmoid(xsc)
        return result

    
def add_new_layers(base_model_path, seq_len, no_of_chromatin_tracks, bin_size, params):
    base_model = build_model(params, seq_len)
    base_model.load_state_dict(torch.load(base_model_path))
    model=bimodal_network(base_model,no_of_chromatin_tracks,params,seq_len,bin_size)
    return model, base_model


def save_metrics(hist_object, pr_history, records_path):
    loss = hist_object['loss']
    val_loss = hist_object['val_loss']
    val_pr = pr_history["val_auprc"]
    
    # Saving the training metrics
    np.savetxt(records_path + 'trainingLoss.txt', loss, fmt='%1.4f')
    np.savetxt(records_path + 'valLoss.txt', val_loss, fmt='%1.4f')
    np.savetxt(records_path + 'valPRC.txt', val_pr, fmt='%1.4f')
    return loss, val_pr


def transfer(train_path, val_path, basemodel, model,
             batchsize, records_path, bin_size, epochs, seed):
    
    def train_one_epoch(epoch_index):
        running_loss = 0.
        batch_avg_loss = 0.

        for i, data in enumerate(train_dataset):
 
            seq,chrom,target,labels = data
            
            # transfer data to GPU
            seq, chrom, labels = seq.to(device), chrom.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(seq,chrom)
            labels=labels.to(torch.float32) 
            loss = loss_fn(outputs, labels)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            batch_avg_loss = running_loss / (i+1) # loss per batch
            print('CHROM - EPOCH {}:  batch {} loss: {}'.format(epoch_index+1, i + 1, batch_avg_loss), end="\r")
        my_lr_scheduler.step()
        print()
        return batch_avg_loss
    
    # GPU
    device = iterutils.getDevice()
    basemodel.to(device)
    model.to(device)

    for param in basemodel.parameters():
        param.requires_grad = False
        
    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
       
    def decayed_learning_rate(step):
        initial_learning_rate = 0.01
        decay_rate = 1e-6
        decay_step = 1.0
        return initial_learning_rate / (1 + decay_rate * step / decay_step)
    my_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=decayed_learning_rate)

    train_dataset = TFdataset(train_path, batchsize, "all", bin_size, seed)
    val_dataset = TFdataset(val_path, batchsize, "all", bin_size, seed)

    print(f"Epochs = {epochs}")
    EPOCHS = epochs

    hist={"loss":[],"val_loss":[]}
    precision_recall_history={"val_auprc":[]}
    
    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch + 1))
           
        model.train(True)
        avg_loss = train_one_epoch(epoch)

        model.train(False)
        
        running_vloss = 0.0
        avg_vloss=0.0
        val_predictions=[]
        val_labels=[]
        for i, vdata in enumerate(val_dataset):
            vseq,vchrom,vtarget,vlabels = vdata
            
            # transfer datat GPU
            vseq, vchrom, vlabels = vseq.to(device), vchrom.to(device), vlabels.to(device)
            
            voutputs = model(vseq,vchrom)
            vlabels=vlabels.to(torch.float32)
            
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += float(vloss)
            avg_vloss = running_vloss / (i + 1)
            print('CHROM - EPOCH {}: LOSS train {} valid {}'.format(epoch + 1, avg_loss, avg_vloss), end="\r")
            val_predictions.append(voutputs.cpu().detach().numpy())
            val_labels.append(vlabels.cpu().detach().numpy())
        print()
        torch.save(model.state_dict(), records_path+'model_epoch{}.torch'.format(epoch+1))
        hist["loss"].append(avg_loss)
        hist["val_loss"].append(avg_vloss)
        val_predictions=np.concatenate(val_predictions)
        val_labels=np.concatenate(val_labels)
        aupr = average_precision_score(val_labels, val_predictions)
        precision_recall_history["val_auprc"].append(aupr)
    
    loss, val_pr = save_metrics(hist, precision_recall_history,records_path=records_path)
    
    return loss, val_pr


def transfer_and_train_msc(train_path, val_path, base_model_path,
                           batch_size, records_path, bin_size, seq_len, params, epochs, seed):

    # Calculate number of chromatin tracks
    no_of_chrom_tracks = len(train_path['chromatin_tracks'])
    model, basemodel = add_new_layers(base_model_path, seq_len, no_of_chrom_tracks, bin_size, params)
    loss, val_pr = transfer(train_path, val_path, basemodel, model,
                    batch_size, records_path, bin_size, epochs=epochs, seed=seed)
    return loss, val_pr
