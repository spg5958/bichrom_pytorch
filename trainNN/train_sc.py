import h5py
import numpy as np
import pandas as pd

from sklearn.metrics import average_precision_score as auprc

import iterutils

import torch
from torch import nn
from train_seq import bichrom_seq,build_model
from datetime import datetime


def TFdataset(path, batchsize, dataflag, bin_size):
    print(f"bin_size = {bin_size}")
    TFdataset_batched = iterutils.train_TFRecord_dataset(path, batchsize, dataflag, transforms={"chrom": lambda x:x.reshape((x.shape[0],bin_size,-1)).mean(axis=1).flatten()})
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
        self.relu2=nn.ReLU()
        self.linear=nn.Linear(5, 1)
        self.tanh=nn.Tanh()
        
    def forward(self,x):
        xc=self._reshape(x, (self.no_of_chromatin_tracks, int(self.seq_len/self.bin_size)))
        xc=self.conv1d(xc)
        xc=self.relu(xc)
        xc=torch.permute(xc, (0, 2, 1))
        xc=self.lstm(xc)[0][:, -1, :]
        xc=self.relu2(xc)
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
        self.linear2=nn.Linear(2, 1)
        self.sigmoid=nn.Sigmoid()
        
    def forward(self,seq_input,chromatin_input):
        self.base_model(seq_input)
        curr_tensor=self.activation['dense_2']
        xs=self.linear(curr_tensor)
        xs=self.tanh(xs)
        xc=self.model(chromatin_input)
        xsc = torch.cat((xs, xc), dim=1)
        xsc=self.linear2(xsc)
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
             batchsize, records_path, bin_size):
    
    for param in basemodel.parameters():
        param.requires_grad = False
        
    ###########################
    print("#"*20)
    for name, param in model.named_parameters():
        print(name, param.requires_grad)
    print("#"*20)
    ########################### 
    
    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
    train_dataset = TFdataset(train_path, batchsize, "all", bin_size)
    val_dataset = TFdataset(val_path, batchsize, "all", bin_size)
    
    def train_one_epoch(epoch_index):
        running_loss = 0.
        batch_avg_vloss = 0.

        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for i, data in enumerate(train_dataset):
            
            print(f"Total Parameters = {sum(p.numel() for p in model.parameters())}")
            print(f"Total Trainable Parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
            
            ###########################
            print("#"*20)
            w1=model.state_dict()['base_model.linear.weight']
            w3=model.state_dict()['model.conv1d.weight']
            print("#"*20)
            ###########################
            
            # Every data instance is an input + label pair
            seq,chrom,target,labels = data
            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            outputs = model(seq,chrom)
            labels=labels.to(torch.float32) 
            # Compute the loss and its gradients
            loss = loss_fn(outputs, labels)
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            batch_avg_vloss = running_loss / (i+1) # loss per batch
            print('  batch {} loss: {}'.format(i + 1, batch_avg_vloss))

         
            ###########################
            print("#"*20)
            w2=model.state_dict()['base_model.linear.weight']
            w4=model.state_dict()['model.conv1d.weight']
            
            print("Base Model Weight Norm = "+str(torch.linalg.norm(torch.sub(w1,w2))))
            print("Model Weight Norm = "+str(torch.linalg.norm(torch.sub(w3,w4))))
            print("#"*20)
            ###########################
            
        return batch_avg_vloss
    
    # Initializing in a separate cell so we can easily add more epochs to the same run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    epoch = 0

    EPOCHS = 2

    best_vloss = 1_000_000.

    hist={"loss":[],"val_loss":[]}
    precision_recall_history={"val_auprc":[]}
    
    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        print(model)
                
        model.train(True)
        avg_loss = train_one_epoch(epoch)

        # We don't need gradients on to do reporting
        model.train(False)
        
        running_vloss = 0.0
        avg_vloss=0.0
        val_predictions=[]
        val_labels=[]
        for i, vdata in enumerate(val_dataset):
            vseq,vchrom,vtarget,vlabels = vdata
            voutputs = model(vseq,vchrom)
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
    
    loss, val_pr = save_metrics(hist, precision_recall_history,records_path=records_path)
    
    return loss, val_pr


def transfer_and_train_msc(train_path, val_path, base_model_path,
                           batch_size, records_path, bin_size, seq_len, params):

    # Calculate number of chromatin tracks
    no_of_chrom_tracks = len(train_path['chromatin_tracks'])
    model, basemodel = add_new_layers(base_model_path, seq_len, no_of_chrom_tracks, bin_size, params)
    loss, val_pr = transfer(train_path, val_path, basemodel, model,
                    batch_size, records_path, bin_size)
    return loss, val_pr
