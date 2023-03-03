import h5py
import numpy as np
import pandas as pd
# import tensorflow as tf

# from sklearn.metrics import average_precision_score as auprc
# from tensorflow.keras.models import Model, load_model
# from tensorflow.keras.layers import Dense, concatenate, Input, LSTM
# from tensorflow.keras.layers import Conv1D, Reshape, Lambda
# from tensorflow.keras.optimizers import SGD
# from tensorflow.keras.callbacks import Callback
# from tensorflow.keras.callbacks import ModelCheckpoint
# import tensorflow.keras.backend as K

# import tensorflow as tf

import iterutils

import torch
from torch import nn
from train_seq import bichrom_seq,build_model
from datetime import datetime

def TFdataset(path, batchsize, dataflag):

    TFdataset_batched = iterutils.train_TFRecord_dataset(path, batchsize, dataflag)
#     print(next(iter(TFdataset_batched)))
    return TFdataset_batched

class bichrom_chrom(nn.Module):
    def __init__(self, no_of_chromatin_tracks):
        super().__init__()
        self.conv1d=nn.Conv1d(no_of_chromatin_tracks, 15, 1, padding="valid")
        self.relu=nn.ReLU()
        self.lstm=nn.LSTM(15, 5, batch_first=True)
        self.linear=nn.Linear(5, 1)
        self.tanh=nn.Tanh()
        
    def forward(self,x):
        xc=self.conv1d(x)
        xc=self.relu(xc)
        xc=torch.permute(xc, (0, 2, 1))
        xc=self.lstm(xc)[0][:, -1, :]
        xc=self.linear(xc)
        xc=self.tanh(xc)
        return xc

class bichrom(nn.Module):
    def __init__(self,base_model,no_of_chromatin_tracks):
        super().__init__()
        self.base_model=base_model
        self.activation = {}
        def get_activation(name):
            def hook(seq_model, input, output):
                self.activation[name] = output.detach()
            return hook
        self.base_model.model_dense_repeat[7].register_forward_hook(get_activation('dense_2'))
        self.linear=nn.Linear(512, 1)
        self.tanh=nn.Tanh()
        print("-->"+str(no_of_chromatin_tracks))
        self.model=bichrom_chrom(no_of_chromatin_tracks)
        self.linear2=nn.Linear(2, 1)
        self.sigmoid=nn.Sigmoid()
        
    def forward(self,seq_input, chromatin_input):
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
    model=bichrom(base_model,no_of_chromatin_tracks)
    return model, base_model

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

# def save_metrics(hist_object, pr_history, records_path):
#     loss = hist_object.history['loss']
#     val_loss = hist_object.history['val_loss']
#     val_pr = pr_history.val_auprc
#     # Saving the training metrics
#     np.savetxt(records_path + 'trainingLoss.txt', loss, fmt='%1.4f')
#     np.savetxt(records_path + 'valLoss.txt', val_loss, fmt='%1.4f')
#     np.savetxt(records_path + 'valPRC.txt', val_pr, fmt='%1.4f')
#     return loss, val_pr


def transfer(train_path, val_path, basemodel, model,
             batchsize, records_path):
    
#     for param in basemodel.parameters():
#         param.requires_grad = False
    
    loss_fn = torch.nn.CrossEntropyLoss()
#     optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_dataset = TFdataset(train_path, batchsize, "all")
    val_dataset = TFdataset(val_path, batchsize, "all")
    
#     layers=list(model.state_dict().keys())
#     print(len(layers))
#     print(len(list(model.parameters())))
#     for l,p in zip(model.state_dict().keys(), model.parameters()):
#         print(l,p.requires_grad)
    
    def train_one_epoch(epoch_index):
        running_loss = 0.
        last_loss = 0.

        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for i, data in enumerate(train_dataset):
            
            print(f"Total Parameters = {sum(p.numel() for p in model.parameters())}")
            print(f"Total Trainable Parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
            
#             w1=model.state_dict()['base_model.linear.weight']
#             w3=model.state_dict()['model.linear.weight']
            
            # Every data instance is an input + label pair
            seq,chrom,target,labels = data
            print(seq.shape,chrom.shape)
            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            outputs = model(seq,chrom)#(torch.rand(512,4,1000),torch.randn(512,2,500))#(seq,chrom)
            labels=labels.type(torch.DoubleTensor) 
            # Compute the loss and its gradients
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
        
#             w2=model.state_dict()['base_model.linear.weight']
#             w4=model.state_dict()['model.linear.weight']
            
#             print("Base Model Weight Norm = "+str(torch.linalg.norm(torch.sub(w1,w2))))
#             print("Model Weight Norm = "+str(torch.linalg.norm(torch.sub(w3,w4))))
            
        return last_loss
    
    # Initializing in a separate cell so we can easily add more epochs to the same run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#     writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
    epoch_number = 0

    EPOCHS = 1

    best_vloss = 1_000_000.

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        
        print(model)
                
        model.train(True)
        avg_loss = train_one_epoch(epoch_number)

        # We don't need gradients on to do reporting
        model.train(False)
        
        running_vloss = 0.0
        avg_vloss=0.0
        for i, vdata in enumerate(val_dataset):
            vseq,vchrom,vtarget,vlabels = vdata
            voutputs = model(vseq,vchrom)
            vlabels=vlabels.type(torch.DoubleTensor)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss
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

def transfer_and_train_msc(train_path, val_path, base_model_path,
                           batch_size, records_path, bin_size, seq_len, params):

    # Calculate number of chromatin tracks
    no_of_chrom_tracks = len(train_path['chromatin_tracks'])
    model, basemodel = add_new_layers(base_model_path, seq_len, no_of_chrom_tracks, bin_size, params)
    loss, val_pr = transfer(train_path, val_path, basemodel, model,
                    batch_size, records_path)
    return loss, val_pr
