from __future__ import division
import h5py
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

# user defined module
import iterutils
from helper import plot_distributions


from functools import partial


def transforms(x,bin_size):
    return x.reshape((x.shape[0],bin_size,-1)).mean(axis=1).flatten()


def TFdataset(path, batchsize, dataflag, bin_size):
    
    transform_frozen = partial(transforms, bin_size = bin_size)
    
    TFdataset_batched = iterutils.train_TFRecord_dataset(path, batchsize, dataflag, shuffle=False, drop_remainder=False, transforms={"chrom": transform_frozen})

    return TFdataset_batched


def test_on_batch(TFdataset, model, outfile, mode):
    """
    Get probabilities for each test data point.
    The reason that this is implemented in a batch is because
    the whole genome cannot be loaded without batching.
    Parameters:
        batch_generator (generator): a generator that yields sequence, chromatin
        and label vectors.
        model (keras Model): A trained Keras model
        outfile (str): The outfile used for storing probabilities.
    Returns: None (Saves an output file with the probabilities for the test set )
    """
    print(model)
    model.train(False)
    probas_list=[]
    true_labels_list=[]
    for batch in TFdataset:
        seq,chrom,target,labels=batch
        if mode=="seqonly":
            p=model(seq).detach().numpy()
        else:
            p=model(seq,chrom).detach().numpy()
        probas_list.append(p)
        true_labels_list.append(labels)
    probas=np.concatenate(probas_list)
    true_labels = np.concatenate(true_labels_list)
    
    # erase the contents in outfile
    file = open(outfile, "w")
    file.close()
    # saving to file: 
    with open(outfile, "a") as fh:
        np.savetxt(fh, probas)
    
    return true_labels, probas


def get_metrics(test_labels, test_probas, records_file, model_name):
    """
    Takes the test labels and test probabilities, and calculates and/or
    plots the following:
    a. P-R Curves
    b. auPRC
    c. auROC
    d. Posterior Distributions of the Recall at FPR=0.01
    Parameters:
        test_labels (ndarray): n * 1 vector with the true labels ( 0 or 1 )
        test_probas (ndarray): n * 1 vector with the network probabilities
        records_file (str): Path to output file
        model_name (str): Model being tested
    Returns: None
    """
    # Calculate auROC
    roc_auc = roc_auc_score(test_labels, test_probas)
    # Calculate auPRC
    precision, recall, thresholds = precision_recall_curve(test_labels, test_probas)
    prc_auc = auc(recall, precision)
    records_file.write('')
    # Write auROC and auPRC to records file.
    records_file.write("Model:{0}\n".format(model_name))
    records_file.write("AUC ROC:{0}\n".format(roc_auc))
    records_file.write("AUC PRC:{0}\n".format(prc_auc))


def get_probabilities(path, model, outfile, mode, bin_size):
    """
    Get network-assigned probabilities
    Parameters:
        filename (str): Input file to be loaded
        seq_len (int): Length of input DNA sequence
    Returns:
         probas (ndarray): An array of probabilities for the test set
         true labels (ndarray): True test-set labels
    """
    # Inputing a range of default values here, can be changed later.
    dataset = TFdataset(path=path, batchsize=1000, dataflag=mode, bin_size=bin_size)
    # Load the keras model
    # model = load_model(model_file)
    true_labels, probas = test_on_batch(dataset, model, outfile, mode)

    return true_labels, probas


def plot_pr_curve(test_labels, test_probas, color, label):
    # Get the PR values:
    precision, recall, _ = precision_recall_curve(y_true=test_labels,
                                                  probas_pred=test_probas)
    plt.plot(recall, precision, c=color, lw=2.5, label=label)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    
def combine_pr_curves(records_file, m_seq_probas, m_sc_probas, labels):
    plot_pr_curve(labels, m_seq_probas, color='#F1C40F', label='Seqnet')
    plot_pr_curve(labels, m_sc_probas, color='#2471A3', label='Bichrom')
    plt.legend(loc='upper right')
    plt.savefig(records_file + '.pr_curves.pdf')

    
def evaluate_models(path, probas_out_seq, probas_out_sc,
                    model_seq, model_sc, records_file_path, bin_size):

    
    # Define the file that contains testing metrics
    records_files = open(records_file_path + '.txt', "w")

    # Get the probabilities for both M-SEQ and M-SC models:
    # Note: Labels are the same for M-SC and M-SEQ
    true_labels, probas_seq = get_probabilities(path=path,
                                                model=model_seq,
                                                outfile=probas_out_seq,
                                                mode='seqonly',bin_size=bin_size)

    _, probas_sc = get_probabilities(path=path, 
                                     model=model_sc, outfile=probas_out_sc,
                                     mode='all',bin_size=bin_size)

    # Get the auROC and the auPRC for both M-SEQ and M-SC models:
    get_metrics(true_labels, probas_seq, records_files, 'MSEQ')
    get_metrics(true_labels, probas_sc, records_files, 'MSC')

    # Plot the P-R curves
    combine_pr_curves(records_file_path, probas_seq, probas_sc, true_labels)

    # Plot the posterior distributions of the recall:
    plot_distributions(records_file_path, probas_seq, probas_sc, true_labels,
                       fpr_thresh=0.01)
