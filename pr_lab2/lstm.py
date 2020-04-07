import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn


class FrameLevelDataset(Dataset):
    def __init__(self, feats, labels, sort=False):
        """
            feats: Python list of numpy arrays that contain the sequence features.
                   Each element of this list is a numpy array of shape seq_length x feature_dimension
            labels: Python list that contains the label for each sequence (each label must be an integer)
        """
        self.lengths =  [len(feat) for feat in feats]# Find the lengths 
        # if sort is true sort each list according to self.lengths
        if sort:
            indexes = list(range(len(self.lengths)))
            indexes.sort(key=self.lengths.__getitem__, reverse=True)
            self.lengths = list(map(self.lengths.__getitem__, indexes))
            feats = list(map(feats.__getitem__, indexes))
            labels = list(map(labels.__getitem__, indexes))
          
        self.feats = self.zero_pad_and_stack(feats)
        if isinstance(labels, (list, tuple)):
            self.labels = np.array(labels).astype('int64')

    def zero_pad_and_stack(self, x):
        """
            This function performs zero padding on a list of features and forms them into a numpy 3D array
            returns
                padded: a 3D numpy array of shape num_sequences x max_sequence_length x feature_dimension
        """
        # --------------- Insert your code here ---------------- #
        max_length = max(self.lengths)
        padded = np.empty([len(x), max_length, x[0].shape[1]])
        for i, item in enumerate(x):
            padded[i] = np.pad(x[i], [(0, max_length - len(x[i])), (0, 0)], 'constant', constant_values=(0)) 
        return padded

    def __getitem__(self, item):#so in my dataloader it will be X,y,and length of x!
        return self.feats[item], self.labels[item], self.lengths[item]

    def __len__(self):
        return len(self.feats)


class BasicLSTM(nn.Module):
    def __init__(self, input_dim, rnn_size, output_dim, num_layers, dropout, bidirectional=False, pad_packed=False):
        super(BasicLSTM, self).__init__()
        self.bidirectional = bidirectional
        self.feature_size = rnn_size  * 2 if self.bidirectional else rnn_size
        # --------------- Insert your code here ---------------- #
        # Initialize the LSTM, Dropout, Output layers
        self.input_size = input_dim
        self.num_layers= num_layers
        self.output_size = output_dim
        self.dropout = dropout
        self.pad_packed = pad_packed
        self.lstm = nn.LSTM(self.input_size, rnn_size, self.num_layers, dropout=self.dropout, 
                            batch_first=True, bidirectional=self.bidirectional)
        self.output = nn.Linear(self.feature_size, self.output_size)

    def forward(self, x, lengths):
        """ 
            x : 3D numpy array of dimension N x L x D
                N: batch index
                L: sequence index
                D: feature index

            lengths: N x 1
         """
        
        # --------------- Insert your code here ---------------- #
        
        # You must have all of the outputs of the LSTM, but you need only the last one (that does not exceed the sequence length)
        # To get it use the last_timestep method
        # Then pass it through the remaining network
        # if stacked is true use pack_padded_sequence and pad_packed_sequence
        if self.pad_packed:
            packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)
            lstm_outs, hidden = self.lstm(packed)
            lstm_outs, lengths = nn.utils.rnn.pad_packed_sequence(lstm_outs, batch_first=True) 
        else:
            lstm_outs, hidden = self.lstm(x)
        last_out = self.last_timestep(lstm_outs, lengths, self.bidirectional)
        output = self.output(last_out)
        last_outputs = nn.functional.log_softmax(output, dim=1)
        return last_outputs

    def last_timestep(self, outputs, lengths, bidirectional=False):
        """
            Returns the last output of the LSTM taking into account the zero padding
        """
        if bidirectional:
            forward, backward = self.split_directions(outputs)
            last_forward = self.last_by_index(forward, lengths)
            last_backward = backward[:, 0, :]
            # Concatenate and return - maybe add more functionalities like average
            return torch.cat((last_forward, last_backward), dim=-1)

        else:
                return self.last_by_index(outputs, lengths)

    @staticmethod
    def split_directions(outputs):
        direction_size = int(outputs.size(-1) / 2)
        forward = outputs[:, :, :direction_size]
        backward = outputs[:, :, direction_size:]
        return forward, backward

    @staticmethod
    def last_by_index(outputs, lengths):
        # Index of the last output for each sequence.
        idx = (lengths - 1).view(-1, 1).expand(outputs.size(0),
                                               outputs.size(2)).unsqueeze(1)
        return outputs.gather(1, idx).squeeze()
