
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack

class Layer(nn.Module):
    """
    A single layer of our very simple autoencoder
    """

    def __init__(self, in_size, out_size):
        super(Layer, self).__init__()
        self.linear = nn.Linear(in_size, out_size, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, inp):
        x = self.linear(inp)
        x = self.tanh(x)
        return x


class AutoEncoder(nn.Module):
    """
    A very simple autoencoder.  No bells, whistles, or convolutions
    """

    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            Layer(1025, 512),
            Layer(512, 256),
            Layer(256, 120))
        self.decoder = nn.Sequential(
            Layer(120, 256),
            Layer(256, 512),
            Layer(512, 1025))

    def forward(self, inp):
        encoded = self.encoder(inp)
        decoded = self.decoder(encoded)
        return decoded

    
class MaskedMSE(nn.Module):
    def __init__(self):
        super(MaskedMSE, self).__init__()
        self.criterion = nn.MSELoss(size_average=False)
        
    # Taken from
    # https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation
    @staticmethod
    def _sequence_mask(sequence_length):
        
        max_len = sequence_length.data.max()
        batch_size = sequence_length.size(0)
        
        seq_range = torch.arange(0, max_len).long()
        seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
        seq_range_expand = Variable(seq_range_expand)
        if sequence_length.is_cuda:
            seq_range_expand = seq_range_expand.cuda()
        seq_length_expand = sequence_length.unsqueeze(1) \
                                           .expand_as(seq_range_expand)
        return (seq_range_expand < seq_length_expand).float()


    def forward(self, input, target, lengths):
        
        # (B, T, 1)
        mask = self._sequence_mask(lengths).unsqueeze(2)
        
        # (B, T, D)
        mask_ = mask.expand_as(input)
        self.loss = self.criterion(input*mask_, target*mask_)
        self.loss = self.loss / mask.sum()
        return self.loss

    
class TimeLayer(nn.Module):
    """
    A single layer of our very simple autoencoder
    """

    def __init__(self, in_size, out_size):
        super(TimeLayer, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_size ,out_channels=out_size, kernel_size=1,bias=False)
        self.relu = nn.ReLU()

    def forward(self, inp):
        x = self.conv1(inp)
        x = self.relu(x)
        return x

class TimeStretch(nn.Module):
    """
    A very simple network.  No bells, whistles, or convolutions
    """

    def __init__(self):
        super(TimeStretch, self).__init__()
        self.stretch = nn.Sequential(
            Layer(120, 300),
            Layer(300, 120))

    def forward(self, inp):
        stretched = self.stretch(inp)
        return stretched