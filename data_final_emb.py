from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import torch
import glob
import audio 
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.autograd import Variable

def data_to_array(audio_path):
    input_data = []
    target_data = []
    pattern = audio_path + '*' + '.npz'
    file_list = glob.glob(pattern)
    counter = 0
    for item in file_list:
        #print(item)
        item = np.load(item)
        input , target = item['input'], item['target']
        
        mini = min(input.shape[0] , target.shape[0])
        target = target[:mini]
        input = input[:mini]
        
        input_data.append(input)
        target_data.append(target)
        
        counter +=1
        if counter % 1000 == 0:
            print(counter)
    return input_data, target_data

# Code taken from:
# https://github.com/dhpollack/programming_notebooks/blob/master/pytorch_attention_audio.py#L245
def collate_by_input_length(batch):
    """"
     Args:
         batch: (list of tuples) [(audio, target)].
             input is a FloatTensor
             target is a FloatTensor
       Output:
         packed_batch: (PackedSequence), see torch.nn.utils.rnn.pack_padded_sequence
         labels: (Tensor), labels from the file names of the wav.
    """
    if len(batch) == 1:
        
        sigs, labels = batch[0][0], batch[0][1]
        sigs = sigs.t()
        lengths = [sigs.size(0)]
        sigs.unsqueeze_(0)
        labels.unsqueeze_(0)
        
    if len(batch) > 1:

        sigs, labels, lengths = zip(*[(a.t(), b.t(), a.size(1)) for (a,b) in sorted(batch, key=lambda x: x[0].size(1), reverse=True)])
        max_len, n_feats = sigs[0].size()
        sigs = [torch.cat((s, torch.zeros(max_len - s.size(0), n_feats)), 0) if s.size(0) != max_len else s for s in sigs]
        labels = [torch.cat((l, torch.zeros(max_len - l.size(0), n_feats)), 0) if l.size(0) != max_len else l for l in labels]

        sigs = torch.stack(sigs, 0)
        labels = torch.stack(labels, 0)
        lengths = torch.LongTensor(lengths)
    #packed_batch_sig = pack(Variable(sigs), lengths, batch_first=True)
    #packed_batch_labels = pack(Variable(labels), lengths, batch_first=True)
    #print(packed_batch_sig.data.size())
    #sigs = Variable(sigs)
    #labels = Variable(labels)
    
    return sigs, labels, lengths

class audioDataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        tuple_xy = self.data[index]
        x, y = self.data[index][0], self.data[index][1]
        return x, y

    def __len__(self):
        return len(self.data)

class AudioLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        kwargs['collate_fn'] = collate_by_input_length
        DataLoader.__init__(self, *args, **kwargs)
        
def audio_data_loader(batch_size, shuffle, num_workers, pin_memory, **kwargs):
    
    input, target = data_to_array('./data/emb/')

    TEST_SET_PROPORTION = 0.1

    # split to test/set
    X_train, X_test, y_train, y_test = train_test_split(input, target, test_size=TEST_SET_PROPORTION, random_state=42)

    print("{} training x in total".format(len(X_train)))
    print("{} validation x in total".format(len(X_test)))
    print("{} training y in total".format(len(y_train)))
    print("{} validation y in total".format(len(y_test)))
    
    train_tuples = [(torch.from_numpy(x.T).float(), torch.from_numpy(y.T).float()) for x , y in zip(X_train, y_train)]
    test_tuples = [(torch.from_numpy(x.T).float(), torch.from_numpy(y.T).float()) for x , y in zip(X_test, y_test)]


    X_test, Y_test, L_test = collate_by_input_length(test_tuples)
    
    train_dataset = AudioLoader(audioDataset(train_tuples),
                                            batch_size=batch_size,
                                            shuffle=shuffle,
                                            num_workers=num_workers,
                                            pin_memory=pin_memory)

    return train_dataset, X_test, Y_test, L_test
