from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import torch
import glob
import audio 
from sklearn.model_selection import train_test_split

def data_to_array(audio_path):
    data = []
    pattern = audio_path + '*' + '.npz'
    file_list = glob.glob(pattern)
    counter = 0
    for item in file_list:

        item = np.load(item)
        spec , piece = item['spec'], item['piece']
        data.append(spec.T)
        counter +=1
        if counter % 1000 == 0:
            print(counter)
    return data

class audio_dataset(Dataset):

    def __init__(self,data):
        self.data = data

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        return self.data[idx]    

def audio_data_loader(batch_size, shuffle, num_workers, pin_memory, **kwargs):
    
    specs = data_to_array('./data/fma_small_preprocess/')

    TEST_SET_PROPORTION = 0.1
    print("specs array size = " , specs[0].shape)
    # split to test/set
    X_train, X_test = train_test_split(specs, test_size=TEST_SET_PROPORTION, random_state=42)

    print("{} training pieces in total".format(len(X_train)))
    print("{} validation pieces in total".format(len(X_test)))
   
    X_train = torch.FloatTensor(np.asarray(X_train))
    X_test = torch.FloatTensor(np.asarray(X_test))

    audioDataset = audio_dataset(X_train)
    train_dataset = DataLoader(audioDataset, 
		batch_size=batch_size,
                shuffle=shuffle,
		num_workers=num_workers,
		pin_memory=pin_memory)
    return train_dataset, X_test
