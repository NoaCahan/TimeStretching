from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import torch
from torch.autograd import Variable

import os
import glob
import audio
from utils import timestretch, load_model
from model import AutoEncoder
import librosa

def encode(spec):
    
    model_path = './restore/'
    model_name = 'Autoencoder1.model'

    net = AutoEncoder()
    net = load_model(net,model_path,model_name)
    cuda_available = torch.cuda.is_available()
    if cuda_available is True:
        net = net.cuda()
        
    net.eval()
    spec = torch.FloatTensor(torch.from_numpy(spec))
    spec = torch.unsqueeze(spec, 0)
    spec = Variable(spec, volatile=True).contiguous()

    if cuda_available is True:
        spec = spec.cuda()

    # Pass input audio to net forward pass   
    out = net.encoder(spec)
    out = out.data.cpu().numpy()
    out = np.squeeze(out)
    return out

def prepare_data(audio_path):

    rates = [0.25, 0.5, 1, 1.5, 2, 2.5]
    enc_path = './data/emb/'
    
    # Create directory for encoding
    if os.path.exists(enc_path) is False:
        os.makedirs(enc_path)
    
    pattern = audio_path + '*' + '.npz'
    file_list = glob.glob(pattern)

    for item in file_list:
        print(os.path.splitext(os.path.basename(item))[0])
        item_ndname = enc_path + os.path.splitext(os.path.basename(item))[0][:4]
        item = np.load(item)
        spec , piece = item['spec'], item['piece']

        # get original piece encoding
        #spec = torch.FloatTensor(torch.from_numpy(spec))
        enc = encode(spec.T)
        
        for rate in rates:
            s = librosa.effects.time_stretch(piece, rate)
            spec_s = audio.spectrogram(s).astype(np.float32)
            enc_s = encode(spec_s.T)
            enc_o = timestretch(enc.T, (1/rate))
            
            #enc_s = torch.FloatTensor(enc_s)
            #enc = torch.FloatTensor(enc.T)
            #print('enc.T = ' , enc_o.T.shape , 'enc_s = ', enc_s.shape )
            new_item = item_ndname + '_' + str(rate)
            
            np.savez( new_item, input=enc_o.T, target = enc_s)

if __name__ == '__main__':
    prepare_data('./data/fma_small_preprocess/')