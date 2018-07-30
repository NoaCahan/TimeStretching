# This code is taken from:
# https://github.com/deep-art-project/Music/blob/master/wavenet/audio_func.py

import numpy as np
import torch
import librosa
import json
import os
import glob
from collections import OrderedDict

from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from skimage.transform import resize

def wrap(data, **kwargs):
    if torch.is_tensor(data):
        var = Variable(data, **kwargs).cuda()
        return var
    else:
        return tuple([wrap(x, **kwargs) for x in data])
    
def get_params(json_dir):
    with open(json_dir,'r') as f:
        params = json.load(f)
    f.close()
    return params

def get_arguments():
	train_params = get_params('./params/train_params.json')
	dataset_params = get_params('./params/dataset_params.json')
	return train_params,dataset_params

def get_e_arguments():
	train_params = get_params('./params/train_params_e.json')
	dataset_params = get_params('./params/dataset_params_e.json')
	return train_params,dataset_params

def get_optimizer(model,optimizer_type,learning_rate,momentum1=False):
	if optimizer_type =='sgd':
		return optim.sgd(model.parameters(),lr=learning_rate,momentum=momentum1)
	if optimizer_type == 'RMSprop':
		return optim.RMSprop(model.parameters(),lr=learning_rate,momentum= momentum1)
	if optimizer_type == 'Adam':
		return optim.Adam(model.parameters(),lr=learning_rate, weight_decay=0.0)
	if optimizer_type == 'lbfgs':
                return optim.LBFGS(model.parameters(), lr = learning_rate)	

def save_model(model,num_epoch,path):
	model_name = 'Autoencoder' + str(num_epoch)+'.model'
	checkpoint_path = path + model_name
	print('Storing checkpoint to {}...'.format(path))
	torch.save(model.state_dict(),checkpoint_path)
	print('done')

def _load(checkpoint_path):
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    return checkpoint

def load_model(model, path, model_name):

    checkpoint_path = path + model_name
    #print("Trying to restore saved checkpoint from ",
    #      "{}".format(checkpoint_path))
    if os.path.exists(checkpoint_path):
    #    print("Checkpoint found, restoring!")
        # Create a new state dict to prevent error when storing a model
        # on one device and restore it from another
        
        state_dict = _load(checkpoint_path)
        keys = list(state_dict.keys())
        if keys[0][:6] == 'module':
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]
                new_state_dict[name] = v
            state_dict = new_state_dict
        model.load_state_dict(state_dict)
        return model
    else:
        print("No checkpoint found!")
        return None

def mu_law_encode(audio, quantization_channels = 256):
    '''
    Arguments:
        audio: type(torch.FloatTensor), size(sequence_length)
        quantization_channels: as the name describes
    Input:
        torch Tensor of shape(sequence length)
    Return:
        A torch tensor with each element ranging from 0 to 255
        The size of return tensor is the same as input tensor
    '''
    mu = torch.Tensor([quantization_channels - 1])
    mu = mu.float()
    safe_audio_abs = torch.abs(torch.clamp(audio, -1.0, 1.0))
    magnitude = torch.log1p(mu * safe_audio_abs) / torch.log1p(mu)
    signal = torch.sign(audio) * magnitude
    encoded = (signal + 1) / 2 * mu + 0.5
    return encoded.long()

def mu_law_decode(output, quantization_channels = 256):
    '''
    Argument:
        output:quantized values, data type(torch Tensor),
               data shape(sequence_length), each element
               in output is an int ranging from 0 to 255
    Return:
        A torch Tensor with each element's absolute value
        less than 1, the size of returned tensor is same as
        Argument 'output'
    '''
    mu = torch.Tensor([quantization_channels - 1])
    mu = mu.float()
    signal = 2.0 * (output.float() / mu) - 1.0
    magnitude = (1.0 / mu) * ((1.0 + mu) ** torch.abs(signal) - 1.0)
    return torch.sign(signal) * magnitude

def check_grad(params, clip_th, ignore_th):
    befgad = torch.nn.utils.clip_grad_norm(params, clip_th)
    return (not np.isfinite(befgad) or (befgad > ignore_th))

def timestretch(encoding, factor):
    
    '''
    timestretch is a utility function that stretchs given encoding.
    It retains the encodings range of values.
    '''
    min_encoding, max_encoding = encoding.min(), encoding.max()
    encodings_norm = (encoding - min_encoding) / (max_encoding - min_encoding)
    timestretches = []
    
    encodings_norm = encodings_norm.reshape(encodings_norm.shape[0] ,encodings_norm.shape[1], 1)
    for encoding_i in encodings_norm:
        stretched = resize(encoding_i, (int(encoding_i.shape[0] * factor), encoding_i.shape[1]), mode='reflect')
        stretched = (stretched * (max_encoding - min_encoding)) + min_encoding
        timestretches.append(stretched)
        
    encoding_timestretched = np.array(timestretches).squeeze(-1)
    return encoding_timestretched