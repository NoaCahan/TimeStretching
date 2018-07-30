import audio
from utils import load_model, timestretch
import json
import numpy as np 

import torch
from torch.autograd import Variable
import torch.nn.functional as F 
import os
from model import AutoEncoder, TimeStretch
import librosa

def transform_encode(model_name, encoding_name, encoding_transform_name):
    model_path = './restore/'
    encoding_path = './encoding/'
    
    # Create directory for encoding
    if os.path.exists(encoding_path) is False:
        os.makedirs(encoding_path)
    
    net = TimeStretch()
    net = load_model(net,model_path,model_name)
    cuda_available = torch.cuda.is_available()
    if cuda_available is True:
        net = net.cuda()
        
    net.eval()
    
    # Load encoding
    enc = np.load(encoding_name).astype(np.float32)

    enc = torch.from_numpy(enc)
    enc = torch.FloatTensor(enc)
    enc = torch.unsqueeze(enc, 0)
    enc = Variable(enc, volatile=True).contiguous()

    if cuda_available is True:
        enc = enc.cuda()
        
    # Pass input audio to net forward pass    
    enc_ts = net(enc)
    enc_ts = enc_ts.data.cpu().numpy()
    #encoding = np.squeeze(encoding)
    
    enc_ts_ndarray = encoding_path + encoding_transform_name + '.npy'
    np.save(enc_ts_ndarray, enc_ts)
    
    
def encode(model_name, piece, encoding_name):

    model_path = './restore/'
    encoding_path = './encoding/'
    
    # Create directory for encoding
    if os.path.exists(encoding_path) is False:
        os.makedirs(encoding_path)
    
    net = AutoEncoder()
    net = load_model(net,model_path,model_name)
    cuda_available = torch.cuda.is_available()
    if cuda_available is True:
        net = net.cuda()
        
    net.eval()
    
    # Load audio for encoding
    piece = audio.load_wav(piece)
    spec = audio.spectrogram(piece).astype(np.float32)

    spec = torch.from_numpy(spec.T)
    spec = torch.FloatTensor(spec)
    spec = torch.unsqueeze(spec, 0)
    spec = Variable(spec, volatile=True).contiguous()

    if cuda_available is True:
        spec = spec.cuda()
        
    # Pass input audio to net forward pass    
    encoding = net.encoder(spec)
    encoding = encoding.data.cpu().numpy()
    #encoding = np.squeeze(encoding)
    
    encoding_ndarray = encoding_path + encoding_name+ '.npy'
    np.save(encoding_ndarray, encoding)
    
    
def decode(model_name, encoding, decoder_name):
    
    """Synthesize audio from an array of embeddings.
    
    Args:
    encodings: Numpy array with shape [batch_size, time, dim].
    save_paths: Iterable of output file names.
    checkpoint_path: Location of the pretrained model. [model.ckpt-200000]
    samples_per_save: Save files after every amount of generated samples.

    """
    decoder_path = './decoding/'
    model_path = './restore/'
    
    # Create directory for encoding
    if os.path.exists(decoder_path) is False:
        os.makedirs(decoder_path)
    
    net = AutoEncoder()
    net = load_model(net,model_path,model_name)
    cuda_available = torch.cuda.is_available()
    if cuda_available is True:
        net = net.cuda()
        
    net.eval()

    # Load Encoding
    encoding_ndarray = np.load(encoding)
    encoding = torch.from_numpy(encoding_ndarray).float()
    encoding = Variable(encoding, volatile=True)
    
    generated_spec = net.decoder(encoding)
    generated_spec = generated_spec.data.cpu().numpy()
    generated_spec = np.squeeze(generated_spec)
    
    dec_name = decoder_path + decoder_name
    np.save(dec_name , generated_spec)
    

def generate(model_path,model_name, generate_path, generate_name, piece):
    
    """Synthesize audio from an array of embeddings.
    
    Args:
    encodings: Numpy array with shape [batch_size, time, dim].
    save_paths: Iterable of output file names.
    checkpoint_path: Location of the pretrained model. [model.ckpt-200000]
    samples_per_save: Save files after every amount of generated samples.

    """
    
    # Create directory for encoding
    if os.path.exists(generate_path) is False:
        os.makedirs(generate_path)

    net = AutoEncoder()
    net = load_model(net,model_path,model_name)
    cuda_available = torch.cuda.is_available()
    if cuda_available is True:
        net = net.cuda()

    net.eval()

    # Load audio for encoding
    piece = audio.load_wav(piece)
    spec = audio.spectrogram(piece).astype(np.float32)
    spec = torch.from_numpy(spec.T)
    spec = torch.FloatTensor(spec)
    
    spec = torch.unsqueeze(spec, 0)
    spec = Variable(spec, volatile=True).contiguous()

    if cuda_available is True:
        spec = spec.cuda()

    generated_spec = net(spec)
    generated_spec = generated_spec.data.cpu().numpy()
    generated_spec = np.squeeze(generated_spec)
    
    waveform = audio.inv_spectrogram(generated_spec.T)
    wav_name = generate_path + generate_name + '.wav'

    audio.save_wav(waveform , wav_name)    
    
def spec_to_wav(decode, wav_name):
    spec = np.load(decode)
    waveform = audio.inv_spectrogram(spec.T)
    audio.save_wav(waveform , wav_name)
    
def testing_performance(orig, synth, rate):
    
    # Naming wavs
    synth_ts = synth[:-4] + '_' + str(rate) + '.wav'
    librosa_synth_ts_name = synth_ts[:-4] + '_librosa.wav'

    # Load wavs to be compared
    orig = audio.load_wav(orig)
    synth = audio.load_wav(synth)
    synth_ts = audio.load_wav(synth_ts)
    
    # Compare results to Librosa timestretch
    librosa_synth_ts = librosa.effects.time_stretch(synth, rate)
    audio.save_wav(librosa_synth_ts, librosa_synth_ts_name)
    
    # Calculate MSE measure between then
    min_len = min(len(orig) , len(synth))
    model_err = ((orig[:min_len] - synth[:min_len]) ** 2).mean()
    
    min_len = min(len(librosa_synth_ts) , len(synth_ts))
    ts_err = ((librosa_synth_ts[:min_len] - synth_ts[:min_len]) ** 2).mean()
    
    print('model reconstruction error {:.4f}'.format(model_err))
    print('time stretching error {:.4f}'.format(ts_err))
    
if __name__ =='__main__':
    
    '''
    # Encoding examples on 3 samples
    encode('Autoencoder200.model', './samples/Imperial_march_cropped.wav','Imperial_march_cropped')
    encode('Autoencoder200.model', './samples/000002.wav','000002')
    encode('Autoencoder200.model', './samples/noa.wav','noa')
    
    # Decooding examples on 3 samples
    decode('Autoencoder200.model','./encoding/Imperial_march_cropped.npy', 'Imperial_march_cropped_AE200' )
    decode('Autoencoder200.model','./encoding/noa.npy', 'noa_AE200' )
    decode('Autoencoder200.model','./encoding/000002.npy', '000002_AE200' )

    # Decoding and time stretching on 3 samples
    decode_with_timestretching('Autoencoder200.model', './encoding/Imperial_march_cropped.npy', 'Imperial_march_cropped_AE200', 2)
    decode_with_timestretching('Autoencoder200.model', './encoding/Imperial_march_cropped.npy', 'Imperial_march_cropped_AE200', 0.5)

    decode_with_timestretching('Autoencoder200.model', './encoding/noa.npy', 'noa_AE200', 2)
    decode_with_timestretching('Autoencoder200.model', './encoding/noa.npy', 'noa_AE200', 0.5)
    
    decode_with_timestretching('Autoencoder200.model', './encoding/000002.npy', '000002_AE200', 2)
    decode_with_timestretching('Autoencoder200.model', './encoding/000002.npy', '000002_AE200', 0.5)
    #generate('./restore/', 'Autoencoder200.model','./generate/','000002_Autoencoder200', './samples/000002.wav')
    '''

    #testing_performance('./samples/noa.wav', './decoding/noa_AE200.wav',2.0)
    #decode_with_timestretching_sgd('Autoencoder200.model', './encoding/noa.npy', 'noa_AE200', 2)
    generate('./restore/','Autoencoder141.model', './generate/', 'noa', './samples/noa.wav')