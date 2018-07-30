import os
import glob
import librosa
import numpy as np
import audio

def mu_law_encode(audio, quantization_channels = 256):
    '''
    Arguments:
        audio: type(np.array), size(sequence_length)
        quantization_channels: as the name describes
    Input:
        np.array of shape(sequence_length)
    Return:
        np.array with each element ranging from 0 to 255
        The size of return array is the same as input tensor
    '''
    mu = quantization_channels - 1
    safe_audio_abs = np.abs(np.clip(audio, -1.0, 1.0))
    magnitude = np.log1p(mu * safe_audio_abs) / np.log1p(mu)
    signal = np.sign(audio) * magnitude
    encoded = (signal + 1) / 2 * mu + 0.5
    return encoded.astype(np.float32)
    
def wav_to_ndarray(audio_dir, ndarray_dir, window_length):
    pattern = audio_dir + '*' + '.wav'
    file_list = glob.glob(pattern)

    for item in file_list:
        item_ndname = ndarray_dir + os.path.splitext(os.path.basename(item))[0] + '.npy'
        audio = audio.load_wav(item)
        item_iter = 0
        
        while(len(item)) > window_length:

            piece = item[: (window_length - 1)]
            #piece = torch.FloatTensor(torch.from_numpy(piece))

            item = item[window_length:]
            new_item_ndname = item_ndname + str(item_iter) + '.npy'
            
            np.save(new_item_ndname , piece)
            item_iter += 1
        

def preprocess(audio_dir, ndarray_dir , window_length):
    pattern = audio_dir + '*' + '.wav'
    file_list = glob.glob(pattern)

    for item in file_list:
        item_ndname = ndarray_dir + os.path.splitext(os.path.basename(item))[0]
        item = audio.load_wav(item)
        
        item_iter = 0
        
        while(len(item)) > window_length:

            piece = item[: (window_length - 1)]
            spec = audio.spectrogram(piece).astype(np.float32)
            #spec = audio.melspectrogram(piece).astype(np.float32)
            
            #spec = torch.FloatTensor(torch.from_numpy(spec.T))
            #piece = torch.FloatTensor(torch.from_numpy(item))

            item = item[window_length:]
            new_item_ndname = item_ndname + str(item_iter) + '.npy'
            
            np.savez(new_item_ndname , piece = piece, spec = spec)
            item_iter += 1

if __name__ == '__main__':
    #wav_to_ndarray('./data/fma_small_wav/', './data/fma_small_not_encoded_split/', audio.hparams["window_length"])
    preprocess('./data/fma_small_wav/', './data/fma_small_preprocess/' , audio.hparams["window_length"])