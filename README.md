# TimeStretching
Pytorch implementation of Time Stretching in Music using an Autoencoder Network 

 ### Problem Description:
 
Time stretching is the process of changing the duration of an audio signal without affecting its pitch.
When using naive rate conversion or resampling, slowing down a recording lowers the pitch, speeding it up raises the pitch.

Although many classical solution exists (in both time and frequency domain), to the best of my knowledge, there is no existing neural network that solves the time stretching problem.

A neural network for time stretching is important for a number of reasons:

1. Deep learning on Music Information retrieval is an active area of research where there are still many tasks that are unexplored. As part of the task of obtaining better understanding of Music audio, time stretching is important.
 
2. The key importance of this model is as part of an end to end audio processing task.
An extended project will be to use neural networks to create a full phase vocoder application.
 
In the proposed method, time stretching will be implemented by scaling in time the encodings from a pre-trained autoencoder, and then adding a discriminator in order to force similarity between the scaled and original encodings. 

### Chosen Model Architecture:

Use a pre-trained autoencoder architecture for music audio and to manipulated the encoding in the time domain, in order to reach the desired time stretching effect. Then the scaled encoded vector will be decoded by adding a discriminator to the encoder, to create a GAN with a loss between the original encoded vector and the scaled one. 

In this approach the audio can be process in two ways:
Raw audio will be an input to the net without any preprocessing. In this method the net should be some kind of RNN for sequential data processing.
Audio will be preprocessed and fed to the net as a spectrogram. Here the net will be CNN based.

Instead of trying to create a network that can be used as part of an end-to-end raw audio processing application, I tried to first solve a simplified version of the problem that will have a shorter training and inference time.
In order to achieve this I've trained an extremely simple autoencoder on spectrogram images (rather than raw audio) on a smaller dataset (7.2 GiB).
The model has only 6 fully connected layers with tanh() activation between them.
For the encoding manipulation, I’ve used a second network which in which I’ve used as a dataset, the encoding of the previous network stretched in different time lengths and corresponding encodings of a time-stretched signal from a phase vocoder.
The model uses 2 fully connected layers with tanh() activation between them.

Notebooks folder: contains notebook for presenting the results.

