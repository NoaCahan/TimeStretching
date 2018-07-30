import warnings
import torch 
from collections import OrderedDict
from data_final import audio_data_loader
from functools import cmp_to_key
from model import AutoEncoder #, MaskedMSE
import os
import glob
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import numpy as np
from utils import check_grad, get_params, get_arguments, get_optimizer, save_model, load_model

def evaluate(net, criterion, epoch, eval_losses, validation, test_loss_log_file, cuda_available):
    
    total = 0
    net.eval()
    valid_num_samples = validation.size(0)
    
    if cuda_available:
        validation = validation.cuda(async=True)

    target = Variable(validation.view(-1), volatile=True)
    validation = Variable(validation, volatile=True)

    output = net(validation)
    loss = criterion(output, target)
    total += loss.data[0]

    avg_loss = total / valid_num_samples

    eval_losses.append(avg_loss)
    test_loss_log_file.writelines('====> Test set loss: {:.4f}'.format(avg_loss))
    test_loss_log_file.flush()
    
    return avg_loss

def train(net, criterion, optimizer, train_losses, train_params, train_loss_log_file, dataloader, cuda_available):
    
    total_loss = 0
    num_trained = 0

    net.train() 
    
    for i_batch,sample_batch in enumerate(dataloader):
        
        optimizer.zero_grad()
        music_spec = sample_batch
        if cuda_available:
            music_spec = music_spec.cuda(async=True)

        target_spec = Variable(music_spec.view(-1))
        music_spec = Variable(music_spec)

        outputs = net(music_spec)
        loss = criterion(outputs,target_spec)
        loss.backward()
        
        if check_grad(net.parameters(), train_params['clip_grad'], train_params['ignore_grad']):
            print('Not a finite gradient or too big, ignoring.')
            optimizer.zero_grad()
            continue

        optimizer.step()
        total_loss += loss.data[0]
        num_trained += 1

        if num_trained % train_params['print_every'] == 0:
            avg_loss = total_loss/train_params['print_every']
            print(num_trained , " ) loss is ", avg_loss)

            train_losses.append(avg_loss)
            train_loss_log_file.writelines('====> Train set loss: {:.4f}'.format(avg_loss))
            train_loss_log_file.flush()
            total_loss = 0
    
def main():
    
    cuda_available = torch.cuda.is_available()
    train_params,dataset_params = get_arguments()
    net = AutoEncoder()
    epoch_trained = 0
    if train_params['restore_model']:
        net = load_model(net,train_params['restore_dir'],train_params['restore_model'])
        if net is None:
            print("Initialize network and train from scratch.")
            net = AutoEncoder()
        else:
            epoch_trained = 0
            
    train_loader, validation = audio_data_loader(**dataset_params)

    if cuda_available is False :
        warnings.warn("Cuda is not avalable, can not train model using multi-gpu.")
    if cuda_available:
        # Remove train_params["device_ids"] for single GPU
        if train_params["device_ids"]:
            batch_size = dataset_params["batch_size"]
            num_gpu = len(train_params["device_ids"])
            assert batch_size % num_gpu == 0
            net = nn.DataParallel(net,device_ids=train_params['device_ids'])
        torch.backends.cudnn.benchmark = True		
        net = net.cuda()
        
    criterion = nn.MSELoss()
    optimizer = get_optimizer(net,train_params['optimizer'],train_params['learning_rate'],train_params['momentum'])
    
    if cuda_available:
        criterion=criterion.cuda()
    if not os.path.exists(train_params['log_dir']) :
        os.makedirs(train_params['log_dir'])
    if not os.path.exists(train_params['restore_dir']):
        os.makedirs(train_params['restore_dir'])
    train_loss_log_file = open(train_params['log_dir']+'train_loss_log.log','a')
    test_loss_log_file = open(train_params['log_dir']+'test_loss_log.log','a')

    # Add print for start of training time
    time = str(datetime.now())
    line = 'Training Started at' + str(time) +' !!! \n'
    train_loss_log_file.writelines(line)
    train_loss_log_file.flush()

    # Keep track of losses
    train_losses = []
    eval_losses = []
    best_eval = float('inf')

    # Begin!
    for epoch in range(train_params['num_epochs']):
        train(net, criterion, optimizer, train_losses, train_params, train_loss_log_file, train_loader, cuda_available)
        eval_loss = evaluate(net, criterion, epoch, eval_losses, validation, test_loss_log_file, cuda_available)
        if eval_loss < best_eval:
            
            save_model(net,1,train_params['restore_dir'])
            
            torch.save(net.state_dict(), train_params['restore_dir'] +'bestmodel.pth')
            best_eval = eval_loss
            
        save_model(net,epoch_trained + epoch + 1,train_params['restore_dir'])
        torch.save([train_losses, eval_losses, epoch], train_params['restore_dir'] + 'data_params')

    # Add print for end of training time
    time = str(datetime.now())
    line = 'Training Ended at' + str(time) +' !!! \n'
    train_loss_log_file.writelines(line)
    train_loss_log_file.flush()
    
    train_loss_log_file.close()
    test_loss_log_file.close()
    
if __name__ == '__main__':
    main()
