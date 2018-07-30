import warnings
import torch 
from collections import OrderedDict
from data_final_emb import audio_data_loader
from functools import cmp_to_key
from model import AutoEncoder , TimeStretch, MaskedMSE
import os
import glob
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import numpy as np
from utils import check_grad, get_e_arguments, get_optimizer, save_model, load_model

def evaluate(net, criterion, epoch, eval_losses, x_val, y_val, lengths , test_loss_log_file, cuda_available):
    
    total = 0
    net.eval()
    
    num_samples = x_val.size(0)
    input , target = x_val, y_val
    
    if cuda_available:
        input = x_val.cuda(async=True)
        target = y_val.cuda(async=True)
        lengths = lengths.cuda(async=True)

    target = Variable(target,volatile=True)
    input = Variable(input, volatile=True)
    lengths = Variable(lengths, volatile=True)

    output = net(input)
    #loss = criterion(output, target)
    loss = criterion(output, target, lengths)
    total += loss.data[0]

    avg_loss = total / num_samples

    eval_losses.append(avg_loss)
    test_loss_log_file.writelines('====> Test set loss: {:.4f}'.format(avg_loss))
    test_loss_log_file.flush()
    
    return avg_loss

def train(net, criterion, optimizer, train_losses, train_params, train_loss_log_file, dataloader, cuda_available):
    
    total_loss = 0
    num_trained = 0

    net.train() 
    
    for i_batch, (b_x, b_y, lengths) in enumerate(dataloader):
        
        optimizer.zero_grad()
        input , target,  lengths = b_x, b_y, lengths
        
        batch_size = b_x.size(0)
        
        if cuda_available:
            input = b_x.cuda(async=True)
            target = b_y.cuda(async=True)
            lengths = lengths.cuda(async=True)
            
        target = Variable(target)
        input = Variable(input)
        lengths = Variable(lengths)
        
        outputs = net(input)
        loss = criterion(outputs, target, lengths)
        #loss = criterion(outputs, target)
        loss.backward()
        
        if check_grad(net.parameters(), train_params['clip_grad'], train_params['ignore_grad']):
            #print('Not a finite gradient or too big, ignoring.')
            optimizer.zero_grad()
            continue

        #print("loss ", loss.data[0])
        #total_loss += loss.data[0]
        total_loss += (loss.data[0] / batch_size)
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
    train_params,dataset_params = get_e_arguments()
    net = TimeStretch()
    epoch_trained = 0
    if train_params['restore_model']:
        net = load_model(net,train_params['restore_dir'],train_params['restore_model'])
        if net is None:
            print("Initialize network and train from scratch.")
            net = TimeStretch()
        else:
            epoch_trained = 0
            
    train_loader, X_val_var, y_val_var, L_test = audio_data_loader(**dataset_params)

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
        
    criterion = MaskedMSE() #nn.MSELoss()#
    optimizer = get_optimizer(net, train_params['optimizer'], train_params['learning_rate'], train_params['momentum'])
    
    if cuda_available:
        criterion=criterion.cuda()
    if not os.path.exists(train_params['log_dir']) :
        os.makedirs(train_params['log_dir'])
    if not os.path.exists(train_params['restore_dir']):
        os.makedirs(train_params['restore_dir'])
    train_loss_log_file = open(train_params['log_dir']+'train_loss_log_e.log','a')
    test_loss_log_file = open(train_params['log_dir']+'test_loss_log_e.log','a')

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
        eval_loss = evaluate(net, criterion, epoch, eval_losses, X_val_var, y_val_var, L_test, test_loss_log_file, cuda_available)
        
        #if eval_loss < best_eval:
            
        #    save_model(net,1,train_params['restore_dir'])
            
        #    torch.save(net.state_dict(), train_params['restore_dir'] +'bestmodel.pth')
        #    best_eval = eval_loss
        if epoch % train_params['check_point_every'] == 0:
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

