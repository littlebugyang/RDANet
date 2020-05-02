from __future__ import print_function
import argparse
from math import log10

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from rbpn import Net as RBPN
from data import get_training_set, get_eval_set
import pdb
import socket
import time
import cv2
import shutil

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--batchSize', type=int, default=8, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=5, help='testing batch size')
parser.add_argument('--start_epoch', type=int, default=1, help='Starting epoch for continuing training')
parser.add_argument('--nEpochs', type=int, default=150, help='number of epochs to train for')
parser.add_argument('--snapshots', type=int, default=5, help='Snapshots')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=0.01')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=8, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=8, type=int, help='number of gpu')
parser.add_argument('--data_dir', type=str, default='./vimeo_septuplet/sequences')
parser.add_argument('--file_list', type=str, default='sep_trainlist.txt')
parser.add_argument('--other_dataset', type=bool, default=False, help="use other dataset than vimeo-90k")
parser.add_argument('--future_frame', type=bool, default=True, help="use future frame")
parser.add_argument('--nFrames', type=int, default=7)
parser.add_argument('--patch_size', type=int, default=64, help='0 to use original frame size')
parser.add_argument('--data_augmentation', type=bool, default=True)
parser.add_argument('--model_type', type=str, default='RBPN')
parser.add_argument('--residual', type=bool, default=False)
parser.add_argument('--pretrained_sr', default='3x_dl10VDBPNF7_epoch_84.pth', help='sr pretrained base model')
parser.add_argument('--pretrained', type=bool, default=False)
parser.add_argument('--save_folder', default='weights/', help='Location to save checkpoint models')
parser.add_argument('--prefix', default='F7', help='Location to save checkpoint models')

# run on my local windows machine:
# python main.py --gpu_mode='' --data_dir='./REDS4/GT' --file_list='train_list.txt' --other_dataset=True

opt = parser.parse_args()

### customization
opt.data_dir = './vimeo90k/'
opt.file_list = 'available_list.txt'
opt.other_dataset = False
# opt.threads = 1
opt.snapshots = 5
opt.batchSize = 1
opt.gpus = 1
opt.nFrames = 7
###

run_start_time = str(time.time()).split('.')[0]

gpus_list = range(opt.gpus)
hostname = str(socket.gethostname())
epoch_result_dir = './epoch_results'
cudnn.benchmark = True
print(opt)

def save_epoch_result(result_dir, epoch, iteration, results):
    # if the variable "results" is type of dict, the code should be more robust
    epoch_dir = os.path.join(result_dir, 'epoch_'+str(epoch).zfill(5))
    if os.path.exists(epoch_dir):
        shutil.rmtree(epoch_dir)
    os.mkdir(epoch_dir)

    input_result_path = os.path.join(epoch_dir, str(iteration).zfill(5) + '_input.png')
    target_result_path = os.path.join(epoch_dir, str(iteration).zfill(5) + '_target.png')
    bicubic_result_path = os.path.join(epoch_dir, str(iteration).zfill(5) + '_bicubic.png')
    flow_result_path = os.path.join(epoch_dir, str(iteration).zfill(5) + '_flow.png')
    prediction_result_path = os.path.join(epoch_dir, str(iteration).zfill(5) + '_prediction.png')

    input_img = results[0].squeeze().clamp(0, 1).numpy().transpose(1, 2, 0)
    target_img = results[1].squeeze().clamp(0, 1).numpy().transpose(1, 2, 0)
    bicubic_img = results[2].squeeze().clamp(0, 1).numpy().transpose(1, 2, 0)
    prediction_img = results[3].squeeze().clamp(0, 1).numpy().transpose(1, 2, 0)
    cv2.imwrite(input_result_path, cv2.cvtColor(input_img*255, cv2.COLOR_BGR2RGB),  [cv2.IMWRITE_PNG_COMPRESSION, 0])
    cv2.imwrite(target_result_path, cv2.cvtColor(target_img*255, cv2.COLOR_BGR2RGB),  [cv2.IMWRITE_PNG_COMPRESSION, 0])
    cv2.imwrite(bicubic_result_path, cv2.cvtColor(bicubic_img*255, cv2.COLOR_BGR2RGB),  [cv2.IMWRITE_PNG_COMPRESSION, 0])
    cv2.imwrite(prediction_result_path, cv2.cvtColor(prediction_img*255, cv2.COLOR_BGR2RGB),  [cv2.IMWRITE_PNG_COMPRESSION, 0])

def train(epoch): # 这里只是train一次，epoch传进来只是为了记录序数，无语
    epoch_loss = 0
    model.train()
    save_result = (epoch % (opt.snapshots) == 0)
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target, neigbor, flow, bicubic = batch[0], batch[1], batch[2], batch[3], batch[4]
        if cuda:
            input = Variable(input).cuda(gpus_list[0])
            target = Variable(target).cuda(gpus_list[0])
            bicubic = Variable(bicubic).cuda(gpus_list[0])
            neigbor = [Variable(j).cuda(gpus_list[0]) for j in neigbor]
            flow = [Variable(j).cuda(gpus_list[0]).float() for j in flow]
        optimizer.zero_grad()
        t0 = time.time()
        prediction = model(input, neigbor, flow)
        if opt.residual:
            prediction = prediction + bicubic

        # save the result of this epoch
        if save_result:
            save_epoch_result(epoch_result_dir, epoch, iteration, [input.cpu().data, target.cpu().data, bicubic.cpu().data, prediction.cpu().data])

        loss = criterion(prediction, target)
        t1 = time.time()
        epoch_loss += loss.data
        loss.backward()
        optimizer.step()

        print("===> Epoch[{}]({}/{}): Loss: {:.4f} || Timer: {:.4f} sec.".format(epoch, iteration, len(training_data_loader), loss.item(), (t1 - t0)))

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))
    return epoch_loss / len(training_data_loader)

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

def checkpoint(epoch):
    model_out_path = opt.save_folder+str(opt.upscale_factor)+'x_'+hostname+'_'+run_start_time+'_'+opt.model_type+opt.prefix+"_epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)
    torch.cuda.empty_cache()

print('===> Loading datasets')
train_set = get_training_set(opt.data_dir, opt.nFrames, opt.upscale_factor, opt.data_augmentation, opt.file_list, opt.other_dataset, opt.patch_size, opt.future_frame)
#test_set = get_eval_set(opt.test_dir, opt.nFrames, opt.upscale_factor, opt.data_augmentation)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=False)
#testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=False)

print('===> Building model ', opt.model_type)
if opt.model_type == 'RBPN':
    model = RBPN(num_channels=3, base_filter=256,  feat = 64, num_stages=3, n_resblock=5, nFrames=opt.nFrames, scale_factor=opt.upscale_factor) 

model = torch.nn.DataParallel(model, device_ids=gpus_list)
criterion = nn.L1Loss()

print('---------- Networks architecture -------------')
print_network(model)
print('----------------------------------------------')

if opt.pretrained:
    model_name = os.path.join(opt.save_folder + opt.pretrained_sr)
    if os.path.exists(model_name):
        #model= torch.load(model_name, map_location=lambda storage, loc: storage)
        model.load_state_dict(torch.load(model_name, map_location=lambda storage, loc: storage))
        print('Pre-trained SR model is loaded.')

if cuda:
    model = model.cuda(gpus_list[0])
    criterion = criterion.cuda(gpus_list[0])

optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)
for epoch in range(opt.start_epoch, opt.nEpochs + 1):
    avg_loss = train(epoch)
    loss_file = open('./epoch_loss.txt', mode='a')
    loss_file.write('Epoch [' + str(epoch) + '] average loss: ' + str(avg_loss.item()) + '\n')
    loss_file.close()
    #test()
    # learning rate is decayed by a factor of 10 every half of total epochs
    if (epoch+1) % (opt.nEpochs/2) == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 10.0
        print('Learning rate decay: lr={}'.format(optimizer.param_groups[0]['lr']))
    
    if (epoch+1) % (opt.snapshots) == 0:
        checkpoint(epoch)
