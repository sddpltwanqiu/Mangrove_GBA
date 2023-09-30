#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: liutang
"""

import warnings

import argparse
import os
import sys
import shutil
import glob
import time
from tqdm import tqdm

import matplotlib 

import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms


from tensorboardX import SummaryWriter

# used for import treecountnet
from models import TreeCountNet_new
from utils.metics import mAE, RMSE, R2
from utils.loss_fun import SSIM,WeightedFocalLoss

from Lookahead import Lookahead
from RAdam import RAdam
matplotlib.use('Agg')

os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str,[i for i in range(4,8)]))


warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='tree counting Training')

parser.add_argument('--epochs', default=1000,
                    type=int, help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0,
                    type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=64, type=int,
                    help='mini-batch size (default: 64)')
parser.add_argument('--lr', default=0.001,
                    type=float, help='initial learning rate')
parser.add_argument('--in_channels', default=3,
                    type=int, help='number of channels of input')
parser.add_argument('--padding_size', default=0,
                    type=int, help='number of padding pixels around the input image')
parser.add_argument('--print-freq', '-p', default=2,
                    type=int, help='print frequency (default: 20)')
parser.add_argument('--deepsupervision', default=True,
                    type=bool, help='')
parser.add_argument('--factor', default=1,
                    type=int, help='')
parser.add_argument('--no-augment', dest='noaugment', action='store_false',
                    help='whether to use standard augmentation (default: True)')
parser.add_argument('--transfer', default='',
                    type=str, help='path to best state for transfer learning (default: none)')
parser.add_argument('--resume', default='',
                    type=str, help='path to resume')
parser.add_argument('--name', default='UAV_seg+density_0.00001',
                    type=str, help='name of experiment')
parser.add_argument('--image_path', default='/home/liutang/TreeCount/Mangrove/dataset/Mapbox_dataset/Mapbox_gauss',
                    type=str, help='path to latest checkpoint (default: none)')
parser.add_argument('--density_path', default='/home/liutang/TreeCount/Mangrove/dataset/Mapbox_dataset/Density',
                    type=str, help='path to latest checkpoint (default: none)')
parser.add_argument('--seg_path', default='/home/liutang/TreeCount/Mangrove/dataset/Mapbox_dataset/Area',
                    type=str, help='path to latest checkpoint (default: none)')
parser.add_argument('--tensorboard',
                    help='Log progress to TensorBoard', action='store_true')


parser.set_defaults(augment=False)
parser.set_defaults(tensorboard=True)

best_prec1 = 1000

args = parser.parse_args()

name = args.name + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

def main():
    global args, best_prec1, writer

    if args.tensorboard: writer = SummaryWriter()
    
    # Data loading code
    
    rs, train_location, valid_location, test_location = Preprocess(args.image_path, args.density_path, args.seg_path)
        
    colorjitter, colorequalization = ColorJitter(), ColorEqualization()
    
    randomrotate, totensor, normalize = RandomRotate(), ToTensor(), LocalPixelNormalize(rs)
    padding = Padding(args.padding_size)
    
    if args.augment:
        transform_train = transforms.Compose([colorjitter, colorequalization, normalize, totensor])
        #transform_train = transforms.Compose([normalize, randomrotate, padding, totensor])
        #transform_train = transforms.Compose([colorequalization,
         #                                 normalize, randomrotate, padding, totensor])
    else:
        transform_train = transforms.Compose([colorequalization, normalize, totensor])
        #transform_train = transforms.Compose([colorequalization, normalize, totensor])
        
    transform_val = transforms.Compose([colorequalization, normalize, totensor])
    #transform_val = transforms.Compose([normalize, totensor])

    kwargs = {'num_workers': 8, 'pin_memory': True}
    
    train_dataset = SatLabelDataset(train_location, transform=transform_train)
    val_dataset = SatLabelDataset(valid_location, transform=transform_val)
    test_dataset = SatLabelDataset(valid_location, transform=transform_val)
    
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,**kwargs)
    
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True,**kwargs)

    # create model
    cudnn.benchmark = True
    
    model = TreeCountNet_new.TreeCountNet(args)
    
    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    
    # for training on multiple GPUs. 
    # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
    nGPUs = torch.cuda.device_count()
    if (nGPUs >= 1):
        model = model.cuda()
        model = torch.nn.DataParallel(model)

    # initialize net
    model.apply(weights_init)
    
    
    # define loss function (criterion) and optimizer
    kwargs = {"gamma": 2.0, "reduction": 'mean', "ignore_index": 255}
    criterion = (torch.nn.MSELoss().cuda(), SSIM().cuda(), WeightedFocalLoss(**kwargs).cuda())
    #criterion = (WeightedFocalLoss(**kwargs).cuda(),
    #             GeneralizedSoftDiceLoss(ignore_index=255).cuda())
    
    #optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay = 1e-4)
    
    optim = RAdam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    optimizer = Lookahead( optim, alpha= 0.6 , k = 10)
    # start training  
    for epoch in range(args.start_epoch, args.epochs):
        
        #adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        premse, premae, presegacc = validate(val_loader, model, criterion, epoch)

        # remember best prec@1 and save checkpoint
        is_best = premse < best_prec1
        best_prec1 = min(premse, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_premse': best_prec1,
            'presegacc':presegacc,
            'rs': rs
        }, is_best)
    
    print('Best rmse: ', best_prec1)
    print('Best mae: ', premae)
    
    writer.close()
    

def train(train_loader, model, criterion, optimizer, epoch):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()
    top3 = AverageMeter()
    top1_seg = AverageMeter()
    
    losses_1 = AverageMeter()
    losses_2 = AverageMeter()
    losses_3 = AverageMeter()
    
    # criterion
    
    density_criterion_1, density_criterion_2, seg_criterion_3 = criterion
    #seg_criterion = criterion

    # switch to train mode
    model.train()

    end = time.time()
    for i, samples in enumerate(train_loader):

        # prepare data
        density = samples["density"]
        density = density.cuda()
        input = samples["image"]
        input = input.cuda()
        seg_label = samples["seglabel"]
        seg_label = seg_label.cuda()

        input_var = torch.autograd.Variable(input)
        density_var = torch.autograd.Variable(density)
        seg_var = torch.autograd.Variable(seg_label)

        # compute output
        if args.deepsupervision:
            out1,out2, out3, out, shadow = model(input_var)
            for num in range(4):
                loss_1 = density_criterion_1(out, density_var)+density_criterion_1(out2, density_var)+density_criterion_1(out3, density_var)+density_criterion_1(out1, density_var)
                #loss_2 = density_criterion_2(out, target_var)+density_criterion_2(out2, target_var)+density_criterion_2(out3, target_var)+density_criterion_2(out1, target_var)
                loss_3 = seg_criterion_3(shadow, seg_var)
                loss = loss_1 + 0.000001 * loss_3 #+ loss_2*0.02
        else:
            out,shadow = model(input_var)
            #loss = seg_criterion_1(out, target_1) + seg_criterion_2(out, target_2)
            loss_1 = density_criterion_1(out, density_var)
            loss_3 = seg_criterion_3(shadow, seg_var)
            #loss_2 = seg_criterion_2(out, target_var)
            loss = loss_1 + 0.000001 *loss_3#+ loss_2*0.02
        
        #loss = seg_criterion(out, target_var)
        # measure accuracy and record loss
        gts, preds = [], []
        pred = out.data.cpu().numpy()
        gt   = density.cpu().numpy()
        for pred_, gt_ in zip(pred, gt):
            preds.append(pred_)
            gts.append(gt_)
        prec1 = density_accuracy(gts, preds, args.factor) # Batch*Class*H*W
        
        gts_s, preds_s = [], []
        pred_s = shadow.data.max(1)[1].cpu().numpy()
        gt_s   = seg_label.cpu().numpy()
        for pred_s_, gt_s_ in zip(pred_s, gt_s):
            preds_s.append(pred_s_)
            gts_s.append(gt_s_)
        prec_seg = seg_accuracy(gts_s, preds_s, n_class=2) # Batch*Class*H*W
        
        top1.update(prec1["RMSE"], input.size(0))
        top2.update(prec1["MAE"], input.size(0))
        top3.update(prec1["R2"], input.size(0))
        top1_seg.update(prec_seg["freqw acc"], input.size(0))

        losses.update(loss.item(), input.size(0))
        losses_1.update(loss_1.item(), input.size(0))
        losses_3.update(loss_3.item(), input.size(0))
        #losses_2.update(loss_2.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'RMSE {top1.val:.3f} ({top1.avg:.3f})\t'
                  'MAE {top2.val:.3f} ({top2.avg:.3f})\t'
                  'ACC {top1_seg.val:.3f}({top1_seg.avg:.3f})'
                  .format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      loss=losses, top1=top1, top2=top2, top1_seg=top1_seg))
    
    # log to TensorBoard   
    
    if args.tensorboard:
        writer.add_scalar('train_loss', losses.avg, epoch)
        writer.add_scalar('train_mae', top1.avg, epoch)
        writer.add_scalar('train_mse', top2.avg, epoch)
        writer.add_scalar('seg_acc', top1_seg.avg, epoch)
       
        writer.add_scalar("density_loss_1", losses_1.avg, epoch)
        writer.add_scalar("density_loss_2", losses_2.avg, epoch)
        writer.add_scalar("seg_loss_3", losses_3.avg, epoch)

def validate(val_loader, model, criterion, epoch):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()
    top3 = AverageMeter()
    top1_seg = AverageMeter()
    # criterion
    density_criterion_1, density_criterion_2, seg_criterion_3 = criterion
    #seg_criterion = criterion

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, samples in enumerate(val_loader):
        # prepare data
        density = samples["density"]
        density = density.cuda()
        input = samples["image"]
        input = input.cuda()
        seg_label = samples["seglabel"]
        seg_label = seg_label.cuda()

        input_var = torch.autograd.Variable(input)
        density_var = torch.autograd.Variable(density)
        seg_var = torch.autograd.Variable(seg_label)

        # compute output
        if args.deepsupervision:
            with torch.no_grad():
                _,_,_, out, shadow = model(input_var)
                loss_1 = density_criterion_1(out, density_var)
                loss_3 = seg_criterion_3(shadow, seg_var)
                #loss_2 = seg_criterion_2(out, target_var)
                loss = loss_1 + 0.000001 *loss_3 #+ loss_2 * 0.2
                #loss = seg_criterion(out, target_var)
        else:
            with torch.no_grad():
                out,shadow = model(input_var)
                loss_1 = density_criterion_1(out, density_var)
                loss_3 = seg_criterion_3(shadow, seg_var)
                #loss_2 = seg_criterion_2(out, target_var)
                loss = loss_1 + 0.000001 *loss_3 #+ loss_2 * 0.2
                #loss = seg_criterion(out, target_var)

        # measure accuracy and record loss
        gts, preds = [], []
        pred = out.data.cpu().numpy()
        gt   = density.cpu().numpy()
        for pred_, gt_ in zip(pred, gt):
            preds.append(pred_)
            gts.append(gt_)
        prec1 = density_accuracy(gts, preds, args.factor) # Batch*Class*H*W
        
        gts_s, preds_s = [], []
        pred_s = shadow.data.max(1)[1].cpu().numpy()
        gt_s   = seg_label.cpu().numpy()
        for pred_s_, gt_s_ in zip(pred_s, gt_s):
            preds_s.append(pred_s_)
            gts_s.append(gt_s_)
        prec_seg = seg_accuracy(gts_s, preds_s, n_class=2) # Batch*Class*H*W
        
        top1.update(prec1["RMSE"], input.size(0))
        top2.update(prec1["MAE"], input.size(0))
        top3.update(prec1["R2"], input.size(0))
        top1_seg.update(prec_seg["freqw acc"], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'RMSE {top1.val:.3f} ({top1.avg:.3f})\t'
                  'MAE {top2.val:.3f} ({top2.avg:.3f})\t'
                  'ACC {top1_seg.val:.3f}({top1_seg.avg:.3f})'
                  .format(
                      epoch, i, len(val_loader), batch_time=batch_time,
                      loss=losses, top1=top1, top2=top2, top1_seg=top1_seg))

    print(' * RMSE {top1.avg:.3f}\t'
        'MAE{top2.avg:.3f}\t'
        'ACC{top1_seg.avg:.3f}\t'
        'R2{top3.avg:.3f}'
        .format(top1=top1, top2=top2, top1_seg=top1_seg, top3=top3))
    
    # log to TensorBoard
    
    plt.close()
    fig = plt.figure()
    p1 = plt.subplot(1,3,1)
    ret=samples["image"][0].numpy().transpose((1,2,0))
    scaled_ret = (ret - ret.min())/(ret.max() - ret.min())   ## Scaled between 0 to 1 to see properly
    p1.imshow(scaled_ret)        
    p2 = plt.subplot(1,3,2)
    scaled_label = samples["density"][0].numpy()
    p2.imshow(scaled_label)
    p3 = plt.subplot(1,3,3)
    #p3.imshow(pred[0])
    
    if args.tensorboard:
        writer.add_scalar('val_loss', losses.avg, epoch)
        writer.add_scalar('val_mae', top1.avg, epoch)
        writer.add_scalar('val_mse', top2.avg, epoch)
        writer.add_scalar('seg_acc', top1_seg.avg, epoch)
        writer.add_figure("val_view", fig, epoch)
    return top1.avg, top2.avg, top1_seg.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    #name = args.name
    global name
    directory = "runs/{}/".format(name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/{}/'.format(name) + 'model_best.pth.tar')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 after 150 and 225 epochs"""
    lr = args.lr * (1.0 - 1.0*epoch/args.epochs)**0.9
    # log to TensorBoard
    if args.tensorboard:
        writer.add_scalar('learning_rate', lr, epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


#++++++++++++++ Evaluation metrics for Density Regression ++++++++++++++++++


def density_accuracy(label_trues, label_preds, n_class, factor=1):
    """Returns accuracy score evaluation result.
      - overall accuracy: ssim
      - mean accuracy error  : mae
      - root mean square error : rmse
      - r-square    : r-square
      adapted from metrics in pytorch-semseg
    """
    mae = mAE(label_preds, label_trues, factor)
    rmse = RMSE(label_preds, label_trues, factor)
    r2 = R2(label_preds, label_trues, factor)

    return {"MAE": mae,
            "RMSE"   : rmse,
            "R2"   : r2,
           }

# ++++++++++++++ Evaluation metrics for Sementic Segmentation ++++++++++++++++++

def seg_accuracy(label_trues, label_preds, n_class, ignore_value=255):
    """Returns accuracy score evaluation result.
      - overall accuracy: acc
      - mean accuracy   : acc_cls
      - mean IU         : mean_iu
      - fwavacc         : FreqW Acc
      adapted from metrics in pytorch-semseg
    """

    mask = (label_trues == ignore_value)
    label_trues = np.ma.array(label_trues, mask=mask)
    label_preds = np.ma.array(label_preds, mask=mask)
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    cls_iu = dict(zip(range(n_class), iu))

    # output fwavacc for the moment
    return {"overall acc": acc,
            "mean acc": acc_cls,
            "mean IoU": mean_iu,
            "freqw acc": fwavacc,
            "class IoU": cls_iu}


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist

    
#++++++++++++++++ Preprocess dataset ++++++++++++++++++++++++++++++++++++++++++
        
def Preprocess(image_path, density_path, seg_path, numofclass=2):
    """
    Motivation:
    this function is used to preprocess and organize data to train and validate.
    Step 1: compute mean image and variance image for standardization
    Step 2: compute class weight for data imbalance
    Step 3: split total dataset into train (60%) and validate (40%) parts at random 
    
    Arguments:
    path: directory name inclusive of images, labels, and their list file
    name: list file name, including image/label pairs
    
    Outputs:
    weight: 
    val_dict:
    train_dict:
    image_ave:
    image_std:
    
    """
    
    if os.path.exists(image_path) and os.path.exists(density_path) and os.path.exists(seg_path):
        images_train = sorted(glob.glob(image_path + "//train/*.TIF"))
        densities_train = sorted(glob.glob(density_path + "//train/*.TIF"))
        segs_train = sorted(glob.glob(seg_path + "//train/*.TIF"))

        images_test = sorted(glob.glob(image_path + "//test/*.TIF"))
        densities_test = sorted(glob.glob(density_path + "//test/*.TIF"))
        segs_test = sorted(glob.glob(seg_path + "//test/*.TIF"))

        images_val = sorted(glob.glob(image_path + "//val/*.TIF"))
        densities_val = sorted(glob.glob(density_path + "//val/*.TIF"))
        segs_val = sorted(glob.glob(seg_path + "//val/*.TIF"))
    else:
        raise Exception("check path for images and labels")
       
    test_location = {}
    train_location = {}      
    valid_location = {}
    rs = RunningStats()
    
    for idx in tqdm(range(len(images_train))):
        
        image_location_train = images_train[idx]
        density_location_train = densities_train[idx]
        seg_location_train = segs_train[idx]
        train_location.update({idx:[image_location_train, density_location_train, seg_location_train]})
        
        image = cv2.imread(image_location_train)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        
        image = cv2.resize(image, (256,256), interpolation=cv2.INTER_NEAREST)
        #label = cv2.resize(label, (256,256), interpolation=cv2.INTER_NEAREST)
        rs.push(image)
    
    for idx in tqdm(range(len(images_test))):
        
        image_location_test = images_test[idx]
        density_location_test = densities_test[idx]
        seg_location_test = segs_test[idx]
        test_location.update({idx:[image_location_test, density_location_test, seg_location_test]})
        
        image = cv2.imread(image_location_test)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        
        image = cv2.resize(image, (256,256), interpolation=cv2.INTER_NEAREST)
        #label = cv2.resize(label, (256,256), interpolation=cv2.INTER_NEAREST)
        rs.push(image)

    for idx in tqdm(range(len(images_val))):
        
        image_location_val = images_val[idx]
        density_location_val = densities_val[idx]
        seg_location_val = segs_val[idx]
        valid_location.update({idx:[image_location_val, density_location_val, seg_location_val]})
        
        image = cv2.imread(image_location_val)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        
        image = cv2.resize(image, (256,256), interpolation=cv2.INTER_NEAREST)
        #label = cv2.resize(label, (256,256), interpolation=cv2.INTER_NEAREST)
        rs.push(image)
    
    #return weight, rs.mean, rs.std, train_location, valid_location
    return rs, train_location, valid_location, test_location
    
#++++++++++++++++ Abstract Dataset ++++++++++++++++++++++++++++++++++++++++++++
        
class SatLabelDataset(torch.utils.data.Dataset):
    """Satellite imagery and its label dataset"""
    
    def __init__(self, location, transform=None):
        """
        Arguments:
            csv_file (string): Path to the file with paris of satellie imagery
            and its label 
            root_dir (string): Directory with all the image pairs
            transform (callable, optional): Optional transform to be applied on
            a sample.
        """
        self.location = location
        self.transform = transform
        
    def __len__(self):
        return len(self.location)
    
    def __getitem__(self, idx):
        image_name = os.path.join(self.location[idx][0])
        density_name = os.path.join(self.location[idx][1])
        seg_name = os.path.join(self.location[idx][1])
        
        image = cv2.imread(image_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        density = cv2.imread(density_name, flags=-1)
        seg = cv2.imread(seg_name, flags=-1)
        #label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB).astype(np.int)
        
        image = cv2.resize(image, (256,256), interpolation=cv2.INTER_LINEAR )
        density = cv2.resize(density, (256,256), interpolation=cv2.INTER_LINEAR )
        seg = cv2.resize(seg, (256,256), interpolation=cv2.INTER_NEAREST )

        sample = {"image": image, "density": density, "seglabel": seg}
                
        if self.transform: 
            sample = self.transform(sample)
        
        return sample


    
#++++++++++++++++ Preprocess input data +++++++++++++++++++++++++++++++++++++++
        

class Padding(object):
    
    def __init__(self, padding_size=0):
        super(Padding, self).__init__()
        self.padding_size = padding_size
        
    def __call__(self, sample):
        image, density, seg = sample["image"], sample["density"], sample["seglabel"] 
        image = np.pad(image, ((self.padding_size, self.padding_size),
                               (self.padding_size, self.padding_size),(0,0)),
                               "reflect")
                
        density = np.pad(density, ((self.padding_size, self.padding_size),
                               (self.padding_size, self.padding_size)),
                               "reflect")

        seg = np.pad(seg, ((self.padding_size, self.padding_size),
                               (self.padding_size, self.padding_size)),
                               "reflect")

        return {"image": image, "density": density, "seglabel": seg}



class RandomRotate(object):
    
    def __init__(self, intensity=0.5, ignore_index=255):        
        super(RandomRotate, self).__init__()
        self.ignore_index = ignore_index
        self.intensity = intensity
    
    
    def __call__(self, sample):
        
        image, density, seg = sample["image"], sample["density"], sample["seglabel"] 
        
        if self.intensity < np.random.uniform(0.0, 1.0, 1):
            return {"image": image, "density": density, "seglabel": seg}
                
        angle = np.random.uniform(0.0, 360., 1)
        scale = np.random.uniform(0.99, 1.01, 1)        
        
        height,  width, _ = image.shape
        
        matRotate = cv2.getRotationMatrix2D((height*0.5, width*0.5), angle, scale)        
        image = cv2.warpAffine(image, matRotate, (height, width),
                               flags=cv2.INTER_NEAREST, borderValue=(0,0,0))
        density = cv2.warpAffine(density.astype(np.float32), matRotate, (height, width),
                               flags=cv2.INTER_LINEAR, borderValue=self.ignore_index).astype(np.float32)
        seg = cv2.warpAffine(seg.astype(np.int32), matRotate, (height, width),
                               flags=cv2.INTER_NEAREST, borderValue=self.ignore_index).astype(np.int64)
        
        return {"image": image, "density": density, "seglabel": seg}
    
class ToTensor(object):
    
    def __init__(self):
        super(ToTensor, self).__init__()
    
    def __call__(self, sample):
        image, density, seg = sample["image"], sample["density"], sample["seglabel"] 
        image =  image.transpose((2, 0, 1))
        return {"image": torch.from_numpy(image).type(torch.FloatTensor),
                "density": torch.from_numpy(density).type(torch.FloatTensor),
                "seglabel": torch.from_numpy(seg).type(torch.LongTensor)}


class LocalPixelNormalize(object):
    
    def __init__(self, rs):
        
        super(LocalPixelNormalize, self).__init__()
        self.rs =  rs
        
    def __call__(self, sample):
        
        image, density, seg = sample["image"], sample["density"], sample["seglabel"] 
        image = self.LocalPixelNorm(image)
        return {"image": image, "density": density, "seglabel": seg}
    
    def LocalPixelNorm(self, image):
        image = (image - self.rs.avg)/self.rs.std

        return image
    
class SimpleNormalize(object):
    
    def __init__(self):        
        super(SimpleNormalize, self).__init__()
        
    def __call__(self, sample):        
        image, density, seg = sample["image"], sample["density"], sample["seglabel"] 
        image = image/255.
        
        return {"image": image, "density": density, "seglabel": seg}
        
    
class ColorJitter(object):
    def __init__(self, intensity=0.5, magnitude=0.05):
        super(ColorJitter, self).__init__()
        self.magnitude = magnitude
        self.intensity = intensity
            
    def __call__(self, sample):
        image, density, seg = sample["image"], sample["density"], sample["seglabel"] 
        if self.intensity < np.random.uniform(0.0, 1.0, 1):
            return {"image": image, "density": density, "seglabel": seg}
        if args.in_channels==4:
            b,g,r,n= cv2.split(image)
            image = cv2.merge[r,g,b]
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV_FULL)
        h,s,v = cv2.split(image)
        h = h + np.mean(h)*self.magnitude*np.random.uniform(-1.,1.)
        s = s + np.mean(s)*self.magnitude*np.random.uniform(-1.,1.)
        v = v + np.mean(v)*self.magnitude*np.random.uniform(-1.,1.)
        h = np.clip(h, 0, 255).astype(np.uint8)
        s = np.clip(s, 0, 255).astype(np.uint8)
        v = np.clip(v, 0, 255).astype(np.uint8)
        image = cv2.cvtColor(cv2.merge([h,s,v]), cv2.COLOR_HSV2RGB)
        if args.in_channels==4:
            r,g,b= cv2.split(image)
            image = cv2.merge[r,g,b,n]
        return {"image": image, "density": density, "seglabel": seg}
        
class ColorEqualization(object):
    
    def __init__(self, clipLimit=2.0, tileGridSize=(8,8)):
        super(ColorEqualization, self).__init__()
        self.clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
            
    def __call__(self, sample):
        image, density, seg = sample["image"].astype(np.uint8), sample["density"].astype(np.float32), sample["seglabel"].astype(np.uint8)
        if args.in_channels == 4:
            r, g, b, n = cv2.split(image)
            r, g, b, n = self.clahe.apply(r), self.clahe.apply(g), self.clahe.apply(b), self.clahe.apply(n)
            image = cv2.merge([r, g, b, n])
        else:
            r, g, b = cv2.split(image)
            r, g, b= self.clahe.apply(r), self.clahe.apply(g), self.clahe.apply(b)
            image = cv2.merge([r,g,b])
        return {"image": image, "density": density, "seglabel": seg}
    
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class StyleTransfer(object):
    
    avgs = np.array([0.485, 0.456, 0.406]).reshape((1,1,-1))
    stds = np.array([0.229, 0.224, 0.225]).reshape((1,1,-1))
    
    def __init__(self, styledir=None, pthfile=None, intensity=0.5, alpha=0.5, device="cpu"):
        super(StyleTransfer, self).__init__()
        self.intensity = intensity
        self.alpha = alpha
        self.device = device
        self.model = Model()
        if pthfile is not None:
            self.model.load_state_dict({k.replace("module.",""):v for k,v in torch.load(pthfile, map_location=device).items()})
        else:
            raise ValueError("styledir must be assigned a value.")
        self.model = self.model.to(device).eval()
        
        if styledir is not None:
            location = sorted(glob.glob(styledir + "/*"))
        else:
            raise ValueError("pthfile must be assigned a value.")
        
        self.styles = []
        for idx in range(len(location)):
            style = cv2.imread(location[idx])
            style = cv2.cvtColor(style, cv2.COLOR_BGR2RGB)
            style = cv2.resize(style, (256,256), interpolation=cv2.INTER_NEAREST)
            self.styles.append(style)        
        return None

    
    def norm2tensor(self, image):        
        image = (image - self.avgs) / self.stds
        image = np.expand_dims(image.transpose((2, 0, 1)), 0)
        tensor = torch.from_numpy(image).to(self.device)
        return tensor.type(torch.FloatTensor)
        

    def denorm2image(self, tensor):
        image = tensor.squeeze().to("cpu").numpy().transpose((1, 2, 0))
        image = np.clip(image * self.stds + self.avgs, 0, 1)
        return image
        
    
    def __call__(self, sample):        
        image, density, seg = sample["image"], sample["density"], sample["seglabel"] 
        if self.intensity < np.random.uniform(0.0, 1.0, 1):
            return {"image": image, "density": density, "seglabel": seg}
        
        style = self.styles[np.random.choice(len(self.styles))]        
        style = self.norm2tensor(style/255.)
        image = self.norm2tensor(image/255.)
                
        with torch.no_grad():
            image = self.model.generate(image, style, torch.FloatTensor(np.random.uniform(0.0, self.alpha, 1)))

        image = (self.denorm2image(image) * 255).astype(np.uint8)
        return {"image": image, "density": density, "seglabel": seg}
            


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        
def weights_init(m):
    
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform(m.weight)
        m.bias.data.zero_()
            
    return None

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        
    
class RunningStats(object):
    def __init__(self):
        self.n = 0
        self.old_m = 0
        self.new_m = 0
        self.old_s = 0
        self.new_s = 0
    def clear(self):
        self.n = 0
    def push(self, x):
        self.n += 1
        if self.n == 1:
            self.old_m = self.new_m = x
            self.old_s = 0
        else:
            self.new_m = self.old_m + (x - self.old_m) / self.n
            self.new_s = self.old_s + (x - self.old_m) * (x - self.new_m)
            self.old_m = self.new_m
            self.old_s = self.new_s
    @property
    def avg(self):
        return np.mean(self.new_m) if self.n else 0.0
    @property
    def var(self):
        return np.mean(self.new_s / (self.n - 1)) if self.n > 1 else 0.0
    @property
    def std(self):
        return np.sqrt(self.var)
        

        
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

if __name__ == '__main__':
    main()

