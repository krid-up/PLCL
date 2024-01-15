import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from utils import ramps, losses, metrics
from Dataloaders.brats2019 import *
from Model.mcnet import MCNet3d_v1

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def sharpening(P):
    T = 1/args.temperature
    P_sharpen = P ** T / (P ** T + (1-P) ** T)
    return P_sharpen

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='./Data/BraTS2019_H5/', help='Name of Dataset')
parser.add_argument('--datalist_path', type=str, default='./Datasetlist/BraTS2019/')
parser.add_argument('--exp', type=str,  default='MCNet', help='exp_name')
parser.add_argument('--model', type=str,  default='mcnet', help='model_name')
parser.add_argument('--max_iteration', type=int,  default=25000, help='maximum iteration to train')
parser.add_argument('--max_samples', type=int,  default=250, help='maximum samples to train')
parser.add_argument('--labeled_bs', type=int, default=2, help='batch_size of labeled data per gpu')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size of labeled data per gpu')
parser.add_argument('--base_lr', type=float,  default=0.1, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--labelnum', type=int,  default=50, help='trained samples')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--consistency', type=float, default=0.1, help='consistency_weight')
parser.add_argument('--consistency_rampup', type=float, default=400.0, help='consistency_rampup')
parser.add_argument('--temperature', type=float, default=0.2, help='temperature of sharpening') # we change here as 0.2, compared with original code
parser.add_argument('--lamda', type=float, default=0.5, help='weight to balance all losses')
args = parser.parse_args()

# snapshot_path = args.root_path + "model/{}_{}_{}_labeled/{}".format(args.dataset_name, args.exp, args.labelnum, args.model)
snapshot_path = "../model/{}/{}".format(args.exp, args.model)
num_classes = 2
train_data_path = args.root_path
batch_size = args.batch_size
max_iterations = args.max_iteration
dataset_list = args.datalist_path
patch_size = (96, 96, 96)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
labeled_bs = args.labeled_bs
base_lr = args.base_lr

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)


if __name__ == "__main__":
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
 
    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    
    model = MCNet3d_v1(n_channels=1, n_classes=num_classes, normalization='batchnorm').cuda()

    db_train = BraTS2019(base_dir=train_data_path, datalist_path=dataset_list,
                         split='train',
                         num=None,
                         transform=transforms.Compose([
                             RandomRotFlip(),
                             RandomCrop(patch_size),
                             ToTensor(),
                         ]))
    labelnum = args.labelnum  
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, args.max_samples))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size-labeled_bs)
    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    writer = SummaryWriter(snapshot_path+'/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))
    consistency_criterion = losses.mse_loss
    dice_loss = losses.Binary_dice_loss
    iter_num = 0
    best_dice = 0
    max_epoch = max_iterations // len(trainloader) + 1
    lr_ = base_lr
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            model.train()
            outputs = model(volume_batch)
            num_outputs = len(outputs)

            y_ori = torch.zeros((num_outputs,) + outputs[0].shape)
            y_pseudo_label = torch.zeros((num_outputs,) + outputs[0].shape)

            loss_seg = 0
            loss_seg_dice = 0 
            for idx in range(num_outputs):
                y = outputs[idx][:labeled_bs,...]
                y_prob = F.softmax(y, dim=1)
                loss_seg += F.cross_entropy(y[:labeled_bs], label_batch[:labeled_bs])
                loss_seg_dice += dice_loss(y_prob[:,1,...], label_batch[:labeled_bs,...] == 1)

                y_all = outputs[idx]
                y_prob_all = F.softmax(y_all, dim=1)
                y_ori[idx] = y_prob_all
                y_pseudo_label[idx] = sharpening(y_prob_all)

            loss_consist = 0
            for i in range(num_outputs):
                for j in range(num_outputs):
                    if i != j:
                        loss_consist += consistency_criterion(y_ori[i], y_pseudo_label[j])
            
            consistency_weight = get_current_consistency_weight(iter_num//150)
            
            loss = args.lamda * loss_seg_dice + consistency_weight * loss_consist
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logging.info('iteration %d : loss : %03f, loss_d: %03f, loss_cosist: %03f' % (iter_num, loss, loss_seg_dice, loss_consist))
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group1 in optimizer.param_groups:
                param_group1['lr'] = lr_
            iter_num = iter_num + 1
            writer.add_scalar('Labeled_loss/loss_seg_dice', loss_seg_dice, iter_num)
            writer.add_scalar('Labeled_loss/loss_seg_ce', loss_seg, iter_num)
            writer.add_scalar('Co_loss/consistency_loss', loss_consist, iter_num)
            writer.add_scalar('Co_loss/consist_weight', consistency_weight, iter_num)
            
            if iter_num % 2500 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
