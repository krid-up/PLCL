import os
import sys
import argparse
import random
import logging

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

# from Dataloaders import utils
from Dataloaders.brats2019 import (BraTS2019, CenterCrop, RandomCrop,
                             RandomRotFlip, ToTensor,
                             TwoStreamBatchSampler)
from Model.mcnet import MCNet3d
from utils import losses, metrics, ramps
from Dataloaders.randaugment import RandAugmentMC
from utils.test_util import test_all_case

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='./Data/BraTS2019_H5/', help='Name of Experiment')
parser.add_argument('--datalist_path', type=str,
                    default='./Datasetlist/BraTS2019/')
parser.add_argument('--exp', type=str,
                    default='PLCL_label', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='mcnet', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=25000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.1,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[96, 96, 96],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')

# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=400.0, help='consistency_rampup')
parser.add_argument('--temperature', type=float, 
                    default=0.2, help='temperature of sharpening')
args = parser.parse_args()

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def sharpening(P):
    T = 1/args.temperature
    P_sharpen = P ** T / (P ** T + (1-P) ** T)
    return P_sharpen

def entropy_map(p, C=2):
    # p N*C*W*H*D
    y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=1) / \
        torch.tensor(np.log(C)).cuda()
    return y1

def train(args, snapshot_path):
    num_classes = 2
    base_lr = args.base_lr
    train_data_path = args.root_path
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    dataset_list = args.datalist_path

    model = MCNet3d(n_channels=1, n_classes=num_classes, normalization='batchnorm').cuda()

    db_train = BraTS2019(base_dir=train_data_path, datalist_path=dataset_list, split='train', num=250,
                         transform=transforms.Compose([
                             RandomRotFlip(),
                             RandomCrop(args.patch_size),
                             ToTensor(),
                         ]))
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
    
    train_loader = DataLoader(db_train, batch_size=batch_size,
                              num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)
    MSE = nn.MSELoss()
    kl_distance = nn.KLDivLoss(reduction='none')
    
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(train_loader)))

    iter_num = 0
    max_epoch = max_iterations // len(train_loader)+1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(train_loader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            model.train()
            output_1, output_2 = model(volume_batch)
            pred_1 = torch.softmax(output_1, dim=1)
            pred_2 = torch.softmax(output_2, dim=1)

            # Calculate the supervised loss
            loss_ce = ce_loss(output_1[:args.labeled_bs], label_batch[:args.labeled_bs]) + ce_loss(output_2[:args.labeled_bs], label_batch[:args.labeled_bs])
            loss_dice = dice_loss(pred_1[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1)) + dice_loss(pred_2[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))
            supervised_loss = 0.5* (loss_ce + loss_dice)

            loss = supervised_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Change the learning rate
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/supervised_loss',
                              supervised_loss, iter_num)
            logging.info(
                'iteration %d : loss : %f, ce_loss: %f, dice_loss: %f' %
                (iter_num, loss.item(), loss_ce.item(), loss_dice.item()))
            
            if iter_num > 1000 and iter_num % 500 == 0:
                model.eval()
                avg_metric = test_all_case(
                    model, args.root_path, test_list="val.txt", num_classes=2, patch_size=args.patch_size,
                    stride_xy=18, stride_z=4)
                if avg_metric[:, 0].mean() > best_performance:
                    best_performance = avg_metric[:, 0].mean()
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                writer.add_scalar('info/val_dice_score',
                                  avg_metric[0, 0], iter_num)
                writer.add_scalar('info/val_hd95',
                                  avg_metric[0, 3], iter_num)
                logging.info(
                    'iteration %d : dice_score : %f hd95 : %f' % (iter_num, avg_metric[0, 0].mean(), avg_metric[0, 3].mean()))
                model.train()
            
            if iter_num % 2500 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num/25) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"

if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


    snapshot_path = "../model/{}/{}".format(
        args.exp, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)