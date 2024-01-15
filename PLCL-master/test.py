import argparse
import os
import shutil
from glob import glob
import numpy

import torch

from Model.unet_3D_dv_semi import unet_3D_dv_semi
from Model.mcnet import unet_3D
from Model.mcnet import MCNet3d
from Model.mcnet import MCNet3d_v1
from utils.test_util import test_all_case

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def Inference(FLAGS):
    snapshot_path = "../model/{}/{}".format(FLAGS.exp, FLAGS.model)
    num_classes = 2
    test_save_path = "../Prediction/{}/".format(FLAGS.exp)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    net = MCNet3d(n_channels=1, n_classes=num_classes, normalization='batchnorm').cuda()
    # net = unet_3D(n_channels=1, n_classes=num_classes, normalization='batchnorm').cuda()
    # net = MCNet3d_v1(n_channels=1, n_classes=num_classes, normalization='batchnorm').cuda()
    save_mode_path = os.path.join(snapshot_path, 'iter_' + str(FLAGS.iter) + '.pth')
    print(save_mode_path)
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()
    avg_dice, avg_hd, avg_tpr, avg_ppv = test_all_case(net, base_dir=FLAGS.root_path, method=FLAGS.model, 
                               test_list=FLAGS.test_dir, num_classes=num_classes,
                               patch_size=(96, 96, 96), stride_x=48, stride_y=48, stride_z=48, test_save_path=test_save_path)
    return avg_dice, avg_hd, avg_tpr, avg_ppv


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str,
                        default='./Data/BraTS2019_H5/', help='Name of Experiment')
    parser.add_argument('--exp', type=str,
                        default="PLCL", help='experiment_name')
    parser.add_argument('--model', type=str,
                        default="mcnet", help='model_name')  # voxresnet unet_3D
    parser.add_argument('--iter', type=int,  default=25000, help='model iteration')
    parser.add_argument('--test_dir', type=str, default="./Datasetlist/BraTS2019/test.txt", help='Test list path')
    FLAGS = parser.parse_args()

    metric = Inference(FLAGS)
    print(metric)