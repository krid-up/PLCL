from Model.mcnet import unet_3D
from Model.unet_3D_dv_semi import unet_3D_dv_semi



def net_factory_3d(net_type="unet_3D", in_chns=1, class_num=2):
    if net_type == "unet_3D":
        net = unet_3D(n_channels=in_chns, n_classes=class_num, normalization='batchnorm').cuda()
    elif net_type == "unet_3D_dv_semi":
        net = unet_3D_dv_semi(n_classes=class_num, in_channels=in_chns).cuda()
    else:
        net = None
    return net
