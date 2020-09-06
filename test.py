import time
import os
import torch
from options.test_options import TestOptions
from data.data_loader import ImageFolder
from models.models import create_model
import torchvision
from torchvision import datasets, transforms
from util import util
from util.visualizer import Visualizer
from util import html
import time
from models.net import ResNet50

from data.data_loader_2 import ImageFolder_2
from torch.autograd import Variable
from torch import nn
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from skimage.io import imread, imsave
from skimage.measure import compare_ssim

import torch.nn.functional as F

import numpy as np
import pandas as pd

import re

def ssim_score(generated_images, reference_images):
    ssim = compare_ssim(reference_images, generated_images, gaussian_weights=True, sigma=1.5,
                            use_sample_covariance=False, multichannel=True)
    return ssim

def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def accuracy(output, target, topk=(1,)):
    target = target.long()
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    result = res[0]
    return result

opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

res50 = ResNet50(250)
res50.load_state_dict(torch.load("/home/jiashu/Data/Bestrouting.pth"))
res50.cuda()
res50.eval()

transform = transforms.Compose([
        transforms.Resize(128),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

R64_transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

R32_transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

R16_transform = transforms.Compose([
    transforms.Resize(16),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

Local_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

data_dir = "/home/jiashu/Data/Balance_PIE_Data_3/test/"
target_dir = "/home/jiashu/Data/Balance_PIE_Data_3/test_target/"

gallery_dir = "/home/jiashu/Data/PIE_Gallery_Test/"
gallery_set = ImageFolder_2(gallery_dir, True, transform)
gallery_loader = torch.utils.data.DataLoader(gallery_set, batch_size=1, pin_memory = False, num_workers =1 ,drop_last=False, shuffle=False)

dset = ImageFolder(data_dir, target_dir, transform, R64_transform, R32_transform,R16_transform, Local_transform)

data_loader = torch.utils.data.DataLoader(dset, batch_size=opt.batchSize, drop_last=True, shuffle=False)

model = create_model(opt)
visualizer = Visualizer(opt)
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))

webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

print(opt.how_many)

model = model.eval()
print(model.training)

opt.how_many = 999999

fake_correct_110_240 = 0.
counter_man_110_240 = 0

fake_correct_120_10 = 0.
counter_man_120_10 = 0

fake_correct_90_200 = 0.
counter_man_90_200 = 0

fake_correct_80_190 = 0.
counter_man_80_190 = 0
           
fake_correct_130_40 = 0.
counter_man_130_40 = 0

fake_correct_140_50 = 0.
counter_man_140_50 = 0

ssim_fake_correct_110_240 = 0.

ssim_fake_correct_120_10 = 0.

ssim_fake_correct_90_200 = 0.

ssim_fake_correct_80_190 = 0.
   
ssim_fake_correct_130_40 = 0.

ssim_fake_correct_140_50 = 0.

gallery_feat_list = []
gallery_feat_list_maker = []

Good_dir = web_dir + "/Good"
Fail_dir = web_dir + "/Fail"

print(Good_dir)
print(Fail_dir)

if not os.path.exists(Good_dir):
    os.makedirs(Good_dir)

if not os.path.exists(Fail_dir):
    os.makedirs(Fail_dir)


with torch.no_grad():
    for data_item in gallery_loader: 
        data, maker_correct = data_item[0], data_item[1]
        data = data.cuda()
        data = Variable(data)
        output, gallery_fc, gallery_feat = res50(data)
        gallery_feat_list.append(gallery_feat)
        gallery_feat_list_maker.append(maker_correct.data.item()) 

with torch.no_grad():
    for data in data_loader:

        model.set_input(data)
        startTime = time.time()

        fake_p2, all_posenum, all_idnum, before_list, after_list, input_p1, input_P2  = model.test()

        fake_res50_prediction, fake_avgx, fake_maxp = res50(fake_p2)

        counter_man = 0

        while counter_man < opt.batchSize:

            idnum = all_idnum[counter_man:counter_man+1].item()
            posenum = all_posenum[counter_man:counter_man+1].item()

            maker = 0

            if int(idnum) == 55: #56?
                counter_man = counter_man + 1
                continue

            maker_correct = gallery_feat_list_maker.index(int(idnum))

            k = 1
            arch = fake_maxp[counter_man:counter_man+1,:]

            tem_gallery_feat =  gallery_feat_list[maker_correct]

            d1 = F.l1_loss(arch, tem_gallery_feat) 

            ssim_index = ssim_score(tensor2im(fake_p2[counter_man:counter_man+1,:,:,: ]), tensor2im(input_P2[counter_man:counter_man+1,:,:,: ]))

            p2path = data[6][counter_man]

            p2path = p2path[p2path.rindex("/"):]

            while maker < 99:
                if(maker == maker_correct):
                    maker = maker + 1
                    continue

                gallery_feat = gallery_feat_list[maker]   # L2-normalize

                d2 = F.l1_loss(arch, gallery_feat) 

                if(d1 > d2):
                    k = -1
                    break
                maker = maker + 1

            if posenum == 110 or posenum == 240:
                # print("posenum :", posenum)
                counter_man_110_240 = counter_man_110_240 + 1
                ssim_fake_correct_110_240 = ssim_fake_correct_110_240 + ssim_index
                # print("counter_man_110_240 :", counter_man_110_240)
            elif posenum == 120 or posenum == 10:
                # print("posenum :", posenum)
                counter_man_120_10 = counter_man_120_10 + 1
                ssim_fake_correct_120_10 = ssim_fake_correct_120_10 + ssim_index
                # print("counter_man_120_10 :", counter_man_120_10)
            elif posenum == 90 or posenum == 200:
                # print("posenum :", posenum)
                counter_man_90_200 = counter_man_90_200 + 1
                ssim_fake_correct_90_200 = ssim_fake_correct_90_200 + ssim_index
                # print("counter_man_90_200 :", counter_man_90_200)
            elif posenum == 80 or posenum == 190:
                # print("posenum :", posenum)
                counter_man_80_190 = counter_man_80_190 + 1
                ssim_fake_correct_80_190 = ssim_fake_correct_80_190 + ssim_index
                # print("counter_man_80_190 :", counter_man_80_190)
            elif posenum == 130 or posenum == 40:
                # print("posenum :", posenum)
                counter_man_130_40 = counter_man_130_40 + 1
                ssim_fake_correct_130_40 = ssim_fake_correct_130_40 + ssim_index
                # print("counter_man_130_40 :", counter_man_130_40)
            elif posenum == 140 or posenum == 50:
                # print("posenum :", posenum)
                counter_man_140_50 = counter_man_140_50 + 1
                ssim_fake_correct_140_50 = ssim_fake_correct_140_50 + ssim_index
                # print("counter_man_140_50 :", counter_man_140_50)
            

            if(k == 1):
                print("Correct ssim_index :", str(ssim_index))
                if posenum == 110 or posenum == 240:
                    fake_correct_110_240 = fake_correct_110_240 + 1
                elif posenum == 120 or posenum == 10:
                    fake_correct_120_10 = fake_correct_120_10 + 1
                elif posenum == 90 or posenum == 200:
                    fake_correct_90_200 = fake_correct_90_200 + 1
                elif posenum == 80 or posenum == 190:
                    fake_correct_80_190 = fake_correct_80_190 + 1
                elif posenum == 130 or posenum == 40:
                    fake_correct_130_40 = fake_correct_130_40 + 1
                elif posenum == 140 or posenum == 50:
                    fake_correct_140_50 = fake_correct_140_50 + 1

                # save_path = Good_dir + p2path
                # utils.save_image_single(fake_p2[counter_man:counter_man+1,:,:,: ], save_path)
                
                save_path = Good_dir + p2path
                util.save_image( tensor2im(fake_p2[counter_man:counter_man+1,:,:,: ]), save_path)
                
            else:
                print("Fail ssim_index :", str(ssim_index))
                # save_path = Fail_dir + p2path
                # utils.save_image_single( fake_p2[counter_man:counter_man+1,:,:,: ], save_path)

                save_path = Fail_dir + p2path
                util.save_image( tensor2im(fake_p2[counter_man:counter_man+1,:,:,: ]), save_path)

            counter_man = counter_man + 1

       #  visualizer.save_images(webpage, visuals, img_path_list,  before_list, after_list, input_p1.cpu().float().numpy(), fake_p2.cpu().float().numpy() )
        
    print(" ############ Completed ##############")
    Total_acc = fake_correct_110_240 + fake_correct_120_10 +fake_correct_90_200 +fake_correct_80_190 +fake_correct_130_40 +fake_correct_140_50 
    Total_counter = counter_man_110_240 + counter_man_120_10 + counter_man_90_200+ counter_man_80_190+ counter_man_130_40+ counter_man_140_50 
    print("Final fake_correct_110_240:", fake_correct_110_240/counter_man_110_240)
    print("Final fake_correct_120_10:", fake_correct_120_10/counter_man_120_10)
    print("Final fake_correct_90_200:", fake_correct_90_200/counter_man_90_200)
    print("Final fake_correct_80_190:", fake_correct_80_190/counter_man_80_190)
    print("Final fake_correct_130_40:", fake_correct_130_40/counter_man_130_40)
    print("Final fake_correct_140_50:", fake_correct_140_50/counter_man_140_50)
    print("Final Total_acc:", Total_acc/Total_counter)

    Total_ssim = ssim_fake_correct_110_240 + ssim_fake_correct_120_10 + ssim_fake_correct_90_200 + ssim_fake_correct_80_190 + ssim_fake_correct_130_40 + ssim_fake_correct_140_50
    print("Final ssim_fake_correct_110_240:", ssim_fake_correct_110_240/counter_man_110_240)
    print("Final ssim_fake_correct_120_10:", ssim_fake_correct_120_10/counter_man_120_10)
    print("Final ssim_fake_correct_90_200:", ssim_fake_correct_90_200/counter_man_90_200)
    print("Final ssim_fake_correct_80_190:", ssim_fake_correct_80_190/counter_man_80_190)
    print("Final ssim_fake_correct_130_40:", ssim_fake_correct_130_40/counter_man_130_40)
    print("Final ssim_fake_correct_140_50:", ssim_fake_correct_140_50/counter_man_140_50)
    print("Final Total_ssim:", Total_ssim/Total_counter)

    log_name_ssim = web_dir + "/log_ssim.txt"
    print("log_name_ssim :", log_name_ssim)
    with open(log_name_ssim, 'a') as log_file:
        log_file.write("Final ssim_fake_correct_110_240:" + str(ssim_fake_correct_110_240/counter_man_110_240))
        log_file.write('\n')
        log_file.write("Final ssim_fake_correct_120_10:" + str(ssim_fake_correct_120_10/counter_man_120_10))
        log_file.write('\n')
        log_file.write("Final ssim_fake_correct_90_200:" + str( ssim_fake_correct_90_200/counter_man_90_200))
        log_file.write('\n')
        log_file.write("Final ssim_fake_correct_80_190:" + str( ssim_fake_correct_80_190/counter_man_80_190))
        log_file.write('\n')
        log_file.write("Final ssim_fake_correct_130_40:" + str( ssim_fake_correct_130_40/counter_man_130_40))
        log_file.write('\n')
        log_file.write("Final ssim_fake_correct_140_50:" + str( ssim_fake_correct_140_50/counter_man_140_50))
        log_file.write('\n')
        log_file.write("Final Total_ssim:" + str( Total_ssim/Total_counter))
        log_file.write('\n')

    log_name = web_dir + "/log_2.txt"
    print("log_name :", log_name)
    with open(log_name, 'a') as log_file:
        log_file.write("Final fake_correct_110_240:" + str(fake_correct_110_240/counter_man_110_240))
        log_file.write('\n')
        log_file.write("Final fake_correct_120_10:" + str(fake_correct_120_10/counter_man_120_10))
        log_file.write('\n')
        log_file.write("Final fake_correct_90_200:" + str( fake_correct_90_200/counter_man_90_200))
        log_file.write('\n')
        log_file.write("Final fake_correct_80_190:" + str( fake_correct_80_190/counter_man_80_190))
        log_file.write('\n')
        log_file.write("Final fake_correct_130_40:" + str( fake_correct_130_40/counter_man_130_40))
        log_file.write('\n')
        log_file.write("Final fake_correct_140_50:" + str( fake_correct_140_50/counter_man_140_50))
        log_file.write('\n')
        log_file.write("Final Total_acc:" + str( Total_acc/Total_counter))
        log_file.write('\n')

    webpage.save()