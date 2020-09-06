import time
import os
from options.train_options import TrainOptions
from data.data_loader import ImageFolder
from models.models import create_model
from util.visualizer import Visualizer
from torchvision import datasets, transforms
import torch
import torchvision.models as models
from models.net import ResNet50
from util import html
opt = TrainOptions().parse()
from torch import nn
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from losses.ssim import SSIM

torch.manual_seed(2019)

def valid_plot_history(errors_list):
    fig = plt.figure(figsize=(15, 15))
    fake_acc = []
    real_acc = []
    for errors in errors_list:
        fake_acc.append(errors[0])
        real_acc.append(errors[1]) 

    ax = fig.add_subplot(1, 1, 1)
    ax.plot(fake_acc , label='fake_acc')
    ax.plot(real_acc, label='real_acc')
    ax.legend(loc=2, bbox_to_anchor=(1.0,1.0), borderaxespad = 0.) 
    plt.savefig(opt.checkpoints_dir+ "/" + opt.name +'/web/images/valid_plot_line_plot_loss.png')
    plt.close()

def plot_history(errors_list):
    fig = plt.figure(figsize=(20, 20))
    pair_L1loss = []
    D_PP = []
    D_PB = []
    pair_GANloss = []
    origin_L1 = []
    perceptual = []   
    # TRV_loss = []
    loss_64_list = []
    loss_32_list = []
    loss_16_list = []
    Featmap_loss = []
    fake_acc = []
    real_acc = []
    multi_scale_loss = []
    Local_loss = []
    ssim_score = []
    local_GAN_loss = []
    for errors in errors_list:
        pair_L1loss.append(errors[0])
        D_PP.append(errors[1])
        D_PB.append(errors[2])
        pair_GANloss.append(errors[3])
        origin_L1.append(errors[4])
        perceptual.append(errors[5])
        # TRV_loss.append(errors[6])
        Featmap_loss.append(errors[6])
        fake_acc.append(errors[7])
        real_acc.append(errors[8])
        multi_scale_loss.append(errors[9])
        ssim_score.append(errors[10])

    ax = fig.add_subplot(6, 1, 1)
    ax.plot(pair_L1loss , label='pair_L1loss')
    ax.plot(pair_GANloss, label='pair_GANloss')
    ax.legend(loc=2, bbox_to_anchor=(1.0,1.0), borderaxespad = 0.) 

    ax2 = fig.add_subplot(6, 1, 2)
    ax2.plot(D_PP, label='D_PP')
    ax2.plot(D_PB, label='D_PB')
    ax2.legend(loc=2, bbox_to_anchor=(1.0,1.0), borderaxespad = 0.)

    ax3 = fig.add_subplot(6, 1, 3)
    ax3.plot(origin_L1,label='origin_L1')
    ax3.plot(perceptual,label='perceptual')
    ax3.plot(multi_scale_loss, label='multi_scale_loss')
    ax3.legend(loc=2, bbox_to_anchor=(1.0,1.0), borderaxespad = 0.) 
    
    ax4 = fig.add_subplot(6, 1, 4)
    # ax4.plot(TRV_loss,label='TRV_loss')
    ax4.plot(Featmap_loss,label='Featmap_loss')
    ax4.legend(loc=2, bbox_to_anchor=(1.0,1.0), borderaxespad = 0.) 

    ax5 = fig.add_subplot(6, 1, 5)
    ax5.plot(fake_acc,label='fake_acc')
    ax5.plot(real_acc,label='real_acc')
    ax5.legend(loc=2, bbox_to_anchor=(1.0,1.0), borderaxespad = 0.) 

    ax6 = fig.add_subplot(6, 1, 6)
    ax6.plot(ssim_score,label='ssim_score')
    ax6.legend(loc=2, bbox_to_anchor=(1.0,1.0), borderaxespad = 0.) 

    plt.savefig(opt.checkpoints_dir+ "/" + opt.name +'/web/images/plot_line_plot_loss.png')
    plt.close()   

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

data_dir = "/home/jiashu/Data/Balance_PIE_Data_3/train/"
target_dir = "/home/jiashu/Data/Balance_PIE_Data_3/train_target/"

valid_data_dir = "/home/jiashu/Data/Balance_PIE_Data_3/larger_mini_test/"
valid_target_dir = "/home/jiashu/Data/Balance_PIE_Data_3/larger_mini_test_target/"

dset = ImageFolder(data_dir, target_dir, transform, R64_transform, R32_transform,R16_transform, Local_transform)

validset = ImageFolder(valid_data_dir, valid_target_dir, transform, R64_transform, R32_transform,R16_transform, Local_transform)

train_loader = torch.utils.data.DataLoader(dset, batch_size=opt.batchSize, pin_memory = True, num_workers =opt.batchSize ,drop_last=True, shuffle=True)
valid_loader = torch.utils.data.DataLoader(validset, batch_size=opt.batchSize, pin_memory = True, num_workers =opt.batchSize ,drop_last=True, shuffle=True)
dataset_size = len(dset)
print('#training images = %d' % dataset_size)

model = create_model(opt)
visualizer = Visualizer(opt)
total_steps = 0

res50 = ResNet50(250)
res50.load_state_dict(torch.load("/home/jiashu/Data/Bestrouting.pth"))
res50.cuda()
res50.eval()

mseloss = torch.nn.MSELoss()
CELoss = torch.nn.CrossEntropyLoss()

v_history_list = []
vv_history_list = []

log_name = opt.checkpoints_dir+ "/" + opt.name + "/" + "log_file.txt"

ssim_loss = SSIM()

for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    epoch_iter = 0

    for data in train_loader:
        iter_start_time = time.time()
        visualizer.reset()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize
        model.set_input(data)
        
        fake_p2, input_P2, idnum = model.forward_parameters()

        ssim_score = (ssim_loss(fake_p2, input_P2))

        idnum = idnum.long()
        idnum = idnum.squeeze()

        fake_res50_prediction, fake_avgx, fake_fc = res50(fake_p2)
        res50_prediction, avgx, fc = res50(input_P2)
 
        featmap_loss = fake_avgx - avgx
        fc_loss = fake_fc- fc
        ip_loss = torch.norm(featmap_loss) + torch.norm(fc_loss)
        
        fake_correct = accuracy(fake_res50_prediction, idnum) 
        real_correct = accuracy(res50_prediction, idnum)

        model.backward_parameters(ip_loss, fake_correct, real_correct, ssim_score)

        A = 0
        if total_steps % opt.display_freq == 0:
            save_result = total_steps % opt.update_html_freq == 0
            visualizer.display_current_results(model.get_current_visuals(), epoch, save_result, A)

        errors = model.get_current_errors()
        t = (time.time() - iter_start_time) / opt.batchSize
        k_history, v_history = visualizer.print_current_errors(epoch, epoch_iter, errors, t, A)
        v_history_list.append(v_history)

        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, total_steps))
            model.save('latest')
        plot_history(v_history_list)

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save('latest')
        model.save(epoch)
    torch.cuda.empty_cache()

    with torch.no_grad():
        for data in valid_loader:
            model.set_input(data)
            startTime = time.time()
            fake_p2, input_P2, idnum = model.forward_parameters()
            save_result = total_steps % opt.update_html_freq == 0
            A = 1
            visualizer.display_current_results(model.get_current_visuals(), epoch, save_result, A)
            idnum = idnum.long()
            idnum = idnum.squeeze()
            fake_res50_prediction, fake_avgx, fake_maxp = res50(fake_p2)
            res50_prediction, avgx, maxp = res50(input_P2)
            perceptual_loss = mseloss(fake_maxp, maxp)
            fake_correct = accuracy(fake_res50_prediction, idnum) 
            real_correct = accuracy(res50_prediction, idnum)
            vv_history_list.append([fake_correct, real_correct])

    valid_plot_history(vv_history_list)

    torch.cuda.empty_cache()
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    learning_rate = model.update_learning_rate(epoch)
    with open(log_name, 'a') as log_file:
        log_file.write("Epoch :" + str(epoch))
        log_file.write("   learning_rate :" + str(learning_rate))
        log_file.write('\n')

