import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
# losses
from losses.L1_plus_perceptualLoss import L1_plus_perceptualLoss
# from models.vgg import Vgg16
import sys
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import torch.autograd as autograd

class ASModel(BaseModel):
    def name(self):
        return 'ASModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        nb = opt.batchSize
        size = opt.fineSize
        self.input_P1_set = self.Tensor(nb, opt.P_input_nc, size, size)
        self.input_BP1_set = self.Tensor(nb, opt.BP_input_nc, size, size)
        self.input_P2_set = self.Tensor(nb, opt.P_input_nc, size, size)
        self.input_BP2_set = self.Tensor(nb, opt.BP_input_nc, size, size)
        self.input_P2_64_set = self.Tensor(nb, opt.P_input_nc, size, size)
        self.input_P2_32_set = self.Tensor(nb, opt.P_input_nc, size, size)
        self.input_P2_16_set = self.Tensor(nb, opt.P_input_nc, size, size)
        self.posenum_set = self.Tensor(nb, size, 1)
        self.posenum_para_set = self.Tensor(nb, size, 1)
        self.idnum_set = self.Tensor(nb, size, 1)

        self.input_P1_64_set = self.Tensor(nb, opt.P_input_nc, size, size)
        self.input_P1_32_set = self.Tensor(nb, opt.P_input_nc, size, size)

        self.left_eye_set = self.Tensor(nb, opt.P_input_nc, size, size)
        self.right_eye_set = self.Tensor(nb, opt.P_input_nc, size, size)
        self.nose_set = self.Tensor(nb, opt.P_input_nc, size, size)
        self.mouth_set = self.Tensor(nb, opt.P_input_nc, size, size)

        self.target_left_eye_set = self.Tensor(nb, opt.P_input_nc, size, size)
        self.target_right_eye_set = self.Tensor(nb, opt.P_input_nc, size, size)
        self.target_nose_set = self.Tensor(nb, opt.P_input_nc, size, size)
        self.target_mouth_set = self.Tensor(nb, opt.P_input_nc, size, size)

        input_nc = [opt.P_input_nc, opt.BP_input_nc+opt.BP_input_nc]
        self.netG = networks.define_G(input_nc, opt.P_input_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids,
                                        n_downsampling=opt.G_n_downsampling)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            if opt.with_D_PB:
                self.netD_PB = networks.define_D(opt.P_input_nc+opt.BP_input_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids,
                                            not opt.no_dropout_D,
                                            n_downsampling = opt.D_n_downsampling)

            if opt.with_D_PP:
                self.netD_PP = networks.define_D(opt.P_input_nc+opt.P_input_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids,
                                            not opt.no_dropout_D,
                                            n_downsampling = opt.D_n_downsampling)

        if not self.isTrain or opt.continue_train == 1:
            which_epoch = opt.which_epoch
            self.load_network(self.netG, 'netG', which_epoch)
            if self.isTrain:
                if opt.with_D_PB:
                    self.load_network(self.netD_PB, 'netD_PB', which_epoch)
                if opt.with_D_PP:
                    self.load_network(self.netD_PP, 'netD_PP', which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            self.fake_PP_pool = ImagePool(opt.pool_size)
            self.fake_PB_pool = ImagePool(opt.pool_size)

            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)

            if opt.L1_type == 'origin':
                self.criterionL1 = torch.nn.L1Loss()
            elif opt.L1_type == 'l1_plus_perL1':
                self.criterionL1 = L1_plus_perceptualLoss(opt.lambda_A, opt.lambda_B, opt.perceptual_layers, self.gpu_ids, opt.percep_is_l1)
            else:
                raise Excption('Unsurportted type of L1!')

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr= (opt.lr/2) , betas=(opt.beta1, 0.999))
            if opt.with_D_PB:
                self.optimizer_D_PB = torch.optim.Adam(self.netD_PB.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            if opt.with_D_PP:
                self.optimizer_D_PP = torch.optim.Adam(self.netD_PP.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_G)
            if opt.with_D_PB:
                self.optimizers.append(self.optimizer_D_PB)
            if opt.with_D_PP:
                self.optimizers.append(self.optimizer_D_PP)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

    def set_input(self, input):
        input_P1, input_BP1 = input[0], input[2]
        input_P2, input_BP2 = input[1], input[3]
        input_P2_64, input_P2_32 = input[7], input[8]

        posenum = input[4]
        idnum = input[9]

        input_P2_16  = input[10]

        input_P1_32 = input[11]
        input_P1_64 = input[12]

        self.input_P1_64_set.resize_(input_P1_64.size()).copy_(input_P1_64)
        self.input_P1_32_set.resize_(input_P1_32.size()).copy_(input_P1_32)

        self.input_P1_set.resize_(input_P1.size()).copy_(input_P1)
        self.input_BP1_set.resize_(input_BP1.size()).copy_(input_BP1)
        self.input_P2_set.resize_(input_P2.size()).copy_(input_P2)
        self.input_BP2_set.resize_(input_BP2.size()).copy_(input_BP2)
        self.input_P2_64_set.resize_(input_P2_64.size()).copy_(input_P2_64)
        self.input_P2_32_set.resize_(input_P2_32.size()).copy_(input_P2_32)
        self.input_P2_16_set.resize_(input_P2_16.size()).copy_(input_P2_16)
        self.idnum_set.resize_(idnum.size()).copy_(idnum)
        self.posenum_set.resize_(posenum.size()).copy_(posenum)

        self.posenum_para_set.resize_(posenum.size()).copy_(posenum)
        self.idnum_set.resize_(idnum.size()).copy_(idnum)

    def forward(self):

        self.input_P1 = Variable(self.input_P1_set)
        self.input_BP1 = Variable(self.input_BP1_set)

        self.input_P2 = Variable(self.input_P2_set)
        self.input_BP2 = Variable(self.input_BP2_set)

        self.input_P1_64 = Variable(self.input_P1_64_set)
        self.input_P1_32 = Variable(self.input_P1_32_set)

        self.input_P2_64 = Variable(self.input_P2_64_set)
        self.input_P2_32 = Variable(self.input_P2_32_set)
        self.input_P2_16 = Variable(self.input_P2_16_set)
        self.idnum = Variable(self.idnum_set)
        self.posenum = Variable(self.posenum_set)
        self.posenum_para = Variable(self.posenum_para_set)

        self.input_BP1 = self.input_BP1.view(-1,1,128,128)
        self.input_BP2 = self.input_BP2.view(-1,1,128,128)

        G_input = [self.input_P1,
                   torch.cat((self.input_BP1, self.input_BP2), 1)]
 
        self.fake_p2,  self.x_64, self.x_32, self.x_16, before_list, after_list = self.netG(G_input)
                            
        return self.fake_p2, self.input_P2, self.idnum


    def test(self):
        self.input_P1 = Variable(self.input_P1_set)
        self.input_BP1 = Variable(self.input_BP1_set)

        self.input_P2 = Variable(self.input_P2_set)
        self.input_BP2 = Variable(self.input_BP2_set)

        self.input_P1_64 = Variable(self.input_P1_64_set)
        self.input_P1_32 = Variable(self.input_P1_32_set)

        self.input_P2_64 = Variable(self.input_P2_64_set)
        self.input_P2_32 = Variable(self.input_P2_32_set)
        self.input_P2_16 = Variable(self.input_P2_16_set)
        self.idnum = Variable(self.idnum_set)
        self.posenum = Variable(self.posenum_set)
        self.posenum_para = Variable(self.posenum_para_set)

        self.input_BP1 = self.input_BP1.view(-1,1,128,128)
        self.input_BP2 = self.input_BP2.view(-1,1,128,128)

        G_input = [self.input_P1,
                   torch.cat((self.input_BP1, self.input_BP2), 1)]
 
        self.fake_p2,  self.x_64, self.x_32, self.x_16, before_list, after_list = self.netG(G_input)
        
        return self.fake_p2, self.posenum, self.idnum, before_list, after_list, self.input_P1, self.input_P2              


    def Adaptive_symmetrical_loss(self, input_features, target_features, adpative_lambda):
        batchSize, c, h_x, w_x = input_features.size()
        adptive_Scale_Loss = 0.
        for i in range(batchSize):
            adptive_Scale_Loss = adptive_Scale_Loss + F.l1_loss(input_features[i,:,:,:], target_features[i,:,:,:]) *  adpative_lambda[i,0]
        return adptive_Scale_Loss

    # get image paths
    def get_image_paths(self):
        return self.image_paths
    
    def Multi_Scale_Loss(self, input_features, target_features):
        Scale_Loss = F.l1_loss(input_features, target_features)
        return Scale_Loss

    def Total_Variation_Regularization(self, x):
        batchSize, c, h_x, w_x = x.size()
        count_h =  (x.size()[2]-1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        TRV_loss = (h_tv/count_h+w_tv/count_w)/batchSize
        return TRV_loss

    def Adv_loss(self, x):
        batchSize, c, h_x, w_x = x.size()
        count_h =  (x.size()[2]-1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        TRV_loss = (h_tv/count_h+w_tv/count_w)/batchSize
        return TRV_loss

    def mseloss(self, x, y):
        mseloss = torch.nn.MSELoss()
        return mseloss(x, y)

    def flip(self, x, dim):
        indices = [slice(None)] * x.dim()
        indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                    dtype=torch.long, device=x.device)
        return x[tuple(indices)]

    def backward_G(self, perceptual_loss, fake_correct, real_correct, ssim_score):

        i = 0
        for posenum in self.posenum:
            if posenum == 110. or posenum == 240.:
                self.posenum_para[i,0]= 1.00
            elif posenum == 120. or posenum == 10.:
                self.posenum_para[i,0] = 0.96
            elif posenum == 90. or posenum == 200.:
                self.posenum_para[i,0]  =  0.86
            elif posenum == 80. or posenum == 190.:
                self.posenum_para[i,0] = 0.7
            elif posenum == 130. or posenum == 40.:
                self.posenum_para[i,0]  = 0.5
            elif posenum == 140. or posenum == 50.:
                self.posenum_para[i,0] =  0.25
            elif posenum == 81. or posenum  == 191.:
                self.posenum_para[i,0]  = 0.5
            i = i + 1

        self.fake_p2_l = self.fake_p2[: , : , : , :64]
        
        self.fake_p2_r = self.fake_p2[: , : , : , 64:]
       
        self.fake_p2_flip_l = self.flip(self.fake_p2_l, -1)
      
        self.symmetrical_loss = self.Adaptive_symmetrical_loss(self.fake_p2_flip_l, self.fake_p2_r, self.posenum_para) * self.opt.lambda_C

        # self.Local_loss = (self.Multi_Scale_Loss(self.n_left_eye, self.target_left_eye) + self.Multi_Scale_Loss( self.n_right_eye, self.target_right_eye) + self.Multi_Scale_Loss( self.n_nose, self.target_nose) + self.Multi_Scale_Loss(self.n_mouth, self.target_mouth)) / 4 * self.opt.lambda_F

        self.multi_scale_loss = (self.Multi_Scale_Loss(self.x_64, self.input_P2_64) + self.Multi_Scale_Loss(self.x_32, self.input_P2_32) + self.Multi_Scale_Loss(self.x_16, self.input_P2_16))/3 * self.opt.lambda_G

        # self.TRV_loss = self.Total_Variation_Regularization(self.fake_p2) * self.opt.lambda_D
        
        if self.opt.with_D_PB:
            pred_fake_PB = self.netD_PB(torch.cat((self.fake_p2, self.input_BP2), 1))
            pred_real_PB = self.netD_PB(torch.cat((self.input_P2, self.input_BP2), 1))
            self.loss_G_GAN_PB = self.criterionGAN(pred_fake_PB, True) + self.criterionGAN(pred_real_PB, False)

        if self.opt.with_D_PP:
            pred_fake_PP = self.netD_PP(torch.cat((self.fake_p2, self.input_P1), 1))
            pred_real_PP = self.netD_PP(torch.cat((self.input_P2, self.input_P1), 1))
            self.loss_G_GAN_PP = self.criterionGAN(pred_fake_PP, True) + self.criterionGAN(pred_real_PP, False)

        # L1 loss
        if self.opt.L1_type == 'l1_plus_perL1' :
            losses = self.criterionL1(self.fake_p2, self.input_P2)
            self.loss_G_L1 = losses[0]
            self.loss_originL1 = losses[1].data.item()
            self.loss_perceptual = losses[2].data.item()
        else:
            self.loss_G_L1 = self.criterionL1(self.fake_p2, self.input_P2) * self.opt.lambda_A

        self.perceptual_loss = perceptual_loss * self.opt.lambda_E

        pair_L1loss = self.loss_G_L1 + self.perceptual_loss  + self.multi_scale_loss 

        if self.opt.with_D_PB:
            pair_GANloss = self.loss_G_GAN_PB * self.opt.lambda_GAN
            if self.opt.with_D_PP:
                pair_GANloss += self.loss_G_GAN_PP * self.opt.lambda_GAN
                pair_GANloss = pair_GANloss / 2
        else:
            if self.opt.with_D_PP:
                pair_GANloss = self.loss_G_GAN_PP * self.opt.lambda_GAN

        if self.opt.with_D_PB or self.opt.with_D_PP:
            pair_loss = pair_L1loss + pair_GANloss
        else:
            pair_loss = pair_L1loss

        pair_loss.backward()

        # self.TRV_loss = self.TRV_loss.data.item()
        self.perceptual_loss = self.perceptual_loss.data.item()
        self.pair_L1loss = pair_L1loss.data.item()
        self.fake_correct = fake_correct.item()
        self.real_correct = real_correct.item()
        # self.symmetrical_loss = self.symmetrical_loss.item()
        self.multi_scale_loss = self.multi_scale_loss.item()
        # self.Local_loss = self.Local_loss.item()
        self.ssim_score = ssim_score.item()

        if self.opt.with_D_PB or self.opt.with_D_PP:
            self.pair_GANloss = pair_GANloss.data.item()


    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True) 
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False) 
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    # D: take(P, B) as input
    def backward_D_PB(self):
        real_PB = torch.cat((self.input_P2, self.input_BP2), 1)
        # fake_PB = self.fake_PB_pool.query(torch.cat((self.fake_p2, self.input_BP2), 1))
        fake_PB = self.fake_PB_pool.query( torch.cat((self.fake_p2, self.input_BP2), 1).data )
        loss_D_PB = self.backward_D_basic(self.netD_PB, real_PB, fake_PB)
        self.loss_D_PB = loss_D_PB.data.item()

    # D: take(P, P') as input
    def backward_D_PP(self):
        real_PP = torch.cat((self.input_P2, self.input_P1), 1)
        # fake_PP = self.fake_PP_pool.query(torch.cat((self.fake_p2, self.input_P1), 1))
        fake_PP = self.fake_PP_pool.query(torch.cat((self.fake_p2, self.input_P1), 1).data)
        loss_D_PP = self.backward_D_basic(self.netD_PP, real_PP, fake_PP)
        self.loss_D_PP = loss_D_PP.data.item()


    def forward_parameters(self):
        # forward
        output_1 , output_2, output_3 = self.forward()
        return output_1 , output_2, output_3

    def backward_parameters(self, perceptual_loss, fake_correct, real_correct, ssim_score):

        self.optimizer_G.zero_grad()
        self.backward_G(perceptual_loss, fake_correct, real_correct, ssim_score)
        self.optimizer_G.step()

        # D_P
        if self.opt.with_D_PP:
            for i in range(self.opt.DG_ratio):
                self.optimizer_D_PP.zero_grad()
                self.backward_D_PP()
                self.optimizer_D_PP.step()
        # D_BP
        if self.opt.with_D_PB:
            for i in range(self.opt.DG_ratio):
                self.optimizer_D_PB.zero_grad()
                self.backward_D_PB()
                self.optimizer_D_PB.step()


    def get_current_errors(self):
        ret_errors = OrderedDict([ ('pair_L1loss', self.pair_L1loss)])
        if self.opt.with_D_PP:
            ret_errors['D_PP'] = self.loss_D_PP
        if self.opt.with_D_PB:
            ret_errors['D_PB'] = self.loss_D_PB
        if self.opt.with_D_PB or self.opt.with_D_PP:
            ret_errors['pair_GANloss'] = self.pair_GANloss
        if self.opt.L1_type == 'l1_plus_perL1':
            ret_errors['origin_L1'] = self.loss_originL1
            ret_errors['perceptual'] = self.loss_perceptual
        # ret_errors['TRV_loss'] = self.TRV_loss
        # ret_errors['symmetrical_loss'] = self.symmetrical_loss
        ret_errors['Feat_loss'] = self.perceptual_loss
        ret_errors['fake_correct'] = self.fake_correct
        ret_errors['real_correct'] = self.real_correct
        ret_errors['multi_scale_loss'] = self.multi_scale_loss
        # ret_errors['Local_loss'] = self.Local_loss
        ret_errors['ssim_score'] = self.ssim_score

        return ret_errors

    def get_current_visuals(self):
        height, width = self.input_P1.size(2), self.input_P1.size(3)
        input_P1 = util.tensor2im(self.input_P1.data)
        input_P2 = util.tensor2im(self.input_P2.data)


        self.input_BP1 = torch.sum(self.input_BP1, 1, keepdim=True, out=None) 
        self.input_BP2 = torch.sum(self.input_BP2, 1, keepdim=True, out=None) 

        input_BP1 = util.tensor2im(self.input_BP1.data)
        input_BP2 = util.tensor2im(self.input_BP2.data)

        fake_p2 = util.tensor2im(self.fake_p2.data)

        vis = np.zeros((height, width*5, 3)).astype(np.uint8) #h, w, c
        vis[:, :width, :] = input_P1
        vis[:, width:width*2, :] = input_BP1
        vis[:, width*2:width*3, :] = input_P2
        vis[:, width*3:width*4, :] = input_BP2
        vis[:, width*4:, :] = fake_p2

        ret_visuals = OrderedDict([('vis', vis)])

        return ret_visuals

    def save(self, label):
        self.save_network(self.netG,  'netG',  label, self.gpu_ids)
        if self.opt.with_D_PB:
            self.save_network(self.netD_PB,  'netD_PB',  label, self.gpu_ids)
        if self.opt.with_D_PP:
            self.save_network(self.netD_PP, 'netD_PP', label, self.gpu_ids)