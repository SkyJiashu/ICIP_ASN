# ICIP_ASN
Code for ICIP 2020 Paper: ATTENTION SELECTIVE NETWORK FOR FACE SYNTHESIS AND POSE-INVARIANTFACE RECOGNITION

Train:

python3 train.py --dataroot ./face_data/ --name ICIP_Weight --model ASN --lambda_GAN 5 --lambda_A 10  --lambda_B 6 --lambda_C 0.01 --lambda_E 0.01 --no_lsgan --n_layers 3 --norm batch --batchSize  12  --resize_or_crop no --gpu_ids 3,2 --which_epoch latest --BP_input_nc 1 --no_flip --which_model_netG ASN --niter 1000 --niter_decay 100000 --checkpoints_dir ./checkpoints --L1_type l1_plus_perL1 --perceptual_layers 5 --DG_ratio 3 --percep_is_l1 1 --n_layers_D 6 --with_D_PP 1 --with_D_PB 1  --display_id 0 --lr 0.002 --lr_decay_iters 1 --lr_policy lambda --display_port 8078

Test:

python3 test.py  --dataroot ./face_data/ --name ICIP_Weight --model ASN --phase test --norm batch --batchSize 12 --resize_or_crop no --gpu_ids 1 --BP_input_nc 1 --no_flip --which_model_netG ASN --checkpoints_dir ./checkpoints --which_epoch 11 --results_dir ./results --display_id 0

Pretrained Model: https://drive.google.com/file/d/1lIG6AsPIGaHCRamVPvUFklb7bN3UzXis/view?usp=sharing

Thanks for the code structure: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

Thanks for the SSIM code: https://github.com/Po-Hsun-Su/pytorch-ssim

