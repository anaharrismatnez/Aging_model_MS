# Ana Harris 06/02/2023
# Training cGAN pix2pix.
# We introduce basal MRI + delta condition
# Generated images will correspond to the follow-up MRI of the patient at timepoint delta.

import os
import sys
main_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, main_folder)

from torch.utils.data import DataLoader
from model_pix2pix import *
import torch
import wandb
from utils.util import *
from utils.utils_image import *
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchmetrics import PeakSignalNoiseRatio



def train(args,train_data,epochs_check,G_optimizer,D_optimizer,device,G,D,models_path):

    train_loader = DataLoader(train_data, batch_size=args.B, num_workers=0, shuffle=True)

    D_scaler = torch.cuda.amp.GradScaler(enabled=True)
    G_scaler = torch.cuda.amp.GradScaler(enabled=True)

    GAN_criterion = torch.nn.BCEWithLogitsLoss()  #inputs image + label (discriminator loss)
    L1_criterion = torch.nn.L1Loss() 

    psnr = PeakSignalNoiseRatio()

    for ep in range(epochs_check,args.e):

        psnr_score = run_D_total_loss = run_GAN = run_D_real = run_D_fake = run_G_total_loss = run_L1 = d_l2 = g_l2 = l_rmse = l_fm = 0

        print(f'epoch: {ep+1}/{args.e}')

        for i,data in tqdm(enumerate(train_loader),total=len(train_loader)):
            
            fup,basal,filename,delta = data
            delta = delta.to(device)

            delta = delta.unsqueeze(1)
            basal = basal.unsqueeze(1)
            basal = basal.to(device).float()
            fup = fup.unsqueeze(1)
            fup = fup.to(device).float()

            mask = generate_mask(basal).cpu()


            # Ground-truth labels real & synth (Depends of the size of the discriminator's output)  
            out_dims = dim_out_layer(args.p,args.pad)
            real_class = torch.ones((basal.shape[0],1,out_dims,out_dims,out_dims)).to(device)
            fake_class = torch.zeros((basal.shape[0],1,out_dims,out_dims,out_dims)).to(device)
            if args.s:
                real_class[real_class==1] = 0.9 # Soft labeling
            
            # Generator forward pass

            fake = G(basal,delta)        

            D_optimizer.zero_grad()

            # Train Discriminator with synthetic images
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    D_fake, D_fake_fmaps = D(torch.cat((fake.detach(), basal), 1))
                D_fake_loss = GAN_criterion(D_fake.detach(), fake_class)
                run_D_fake += D_fake_loss

            # Train Discriminator with real images
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    D_real, D_real_fmaps = D(torch.cat((fup, basal), 1))
                D_real_loss = GAN_criterion(D_real.detach(), real_class)
                run_D_real += D_real_loss

            # L2 regularization
            l2_reg_d = 0
            for name, param in D.named_parameters():
                if 'weight' in name:
                    l2_reg_d += torch.norm(param, p=2) ** 2

            # Discriminator total loss
            with torch.cuda.amp.autocast():
                D_total_loss = ((D_real_loss + (args.vd * l2_reg_d)) + D_fake_loss) * 0.5

            D_scaler.scale(D_total_loss).backward()
            D_scaler.step(D_optimizer)
            D_scaler.update()

            run_D_total_loss += D_total_loss
            d_l2 += args.vd*l2_reg_d

            D_real_fmaps_detached = [fmap.detach() for fmap in D_real_fmaps]
            D_fake_fmaps_detached = [fmap.detach() for fmap in D_fake_fmaps]

            rmse_loss = brain_RMSE_loss(fup.detach(),fake.detach())

            # Feature-matching loss function:
            f_matching_loss = torch.mean(torch.stack([torch.mean(torch.abs(D_real_fmaps_detached[i]-D_fake_fmaps_detached[i])) for i in range(len(D_real_fmaps_detached))]))

            # Train generator 
            G_optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    G_fake,_ = D(torch.cat((fake.detach(),basal),1))
                Loss_G_GAN = GAN_criterion(G_fake.detach(),real_class)
                Loss_G_l1 = L1_criterion(fake,fup)

            # L2 regularization
            l2_reg_g = 0
            for name,param in D.named_parameters():
                if 'weight' in name:
                    l2_reg_g += torch.norm(param, p=2) ** 2

            with torch.cuda.amp.autocast():
                G_loss = Loss_G_GAN + (Loss_G_l1 *args.l) + (args.mu_fm_loss * f_matching_loss) + (args.g * rmse_loss) + (args.vg*l2_reg_g)
                

            G_scaler.scale(G_loss).backward()
            G_scaler.step(G_optimizer)
            G_scaler.update()

            run_L1 += Loss_G_l1
            run_GAN += Loss_G_GAN
            run_G_total_loss += G_loss
            g_l2 += l2_reg_g * args.vg
            l_fm += f_matching_loss * args.mu_fm_loss
            l_rmse += rmse_loss * args.g


            masked_fake = torch.mul(fake.detach().cpu(),mask)
            masked_fup = torch.mul(fup.detach().cpu(),mask)

            psnr_score += psnr(masked_fake,masked_fup)


        psnr_ep = (torch.sum(psnr_score))/len(train_loader)
        print('G_loss:', round(run_G_total_loss.item()/len(train_loader),3), 'L_rmse',round(l_rmse.item()/len(train_loader),3),'PSNR:', round(psnr_ep.item(),3))


        if args.w:
            wandb.log({
                'epoch': ep +1,
                'G_loss': (run_G_total_loss.item()/len(train_loader)), 
                'G_L1': (run_L1.item()/len(train_loader)),
                'GAN_loss': (run_GAN.item()/len(train_loader)), 
                'D_real': (run_D_real.item()/len(train_loader)), 
                'D_fake': (run_D_fake.item()/len(train_loader)), 
                'D_total':(run_D_total_loss.item()/len(train_loader)), 
                'D_l2' : (d_l2.item()/len(train_loader)),
                'G_l2' : (g_l2.item()/len(train_loader)),
                'L_fm' : (l_fm.item()/len(train_loader)),
                'L_rmse' : (l_rmse.item()/len(train_loader)),
                'PSNR':psnr_ep
            }) 

            #Report random image from training in wandb
            img = fake[0].detach().squeeze().cpu().numpy()
            figs=plt.figure(figsize=(6,6))
            plt.axis("off")
            plt.title(f'epoch: {(ep+1)}')
            plt.imshow(img[83,:,:].T, cmap='gray', origin='lower')  # MOVEAXIS(0,2), 1st index is the axial in nibabel 

            plots = wandb.Image(figs)
            plt.close(figs)
            wandb.log({f"example: {filename[0]}, epoch {(ep+1)}": plots}) 
        

        if (ep+1) == 1 or epochs_check != 0:
            best_loss = run_G_total_loss/len(train_loader)

        
        if run_G_total_loss/len(train_loader) < best_loss:
            torch.save({
                'epoch': ep+1,
                'model_state_dict': G.state_dict(),
                'optimizer_state_dict': G_optimizer.state_dict(),
                'gan_loss': run_GAN/len(train_loader),
                'l1_loss' : run_L1/len(train_loader),
                'G_loss' : run_G_total_loss/len(train_loader),
                    }, f'{models_path}/best_model_generator')
            if args.w:
                wandb.save(f'{models_path}/best_model_generator')

            torch.save({
                'epoch': ep+1,
                'model_state_dict': D.state_dict(),
                'optimizer_state_dict': D_optimizer.state_dict(),
                'D_fake': run_D_fake/len(train_loader),
                'D_real' : run_D_real/len(train_loader),
                'D_total' : run_D_total_loss/len(train_loader),
                    }, f'{models_path}/best_model_discriminator')

            print('Checkpoint saved!')

            best_loss = run_G_total_loss/len(train_loader)

        if (ep+1) % 10 == 0:
            torch.save({
                'epoch': ep+1,
                'model_state_dict': G.state_dict(),
                'optimizer_state_dict': G_optimizer.state_dict(),
                'gan_loss': run_GAN/len(train_loader),
                'l1_loss' : run_L1/len(train_loader),
                'G_loss' : run_G_total_loss/len(train_loader),
                    }, f'{models_path}/{ep+1}_generator')


            torch.save({
                'epoch': ep+1,
                'model_state_dict': D.state_dict(),
                'optimizer_state_dict': D_optimizer.state_dict(),
                'D_fake': run_D_fake/len(train_loader),
                'D_real' : run_D_real/len(train_loader),
                'D_total' : run_D_total_loss/len(train_loader),
                    }, f'{models_path}/{ep+1}_discriminator')

        if (ep+1) == args.e and args.w == True:
            wandb.save(f'{models_path}/{ep+1}_generator')


    print('Training completed')  
    
