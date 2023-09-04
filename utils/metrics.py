import argparse
import numpy as np
import sys
import pandas as pd

import os
main_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, main_folder)

from utils.utils_image import *
from utils.SSIM import compute_ssim, maps
from torchmetrics import PeakSignalNoiseRatio, MeanAbsoluteError
import json

def get_data(
    data_path: str,
    syn_path: str,
    delta: int,

):
    data_dicts = []
    for folder in os.listdir(syn_path):
        if folder == 'bad_results' or folder =='report':
            continue
        else:
            if not folder.endswith('.csv'):
                site = folder.split('_')[1]
                name = folder.split('_')[1]+'_'+folder.split('_')[2]
                fold_name = folder.split('.npy')[0]
                delta = json.load(open(os.path.join(data_path,site,fold_name,'info.json'),'r'))['delta']
                if delta == args.d:
                    if folder.startswith('1_'):
                        data_dicts.append(
                            {
                                "fake": f"{syn_path}/{folder}",
                                "basal": f"{data_path}/{site}/{fold_name}/{name}",
                                "gt" : f"{data_path}/{site}/{fold_name}/{name}",
                                "delta": delta
                            }
                        )
                    else:
                        data_dicts.append(
                            {
                                "fake": f"{syn_path}/{folder}",
                                "basal": f"{data_path}/{site}/{fold_name}/{name}",
                                "gt" : f"{data_path}/{site}/{fold_name}/r_{name}",
                                "delta": delta
                                
                            }
                        )

                        

    print(f"Found {len(data_dicts)} subjects.")
    return data_dicts


def main(args):
    df = pd.DataFrame()
    df_bad = pd.DataFrame()
    print('---------Computing metrics:')

    dataset = get_data(args.data_path,args.syn_path,args.d)

    if not os.path.exists(os.path.join(args.syn_path,'report')):
        os.mkdir(os.path.join(args.syn_path,'report'))

    psnr = PeakSignalNoiseRatio()
    mae = MeanAbsoluteError()

    for i in range(len(dataset)):
        
        folder = dataset[i]['basal'].split('/')[-2]

        print(folder)

        fake = np.load(dataset[i]['fake'])
        gt = np.load(dataset[i]['gt'])

        if fake.shape == (128,128,128):
            fake = np.moveaxis(fake,0,2)
            fake = data_transformation(fake,img_size=None,range=(0,1))
            gt = data_transformation(gt,img_size=128,range=(0,1))
        else:
            fake = data_transformation(fake,img_size=None,range=(0,1))
            gt = data_transformation(gt,img_size=None,range=(0,1))

        mask_fake = np.zeros_like(fake)
        mask_fake[fake > 0.1] = 1

        mask_gt = np.zeros_like(gt)
        mask_gt[gt != 0] = 1

        mask = mask_gt + mask_fake
        mask[mask != 2] = 0
        mask[mask == 2] = 1

        fake[mask == 0] = 0
        gt[mask == 0] = 0

        fake = fake.unsqueeze(0).unsqueeze(0).float()
        gt = gt.unsqueeze(0).unsqueeze(0).float()

        luminosity,contrast,structural,ssim_map,structural_map = compute_ssim(fake,gt,mask,map=True)
        ssim = luminosity * contrast * structural

        dssim_map = (1 - ssim_map)
        dssim_map[mask == 0] = 0

        d_structural_map = (1 - structural_map)
        d_structural_map[mask == 0] = 0
        
    
        if ssim < 0.5:

            df_bad.at[i,'Subject'] = f'{folder}'
            df_bad.at[i,'MAE'] = mae(fake,gt).numpy()
            df_bad.at[i,'PSNR'] = psnr(fake,gt).numpy()
            df_bad.at[i,'SSIM'] = ssim
            df_bad.at[i,'DSSIM'] = (1-ssim)
            df_bad.at[i,'structural_score'] = structural
            df_bad.at[i,'D_structural_score'] = (1-structural)
            df_bad.at[i,'luminosity_score'] = luminosity
            df_bad.at[i,'contrast_score'] = contrast


        df.at[i,'Subject'] = f'{folder}'
        df.at[i,'MAE'] = mae(fake,gt).numpy()
        df.at[i,'PSNR'] = psnr(fake,gt).numpy()
        df.at[i,'SSIM'] = ssim
        df.at[i,'DSSIM'] = (1-ssim)
        df.at[i,'structural_score'] = structural
        df.at[i,'D_structural_score'] = (1-structural)
        df.at[i,'luminosity_score'] = luminosity
        df.at[i,'contrast_score'] = contrast


    dssim_map = maps(dssim_map,method='dssim')
    dssim_map.savefig(f'{args.syn_path}/report/folder_DSSIM_map_{args.d}year.png')

    ssim_map = maps(ssim_map,method='ssim')
    ssim_map.savefig(f'{args.syn_path}/report/folder_SSIM_map_{args.d}year.png')

    d_structural_map = maps(d_structural_map,method='d_structural')
    d_structural_map.savefig(f'{args.syn_path}/report/folder_d_structural_map_{args.d}year.png')
    
    
    means,devs = ['mean'],['dev']
    for metric in df.columns:
        if metric != 'Subject':
            means.append(round(np.mean(df[f'{metric}']),3))
            devs.append(round(np.std(df[f'{metric}']),3))

    df = df.reset_index(drop = True)
    df.loc[len(df)] = means
    df.loc[len(df)] = devs

    print('Done')

    df.to_csv(f'{args.syn_path}/report/metrics_report_{args.d}year.csv',index=False)

    df_bad.to_csv(f'{args.syn_path}/bad_results_report.csv',index=False)


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser(description='Metric evaluation. Scores will be added in the input csv.')
    parser.add_argument('-data_path',type=str,help='Directory of the data path (.npy).',required=True)
    parser.add_argument('-syn_path',type=str,help='Directory of the generated images (.npy).',required=True)
    parser.add_argument('-d',type=int,help='Delta',required=True)

    args = parser.parse_args()

    main(args)