import os 
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
from torch.optim import *
from argparse import ArgumentParser
import dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from torch.utils.data import DataLoader

import trainer
import models

def main(args,k):
    
    CKPT_PATH=args.ckpt_path

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    hparam ={"K":k,"backbone":args.backbone,'DS':args.dataset}
    
    
    root_path = args.data_path
    seg_ckpt_path=args.seg_ckpt_path
    seg_ckpt = torch.load(seg_ckpt_path)['MODEL']
    transform = {'flip': 
                           A.Compose([
                            A.HorizontalFlip(always_apply=True),
                          ]),
                                          
                  "origin": A.Compose([
                                        A.NoOp(always_apply=True)
                            ] )
                }

    test_ds = getattr(dataset,args.dataset)(root_path=root_path, transform=transform, K=k,mode='test')
    test_loader = DataLoader(dataset=test_ds,batch_size=1,pin_memory=True, shuffle= False,num_workers=0)
    

    model = {"classification":getattr(models,args.backbone)(),"segmentation":models.UNet(3,2)}
    model['segmentation'].load_state_dict(seg_ckpt)
    for param in model['segmentation'].parameters():
        param.requires_grad = False
    model = models.RBSG-Net.RBSGNET(model)
    
    history = trainer.test(model,test_loader,device,ckpt_path=CKPT_PATH)
    print(history)
    
if __name__ =='__main__':

    parser = ArgumentParser()
    parser.add_argument("--backbone", default='davit_base',type = str)
    parser.add_argument("--dataset", default='SYSU',type = str)
    
    parser.add_argument("--ckpt_path",type = str)
    parser.add_argument("--seg_ckpt_path",type = str)
    parser.add_argument("--data_path",type = str)
    args = parser.parse_args()
    if args.dataset == 'SYSU':
        for i in range(1,3):
            main(args,i)
    elif args.dataset == 'RegDB':
        for i in range(1,6):
            main(args,i)

    
    

        

