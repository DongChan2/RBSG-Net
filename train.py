import torch
from torch.optim import *
from argparse import ArgumentParser
import dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

import trainer
from models import *


def main(args,k):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hparam ={'EPOCH':args.epoch,'BATCH_SIZE':args.batch_size,'lr':args.lr,'weight_decay':args.weight_decay,"K":k,"backbone":args.backbone,'DS':args.dataset,'memo':"Proposed"}
    root_path = args.img_path
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
    #@@  dataset
    train_ds = getattr(dataset,args.dataset)(root_path=root_path, transform=transform, K=k,mode='train')
    valid_ds = getattr(dataset,args.dataset)(root_path=root_path, transform=transform, K=k,mode='validation')
  

    train_loader = DataLoader(dataset=train_ds,batch_size=args.batch_size,pin_memory=True, shuffle= True,num_workers=0)
    valid_loader = DataLoader(dataset=valid_ds,batch_size=args.batch_size,pin_memory=True, shuffle= False,num_workers=0)
    
    loaders=[train_loader,valid_loader]
    model = {"classification":getattr(models,args.backbone)(),"segmentation":SegmentationModel.UNet(3,2)}
    model['segmentation'].load_state_dict(seg_ckpt)
    for param in model['segmentation'].parameters():
        param.requires_grad = False
    model = models.RBSG-Net.RBSGNET(model)
    optimizer = Adam([{'params':filter(lambda p: p.requires_grad, model.parameters())}],lr=hparam['lr'],weight_decay=hparam['weight_decay'])
    lr_scheduler= torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=100,eta_min=1e-6)
    trainer.train(model,loaders,optimizer,hparam,device,lr_scheduler=lr_scheduler,save_ckpt=True)


if __name__ =='__main__':
    parser = ArgumentParser()
    parser.add_argument("--batch_size", default=8,type = int)
    parser.add_argument("--epoch", default=30,type = int)
    parser.add_argument("--lr", default=1e-4,type = float)
    parser.add_argument("--weight_decay", default=1e-4,type = float)
    parser.add_argument("--backbone", default='davit_base',type = str)
    parser.add_argument("--dataset", default='SYSU',type = str)
    
    parser.add_argument("--img_path",type = str)
    parser.add_argument("--seg_ckpt_path",type = str)

    args = parser.parse_args()

    if args.dataset == 'RegDB':
        for i in range(1,6):
            main(args,k=i)
    elif args.dataset == 'SYSU': 
        for i in range(1,3):
            main(args,k=i)

        



        

