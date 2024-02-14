import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as TF
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve,confusion_matrix,ConfusionMatrixDisplay
from ray import tune
from tqdm import tqdm 
import time
from datetime import datetime
import numpy as np 
import random 
import matplotlib.pyplot as plt 


from torch.utils.tensorboard import SummaryWriter
from mp_dataloader import DataLoader_multi_worker_FIX
import models
import csv


def save_checkpoint(epoch,model,optimizer,eer,name,scheduler=None):
    
    PATH= "./"
    NEW_PATH=PATH+name
    os.makedirs(NEW_PATH,exist_ok=True)
    ckpt={"MODEL":model.classification.state_dict(),
          "OPTIMIZER":optimizer.state_dict(),
          'EPOCH':epoch,
          "NAME":name}
    if scheduler is not None:
        ckpt.update({"SCHEDULER_STATE":scheduler.state_dict()})
  
    eer = eer*100
    torch.save(ckpt,NEW_PATH+f"/Epoch_{epoch},EER_{eer:.3f}.pth.tar")
    print(f"Model Saved,when Epoch:{epoch}")


def compute_eer(preds,targets):
    

    fpr, tpr, thresholds = roc_curve(targets, preds, pos_label=1,drop_intermediate=True)

    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
   
    return eer


def compute_accuracy(preds,target):
    preds = (torch.argmax(preds,dim=1)).view(-1)
    target=target.view(-1)
    accuracy = ((preds==target).double().sum())/len(target)
    
    return accuracy
    

def _train_one_step(model,data,optimizer,device,**kwargs):
    logger = kwargs['logger']
    criterion = kwargs['criterion']
    x,y = data
    x,y = x.to(device),y.to(device)

    pred = model(x)

    optimizer.zero_grad()
       
    loss =  criterion['CE'](pred,y)
    accuracy = compute_accuracy(pred,y)
    
    logger.add_scalar("loss/step",loss.item(),kwargs['iter']) #@@@
    logger.add_scalar("accuracy/step",accuracy.item(),kwargs['iter']) #@@@
    
    loss.backward()
    optimizer.step()

    
    return {'loss':loss.item(),'accuracy':accuracy.item()}
    

def _train_one_epoch(model,dataloader,optimizer,device,**kwargs):

    model.train()
    model.segmentation.eval()
    total_loss = 0
    total_accuracy = 0
    
    for batch_index,data in enumerate(tqdm(dataloader)):
        history = _train_one_step(model,data,optimizer,device,logger=kwargs['logger'],iter=(len(dataloader)*(kwargs['epoch_index'])+(batch_index)), criterion = kwargs['criterion'])
        total_loss += history['loss']
        total_accuracy += history['accuracy']

    return {'loss':total_loss,'accuracy':total_accuracy}



def _validate_one_step(model,data,device,*args,**kwargs):
    logger = kwargs['logger']
    criterion = kwargs['criterion']
    x,y = data
    x=x.to(device)
    y=y.to(device)
    
    with torch.no_grad():
        pred = model(x)
    loss =  criterion['CE'](pred,y)
    accuracy = compute_accuracy(pred,y)
    logger.add_scalar("loss/step",loss.item(),kwargs['iter'])  #@@@
    logger.add_scalar("accuracy/step",accuracy.item(),kwargs['iter']) #@@@
    
    return {'loss':loss.item(),'accuracy':accuracy.item()}
    

def _validate_one_epoch(model,dataloader,device,**kwargs):
    model.eval()
    total_loss = 0
    total_accuracy = 0
    for batch_index,data in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            history = _validate_one_step(model,data,device,logger=kwargs['logger'],iter=(len(dataloader)*(kwargs['epoch_index'])+(batch_index)), criterion = kwargs['criterion'])
        total_loss += history['loss']
        total_accuracy += history['accuracy']
    
    return {'loss':total_loss,'accuracy':total_accuracy}



def _test_one_step(model,data,device,*args,**kwargs):
    x,y=data
    x=x.to(device)
    y=y.to(device)
    with torch.no_grad():
        pred =model(x)
        
    return pred,y
    

def _test_one_epoch(model,dataloader,device,*args,**kwargs):
    model.eval()
    criterion=kwargs['criterion']
    total_loss =0
    total_acc=0
    total_pred=torch.tensor([])
    total_y=torch.tensor([])

    for data in tqdm(dataloader):
        with torch.no_grad():
            pred,y = _test_one_step(model,data,device)
        total_loss+=criterion['CE'](pred,y)
        total_acc+=compute_accuracy(pred,y)
        total_pred=torch.cat([total_pred,torch.softmax(pred.detach().cpu(),dim=1)[:,1].view(-1)],dim=0)
        total_y=torch.cat([total_y,y.detach().cpu().view(-1)],dim=0)
    eer = compute_eer(total_pred,total_y)
    acc = total_acc/len(dataloader)
    loss = total_loss/len(dataloader)

    return {'eer':eer,'loss':loss.item(),'acc':acc.item()} 
        

def train(model,loaders,optimizer,hparam,device,lr_scheduler=None,save_ckpt=True):
    dataloader,valid_dataloader = loaders
    print(f"Training Start\tK={hparam['K']}")
    print("="*100)
    name_list=[]
    for k,v in hparam.items():
        name_list.append(k + ";"+ str(v))
    
    t= datetime.today().strftime("%m_%d_%H;%M;%S")
    name =",".join(name_list)+f"/{t}"
    
    train_logger = SummaryWriter(log_dir = f"./{name}/train")
    valid_logger = SummaryWriter(log_dir = f"./{name}/validation")
    test_logger = SummaryWriter(log_dir = f"./{name}/test")

    model.to(device)
    
    epochs = hparam['EPOCH']
    criterion = {'CE':nn.CrossEntropyLoss()}

    
    for idx,epoch in (enumerate(range(epochs))):
       
        print(f"\rEpoch :{idx+1}/{epochs}")
        history = _train_one_epoch(model,dataloader,optimizer,device,epoch_index=idx,logger=train_logger,criterion=criterion)
        epoch_loss = history['loss'] / len(dataloader)
        epoch_accuracy = history['accuracy'] / len(dataloader)
        train_logger.add_scalar("loss/epoch",epoch_loss,idx)
        train_logger.add_scalar("accuracy/epoch",epoch_accuracy,idx)
        
        val_history = _validate_one_epoch(model,valid_dataloader,device,epoch_index=idx,logger=valid_logger,criterion=criterion)
        epoch_val_loss = val_history['loss'] / len(valid_dataloader)
        epoch_val_accuracy = val_history['accuracy'] / len(valid_dataloader)
        valid_logger.add_scalar("loss/epoch",epoch_val_loss,idx)
        valid_logger.add_scalar("accuracy/epoch",epoch_val_accuracy,idx)
        torch.cuda.empty_cache()
        
        print(f"loss:{epoch_loss},acc:{epoch_accuracy},valid_loss:{epoch_val_loss},valid_acc:{epoch_val_accuracy}")
        print(test_history)
        
        if lr_scheduler is not None:
            lr_scheduler.step()
        if save_ckpt:
            save_checkpoint(epoch,model,optimizer,test_history['eer'],name=name,scheduler=lr_scheduler)
        history.clear()
        val_history.clear()
        test_history.clear()
        
    train_logger.close()
    valid_logger.close()
    test_logger.close()
    print('Training End')
    print("="*100)



def test(model,dataloader,device,ckpt_path):
    print("Testing Start")
    print("="*20)
    
    ckpt = torch.load(ckpt_path)
    ckpt_model = ckpt['MODEL']
    model.classification.load_state_dict(ckpt_model)
    model.to(device)
    criterion = {'CE':nn.CrossEntropyLoss()}
    test_history = _test_one_epoch(model,dataloader,device,criterion=criterion)

    print(f"eer:{test_history['eer']},acc:{test_history['acc']},loss:{test_history['loss']}")
    print("Testing End")
    print("="*20)
    return test_history

