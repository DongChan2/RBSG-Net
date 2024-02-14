import torch
import torch.nn as nn 
import torch.optim as optim 
import numpy as np 
import cv2
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
import random
from tqdm import tqdm
import glob

def gender_labeling_reg(path):
    if (path.split("\\")[-1]).split("_")[0] == 'male':
        return torch.tensor(0)
    elif (path.split("\\")[-1]).split("_")[0] == 'female':
        return torch.tensor(1)
    
def gender_labeling_sysu(path):
    if (path.split("\\")[-1]).split("_")[0] == 'Male':
        return torch.tensor(0)
    elif (path.split("\\")[-1]).split("_")[0] == 'Female':
        return torch.tensor(1)
    
def split_validation_reg(paths):
    train_paths=paths.copy()
    persons=[]
    valid_paths=[]
    for i in paths:
        persons.append(i.split("\\")[-1].split("_")[-1].split(".")[0]) 
    persons=[int(i) for i in persons] 
    persons = np.unique(persons)
    np.random.seed(100)
    sample = np.random.choice(persons,int(len(persons)*0.1),replace=False)
    for path in paths:
        j=int(path.split("\\")[-1].split("_")[-1].split(".")[0])
        if j in sample:
            valid_paths.append(path)
    for i in valid_paths:
        train_paths.remove(i)       
    return train_paths,valid_paths 

def split_validation_sysu(paths):
    train_paths=paths.copy()
    persons=[]
    valid_paths=[]
    for i in paths:
        persons.append(i.split("\\")[-1].split("_")[-2])
    persons=[int(i) for i in persons] 
    persons = np.unique(persons)
    np.random.seed(100)
    sample = np.random.choice(persons,int(len(persons)*0.1),replace=False)
    for path in paths:
        j=int(path.split("\\")[-1].split("_")[-2])
        if j in sample:
            valid_paths.append(path)
    for i in valid_paths:
        train_paths.remove(i)       
    return train_paths,valid_paths 


class DBGenderDB2(Dataset):
    def __init__(self,root_path,transform,img_size=(384,128),mode='train', K=1):
        super().__init__()
        print("\n---[ RegDB init ]---\n")
        self.root_path = root_path
        self.K = K 
        self.mode = mode
        self.train_paths=[]
        self.transform = transform
        self.to_tensor = A.Compose([A.Resize(*img_size),A.ToFloat(255),ToTensorV2()])
        folds=[1,2,3,4,5]
        folds.remove(K)
        for i in folds:
            self.train_paths.extend(glob.glob(self.root_path+f"\\DBGender-DB2\\{i}\\*.bmp"))
        self.train_paths,self.valid_paths = split_validation_reg(self.train_paths)
        if self.mode =='train':  
           self.path = sorted(self.train_paths)
        elif self.mode =='test':
            self.test_paths = sorted(glob.glob(self.root_path+f"\\DBGender-DB2\\{K}\\*.bmp"))
            self.path = sorted(self.test_paths)
        elif self.mode =='validation':
            self.path = sorted(self.valid_paths)
        self.do_transform()
        
    def do_transform(self):
        self.transformed={'image':[],'label':[]}
        if self.mode == 'train':
            for idx,img in enumerate(tqdm(self.path)):
                img = cv2.imread(img,0)
                img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
                label = gender_labeling_reg(self.path[idx])
                self.transformed['image'].extend((lambda ts:[t(image=img)['image'] for t in ts.values()])(self.transform))
                self.transformed['label'].extend([label]*len(self.transform))
            
        else:
            for idx,img in enumerate(tqdm(self.path)):
                img = cv2.imread(img,0)
                img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
                label = gender_labeling_reg(self.path[idx])
                transformed= self.transform['origin'](image = img)
                self.transformed['image'].append(transformed['image'])
                self.transformed['label'].append(label)    
                
    def __len__(self):
        if self.mode == 'train':
            return len(self.path)*len(self.transform)
        else:
            return len(self.path)

    def __getitem__(self, idx):
        img = self.transformed['image'][idx]
        img = self.to_tensor(image=img)['image']
        return img,self.transformed['label'][idx]
    

class SYSU(Dataset):
    def __init__(self,root_path,transform,img_size=(384,128),mode='train', K=1):
        super().__init__()
        print("\n---[ SYSU init ]---\n")
        self.root_path = root_path
        self.K = K 
        self.mode = mode
        self.train_paths=[]
        self.transform = transform
        self.to_tensor = A.Compose([A.Resize(*img_size),A.ToFloat(255),ToTensorV2()])
        folds=[1,2]
        folds.remove(K)
        for i in folds:
            self.train_paths.extend(glob.glob(self.root_path+f"\\SYSU\\{i}\\*.jpg"))
        self.train_paths,self.valid_paths = split_validation_sysu(self.train_paths)
        if self.mode =='train':  
           self.path = sorted(self.train_paths)
        elif self.mode =='test':
            self.test_paths = sorted(glob.glob(self.root_path+f"\\SYSU\\{K}\\*.jpg"))
            self.path = self.test_paths
        elif self.mode =='validation':
            self.path = self.valid_paths
        self.do_transform()
        
    def do_transform(self):
        self.transformed={'image':[],'label':[]}
        if self.mode == 'train':
            for idx,img in enumerate(tqdm(self.path)):
                img = cv2.imread(img,0)
                img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
                label = gender_labeling_sysu(self.path[idx])
                self.transformed['image'].extend((lambda ts:[t(image=img)['image'] for t in ts.values()])(self.transform))
                self.transformed['label'].extend([label]*len(self.transform))
            
        else:
            for idx,img in enumerate(tqdm(self.path)):
                img = cv2.imread(img,0)
                img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
                label = gender_labeling_sysu(self.path[idx])
                transformed= self.transform['origin'](image = img)
                self.transformed['image'].append(transformed['image'])
                self.transformed['label'].append(label)    
                
    def __len__(self):
        if self.mode == 'train':
            return len(self.path)*len(self.transform)
        else:
            return len(self.path)

    def __getitem__(self, idx):
        img = self.transformed['image'][idx]
        img = self.to_tensor(image=img)['image']
        return img,self.transformed['label'][idx]