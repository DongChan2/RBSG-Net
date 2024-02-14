import torch 
import torch.nn as nn 
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.ops import masks_to_boxes


def split_body(x):
    B,C,H,W = x.size()
    head = x[:,:,:int(H*0.15),:] 
    upper = x[:,:,int(H*0.15):int(H*0.6),:]
    lower = x[:,:,int(H*0.6):,:]

    return head,upper,lower

def extract_roi(inputs,masks,th):
    """
    preds shape == B,C,H,W
    """
    assert len(masks.shape) == 3, "mask의 shape은 B,H,W임"

    
    preds2=inputs.clone()
    targets2=masks.clone()
    output = torch.empty_like(preds2)
    B,C,H,W = preds2.shape
    masks = torch.where(targets2>=th,True,False)
    boxes = masks_to_boxes(masks).to(torch.long)  #(B,4)
    for idx,(pred,box) in enumerate(zip(preds2,boxes)):
        x1,y1,x2,y2=box
        buf = pred[:,y1:y2,x1:x2]
        output[idx]=TF.resize(buf,(H,W))
      
    return output


    
class RBSGNet(nn.Module):
    def __init__(self,model):
        super().__init__()
        self.segmentation = model['segmentation']
        self.classification = model['classification']
        


    def selective_attention(self,preds,ratio):
        pixel_ratio = torch.argmax(preds,dim=1).float()
        pixel_ratio = pixel_ratio.view(pixel_ratio.size(0),-1)
        pixel_ratio = pixel_ratio.mean(dim=1)
        return torch.where((ratio[0]<=pixel_ratio) & (pixel_ratio<=ratio[1]),True,False)
        

    def seg_attention(self,x):
        with torch.no_grad():
            infer = torch.softmax(self.segmentation(x),dim=1).detach()
            mask = infer[:,1,:,:]
            human_mask = extract_roi(infer,mask,th=0.5)
            pooled_infer = extract_roi(infer,mask,th=0.5)
            pooled_x = extract_roi(x,mask,th=0.5)
        head,upper,lower = split_body(human_mask)
        h_attn = self.selective_attention(head,[0.3,0.6]) 
        u_attn = self.selective_attention(upper,[0.6,0.9])
        l_attn = self.selective_attention(lower,[0.4,0.6])      
        fore = torch.ones(x.size(0),1,x.size(2),x.size(3),device='cuda')
        for idx,(h,u,l) in enumerate(zip(h_attn,u_attn,l_attn)):
            if h and u and l:
                fore[idx]=torch.where(pooled_infer[idx,1,:,:].unsqueeze(0)>=0.2,torch.tensor(1.0, dtype=x.dtype,device='cuda'),0.8+pooled_infer[idx,1,:,:].unsqueeze(0))

            else:
                continue
        fore = torch.clamp(fore,0.,1.)
        return fore*pooled_x
    

    def forward(self, x): 
        x=self.seg_attention(x)
        pred = self.classification(x)
        return pred
    
  
    
    


