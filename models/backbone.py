import timm 

def davit_base(pretrained=True):
    model=timm.create_model('davit_base.msft_in1k',pretrained=pretrained,num_classes=2)
    return model
def resnet101(pretrained=True):
    return timm.create_model('resnet101.a1_in1k',pretrained=pretrained,num_classes=2)
def inception_v3(pretrained=True):
    return timm.create_model('inception_v3.tv_in1k',pretrained=pretrained,num_classes=2)
def convnext_base(pretrained=True):
    return timm.create_model('convnext_base.fb_in1k',pretrained=pretrained,num_classes=2)
def swin_base(pretrained=True):
    return timm.create_model('microsoft/swin-base-patch4-window7-224',pretrained=pretrained,num_classes=2)


    

    