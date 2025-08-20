import cv2 as cv 

import numpy as np 
import torch
from torchvision import transforms
from PIL import Image



class CustomPreprocess:
    def __init__(self,size=(48,48),normalize_mean=0.5,normalize_std=0.5,augment=False):
        self.size=size 
        self.normalize_mean=normalize_mean 
        self.normalize_std=normalize_std

        self.aug_transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2,contrast=0.2),
        ])

    def __call__(self,img):

        
        img = np.array(img)
        img=cv.cvtColor(img,cv.COLOR_BGR2GRAY) 

        clahe=cv.createCLAHE(clipLimit=4)
        img = clahe.apply(img)

        img=cv.equalizeHist(img)

        img=cv.GaussianBlur(img,(5,5),0) 
        img=cv.GaussianBlur(img,(5,5),0)  
        # canny_output=cv.Canny(gaussian_img,80,120)
        # img = canny_output.astype(np.float32) / 255.0  
        img = cv.resize(img, self.size)
        img = Image.fromarray(img)
        if self.augment: 
            img=self.aug_transform(img)
        img=transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[self.normalize_mean], std=[self.normalize_std])(img)


        return img


    
