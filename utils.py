from torchvision import transforms as T
import torch
import torch.nn.functional as F
import streamlit as st
from art.attacks.evasion import *
from art.estimators.classification import PyTorchClassifier
import numpy as np 
from PIL import Image
import itertools


with open('imagenet_classes.txt') as f:
  classes = [line.strip() for line in f.readlines()]


def preprocess(image):
    transform = T.Compose([            
    T.Resize(256),                    
    T.CenterCrop(224),                
    T.ToTensor(),                     
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
    ])
    transformed_img = transform(image)
    img = transformed_img.unsqueeze(0)
    return img

def displayAdv(input_image, advimg):
    display_trans = T.Compose([
    T.Resize(256),
    T.CenterCrop(224)])
    orgimg = display_trans(input_image)
    images=[orgimg, advimg]
    captions=['Uploaded Image', 'Image after Adversarial Attack']
    image_iter = paginator("",images)
    index, fimg = map(list, zip(*image_iter))
    st.image(fimg, width = 328, caption = captions)
      

def paginator(label, items, items_per_page=2, on_sidebar=True):
    max_index =  items_per_page
    return itertools.islice(enumerate(items), max_index)


def classify(model, image, classes, model_select, choice):
  image = preprocess(image)
  model.eval()  
  if torch.cuda.is_available():
    image = image.to('cuda')
    model.to('cuda')
  with torch.no_grad():
    output = model(image)
  _, index = torch.max(output, 1)
  percentage = F.softmax(output, dim=1)[0] * 100 
  st.success('Hey! {} classified your image as a {} with confidence {:.3f}'.format(model_select, classes[index[0]], percentage[index[0]].item()))
  

def adversarial(image, model, attack, attack_select, epsilon): 
  image = preprocess(image)
  advx = attack.generate(x=image)
  rescaled = (advx - advx.min()) / (advx.max() - advx.min())
  rescaled = (255*np.transpose(np.squeeze(rescaled))).astype(np.uint8)
  rescaled = np.moveaxis(rescaled ,0, 1)
  img = Image.fromarray(rescaled)
  return img





