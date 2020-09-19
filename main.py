#%%writefile main.py
import streamlit as st
PAGE_CONFIG = {"page_title":"Image Classifier Visualizer with Adv Attacks.io", "layout":"centered"}
st.beta_set_page_config(**PAGE_CONFIG) 
import numpy as np 
import time
import torch
import torchvision.models as models
from torchvision import transforms as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import urllib.request
from art.attacks.evasion import *
from art.estimators.classification import PyTorchClassifier
from utils import *


with open('imagenet_classes.txt') as f:
  classes = [line.strip() for line in f.readlines()]

st.set_option('deprecation.showfileUploaderEncoding', False)
html_temp = '''
    <div style = "background-color: rgba(25,25,112,0.20); padding: 5px; padding-left: 5px; padding-right: 5px">
    <center><h1>Image Classifier</h1></center>
    <center><h2>with</h2></center>
    <center><h1>Adversarial Attack</h1></center>
    <h5></h5>
    </div>
    '''
st.markdown(html_temp, unsafe_allow_html=True)
 
html_temp = '''
    <div>
    <h2></h2>
    <center><h3>Please upload Image for Classification</h3></center>
    </div>
    '''
st.markdown(html_temp, unsafe_allow_html=True)
opt = st.selectbox("How do you want to upload the image for classification?\n", ('Please Select', 'Upload image via link', 'Upload image from device'))
if opt == 'Upload image from device':
    file = st.file_uploader('Select', type = ['jpg', 'png', 'jpeg'])
    st.set_option('deprecation.showfileUploaderEncoding', False)
    if file is not None:
        input_image = Image.open(file)

elif opt == 'Upload image via link':
    img = st.text_input('Enter the Image Address')
    input_image = Image.open(urllib.request.urlopen(img))


model_select = st.selectbox('Please select a Classification Model from the list',('','AlexNet', 'DenseNet', 'GoogLeNet', 'InceptionNet', 'ResNet','VGG'))
attack_select = st.selectbox('Please select an Adversarial Attack from the list:',('None','DeepFool','FGSM','PGD','AutoAttack','BasicIterativeMethod')) 
if attack_select != 'None': 
      epsilon = st.number_input('Enter Epsilon value(e>0):')
      if epsilon < 0:
        warn = st.error("Please enter a positive value for Epsilon")
        time.sleep(3)
        warn.empty()

if st.button('Classify'): 


  image = input_image
  print("Transformed")
  if attack_select == 'None': 
    try:
      if model_select == 'VGG':
         model = models.vgg16(pretrained=True)
      elif model_select == 'AlexNet':
         model = models.alexnet(pretrained=True)
      elif model_select == 'DenseNet':
         model = models.densenet121(pretrained=True)
      elif model_select == 'ResNet':
         model = models.resnet101(pretrained=True)
      elif model_select == 'GoogLeNet':
         model = models.googlenet(pretrained=True)
      elif model_select == 'InceptionNet':
         model = models.inception_v3(pretrained=True)
      st.image(input_image, width = 300, caption = 'Uploaded Image')
      classify(model, image, classes, model_select, 1)
    except:
      pass

  else:
   if model_select == 'VGG':
       model = models.vgg16(pretrained=True)
       
   elif model_select == 'AlexNet':
        model = models.alexnet(pretrained=True)
        
   elif model_select == 'DenseNet':
        model = models.densenet121(pretrained=True)
        
   elif model_select == 'ResNet':
        model = models.resnet101(pretrained=True)

   elif model_select == 'GoogLeNet':
         model = models.googlenet(pretrained=True)

   elif model_select == 'InceptionNet':
         model = models.inception_v3(pretrained=True)

   if model_select != '':
      criterion = nn.CrossEntropyLoss()
      optimizer = optim.Adam(model.parameters(), lr=0.01)
      classifier = PyTorchClassifier(
        model=model,
        clip_values=(0, 1),
        loss=criterion,
        optimizer=optimizer,
        input_shape=(3, 224, 224),
        nb_classes=1000,
      )
   try:
      if attack_select == 'DeepFool':
        attack = DeepFool(classifier, epsilon=epsilon)
        advimg = adversarial(image, model, attack,  attack_select, epsilon)
        displayAdv(input_image,advimg)
        classify(model, advimg, classes,model_select, 1)
      elif attack_select == 'FGSM':
        attack = FastGradientMethod(estimator = classifier, eps = epsilon) 
        advimg = adversarial(image, model, attack,  attack_select, epsilon)
        displayAdv(input_image,advimg)
        classify(model, advimg,classes,model_select, 1)
      elif attack_select == 'PGD':
        attack = ProjectedGradientDescent(classifier, eps = epsilon)
        advimg = adversarial(image, model, attack,  attack_select, epsilon)
        displayAdv(input_image,advimg)
        classify(model, advimg, classes, model_select, 1)
      elif attack_select == 'AutoAttack':
        attack = AutoAttack(estimator = classifier, eps = epsilon)
        advimg = adversarial(image, model, attack,  attack_select, epsilon)
        displayAdv(input_image,advimg)
        classify(model, advimg, classes, model_select, 1)
      elif attack_select == 'BasicIterativeMethod':
        attack = BasicIterativeMethod(classifier, epsilon)
        advimg = adversarial(image, model, attack,  attack_select, epsilon)
        displayAdv(input_image,advimg)
        classify(model, advimg, classes, model_select, 1)
   except:
      pass

   if opt == "Please Select":
        pass

