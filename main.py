import streamlit as st
import torch
import time
import torchvision.models as models
import urllib.request

from PIL import Image
from art.attacks.evasion import *
from utils import *

PAGE_CONFIG = {"page_title":"Image Classifier Visualizer with Adv Attacks.io", "layout":"centered"} 
st.beta_set_page_config(**PAGE_CONFIG)


st.set_option('deprecation.showfileUploaderEncoding', False)
html_temp = '''
    <div style = "background-color: rgba(25,25,112,0.06); padding: 5px; padding-left: 5px; padding-right: 5px">
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
    try:
      img = st.text_input('Enter the Image Address')
      input_image = Image.open(urllib.request.urlopen(img))
    except:
      if st.button('Submit'):
         show = st.error("Please Enter a valid Image Address!")
         time.sleep(4)
         show.empty()

try:
  if input_image is not None:
    model_select = st.selectbox('Please select a Classification Model from the list',('','AlexNet', 'DenseNet', 'GoogLeNet', 'InceptionNet', 'ResNet','VGG'))
    if model_select != '':
        attack_select = st.selectbox('Please select an Adversarial Attack from the list:',('None','AutoAttack','Basic Iterative Method','Carlini L2 Method','DeepFool','ElasticNet','FGSM','Hop Skip Jump','NewtonFool','Projected Gradient Descent','Universal Perturbation','ZOO Attack')) 
        if attack_select != 'None':
          epsilon = st.number_input('Enter Epsilon value')
          if epsilon < 0:
            warn = st.error("Please enter a positive value for Epsilon")
            time.sleep(3)
            warn.empty()

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
except:
  pass

try:
 if model_select != '':
  if st.button('Run'): 
   try:
    if attack_select == 'None':
      display(input_image)
      classify(model, input_image, model_select, 1)
    else:
      classifier = getClassifier(model)
      advimg = None
      if attack_select == 'DeepFool':
        attack = DeepFool(classifier, epsilon=epsilon)
        advimg = adversarial(input_image, model, attack,  attack_select, epsilon)
      elif attack_select == 'FGSM':
        attack = FastGradientMethod(estimator = classifier, eps = epsilon) 
        advimg = adversarial(input_image, model, attack,  attack_select, epsilon)
      elif attack_select == 'Projected Gradient Descent':
        attack = ProjectedGradientDescent(classifier, eps = epsilon)
        advimg = adversarial(input_image, model, attack,  attack_select, epsilon)
      elif attack_select == 'AutoAttack':
        attack = AutoAttack(estimator = classifier, eps = epsilon)
        advimg = adversarial(input_image, model, attack,  attack_select, epsilon)
      elif attack_select == 'Basic Iterative Method':
        attack = BasicIterativeMethod(classifier, epsilon)
        advimg = adversarial(input_image, model, attack,  attack_select, epsilon)
      elif attack_select == 'ElasticNet':
        attack = ElasticNet(classifier)
        advimg = adversarial(input_image, model, attack,  attack_select, 1)
      elif attack_select == 'Carlini L2 Method':
        attack = CarliniL2Method(classifier)
        advimg = adversarial(input_image, model, attack,  attack_select, 1)
      elif attack_select == 'Hop SKip Jump':
        attack = CarliniL2Method(classifier)
        advimg = adversarial(input_image, model, attack,  attack_select, 1)
      elif attack_select == 'NewtonFool':
        attack = NewtonFool(classifier, eta=epsilon)
        advimg = adversarial(input_image, model, attack,  attack_select, 1)
      elif attack_select == 'Universal Perturbation':
        attack = UniversalPerturbation(classifier = classifier, eps=epsilon)
        advimg = adversarial(input_image, model, attack,  attack_select, epsilon)
      elif attack_select == 'ZOO Attack':
        attack = ZooAttack(classifier = classifier, confidence=epsilon)
        advimg = adversarial(input_image, model, attack,  attack_select, epsilon)
      display(input_image,advimg)
      classify(model, advimg, model_select, 1)

   except:
     pass

except:
  pass





      



