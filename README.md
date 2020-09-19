## Team:Fluorine
# Adversarial Attack & Classifications Vizualizer.


<a href="https://pytorch.org"><img src="https://upload.wikimedia.org/wikipedia/commons/9/96/Pytorch_logo.png" height="18" ></a><a href="https://github.com/Trusted-AI/adversarial-robustness-toolbox"><img src="https://raw.githubusercontent.com/TanmayKhot/Fluorine/adv/Images/advtool.JPG" height="24"></a>  <a href="https://www.streamlit.io/"><img src="https://raw.githubusercontent.com/TanmayKhot/Fluorine/adv/Images/streamlit.png" height="18"></a>
<br>
<br>
It is a web-based application to analyze the state of the art models based on their susceptibility against various adversarial attacks and also to find a model perfect for user's needs according to the dataset. 
<br>

# Features
- Visualize the classification task by the various state of the art models with your own images.
- The input images can be either uploaded from local storage or image link.
- Test out various adversarial attacks on your image and check the misclassification.
- Visualize, compare and contrast the classification on the image before and after the attack.
- Would help one to analyze which model is better for their personal use.
 
# Classification Models Included
*Imported from <a href="https://pytorch.org/docs/stable/torchvision/models.html">PyTorch models</a>* 
- AlexNet
- DenseNet
- GoogleNet
- InceptionNet
- ResNet50
- VGG16

# Adversarial Attacks Included
*Imported from <a href="https://adversarial-robustness-toolbox.readthedocs.io/en/latest/index.html">Adversarial Robustness Toolbox</a>* 
- DeepFool
- FGSM
- PGD
- AutoAttack
- BasicIterativeMethod

# Steps for usage
- Setting up the Python Environment with dependencies:

        pip install streamlit
        pip install adversarial-robustness-toolbox

- Cloning the Repository: 

        git clone https://github.com/TanmayKhot/Fluorine.git
 - Enter The directory: 

        cd Flourine
- Run the App:

        streamlit run main.py
      
