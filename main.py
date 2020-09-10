import streamlit as st
import time
from PIL import Image
st.title("Adversarial Attacks Visualizer")
import base64
@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(jpg_file):
    bin_str = get_base64_of_bin_file(jpg_file)
    page_bg_img = '''
    <style>
    body {
    background-image: url("data:image/jpg;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

set_png_as_page_bg('bg.jpg')

select =  st.sidebar.radio('Please enter your choice :',('Home','Image Classification','Mathematical Operation'))
if select == 'Image Classification':
    st.header('Please enter the image for classification')
    st.subheader('Image input: ')
    opt = st.selectbox("How do you want to upload the image for classification?\n", ('Please Select', 'Upload image via link', 'Upload image from device'))

    if opt == 'Upload image from device':
    
        file = st.file_uploader('Select', type = ['jpg', 'png'])

        st.set_option('deprecation.showfileUploaderEncoding', False)
        if file is not None:
            image = Image.open(file)
            st.subheader('Displaying the uploaded image')
            st.subheader('Please wait...',)
            time.sleep(2)
            st.image(image, width = 300, caption = 'Uploaded Image')
    
    if opt == 'Upload image via link':
        img = st.text_input('Enter Link for the image')
        if st.button('Submit'):
            import urllib.request
            response = Image.open(urllib.request.urlopen(img))
            st.subheader('Displaying the uploaded image')
            st.subheader('Please wait...',)
            time.sleep(2)
            st.image(response, width = 300, caption = 'Uploaded Image')

    if opt == "Please Select":
        pass

if select == 'Mathematical Operation':       
    num = st.number_input('Enter number to add 1.0 to it: ')
    st.subheader('Result after adding 1.0 is: ')
    st.subheader(num+1.0)

if select == 'Home':
    st.balloons()
    st.success('Hey there! Welcome to Adversarial Attacks Visualizer.')
    

