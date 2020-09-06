import streamlit as st
st.title("App Testing")
st.header('Please enter the image for classification')
st.subheader('Image input: ')

from PIL import Image
if st.checkbox('Upload image from device'):
    
    file = st.file_uploader('Select', type = ['jpg', 'png'])

    st.set_option('deprecation.showfileUploaderEncoding', False)
    if file is not None:
        image = Image.open(file)
        st.subheader('Displaying the uploaded image')
        st.image(image, width = 300, caption = 'Uploaded Image')

if st.checkbox('Upload image via link'):
    img = st.text_input('Enter Link for the image')
    
    if st.button('Submit'):
        import urllib.request
        response = Image.open(urllib.request.urlopen(img))
        st.image(response, width = 300, caption = 'Uploaded Image')
        


