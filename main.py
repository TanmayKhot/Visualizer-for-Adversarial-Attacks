import streamlit as st
st.title("App Testing")
st.header('Working with images')
st.subheader('Image input')

from PIL import Image
if st.checkbox('To upload image for clasification, check the box'):
    if st.button("Choose an image..."):
        file = st.file_uploader('Select', type = 'jpg')

        st.set_option('deprecation.showfileUploaderEncoding', False)
        if file is not None:
            image = Image.open(file)
            st.subheader('Displaying the uploaded image')
            st.image(image, width = 300, caption = 'Uploaded Image')