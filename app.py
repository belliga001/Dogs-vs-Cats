import streamlit as st

from PIL import Image, ImageOps

import numpy as np
import tensorflow as tf

st.set_option('deprecation.showfileUploaderEncoding',False)

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model("model.h5")
    return model

model = load_model()

st.markdown("# Aplicativo para detectar os motores defeituosos", unsafe_allow_html = True)
st.markdown("---", unsafe_allow_html = True)
st.markdown("* Insira a imagem do motor e o detector irá avalia-lo ", unsafe_allow_html = True)
st.markdown("* xxxxx", unsafe_allow_html = True)
st.markdown("* xxxxx",  unsafe_allow_html = True)
st.markdown("---", unsafe_allow_html = True)

st.markdown("# Predicão", unsafe_allow_html = True)
st.markdown("---", unsafe_allow_html = True)
st.markdown("* Insira uma imagem e  clique em 'Predicão', o sistema retornarã sua predição", unsafe_allow_html = True)
uploaded_file = st.file_uploader("Imagem do motor.", type=['png','jpeg','jpg'])

if uploaded_file is not None:

    st.write("File uploaded! File type: "+uploaded_file.type+".")
    
    image = Image.open(uploaded_file)
    st.image(image, caption = 'Uploaded file.', use_column_width = True)
    
    bl = st.button("Predicão")
    
    if bl:
        
        size = (150, 150)
        
        image = np.asarray(image)
        image = tf.image.resize(image, [150, 150])
        image = np.asarray(image)
        image = np.reshape(image, (1, 150, 150, 3))
        image = image.copy()
        
        image /= 255
        
        label = model.predict_classes(image)
        
        label = label[0][0]
                          
        if label==1:
            st.markdown("* ### A iamgem não tem defeito.", unsafe_allow_html = True)
        else:
            st.markdown("* ### A imagem tem defeito", unsafe_allow_html = True)
        
        st.markdown("---", unsafe_allow_html = True)
        st.markdown("### Muito obrigado por usar este aplicativo", unsafe_allow_html = True)
        st.markdown("---", unsafe_allow_html = True)
    else:
        print("Por favor pressione o botão e façãouma predição.")
