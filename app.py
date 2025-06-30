import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Rock Paper Scissors Classifier",
    page_icon="✊",
    layout="wide"
)

st.markdown("""
    <style>
        /* Main content background */
        section.main {
            background-color: #f0f2f6;
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #2c3e50 !important;
        }
        
        /* Button styling */
        .stButton>button {
            background: linear-gradient(45deg, #4a90e2, #357ab8);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 10px 20px;
            font-weight: 600;
            transition: all 0.3s ease;
            width: 100%;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        
        /* File uploader */
        .st-cq {
            background-color: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        /* Prediction result card */
        .prediction-card {
            background: white;
            border-radius: 12px;
            padding: 24px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.1);
            margin: 20px 0;
        }
    </style>
""", unsafe_allow_html=True)

EMOJI_MAP = {
    "Rock": "✊",
    "Paper": "✋",
    "Scissors": "✌️"
}

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('rps_best.h5')
    return model

def preprocess_image(image):
    img = image.resize((300, 200))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    img_array = img_array / 255.0

    return img_array

def main():
    st.title("🎮 Rock Paper Scissors Classifier")
    st.markdown("### Lade ein Bild hoch und lass die KI raten, ob es sich um Schere, Stein oder Papier handelt!")
    
    if 'prediction' not in st.session_state:
        st.session_state.prediction = None
        st.session_state.confidence = 0
        st.session_state.show_result = False

    with st.sidebar:
        st.title("Info")
        st.markdown("""
        Diese App erkennt Handzeichen für das Spiel Schere-Stein-Papier.
        
        - ✊ Stein
        - ✋ Papier
        - ✌️ Schere
        
        Lade einfach ein Bild hoch und die KI sagt dir, was sie erkennt!
        """)

    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Bild hochladen")
        uploaded_file = st.file_uploader("Wähle ein Bild aus...", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Hochgeladenes Bild', use_container_width=True)
            
            if st.button('🔍 Analyse starten', use_container_width=True):
                with st.spinner('Analysiere...'):
                    try:
                        model = load_model()
                        
                        processed_image = preprocess_image(image)
                        predictions = model.predict(processed_image)
                        
                        class_names = ['Paper', 'Rock', 'Scissors']
                        predicted_class = class_names[np.argmax(predictions[0])]
                        confidence = round(np.max(predictions[0]) * 100, 2)
                        
                        st.session_state.prediction = predicted_class
                        st.session_state.confidence = confidence
                        st.session_state.show_result = True
                        
                    except Exception as e:
                        st.error(f"Ein Fehler ist aufgetreten: {str(e)}")
    
    with col2:
        if st.session_state.show_result:
            st.subheader("🎯 Vorhersage")
            
            with st.container():
                st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                
                st.markdown(f"<h1 style='text-align: center; font-size: 5rem;'>{EMOJI_MAP.get(st.session_state.prediction, '❓')}</h1>", 
                           unsafe_allow_html=True)
                
                st.markdown(f"""
                <div style="text-align: center; margin: 20px 0;">
                    <h2>Das ist {st.session_state.prediction}! ✨</h2>
                    <p style="font-size: 1.2rem;">Sicherheit: <strong>{st.session_state.confidence}%</strong></p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.session_state.confidence > 90:
                    st.success("🎉 Ich bin mir sehr sicher!")
                elif st.session_state.confidence > 70:
                    st.info("👍 Ich bin mir ziemlich sicher!")
                else:
                    st.warning("🤔 Ich bin mir nicht ganz sicher. Versuche ein anderes Bild!")
                
                if st.button('🔄 Neues Bild versuchen', key="try_again", use_container_width=True):
                    st.session_state.show_result = False
                    st.rerun()
                
                st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
