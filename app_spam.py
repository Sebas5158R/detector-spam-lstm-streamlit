import streamlit as st
import tensorflow as tf
import numpy as np
import json
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(page_title="Filtro de Spam AI", page_icon="📨")

st.title("📨 Detector Inteligente de Spam (Rnn/LSTM)")
st.write("Escribe o pega el contenido de un correo o mensaje de texto en **español** para verificar si es seguro o si es Spam.")

MAX_LEN = 20

# Cargar el modelo y el Tokenizer usando caché de Streamlit
@st.cache_resource
def cargar_recursos():
    modelo = tf.keras.models.load_model('modelo_spam_lstm.keras')
    with open('tokenizer_config.json', 'r', encoding='utf-8') as f:
        tokenizer_data = json.load(f)
        tokenizer = tokenizer_from_json(tokenizer_data)
    return modelo, tokenizer

try:
    model, tokenizer = cargar_recursos()
    st.success("🤖 ¡Filtro Antispam LSTM activo y listo!")
except Exception as e:
    st.error("Asegúrate de colocar 'modelo_spam_lstm.keras' y 'tokenizer_config.json' en la misma carpeta.")
    st.stop()

# Entrada del usuario
texto_usuario = st.text_area("Copia el texto del mensaje aquí abajo:", placeholder="Ejemplo: Hola, te envío los documentos de la tarea...", height=120)

if st.button("Escanear Mensaje"):
    if texto_usuario.strip() == "":
        st.warning("Por favor escribe un mensaje válido para poder analizarlo.")
    else:
        with st.spinner("Analizando la secuencia de palabras..."):
            # 1. Transformar el texto ingresado tal como se hizo en el entrenamiento
            secuencia = tokenizer.texts_to_sequences([texto_usuario])
            datos_padded = pad_sequences(secuencia, maxlen=MAX_LEN)
            
            # 2. Hacer la predicción
            probabilidad = float(model.predict(datos_padded)[0][0])
            
            # Determinar clase
            es_spam = probabilidad >= 0.5
            confianza = probabilidad if es_spam else (1.0 - probabilidad)
            
            # 3. Mostrar resultados visuales limpios e intuitivos
            st.subheader("Resultado del Análisis:")
            if es_spam:
                st.error("🚨 ¡ALERTA! Este mensaje ha sido clasificado como SPAM o ESTAFA.")
                st.markdown("⚠️ *Se recomienda no hacer clic en ningún enlace ni compartir información confidencial.*")
            else:
                st.success("✅ CORREO SEGURO. Este mensaje parece ser un correo normal (Ham).")
                
            st.metric(label="Nivel de Certeza de la Inteligencia Artificial", value=f"{confianza * 100:.2f}%")