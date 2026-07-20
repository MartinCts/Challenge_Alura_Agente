import os
import streamlit as st
import time
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from google import genai
from main import GeminiEmbeddingsPuro

load_dotenv()
try:
    # Inicializa el cliente usando la GEMINI_API_KEY que cargó el load_dotenv()
    client = genai.Client() 
except Exception as e:
    st.error(f"Error al inicializar el cliente de Gemini: {e}")

# Configuración de la página web
st.set_page_config(page_title="Super agente: Mercado central 24H", page_icon="🤖")
st.title("🤖 ¡Hola! Soy el agente informativo de mercado central 24H")
st.write("Puedo responder dudas del inventario y de las políticas al mismo tiempo.")

# 1. Cargar bases de datos vectoriales usando caché de Streamlit para que sea ultra veloz
@st.cache_resource
def cargar_bases_datos():
    embeddings = GeminiEmbeddingsPuro()
    db_csv = FAISS.load_local("index2_inventario", embeddings, allow_dangerous_deserialization=True)
    db_pdf = FAISS.load_local("pdf_indice_vectorial", embeddings, allow_dangerous_deserialization=True)
    return db_csv, db_pdf

try:
    db_csv, db_pdf = cargar_bases_datos()
    client = genai.Client()
except Exception as e:
    st.error(f"Error al inicializar los servicios: {e}")

# 2. Inicializar el historial de chat en la sesión de la página web
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar los mensajes anteriores del historial
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. Entrada del usuario (Caja de texto de chat web)
if user_query := st.chat_input("Escribe tu pregunta aquí..."):
    # Mostrar el mensaje del usuario en la pantalla
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    # Generar la respuesta del Agente
    with st.chat_message("assistant"):
        with st.spinner("🔍 Buscando en registros e índices vectoriales..."):
            try:
                # Recuperar contextos de ambos mundos
                docs_csv = db_csv.similarity_search(user_query, k=4)
                docs_pdf = db_pdf.similarity_search(user_query, k=3)
                
                time.sleep(0.5) # Pausa de protección de cuota
                
                contexto_csv = "\n".join([f"- {doc.page_content}" for doc in docs_csv])
                contexto_pdf = "\n".join([f"- {doc.page_content}" for doc in docs_pdf])
                
                # Súper Prompt Estructurado
                prompt_final = f"""
Eres un asistente virtual experto e inteligente para el supermercado. Tu misión es responder la pregunta del usuario de forma clara, natural y precisa utilizando ÚNICAMENTE la información provista en los dos contextos de abajo.

Instrucciones de respuesta:
1. Si la pregunta es sobre productos, precios, marcas o stock, básate en el "Contexto del Inventario (CSV)".
2. Si la pregunta es sobre devoluciones, atención, horarios o reglas, básate en el "Contexto de las Políticas (PDF)".
3. Si la pregunta involucra ambos temas, combina la información de manera lógica y coherente.
4. Si la información necesaria no aparece en ninguno de los dos contextos, responde amablemente indicando que no cuentas con esos datos.

---
CONTEXTO DEL INVENTARIO (CSV):
{contexto_csv}

---
CONTEXTO DE LAS POLÍTICAS (PDF):
{contexto_pdf}
---

Pregunta del usuario: {user_query}
Respuesta profesional fundamentada en los datos provistos:
"""
                # Llamada a Gemini
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt_final,
                )
                
                respuesta_final = response.text
                st.markdown(respuesta_final)
                st.session_state.messages.append({"role": "assistant", "content": respuesta_final})
                
            except Exception as e:
                st.error(f"Ocurrió un error al procesar la solicitud: {e}")