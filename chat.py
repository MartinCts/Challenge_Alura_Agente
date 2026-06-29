import os
from langchain_community.vectorstores import FAISS
from google import genai

# Importamos la clase que ya funciona desde tu main.py
from main import GeminiEmbeddingsPuro

# 1. Inicializar los embeddings y cargar la base de datos de FAISS
print("Cargando base de datos vectorial...")
embeddings = GeminiEmbeddingsPuro()

# allow_dangerous_deserialization es necesario para cargar archivos FAISS locales de forma segura
db = FAISS.load_local("mi_indice_vectorial", embeddings, allow_dangerous_deserialization=True)

# 2. Inicializar el cliente de Gemini para las respuestas de texto
client = genai.Client()

print("\n🤖 ¡Agente RAG Activo! Escribe tu pregunta (o 'salir' para terminar):\n")

while True:
    user_query = input("👤 Tú: ")
    if user_query.lower() in ["salir", "exit", "quit"]:
        print("🤖 ¡Hasta luego!")
        break
        
    if not user_query.strip():
        continue

    print("🔍 Buscando en los documentos...")
    
    # 3. Recuperar los 3 fragmentos más relevantes del PDF
    docs = db.similarity_search(user_query, k=3)
    contexto = "\n\n".join([doc.page_content for doc in docs])
    
    # 4. Construir el prompt inyectando el contexto (La magia del RAG)
    prompt_final = f"""
Eres un asistente inteligente y experto. Tu misión es responder la pregunta del usuario utilizando ÚNICAMENTE la información provista en el Contexto.
Si la respuesta no se encuentra en el contexto, sé honesto y di amablemente que no dispones de esa información en los documentos cargados.

Contexto:
{contexto}

Pregunta: {user_query}
Respuesta fundamentada:
"""

    try:
        # 5. Pedirle a Gemini que redacte la respuesta final basándose en el PDF
        response = client.models.generate_content(
            model="gemini-2.5-flash", # El modelo rápido ideal que vimos en tu catálogo
            contents=prompt_final,
        )
        print(f"\n🤖 Agente: {response.text}\n")
        print("-" * 50)
        
    except Exception as e:
        print(f"\n❌ Ocurrió un error al consultar a Gemini: {e}\n")