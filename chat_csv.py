import os
from langchain_community.vectorstores import FAISS
from google import genai

# Importamos la clase que ya funciona desde tu main.py
from main import GeminiEmbeddingsPuro

# 1. Inicializar los embeddings y cargar la base de datos de FAISS (Asegúrate de que aquí guardaste el CSV)
print("Cargando base de datos vectorial del CSV...")
embeddings = GeminiEmbeddingsPuro()

# CARGA AQUÍ EL ÍNDICE QUE GENERASTE CON TU CSV
db = FAISS.load_local("index2_inventario", embeddings, allow_dangerous_deserialization=True)

# 2. Inicializar el cliente de Gemini para las respuestas de texto
client = genai.Client()

print("\n📊 ¡Agente de Análisis de Datos Activo! Escribe tu pregunta (o 'salir' para terminar):\n")

while True:
    user_query = input("👤 Tú: ")
    if user_query.lower() in ["salir", "exit", "quit"]:
        print("🤖 ¡Hasta luego!")
        break
        
    if not user_query.strip():
        continue

    print("🔍 Analizando filas y registros relevantes...")
    
    # 3. Recuperar los fragmentos/filas más relevantes del CSV
    # Nota: Si tu CSV es grande o requiere comparar varias filas, podrías subir k a 5 o 10.
    docs = db.similarity_search(user_query, k=5) 
    contexto = "\n\n".join([doc.page_content for doc in docs])
    
    # 4. Construir el prompt optimizado para datos tabulares (CSV)
    prompt_final = f"""
Eres un asistente experto en análisis de datos. Tu misión es responder la pregunta del usuario utilizando ÚNICAMENTE la información provista en el Contexto, el cual proviene de un archivo CSV estructurado.

Interpreta las filas y columnas provistas para dar respuestas precisas, sumas, comparaciones o datos específicos si se te solicitan.
Si la respuesta no se puede deducir de los datos del contexto, sé honesto y di amablemente que no dispones de esa información en los registros cargados.

Contexto (Datos del CSV):
{contexto}

Pregunta: {user_query}
Respuesta fundamentada basada en los datos:
"""

    try:
        # 5. Pedirle a Gemini que redacte la respuesta final basándose en el CSV
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt_final,
        )
        print(f"\n🤖 Agente: {response.text}\n")
        print("-" * 50)
        
    except Exception as e:
        print(f"\n❌ Ocurrió un error al consultar a Gemini: {e}\n")