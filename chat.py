import os
import time
from langchain_community.vectorstores import FAISS
from google import genai

# Importamos la clase que ya funciona desde tu main.py
from main import GeminiEmbeddingsPuro

# 1. Inicializar los embeddings compartidos
embeddings = GeminiEmbeddingsPuro()

# 2. Cargar ambas bases de datos vectoriales
print("📂 Cargando base de datos del inventario...")
db_csv = FAISS.load_local("index2_inventario", embeddings, allow_dangerous_deserialization=True)

print("📂 Cargando base de datos de las políticas...")
db_pdf = FAISS.load_local("pdf_indice_vectorial", embeddings, allow_dangerous_deserialization=True)

# 3. Inicializar el cliente de Gemini
client = genai.Client()

print("\n🚀 ¡Agente RAG Integrado Activo!")
print("Puedo responder dudas del inventario y de las políticas al mismo tiempo.")
print("Escribe tu pregunta (o 'salir' para terminar):\n")

while True:
    user_query = input("👤 Tú: ")
    if user_query.lower() in ["salir", "exit", "quit"]:
        print("🤖 ¡Hasta luego!")
        break
        
    if not user_query.strip():
        continue

    print("🔍 Consultando en la base de datos...")
    
    # 4. Recuperar información relevante de ambos mundos
    # Buscamos en el inventario (traemos 4 filas del CSV por si requiere comparar o sumar)
    docs_csv = db_csv.similarity_search(user_query, k=4)
    # Buscamos en las políticas (traemos los 3 fragmentos de PDF más relevantes)
    docs_pdf = db_pdf.similarity_search(user_query, k=3)
    
    # Damos una pequeña pausa para no saturar las llamadas de embeddings de las búsquedas
    time.sleep(0.5)
    
    # 5. Estructurar los contextos por separado para que Gemini distinga las fuentes
    contexto_csv = "\n".join([f"- {doc.page_content}" for doc in docs_csv])
    contexto_pdf = "\n".join([f"- {doc.page_content}" for doc in docs_pdf])
    
    # 6. Construir el prompt maestro unificado
    prompt_final = f"""
Eres un asistente virtual experto e inteligente para el supermercado. Tu misión es responder la pregunta del usuario de forma clara, natural y precisa utilizando ÚNICAMENTE la información provista en los dos contextos de abajo.

Instrucciones de respuesta:
1. Si la pregunta es sobre productos, precios, marcas o stock, básate en el "Contexto del Inventario (CSV)".
2. Si la pregunta es sobre devoluciones, atención, horarios o reglas, básate en el "Contexto de las Políticas (PDF)".
3. Si la pregunta involucra ambos temas (ej. "¿puedo devolver un arroz marca X que compré ayer?"), combina la información de manera lógica y coherente en una sola respuesta.
4. Si la información necesaria para responder no aparece en ninguno de los dos contextos, responde amablemente indicando que no cuentas con esos datos en tus registros.

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

    try:
        # 7. Consultar a Gemini con el súper prompt estructurado
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt_final,
        )
        print(f"\n🤖 Agente:\n{response.text}\n")
        print("-" * 50)
        
    except Exception as e:
        if "503" in str(e):
            print("\n⚠️ El servidor de Google está experimentando alta demanda. Por favor, intenta de nuevo en unos segundos.\n")
        else:
            print(f"\n❌ Ocurrió un error al consultar a Gemini: {e}\n")