import os
import time
from pypdf import PdfReader
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from google import genai

load_dotenv()
try:
    # Inicializa el cliente usando la GEMINI_API_KEY que cargó el load_dotenv()
    client = genai.Client() 
except Exception as e:
    st.error(f"Error al inicializar el cliente de Gemini: {e}")
    
# La clase con control integrado para respetar las 15 peticiones por minuto de Google
class GeminiEmbeddingsPuro(Embeddings):
    def __init__(self):
        self.client = genai.Client()
        self.model_name = "gemini-embedding-2"
    
    def embed_documents(self, texts):
        valores_vectores = []
        for i, text in enumerate(texts, start=1):
            print(f"   Generando embedding {i}/{len(texts)}...")
            response = self.client.models.embed_content(
                model=self.model_name,
                contents=text
            )
            valores_vectores.append(response.embeddings[0].values)
            
            # ⏱️ Pausa obligatoria para aguantar la cuota gratuita con muchos fragmentos
            time.sleep(4.0)
        return valores_vectores
    
    def embed_query(self, text):
        response = self.client.models.embed_content(
            model=self.model_name,
            contents=text
        )
        return response.embeddings[0].values


if __name__ == "__main__":
    # 1. Definimos la lista con tus 4 archivos PDF exactos
    # Pone los nombres reales de tus otros 3 archivos aquí:
    archivos_pdf = [
        "Política_de_Atencion_al Cliente_y_Devoluciones_Mercado_Central_24h_(Mexico).pdf",
        "Manual_de_Proveedores_y_Política_de_Compras_Mercado_Central_24h_(México).pdf",
        "Preguntas _Frecuentes_(FAQ)_Mercado_Central_24h_(México).pdf",
        "Reglamento_Interno_y_Procedimientos_Operativos_Mercado_Central_24h_(México).pdf"
    ]
    
    texto_completo = ""
    
    # Recorremos cada uno de los archivos para extraer e integrar todo el texto
    for archivo in archivos_pdf:
        # Validamos que el archivo exista en la carpeta para evitar caídas
        if os.path.exists(archivo):
            print(f"📖 Leyendo y extrayendo texto de: {archivo}...")
            reader = PdfReader(archivo)
            for num_pag, pagina in enumerate(reader.pages, start=1):
                texto_extracted = pagina.extract_text()
                if texto_extracted:
                    texto_completo += texto_extracted + "\n"
        else:
            print(f"⚠️ Alerta: El archivo '{archivo}' no se encontró en esta carpeta. Lo saltaremos.")

    # 2. Cortar el texto acumulado de todos los PDFs en fragmentos (chunks)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(texto_completo)
    print(f"\n✓ Todo el conocimiento se dividió en {len(chunks)} fragmentos en total.")

    # Inicializamos nuestro generador de vectores
    embeddings = GeminiEmbeddingsPuro()

    print("\n🚀 Enviando fragmentos a Google AI Studio (esto tomará un tiempo por la cuota)...")

    # 4. Crear el Vector Store combinando todos los textos
    db = FAISS.from_texts(chunks, embeddings)

    # 5. Guardar la base de datos unificada en la carpeta local
    db.save_local("pdf_indice_vectorial")

    print("\n✨ ¡Perfecto! Tu base de datos vectorial unificada con los 4 PDFs se ha creado en 'pdf_indice_vectorial'.")