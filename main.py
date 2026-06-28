import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
# Usamos el cliente moderno de Google que ya tienes instalado
from google import genai

# 1. Extraer el texto del PDF
reader = PdfReader("Política_de_Atencion_al Cliente_y_Devoluciones_Mercado_Central_24h_(Mexico).pdf")
texto_completo = ""
for pagina in reader.pages:
    texto_completo += pagina.extract_text() + "\n"

# 2. Cortar el texto en fragmentos (chunks)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_text(texto_completo)
print(f"✓ Texto dividido con éxito en {len(chunks)} fragmentos.")


# 3. Adaptador formal de Embeddings integrado con el nuevo SDK de Google
class GeminiEmbeddingsPuro(Embeddings):
    def __init__(self):
        self.client = genai.Client()
        self.model_name = "gemini-embedding-2"
    
    def embed_documents(self, texts):
        valores_vectores = []
        for text in texts:
            response = self.client.models.embed_content(
                model=self.model_name,
                contents=text
            )
            # CORRECCIÓN: Usamos .embeddings[0].values porque viene dentro de una lista
            valores_vectores.append(response.embeddings[0].values)
            
        return valores_vectores
    
    def embed_query(self, text):
        response = self.client.models.embed_content(
            model=self.model_name,
            contents=text
        )
        # CORRECCIÓN: También aquí en plural para cuando hagas preguntas
        return response.embeddings[0].values


# Inicializamos nuestro generador de vectores
embeddings = GeminiEmbeddingsPuro()

print("Generando vectores uno a uno con 'gemini-embedding-2' y guardando en FAISS...")

# 4. Crear el Vector Store combinando los textos con los embeddings puros
db = FAISS.from_texts(chunks, embeddings)

# 5. Guardar la base de datos en una carpeta local
db.save_local("mi_indice_vectorial")

print("¡Por fin! Tu base de datos vectorial se ha creado correctamente en la carpeta 'mi_indice_vectorial'.")