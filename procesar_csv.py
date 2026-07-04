import os
import csv
import time
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from google import genai

# 1. Configurar cliente de Gemini
load_dotenv()
client = genai.Client()

# 2. Envoltura compatible con FAISS
class GeminiEmbeddings(Embeddings):
    def embed_documents(self, texts):
        # Aunque FAISS pida procesar en lista, aquí los manejamos uno a uno con seguridad
        vectores = []
        for texto in texts:
            response = client.models.embed_content(
                model="gemini-embedding-2",
                contents=texto
            )
            if isinstance(response.embeddings, list):
                vectores.append(response.embeddings[0].values)
            else:
                vectores.append(response.embeddings.values)
            # ⏱️ Pausa micro entre textos si FAISS llega a enviar varios juntos
            time.sleep(0.8)
        return vectores

    def embed_query(self, text):
        response = client.models.embed_content(
            model="gemini-embedding-2",
            contents=text
        )
        if isinstance(response.embeddings, list):
            return response.embeddings[0].values
        return response.embeddings.values

    def __call__(self, text):
        return self.embed_query(text)

# Inicializar embeddings
embeddings = GeminiEmbeddings()

ruta_csv = "inventario_de_supermercado_latam.csv"
vectorstore = None
contador = 0

print("📖 Iniciando lectura y vectorización fila por fila...")

# 3. Leer y procesar uno a uno con freno de mano
with open(ruta_csv, mode='r', encoding='latin-1') as archivo:
    # Tu archivo usa punto y coma (;) como separador
    lector = csv.DictReader(archivo, delimiter=';')
    
    for fila in lector:
        # Obtenemos los datos usando un método más seguro (.get)
        # Limpiamos los espacios en blanco alrededor de las claves por si acaso
        fila_limpia = {k.strip() if k else '': v for k, v in fila.items()}
        
        # Extraemos el SKU (que es la primera columna de tu archivo)
        sku = fila_limpia.get('SKU', '')
        
        # 🛑 DETENER EL PROGRAMA: Si llegamos a las filas fantasmas del final,
        # el SKU estará vacío o será None. En lugar de saltar, rompemos el ciclo (break)
        if not sku or sku.strip() == "":
            print(f"🏁 Se detectó el final de los datos reales en la fila {contador + 1}. Guardando base de datos...")
            break
            
        contador += 1
        
        # Mapeo de tus columnas reales
        producto = fila_limpia.get('Descripción', 'N/A')
        marca = fila_limpia.get('Marca', 'N/A')
        categoria = fila_limpia.get('Categoría', 'N/A')
        precio = fila_limpia.get('Precio de Venta Unitario', 'N/A')
        stock = fila_limpia.get('Stock Actual', 'N/A')
        
        # Construir contenido y metadatos
        texto_contenido = f"Producto: {producto}. Marca: {marca}. Categoría: {categoria}."
        metadatos = {
            "sku": sku, 
            "precio": precio, 
            "stock": stock
        }
        
        # Crear un documento único para esta fila
        doc_actual = [Document(page_content=texto_contenido, metadata=metadatos)]
        
        print(f"🔹 Vectorizando fila {contador}: {producto}...")
        
        # Guardar en FAISS de manera inmediata
        if vectorstore is None:
            vectorstore = FAISS.from_documents(doc_actual, embeddings)
        else:
            vectorstore.add_documents(doc_actual)
        
        # ⏱️ Pausa obligatoria para cuidar tu cuota de la API
        time.sleep(0.7)

# 4. Guardar resultado final
if vectorstore:
    vectorstore.save_local("faiss_index_inventario")
    print(f"✅ ¡Proceso terminado con éxito! Se indexaron {contador} productos en 'faiss_index_inventario'.")
else:
    print("❌ No se procesó ningún documento.")