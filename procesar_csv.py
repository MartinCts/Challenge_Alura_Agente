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
            
            # Pausa de seguridad
            time.sleep(4)
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

# Volvemos a usar utf-8-sig para evitar caracteres raros ocultos de Excel
ruta_csv = "inventario_de_supermercado_latam.csv"
vectorstore = None
contador = 0

print("📖 Iniciando lectura y vectorización dinámica fila por fila...")

with open(ruta_csv, mode='r', encoding='utf-8-sig') as archivo:
    lector = csv.DictReader(archivo, delimiter=';')
    
    for fila in lector:
        # Limpiamos los espacios en blanco alrededor de las columnas (claves) y de los valores
        fila_limpia = {str(k).strip(): str(v).strip() for k, v in fila.items() if k is not None}
        
        # Extraemos el SKU usando tu clave exacta
        sku = fila_limpia.get('SKU', '')
        
        # 🛑 DETENER EL PROGRAMA: Si llegamos a filas vacías al final
        if not sku or sku.strip() == "":
            print(f"🏁 Se detectó el final de los datos reales en la fila {contador + 1}. Guardando base de datos...")
            break
            
        contador += 1
        
        # 🚀 MAPEO DINÁMICO DE TODAS LAS COLUMNAS:
        # Recorremos cada columna presente en esta fila específica del CSV de forma automática
        partes_del_texto = []
        for columna, valor in fila_limpia.items():
            # Filtramos para meter solo datos útiles al agente
            if valor and valor.lower() != "n/a":
                partes_del_texto.append(f"{columna}: {valor}")
        
        # Juntamos todas las características en un solo texto plano descriptivo
        texto_contenido = ", ".join(partes_del_texto) + "."
        
        # El nombre del producto para nuestro print en consola
        producto_print = fila_limpia.get('Descripción', f'Producto #{contador}')
        
        # Creamos el Documento para FAISS (pasando la info completa en el contenido principal)
        doc_actual = [Document(page_content=texto_contenido)]
        
        print(f"🔹 Vectorizando fila {contador}: {producto_print}...")
        
        # Guardar en FAISS de manera inmediata
        if vectorstore is None:
            vectorstore = FAISS.from_documents(doc_actual, embeddings)
        else:
            vectorstore.add_documents(doc_actual)
        
        # ⏱️ Pausa obligatoria por fila para cuidar tu cuota (puedes subirlo a 4 si te da Error 429)
        # time.sleep(0.7)

# 4. Guardar resultado final
if vectorstore:
    # Recuerda apuntar tu chat.py a este mismo nombre de carpeta
    vectorstore.save_local("index2_inventario")
    print(f"✅ ¡Proceso terminado con éxito! Se indexaron {contador} productos con TODAS sus columnas en 'faiss_index_inventario'.")
else:
    print("❌ No se procesó ningún documento.")