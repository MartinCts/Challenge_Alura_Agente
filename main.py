from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Leer el PDF con la librería nativa (es súper rápida y no se traba)
reader = PdfReader("Política_de_Atencion_al Cliente_y_Devoluciones_Mercado_Central_24h_(Mexico).pdf")
texto_completo = ""

for pagina in reader.pages:
    texto_completo += pagina.extract_text() + "\n"

# 2. Configurar el cortador de texto de LangChain
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# 3. Crear los chunks desde el texto extraído
chunks = text_splitter.split_text(texto_completo)

print(f"El documento se dividió en {len(chunks)} fragmentos.")