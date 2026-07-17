Agente del challenge de ALURA LATAM

## 🚀 Guia rápida para ejecutar el proyecto localmente

Sigue estos pasos para configurar y ejecutar el agente RAG en tu computadora:

1. **Clonar el repositorio:**
   ```bash
   git clone <URL_DE_TU_REPOSITORIO>
   cd <NOMBRE_DE_TU_CARPETA>

2. **Crear y activar el entorno virtual (Bash/Git Bash):**
   python -m venv venv
   source venv/Scripts/activate

3. **Instalar dependencias:**
   pip install -r requirements.txt

4. **Configurar la API Key:**
   Duplica el archivo .env.example y renómbralo a .env.
   Abre el archivo .env y reemplaza tu_api_key_aqui por tu clave personal de Google AI Studio.

5. **Preparar la base de datos vectorial (Ejecutar una sola vez):**
   python procesar_csv.py
   python main.py

6. **Iniciar el chat con el agente**
  python chat.py
