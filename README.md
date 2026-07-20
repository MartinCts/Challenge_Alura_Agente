# Agente informativo del mercado central 24H

Este proyecto consiste en la creación de un agente inteligente que puede responder preguntas a un usuario dado acerca de un comercio ficticio pero que cumple con documentación realista de una empresa real. Dicho proyecto corresponde al challenge individual del curso de Alura LATAM One Oracle Next Education.

## Comportamiento general y tecnologías utilizadas

Como información de entrada se tienen 4 archivos pdf que corresponden a las políticas de dicha empresa ficticia y un archivo csv que indica el inventario en tienda. Utilizando el lenguaje de programación de Python se crea un programa que sea capaz de leer los archivos de entrada, dividir la información en chunks y realizar una vectorización de la información. Paara realizar dicha vectorización se utiliza una herramienta de IA perteneciente a los servicios de Google AI Studio, dicho modelo es el gemini-embedding-2.

Una vez que la información está vectorizada y contenida en las carpetas de indexación se utiliza el modelo gemini-2.5-flash igual perteneciente a Google AI Studio para desarrollar el chat con el usuario, dicho modelo tiene la tarea de entender las preguntas del usuario, buscar la información en los vectores creados previamente y transformar esa información a texto para poder responderle al usuario.

En adición, se utiliza el IDE de Visual Studio Code para editar, desarrollar el código y crear el entorno Git para posteriormente subirlo a un repositorio público en Github para poder ser consultado en la comunidad. Finalmente el proyecto es subido a la plataforma de Oracle Cloud Infrastructure (OCI) para poder ser consultado por más comunidad. Las evidencias del comportamiento del agente igual son mostradas al final de este documento mientras se utiliza en OCI. 

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

## Flujo de trabajo y arquitectura

El proyecto está diseñado bajo la arquitectura **RAG (Generación Aumentada por Recuperación)**. Esto permite combinar la capacidad de redacción natural de un Modelo de Lenguaje Grande (LLM) con el conocimiento específico, privado y actualizado de nuestra empresa, almacenado en archivos locales sin exponer datos confidenciales a reentrenamientos públicos.

A grandes rasgos, el sistema opera en dos fases críticas independientes:

### 1. Fase de Indexación (Inyección de Datos)
Es el proceso que se ejecuta una sola vez para preparar la información antes de que el usuario interactúe con el chat.

* **Extracción de Fuentes:** Los scripts acceden al inventario en formato tabular (`.csv`) con codificación estructurada, y a las normativas de la empresa mediante múltiples archivos de texto (`.pdf`).
* **Segmentación Dinámica (Chunking):** El texto de los PDFs se fragmenta en bloques lógicos utilizando `RecursiveCharacterTextSplitter` con un tamaño de 1,000 caracteres y un solapamiento de 200, garantizando que no se pierda el contexto entre fragmentos vecinos.
* **Vectorización Eficiente:** Cada fila del CSV y fragmento del PDF se envía de forma segura a la API de Google utilizando el modelo de última generación `gemini-embedding-2`. Para respetar estrictamente la cuota gratuita de 15 RPM (Solicitudes por Minuto), se integró un freno de mano mediante pausas controladas (`time.sleep`).
* **Persistencia Local:** Los vectores numéricos resultantes se indexan y guardan localmente en el disco duro dentro de dos directorios de bases de datos vectoriales independientes gestionadas por **FAISS**.

https://encrypted-tbn3.gstatic.com/licensed-image?q=tbn:ANd9GcTuGrBzX4s9Tq9j60HIjvfuZmahlsgzPRDI0orMtOkReTsmAlaZE7RKr95nW2m7Ll8q9TgXQy5F7Emqjro

### 2. Fase de Consulta (Ciclo RAG en Tiempo Real)
Es el flujo dinámico que ocurre cada vez que el usuario escribe en la interfaz de la terminal de `chat.py`:
```text
[Usuario] ──> Pregunta en lenguaje natural (Texto)
│
├──> 1. API Gemini (Embedding) ──> Convierte texto a Vector Matemático
│
├──> 2. Búsqueda Local (FAISS) ──> Encuentra coincidencias en Índices (CSV + PDFs)
│
├──> 3. Construcción del Prompt ─> Combina Pregunta + Contextos Encontrados
│
└──> 4. LLM (gemini-2.5-flash) ──> Genera Respuesta Final Fundamentada
```

1. **Traducción Vectorial:** La pregunta del usuario se convierte instantáneamente en un vector matemático utilizando el mismo modelo de embeddings.
2. **Búsqueda Semántica Paralela:** El motor de **FAISS** analiza la base de datos del inventario (extrayendo los 4 registros más relevantes) y la base de datos de políticas (extrayendo los 3 fragmentos más cercanos) en milisegundos de forma local.
3. **Inyección de Contexto Estructurado:** El script unifica ambas fuentes de información y las empaqueta de forma ordenada dentro de un "Súper Prompt" institucional con instrucciones explícitas de control (evitando alucinaciones si el dato no existe).
4. **Generación de la Respuesta:** El modelo de lenguaje `gemini-2.5-flash` recibe el paquete, interpreta los datos tabulares del CSV junto con las reglas del PDF y redacta una respuesta coherente, natural y 100% fundamentada en la realidad de la empresa.

## Obteniendo la API Key de Google AI Studio

Ya que este programa utiliza un embedding vectorial para indexar la inc¿formación y un modelo de inteligencia artificial para generar las respuestas a las preguntas del usuario es necesario utiliar una herramienta externa para llevar esta labor a cabo. Este proyecto no tiene como objetivo realizar ningún desarrollo de IA, por lo tanto, es necesario obtener este servicio de alguna herramienta externa, en nuestro caso se utilizo el servicio de Google AI Studio, el cual es gratuito aunque con algunas limitantes.

Para obtener la API Key necesaria para correr este proyecto se debe contar con una cuenta en google y seguir los siguientes sencillos pasos:
1. Ingresar a Google AI Studio. Entra al sitio oficial desde tu navegador: aistudio.google.com. Asegúrate de iniciar sesión con tu cuenta de Google (Gmail).
2. Habilitar la autenticación en dos pasos (puedes utilizar la app de Google Authenticator).
3. En la esquina superior izquierda, haz clic en el botón azul que dice "Get API key".
4. Selecciona la opción "Create API key" y elige si quieres crearla en un proyecto de Google Cloud nuevo o en uno que ya tengas existente.
5. En la pantalla aparecerá una ventana flotante con tu clave secreta (una cadena larga de letras y números), copiala y pegala en tu entorno .venv, recuerda no compartirla con nadie.

## Alcance del chat con el agente y del modelo usado

Como se menciono anteriormente el servicio de Google AI Studio que se utiliza para realizar la vectorización y el embedding (gemini-embedding-2) y el modelo para gestionar la información y responder las preguntas del usuario (gemini-2.5-flash) son servicios gratuitos que son suficientes para nuestro proyecto de agente inteligente.

Sin embargo, estos presentan limitaciones tanto a la hora de vectorizar como de contestar. El modelo de vectorización tiene un máximo de 15 peticiones por minutos, esto significa que solo puede procesar 15 partes (chunks) de nuestra información cada minuto. Por ejemplo, la información de uno de los pdf's tiene aproximadamente 80 chunks, eso hace que la vectorización de ese documento tome de 5 a 6 minutos en ejecutarse.

Así que nuestra primera restricción es la cuota de procesamiento, por eso es que el programa tiene un retraso de 4 segundos para cada vectorización. La vectorización de los 4 pdf toma alrededor de 20 minutos y la del .csv unos 15 minutos.

Por otra parte, el modelo para le chat presenta menos inconvenientes. Algunas veces después de que el usuario hace una pregunta aparece un mensaje que indica que el modelo está temporalemente deshabilitado o lleno, esto ocurre porque al ser un servicio gratuito mucha gente lo utiliza y el servidor está totalmente ocupado, para solventar este problema el usuario simplemente debe esperar algunos segundos y volver a ahcer la misma pregunta, por lo general a la segunda oportunidad el agente responde correctamente.

NOTA: igualmente, debido a las limitaciones de los modelos se pueden hacer unas 20 preguntas al agente por día

## Preguntas de prueba hechas al agente y sus respectivas respuestas
### Preguntas como cliente
* ¿En qué pasillo encuentro el arroz?
* ¿Qúe pasta dental es la más barata?
* ¿Tengo estacionamiento grátis?
### Preguntas como empleado
* ¿Cuál es la misión de mercado central 24H?
* ¿Cuál es el stock de azucar?
* Si he trabajado 20 años ¿cuántos días de vacaciones tengo?
