EMOTIONNET - Web App para Detección de Emociones en Texto
==========================================================

Descripción
-----------
EmotionNet es una aplicación web desarrollada con Django que permite detectar emociones en frases en inglés utilizando un modelo de Machine Learning. Se entrena con scikit-learn y convierte texto en vectores usando TfidfVectorizer. Las emociones que puede detectar incluyen: joy, anger, sadness, entre otras.

Estructura del Proyecto
------------------------
![imagen](./assets/estructura.png)

Requisitos
----------
- Python 3.10.0
- wget

Instalación
-----------
1. Clonar el repositorio:
```
   git clone https://github.com/VicDel05/emotionNet.git
   cd emotionNet
```

2. Crear entorno virtual:
MacOS y Linux
```
   python -m venv venv
   source venv/bin/activate 
```
Windows
```
   python -m venv venv
   .\venv\Scripts\activate
```

3. Instalar dependencias:
```
   pip install -r requirements.txt
```

Entrenar el Modelo y obtener datos
-----------------------------
Los modelos y datos no se encuentran en este repositorio debido a su peso, por lo cual debes seguir las siguientes instrucciones para tener los datos en tu equipo:

1. Entrar al siguiente link del [Drive](https://drive.google.com/drive/folders/1kiOj-X2khVPAnzWWziV50xLcijOXlBEJ?usp=sharing) para descargar los archivos con los datos.

2. Crear una carpeta llamada Data/full_dataset y dentro colocar los archivos descargados.

3. El siguiente paso es ejecutar el comando `python -m src.train` para iniciar con el entrenamiento de la neurona, este proceso puede demorar algo de tiempo.

4. Después de entrenar el modelo ejcutar el comando para realizar las pruebas de predicción `python -m src.predict`, esto activara la interacción por terminal para ingresar oraciones y ver la predicción de la red nuronal.

Ejecutar la Aplicación Web
--------------------------
Desde la carpeta web:

1. Aplicar migraciones:
   python manage.py migrate

2. Iniciar el servidor:
   python manage.py runserver

3. Ir al navegador y visitar:
   http://127.0.0.1:8000/

Funcionamiento
--------------
1. El usuario ingresa una frase en inglés.
2. El texto se vectoriza usando TfidfVectorizer.
3. El modelo predice la emoción.
4. Se muestra el resultado en la interfaz web.

Contenido de requirements.txt
-----------------------------
```
pandas==2.2.3
scikit-learn==1.6.1
django==5.2.1
tensorflow==2.16.2
numpy==1.26.4
joblib==1.5.0
```

Mejoras Futuras
---------------
- Soporte para español
- Aplicación web en Django
- API REST con Django REST Framework
- Interfaz visual mejorada con Bootstrap o Tailwind
- Registro de usuarios e historial de análisis
- Clasificación multietiqueta (más de una emoción por texto)

Autor
-----
Desarrollado por: Victor Delgado
Contacto: victor.delgadobau@gmail.com

Licencia
--------
Este proyecto está licenciado bajo la Licencia MIT.
