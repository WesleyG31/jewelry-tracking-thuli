
Alcance: ¿qué parte resolveré?
Seguimiento de dedos y anillos, Es el núcleo técnico del reto. Tienes experiencia con YOLO y tracking.

Supuestos: ¿qué limitaciones tengo?
1 week
Adaptabilidad mobile,Documentamos cómo el pipeline puede funcionar en tiempo real y adaptarse.

Estrategia: ¿profundizo o abarco?

Primero empezaré con el tracking de anillos y dedos

Stack: ¿qué herramientas y modelos usaré?

Por el tiempo, utilizaré un modelo ya entrenado como mediapipe.
Python	Lenguaje principal	Tu dominio + comunidad + rapidez
OpenCV	Procesamiento de imágenes, video, visualización	Ligero, flexible
MediaPipe Hands	Detección de mano y dedos	Preciso, en tiempo real, ideal para mobile-first
YOLOv5 (opcional)	Detección más precisa de anillos	Solo si lo básico con heurística falla
Google Colab / Local	Desarrollo inicial	Rápido de montar y compartir
Draw.io	Diagramas de arquitectura	Para documentación técnica
Google Drive	Envío de entregables	Solicitado por la empresa


Realice el hand tracking con mediapipe, ahora puedo ver los puntos de la mano (articulaciones) (me piden seguimiento de dedos y joyas)
Ahora el siguiente paso es detectar el anillo: me piden centrarme en el seguimiento y localizacion preciso del anillo.

-Primero sera por landmarks de mediapipe
🧠 ¿Qué hará ring_detector.py?
Recibirá los landmarks de una mano y el frame de imagen.


https://github.com/user-attachments/assets/e1dc3017-fae5-4062-9adf-3b18c2bc83aa


Identificará las zonas de los dedos donde normalmente se usan anillos.

Extraerá esas regiones.

Aplicará una detección basada en color o contorno.

Devolverá True si encuentra algo con características de un anillo.

ELegi 4 colores distintos porque son normales el dorado y plateado pero yo tengo azul y negro
