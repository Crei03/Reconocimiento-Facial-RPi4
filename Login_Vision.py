#--------------------------------------Importamos librerias--------------------------------------------

from tkinter import *
import os
import unicodedata
import cv2
from matplotlib import pyplot
from mtcnn import MTCNN
import numpy as np
import RPi.GPIO as GPIO
import time
import sounddevice as sd
from vosk import Model, KaldiRecognizer
import unicodedata

THEME_COLOR = "#2C3E50"  # Dark blue-gray
ACCENT_COLOR = "#3498DB"  # Light blue
TEXT_COLOR = "#ECF0F1"   # Light gray
BUTTON_COLOR = "#2980B9"  # Slightly darker blue
SUCCESS_COLOR = "#27AE60"  # Green
ERROR_COLOR = "#E74C3C"   # Red
FONT_MAIN = ("Helvetica", 12)
BUTTON_FONT = ("Helvetica", 11)

# Agregar después de las constantes existentes
MESSAGE_STYLES = {
    "success": {
        "bg": SUCCESS_COLOR,
        "fg": TEXT_COLOR,
        "font": ("Helvetica", 12, "bold"),
        "pady": 15
    },
    "error": {
        "bg": ERROR_COLOR,
        "fg": TEXT_COLOR,
        "font": ("Helvetica", 12, "bold"),
        "pady": 15
    }
}

def show_message(parent, text, type="success"):
    """Función para mostrar mensajes estilizados"""
    frame = Frame(parent, bg=MESSAGE_STYLES[type]["bg"], pady=10)
    frame.pack(fill=X, padx=20, pady=MESSAGE_STYLES[type]["pady"])
    
    Label(frame, text=text, 
          bg=MESSAGE_STYLES[type]["bg"],
          fg=MESSAGE_STYLES[type]["fg"],
          font=MESSAGE_STYLES[type]["font"]).pack()
    
    # Auto-destruir el mensaje después de 3 segundos
    parent.after(3000, frame.destroy)

def custom_button(parent, text, command, width=20):
    btn = Button(parent, text=text, command=command, width=width,
                bg=BUTTON_COLOR, fg=TEXT_COLOR, font=BUTTON_FONT,
                relief='flat', activebackground=ACCENT_COLOR, activeforeground=TEXT_COLOR)
    btn.bind('<Enter>', lambda e: btn.config(background=ACCENT_COLOR))
    btn.bind('<Leave>', lambda e: btn.config(background=BUTTON_COLOR))
    return btn

# Agregar después de custom_button
def small_exit_button(parent, command):
    btn = Button(parent, text="Regresar", command=command, 
                bg=ERROR_COLOR, fg=TEXT_COLOR, font=("Helvetica", 9),
                width=8, height=1, relief='flat')
    btn.bind('<Enter>', lambda e: btn.config(background='#c0392b'))  # Rojo más oscuro al hover
    btn.bind('<Leave>', lambda e: btn.config(background=ERROR_COLOR))
    return btn

# Vosk modelo
modelo_ruta = "./vosk-model-small-es-0.42"  
modelo = Model(modelo_ruta)
recognizer = KaldiRecognizer(modelo, 16000)

def transcribe_audio():
    """Función para transcribir audio usando Vosk"""
    duration = 5  # duración de la grabación en segundos
    fs = 16000    # frecuencia de muestreo
    
    print("Grabando...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    print("Grabación completada")
    
    # Convertir el audio a formato compatible con Vosk
    audio_data = (recording * 32767).astype(np.int16)
    
    # Procesar el audio con Vosk
    if recognizer.AcceptWaveform(audio_data.tobytes()):
        result = recognizer.Result()
        text = eval(result)["text"]
        return text
    return ""

def mic_button(parent, entry_widget):
    """Crear botón de micrófono con estilo consistente"""
    mic_frame = Frame(parent, bg=THEME_COLOR)
    mic_frame.pack(pady=5)
    
    def on_mic_click():
        # Deshabilitar el botón mientras graba
        mic_btn.config(state='disabled')
        
        # Mostrar mensaje de grabación
        show_message(parent, "🎤 Grabando...", "success")
        
        # Esperar un momento para que el mensaje sea visible
        parent.after(500, lambda: grab_audio())
        
    def grab_audio():
        text = transcribe_audio()
        if text:
            entry_widget.delete(0, END)
            entry_widget.insert(0, text)
            show_message(parent, "✓ Texto transcrito exitosamente")
        else:
            show_message(parent, "❌ No se detectó voz", "error")
        # Reactivar el botón
        mic_btn.config(state='normal')
    
    mic_btn = Button(mic_frame, text="🎤", command=on_mic_click,
                    bg=ACCENT_COLOR, fg=TEXT_COLOR,
                    font=("Helvetica", 12), width=3, height=1,
                    relief='flat', activebackground=BUTTON_COLOR)
    mic_btn.pack(side=RIGHT, padx=5)
    return mic_btn

#--------------------------- Funcion para almacenar el registro facial --------------------------------------

def normalize_text(text):
    """Normaliza el texto removiendo tildes y caracteres especiales"""
    # Normalizar texto a NFD y eliminar diacríticos
    text_normalized = unicodedata.normalize('NFD', text)
    text_normalized = ''.join(c for c in text_normalized if not unicodedata.combining(c))  # Elimina diacríticos
    
    # Reemplazar espacios por guiones bajos y eliminar caracteres no alfanuméricos
    text_normalized = ''.join(c for c in text_normalized if c.isalnum() or c == ' ')
    
    return text_normalized.lower()

def registro_facial():
    cap = cv2.VideoCapture(0)
    while(True):
        ret, frame = cap.read()
        frame_clean = frame.copy()  # Copia limpia del frame para guardar
        height, width = frame.shape[:2]

        # Proporciones del rectángulo (relación ancho:alto 4:5)
        box_width = (width * 2) // 5
        box_height = (height * 3) // 5
        x1 = (width - box_width) // 2
        y1 = (height - box_height) // 2
        x2 = x1 + box_width
        y2 = y1 + box_height

        # Líneas de referencia
        cv2.line(frame, (x1 - 20, (y1 + y2) // 2), (x1, (y1 + y2) // 2), (0, 255, 255), 2)  # Línea izquierda
        cv2.line(frame, (x2, (y1 + y2) // 2), (x2 + 20, (y1 + y2) // 2), (0, 255, 255), 2)  # Línea derecha
        cv2.line(frame, ((x1 + x2) // 2, y1 - 20), ((x1 + x2) // 2, y1), (0, 255, 255), 2)  # Línea superior
        cv2.line(frame, ((x1 + x2) // 2, y2), ((x1 + x2) // 2, y2 + 20), (0, 255, 255), 2)  # Línea inferior

        # Rectángulo principal con borde más grueso
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # Mensajes más visibles con fondo
        cv2.rectangle(frame, (10, 10), (width - 10, 60), (0, 0, 0), -1)  # Fondo negro semi-transparente
        cv2.putText(frame, "Centre su rostro en el rectangulo verde", 
                    (width // 8, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)

        # Mensaje en la parte inferior
        cv2.rectangle(frame, (10, height - 60), (width - 10, height - 10), (0, 0, 0), -1)
        cv2.putText(frame, "Presione ESC cuando este listo", 
                    (width // 4, height - 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.imshow('Registro Facial', frame)
        if cv2.waitKey(1) == 27:
            # Guardamos la imagen limpia sin las líneas guía
            usuario_img = normalize_text(usuario.get())
            cv2.imwrite(usuario_img+".jpg", frame_clean)
            break
    usuario_img = normalize_text(usuario.get())
    cv2.imwrite(usuario_img+".jpg",frame)
    cap.release()
    cv2.destroyAllWindows()
    usuario_entrada.delete(0, END)
    show_message(pantalla1, "✓ Registro Facial Completado con Éxito")

    #----------------- Detectamos el rostro y exportamos los pixeles --------------------------
    
    def reg_rostro(img, lista_resultados):
        data = pyplot.imread(img)
        for i in range(len(lista_resultados)):
            x1,y1,ancho, alto = lista_resultados[i]['box']
            x2,y2 = x1 + ancho, y1 + alto
            pyplot.subplot(1, len(lista_resultados), i+1)
            pyplot.axis('off')
            cara_reg = data[y1:y2, x1:x2]
            cara_reg = cv2.resize(cara_reg,(150,200), interpolation = cv2.INTER_CUBIC) #Guardamos la imagen con un tamaño de 150x200
            cv2.imwrite(usuario_img+".jpg",cara_reg)
            pyplot.imshow(data[y1:y2, x1:x2])
        pyplot.show()

    img = usuario_img+".jpg"
    pixeles = pyplot.imread(img)
    detector = MTCNN()
    caras = detector.detect_faces(pixeles)
    reg_rostro(img, caras)   
    
    # Después de completar el registro, volvemos a la pantalla principal
    pantalla1.after(2000, lambda: [pantalla.deiconify(), pantalla1.destroy()])

#------------------------Crearemos una funcion para asignar al boton registro --------------------------------
def registro():
    global usuario
    global usuario_entrada
    global pantalla1
    
    pantalla.withdraw()  # Oculta la ventana principal
    
    usuario = StringVar()
    
    pantalla1 = Toplevel(pantalla)
    pantalla1.title("Registro")
    pantalla1.geometry("400x400")
    pantalla1.configure(bg=THEME_COLOR)
    
    # Agregar protocolo para cuando se cierre la ventana
    def on_closing_registro():
        pantalla.deiconify()  # Muestra la ventana principal
        pantalla1.destroy()
    
    pantalla1.protocol("WM_DELETE_WINDOW", on_closing_registro)
    
    # Frame para el botón de salida en la esquina superior derecha
    exit_frame = Frame(pantalla1, bg=THEME_COLOR)
    exit_frame.pack(anchor='ne', padx=10, pady=5)
    
    small_exit_button(exit_frame, on_closing_registro).pack()
    
    Label(pantalla1, text="Registro de Usuario", bg=THEME_COLOR, fg=TEXT_COLOR, 
          font=("Helvetica", 14, "bold")).pack(pady=20)
    
    Label(pantalla1, text="Nombre de Usuario", bg=THEME_COLOR, fg=TEXT_COLOR, 
          font=FONT_MAIN).pack(pady=5)
    
    usuario_entrada = Entry(pantalla1, textvariable=usuario, font=FONT_MAIN,
                          bg=TEXT_COLOR, fg=THEME_COLOR, width=30)
    usuario_entrada.pack(pady=10)
    
    # Agregar botón de micrófono
    mic_button(pantalla1, usuario_entrada)
    
    # Solo dejamos el botón de registro facial
    custom_button(pantalla1, "Registrar Rostro", registro_facial).pack(pady=10)

#------------------------------------------- Funcion para verificar los datos ingresados al login ------------------------------------
    
def verificacion_login():
    log_usuario = verificacion_usuario.get()

    usuario_entrada2.delete(0, END)

    lista_archivos = os.listdir()   #Vamos a importar la lista de archivos con la libreria os
    if log_usuario in lista_archivos:   #Comparamos los archivos con el que nos interesa
        print("Inicio de sesion exitoso")
        Label(pantalla2, text = "Inicio de Sesion Exitoso", fg = "green", font = ("Calibri",11)).pack()
    else:
        print("Usuario no encontrado")
        Label(pantalla2, text = "Usuario no encontrado", fg = "red", font = ("Calibri",11)).pack()
    
#--------------------------Funcion para el Login Facial --------------------------------------------------------
def login_facial():
#------------------------------Vamos a capturar el rostro-----------------------------------------------------
    cap = cv2.VideoCapture(0)
    while(True):
        ret, frame = cap.read()
        frame_clean = frame.copy()  # Copia limpia del frame para guardar
        height, width = frame.shape[:2]

        # Proporciones del rectángulo (relación ancho:alto 4:5)
        box_width = (width * 2) // 5
        box_height = (height * 3) // 5
        x1 = (width - box_width) // 2
        y1 = (height - box_height) // 2
        x2 = x1 + box_width
        y2 = y1 + box_height

        # Líneas de referencia
        cv2.line(frame, (x1 - 20, (y1 + y2) // 2), (x1, (y1 + y2) // 2), (0, 255, 255), 2)  # Línea izquierda
        cv2.line(frame, (x2, (y1 + y2) // 2), (x2 + 20, (y1 + y2) // 2), (0, 255, 255), 2)  # Línea derecha
        cv2.line(frame, ((x1 + x2) // 2, y1 - 20), ((x1 + x2) // 2, y1), (0, 255, 255), 2)  # Línea superior
        cv2.line(frame, ((x1 + x2) // 2, y2), ((x1 + x2) // 2, y2 + 20), (0, 255, 255), 2)  # Línea inferior

        # Rectángulo principal con borde más grueso
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # Mensajes más visibles con fondo
        cv2.rectangle(frame, (10, 10), (width - 10, 60), (0, 0, 0), -1)  # Fondo negro semi-transparente
        cv2.putText(frame, "Centre su rostro en el rectangulo verde", 
                    (width // 8, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)

        # Mensaje en la parte inferior
        cv2.rectangle(frame, (10, height - 60), (width - 10, height - 10), (0, 0, 0), -1)
        cv2.putText(frame, "Presione ESC cuando este listo", 
                    (width // 4, height - 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)

        # Mostrar el cuadro
        cv2.imshow('Login Facial', frame)

        # Salir al presionar ESC
        if cv2.waitKey(1) == 27:
            # Guardamos la imagen limpia sin las líneas guía
            usuario_login = normalize_text(verificacion_usuario.get())
            cv2.imwrite(usuario_login + "LOG.jpg", frame_clean)
            break
    usuario_login = normalize_text(verificacion_usuario.get())
    cv2.imwrite(usuario_login+"LOG.jpg",frame)
    cap.release()
    cv2.destroyAllWindows()

    #----------------- Funcion para guardar el rostro --------------------------
    
    def log_rostro(img, lista_resultados):
        data = pyplot.imread(img)
        for i in range(len(lista_resultados)):
            x1,y1,ancho, alto = lista_resultados[i]['box']
            x2,y2 = x1 + ancho, y1 + alto
            pyplot.subplot(1, len(lista_resultados), i+1)
            pyplot.axis('off')
            cara_reg = data[y1:y2, x1:x2]
            cara_reg = cv2.resize(cara_reg,(150,200), interpolation = cv2.INTER_CUBIC) #Guardamos la imagen 150x200
            cv2.imwrite(usuario_login+"LOG.jpg",cara_reg)
            return pyplot.imshow(data[y1:y2, x1:x2])
        pyplot.show()

    #-------------------------- Detectamos el rostro-------------------------------------------------------
    
    img = usuario_login+"LOG.jpg"
    pixeles = pyplot.imread(img)
    detector = MTCNN()
    caras = detector.detect_faces(pixeles)
    log_rostro(img, caras)

    #-------------------------- Funcion para comparar los rostros --------------------------------------------
    def orb_sim(img1,img2):
        orb = cv2.ORB_create()  #Creamos el objeto de comparacion
 
        kpa, descr_a = orb.detectAndCompute(img1, None)  #Creamos descriptor 1 y extraemos puntos claves
        kpb, descr_b = orb.detectAndCompute(img2, None)  #Creamos descriptor 2 y extraemos puntos claves

        comp = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True) #Creamos comparador de fuerza

        matches = comp.match(descr_a, descr_b)  #Aplicamos el comparador a los descriptores

        regiones_similares = [i for i in matches if i.distance < 70] #Extraemos las regiones similares en base a los puntos claves
        if len(matches) == 0:
            return 0
        return len(regiones_similares)/len(matches)  #Exportamos el porcentaje de similitud
        
    #---------------------------- Importamos las imagenes y llamamos la funcion de comparacion ---------------------------------
    
    im_archivos = os.listdir()   
    if usuario_login+".jpg" in im_archivos:   
        rostro_reg = cv2.imread(usuario_login+".jpg",0)     
        rostro_log = cv2.imread(usuario_login+"LOG.jpg",0)  
        similitud = orb_sim(rostro_reg, rostro_log)
        if similitud >= 0.95:
            def sequence_actions():
                # Primer mensaje
                mensaje_frame = Frame(pantalla2, bg=SUCCESS_COLOR, pady=10)
                mensaje_frame.pack(fill=X, padx=20, pady=15)
                
                Label(mensaje_frame, 
                      text=f"✓ Acceso Concedido\nBienvenido: {usuario_login}",
                      bg=SUCCESS_COLOR,
                      fg=TEXT_COLOR,
                      font=("Helvetica", 12, "bold")).pack()
                
                # Destruir primer mensaje después de 2 segundos
                pantalla2.after(2000, mensaje_frame.destroy)
                
                # Mostrar segundo mensaje después de 2.1 segundos
                pantalla2.after(2100, lambda: show_message(pantalla2, "Cerradura Desbloqueada\nPuerta Abierta", "success"))
                
                # Ejecutar desbloqueo después de 3 segundos y regresar a pantalla principal después de 8 segundos (5 de la cerradura + 3 de espera)
                pantalla2.after(3000, manejar_cerradura)

            # Iniciar secuencia
            sequence_actions()
            
            print("Bienvenido al sistema usuario: ",usuario_login)
            print("Compatibilidad con la foto del registro: ",similitud)
        else:
            show_message(pantalla2, "❌ No coincide con el rostro registrado", "error")
            print("Rostro incorrecto, Verifique su usuario")
            print("Compatibilidad con la foto del registro: ",similitud)
    else:
        show_message(pantalla2, "❌ Usuario no encontrado", "error")

#--------------- Funcion para desbloquear la cerradura despues de un inicio de sesion exitoso ----------------------------
# Configuración inicial del GPIO
RELAY = 17  # Asegúrate de que este número es el correcto para tu pin
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(RELAY, GPIO.OUT)
GPIO.output(RELAY, GPIO.LOW)  # Asegúrate de que comienza apagado

# Función para manejar la cerradura
def manejar_cerradura():
    """Función para desbloquear la cerradura y bloquearla después de 5 segundos"""
    global RELAY
    print("[INFO] Cerradura desbloqueada")
    GPIO.output(RELAY, GPIO.HIGH)
    time.sleep(5)
    GPIO.output(RELAY, GPIO.LOW)
    print("[INFO] Cerradura bloqueada")
    # Regresar a la pantalla principal
    pantalla.deiconify()
    pantalla2.destroy()

#------------------------Funcion que asignaremos al boton login -------------------------------------------------
        
def login():
    global pantalla2
    global verificacion_usuario
    global usuario_entrada2
    
    pantalla.withdraw()  # Oculta la ventana principal
    
    verificacion_usuario = StringVar()
    
    pantalla2 = Toplevel(pantalla)
    pantalla2.title("Login")
    pantalla2.geometry("400x400")
    pantalla2.configure(bg=THEME_COLOR)
    
    # Agregar protocolo para cuando se cierre la ventana
    def on_closing_login():
        pantalla.deiconify()  # Muestra la ventana principal
        pantalla2.destroy()
    
    pantalla2.protocol("WM_DELETE_WINDOW", on_closing_login)
    
    # Frame para el botón de salida en la esquina superior derecha
    exit_frame = Frame(pantalla2, bg=THEME_COLOR)
    exit_frame.pack(anchor='ne', padx=10, pady=5)
    
    small_exit_button(exit_frame, on_closing_login).pack()
    
    Label(pantalla2, text="Inicio de Sesión", bg=THEME_COLOR, fg=TEXT_COLOR, 
          font=("Helvetica", 14, "bold")).pack(pady=20)
        
    Label(pantalla2, text="Nombre de Usuario", bg=THEME_COLOR, fg=TEXT_COLOR, 
          font=FONT_MAIN).pack(pady=5)
    
    usuario_entrada2 = Entry(pantalla2, textvariable=verificacion_usuario, 
                           font=FONT_MAIN, bg=TEXT_COLOR, fg=THEME_COLOR, width=30)
    usuario_entrada2.pack(pady=10)
    
    # Agregar botón de micrófono
    mic_button(pantalla2, usuario_entrada2)
    
    # Solo dejamos el botón de login facial
    custom_button(pantalla2, "Iniciar Reconocimiento", 
                 login_facial).pack(pady=10)

#------------------------- Funcion de nuestra pantalla principal ------------------------------------------------
    
def pantalla_principal():
    global pantalla
    pantalla = Tk()
    pantalla.geometry("400x500")
    pantalla.title("Sistema de Login Facial")
    pantalla.configure(bg=THEME_COLOR)
    
     # Add icon label
    Label(text="👤📷", bg=THEME_COLOR, fg=TEXT_COLOR, 
          font=("Helvetica", 50)).pack(pady=10)
    
    Label(text="Sistema para Desbloquear Cerradura\ncon Reconocimiento Facial", 
          bg=THEME_COLOR, fg=TEXT_COLOR, 
          font=("Helvetica", 16, "bold")).pack(pady=30)
    
    custom_button(pantalla, "Iniciar Sesión", login, width=25).pack(pady=15)
    custom_button(pantalla, "Crear Nuevo Perfil Facial", registro, width=25).pack(pady=15)
    custom_button(pantalla, "Cerrar Sistema", pantalla.destroy, width=25).pack(pady=15)
    
    pantalla.mainloop()

pantalla_principal()
