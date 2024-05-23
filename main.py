import cv2 #visão computacional
from cv2 import textureFlattening
import numpy as np #manupulação de imagem
import h5py
from tensorflow import keras #aprendizado profundo
from kivy.app import App #interface gráfica
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image 
from kivy.clock import Clock
from kivy.graphics.texture import Texture

#classe de criação da aplicação
class EmotionDetectionApp(App):

    #criar a interface
    def build(self):
        self.emotions_model = keras.models.load_model('fer2013_mini_XCEPTION.102-0.66.hdf5') #rede neural pré treinada
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #detecção de faces
        
        #configura a interface
        self.layout = BoxLayout(orientation='vertical')
        self.image = Image() #exibir camera
        self.start_button = Button(text='Iniciar Detecção')
        self.start_button.bind(on_press=self.start_detection)
        
        self.layout.add_widget(self.image)
        self.layout.add_widget(self.start_button)
        
        #captura da camera usando openCV
        self.capture = cv2.VideoCapture(1)
        self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.capture.set(cv2.CAP_PROP_FPS, 30)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Defina o tamanho desejado
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Defina o tamanho desejado
        self.started = False
        Clock.schedule_interval(self.update, 1.0/30.0)
        
        return self.layout

    #tenta carregar o modelo de emoções
    def load_emotions_model(self):
        try:
            with h5py.File('fer2013_mini_XCEPTION.102-0.66.hdf5', 'r') as f:
                self.emotions_model = f.get('model_weights').get('sequential_1')
        except Exception as e:
            print(f"Erro ao carregar o modelo: {e}")
            self.emotions_model = None

    #chamado quando o botão é pressionado
    def start_detection(self, instance):
        self.started = not self.started
        self.start_button.text = 'Parar Detecção' if self.started else 'Iniciar Detecção'
    
    #atualização do kivy
    def update(self, dt):
        ret, frame = self.capture.read()
        if self.started and ret:
            frame = cv2.flip(frame, 1) #inverte a camera
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #escala de cinza
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(120, 120)) #detecta face
            
            #loop de detecção
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                roi_gray = cv2.resize(roi_gray, (64, 64), interpolation=cv2.INTER_AREA)
                roi_gray = roi_gray.astype('float') / 255.0
                roi_gray = np.expand_dims(roi_gray, axis=0)
                roi_gray = np.expand_dims(roi_gray, axis=-1)
                predictions = self.emotions_model.predict(roi_gray)
                
                #mapeia
                emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', '']
                max_index = np.argmax(predictions)
                if max_index < len(emotion_labels):
                    emotion = emotion_labels[max_index]
                else:
                    emotion = ""
                
                #desenha um retangulo na face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, emotion, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            
            buf1 = cv2.flip(frame, 0)
            buf = buf1.tostring()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.image.texture = texture
    
    def on_stop(self):
        self.capture.release()

if __name__ == '__main__':
    EmotionDetectionApp().run()
