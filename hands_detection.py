import cv2
import mediapipe as mp
from djitellopy import Tello

# Inicializar mediapipe y Tello
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
tello = Tello()

use_webcam = True 
use_Tello = True
isFlying=False

# Conectar con el dron
if use_Tello:
    tello.connect()
    tello.streamoff()
    tello.streamon()
    print("Batería: ", tello.get_battery(), "%")


# Configurar fuente de video (webcam o cámara del dron)
if use_webcam:
    cap = cv2.VideoCapture(0)
else:
    cap = tello.get_video_capture()
    cap.set(cv2.CAP_PROP_FPS, 15)
    cap.set(3, 640)
    cap.set(4, 480)


def send_tello_command(gesture, tello):
    isFlying=False
    # Enviar comandos al dron Tello basados en el gesto detectado
    if use_Tello:
        if gesture == 'takeoff':
            tello.takeoff()
            isFlying=True
        if isFlying:
            if gesture == 'down':
                tello.up_down_velocity=-20
                #tello.move_down(50)
            elif gesture == 'forward':
                tello.move_forward(50)
            elif gesture == 'back':
                tello.move_forward(-50)
            elif gesture == 'flip':
                tello.flip('r')
            elif gesture == 'up':
                tello.up_down_velocity=20
            elif gesture == 'land':
                tello.land()
    else:
        if gesture == 'takeoff':
            print("takeoff")
        elif gesture == 'down':
            print("down")
        elif gesture == 'forward':
            print("forward")
        elif gesture == 'back':
            print("back")
        elif gesture == 'flip':
            print("flip")
        elif gesture == 'up':
            print("up")
        elif gesture == 'land':
            print("land")
pass

def interpret_gesture(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    pinky_pip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]
    palm = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    
    # Despegar el dron con el pulgar hacia arriba
    if thumb_tip.y < index_pip.y < pinky_tip.y:
        return 'takeoff'

    # Aterrizar el dron cuando el pulgar está hacia abajo
    elif thumb_tip.y > index_tip.y > palm.y and thumb_tip.y > thumb_ip.y:
        return 'land'

    # Acercar el dron con la palma abierta
    elif (thumb_tip.y < thumb_ip.y and index_tip.y < index_pip.y and middle_tip.y < middle_pip.y 
        and pinky_tip.y < pinky_pip.y):
        return 'forward'

    # Alejar el dron con el puño cerrado
    elif (index_tip.y > index_pip.y and middle_tip.y > middle_pip.y 
        and pinky_tip.y > pinky_pip.y):
        return 'back'

    # Dar una vuelta sobre si mismo cuando los dedos anular y medio están doblados
    elif thumb_ip.y > thumb_tip.y and middle_pip.y < middle_tip.y and pinky_tip.y < pinky_pip.y:
        return 'flip'

    #Levantar el dron si el dedo índice está hacia arriba
    elif index_tip.y < middle_tip.y < palm.y and index_tip.y < index_pip.y:
        return 'up'

    # Bajar el dron cuando el índice está hacia abajo
    elif index_tip.y > middle_tip.y > palm.y:
        return 'down'
pass

# Inicializar mediapipe hands
with mp_hands.Hands(
    min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Espejo y cambio de BGR a RGB
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Procesar la imagen
        results = hands.process(frame_rgb)

        # Dibujar puntos y conexiones de la mano
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Obtener gesto y enviar comando al dron
                gesture = interpret_gesture(hand_landmarks)
                send_tello_command(gesture, tello)

        # Mostrar imagen
        cv2.imshow('Dron Tello - Control por gestos', frame)

        # Salir con la tecla 'q'
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
#tello.land()
tello.streamoff()
tello.end()








