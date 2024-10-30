import cv2
import mediapipe as mp
import pyautogui
import threading
import time

# Coordenadas dos pontos
mp_maos = mp.solutions.hands
mp_desenho = mp.solutions.drawing_utils

# Detecta mãos
maos = mp_maos.Hands()

camera = cv2.VideoCapture(0)
resolucao_x = 1280
resolucao_y = 720
camera.set(cv2.CAP_PROP_FRAME_WIDTH, resolucao_x)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, resolucao_y)


def pegar_coordenadas(img, lado_invertido=False):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resultado = maos.process(img_rgb)
    todas_maos = []

    if resultado.multi_hand_landmarks:
        for lado_mao, marcacao_maos in zip(resultado.multi_handedness, resultado.multi_hand_landmarks):
            info_maos = {}
            coordenadas = []

            for marcacao in marcacao_maos.landmark:
                coord_x, coord_y, coord_z = int(marcacao.x * resolucao_x), int(marcacao.y * resolucao_y), int(marcacao.z * resolucao_x)
                coordenadas.append((coord_x, coord_y, coord_z))

            info_maos['coordenadas'] = coordenadas

            # Invertendo lado da mão, se necessário
            if lado_invertido:
                if lado_mao.classification[0].label == 'Left':
                    info_maos['lado'] = 'Right'
                else:
                    info_maos['lado'] = 'Left'
            else:
                info_maos['lado'] = lado_mao.classification[0].label

            todas_maos.append(info_maos)

            mp_desenho.draw_landmarks(img, marcacao_maos, mp_maos.HAND_CONNECTIONS)

    return img, todas_maos


def virar_lados(lado):
    if lado == 'Right':
        pyautogui.keyDown('right')

    elif lado == 'Left':
        pyautogui.keyDown('left')
        
    pyautogui.keyUp('right')
    pyautogui.keyUp('left')
    
def principlloop():
    while True:
        sucesso, img = camera.read()
        img = cv2.flip(img, 1)
        img, todas_maos = pegar_coordenadas(img)

        if len(todas_maos) > 0:
            lado = todas_maos[0]['lado']
            virar_lados(lado)  # Aciona a função para virar de acordo com o lado detectado

        cv2.imshow('imagem', img)

        tecla = cv2.waitKey(1)
        if tecla == 27:
            break


# Iniciando a thread principal do loop
thread1 = threading.Thread(target=principlloop)
thread1.start()
thread1.join()

camera.release()
cv2.destroyAllWindows()
