import cv2
from pynput.keyboard import Key, Controller
import mediapipe as mp
import math

# Definindo a resolução da janela da câmera
wCan, hCan = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wCan)
cap.set(4, hCan)

# Inicializando o Mediapipe para detecção de mãos
hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2)
mpDraw = mp.solutions.drawing_utils

# Carregando as imagens de setas e redimensionando
setaDir = cv2.resize(cv2.imread('./controle/seta direita.png'), (100, 100))
setaEsq = cv2.resize(cv2.imread('./controle/seta esquerda.png'), (100, 100))

# Inicializando o controlador de teclado
kb = Controller()

# Variáveis globais de controle
center_x, center_y = wCan // 2, hCan // 2
top_y = center_y - 50
left_x = center_x - 50
bottom_y = top_y + 100
right_x = left_x + 100

def calcular_distancia(m1, m2, posicao=11):
    """Calcula a distância entre dois pontos específicos das mãos."""
    x1, y1 = m1[posicao]
    x2, y2 = m2[posicao]
    return math.hypot(x2 - x1, y2 - y1), (x1 + x2) // 2, (y1 + y2) // 2

def controlar_teclas(diff):
    """Controla as teclas de acordo com a diferença entre as alturas das mãos."""
    if diff >= 100:
        kb.press(Key.left)
        kb.release(Key.right)
    elif diff <= -100:
        kb.press(Key.right)
        kb.release(Key.left)
    else:
        kb.release(Key.left)
        kb.release(Key.right)

def desenhar_setas(img, diff):
    """Desenha as setas indicando a direção do movimento."""
    if diff >= 100:
        img[top_y - 150:bottom_y - 150, left_x - 250:right_x - 250] = setaEsq
    elif diff <= -100:
        img[top_y - 150:bottom_y - 150, left_x + 250:right_x + 250] = setaDir

def processar_mãos(img):
    """Processa a imagem, identifica as mãos e controla as ações."""
    myhands = []
    frameRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(frameRGB)

    if results.multi_hand_landmarks:
        for handLandMarks in results.multi_hand_landmarks:
            myHand = [(int(Landmark.x * wCan), int(Landmark.y * hCan)) for Landmark in handLandMarks.landmark]
            myhands.append(myHand)

            if len(myhands) >= 2:
                m1, m2 = myhands[0], myhands[1]
                length, cx, cy = calcular_distancia(m1, m2)

                if 100 <= length <= 250:
                    kb.press(Key.up)
                else:
                    kb.release(Key.up)

                if 200 <= length <= 600: # mudar valores pra afastara mao
                    cv2.line(img, m1[11], m2[11], (92, 152, 0), 3)
                    cv2.circle(img, (cx, cy), 15, (92, 152, 0), cv2.FILLED)

                    # Diferenciação entre mão esquerda e direita
                    moEsqX, moEsqY = (m1[11] if m1[11][0] > m2[11][0] else m2[11])
                    moDirX, moDirY = (m2[11] if m1[11][0] > m2[11][0] else m1[11])

                    diff = moEsqY - moDirY
                    controlar_teclas(diff)
                    desenhar_setas(img, diff)

                    cv2.putText(img, "ESQUERDA", (moEsqX - 50, moEsqY + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
                    cv2.putText(img, "DIREITA", (moDirX - 50, moDirY + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

    return img

while True:
    success, img = cap.read()
    if not success:
        break

    img = processar_mãos(img)
    cv2.imshow('Imagem', img)
    
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
