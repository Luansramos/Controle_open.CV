import cv2 #para processamento de imagem
import mediapipe as mp #para imagems
import pyautogui

#cordenadas dos pontos
mp_maos = mp.solutions.hands
mp_desenho = mp.solutions.drawing_utils

#detecta maos
maos = mp_maos.Hands()

camera = cv2.VideoCapture(0) #funçao q conecta camera com python
resolucao_x = 1280
resolucao_y = 720
camera.set(cv2.CAP_PROP_FRAME_WIDTH, resolucao_x)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, resolucao_y)


def pegar_coordenadas( img, lado_invertido = False ):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #transformando em rgb
    #(cv2.COLOR_BGR2RGB) == cv2 cor BGR para cor RGB    
    resultado = maos.process(img_rgb)#retorna 3 valores: 1:coordenandas das maos com base no tamanho da imagem;2:coordenas no mundo real (metro);3:lado da mao (direita e esquerda );
    #multi_hand_landmark é o oprimeiro valor retornado
    todas_maos = []
    if resultado.multi_hand_landmarks:# primeiro valor retornado da funçao
        
        for lado_mao, marcacao_maos in zip(resultado.multi_handedness, resultado.multi_hand_landmarks):
            #metodo "zip()" justa os elementos de duas ou mais listas
            info_maos = {}
            coordenadas = []
            for marcacao in marcacao_maos.landmark:
             #landmarks retorna 3 valores de cordenadas de 0 a 1 x,y,z 
                coord_x, coord_y, coord_z = int ( marcacao.x * resolucao_x ), int ( marcacao.y * resolucao_y ), int ( marcacao.z * resolucao_x ) # multicas os valores retornados pela resoluçao para ter um  valor  de coordenadas em px
            coordenadas.append((coord_x, coord_y, coord_z)) #no curso tem um tab aqui pra ficar dentro do for mas achei estranho e tirei
            info_maos['coordenadas'] = coordenadas
            
            #if para trocar o lado da mao e responder o lado certo
            if lado_invertido:# = True
                #inverte os lados e cria a chave 'lado' no diconario
                if  lado_mao.classification[0].label == 'Left':
                    info_maos['lado'] = 'Right'
                else:
                    info_maos['lado'] = 'Left'
            else:
                info_maos['lado'] = lado_mao.classification[0].label 

            
            #lado_mao.classification[0].label -->> acessa o lado da mao 
            todas_maos.append(info_maos)
                    #draw_landmarks desenho da mao
            mp_desenho.draw_landmarks (img, #imagem 
                                    marcacao_maos, #coordenadas dos pontos
                                    mp_maos.HAND_CONNECTIONS) #linha entre os pontos
    return img, todas_maos


def virar_lados( lado ):
    
    if lado:    
        if lado == 'Right':            
            pyautogui.keyDown('right')
            cv2.waitKey(10)
            pyautogui.keyUp('right')
            #pyautogui.press('right')

        if lado == 'Left':
            pyautogui.keyDown('left')
            cv2.waitKey(10)
            pyautogui.keyUp('left')





while True:

    
    sucesso, img = camera.read() 
    # metodo read() captutra um frame da camera e restorna true se a a captura for bem sucedida e false caso nn, img pega a imagem q esta sendo analisada
    img = cv2.flip(img,1)
    
    img, todas_maos = pegar_coordenadas(img)
    
    if len(todas_maos) > 0:
        virar_lados( todas_maos[0]['lado'] )
        
        
        
    '''print(todas_maos)'''
    
    cv2.imshow('imagem',img) #imshow() mostra a img da camera, com o nome da janela e o parametro de imagem;  "imagem" corresponde ao nome da janela q se abre
    
    tecla = cv2.waitKey(1) #witKey faz o codigo parar por 1 milisegundo e guarda uma tecla do teclado

    if tecla == 27:
        break
