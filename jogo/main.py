from ursina import *
import random

# Inicializando a engine Ursina
app = Ursina()

# Definindo a janela do jogo
window.title = 'Corrida Python 3D'
window.borderless = False
window.fullscreen = False
window.exit_button.visible = False
window.fps_counter.enabled = True

# Variável para controlar a pontuação
score = 0
menu_elements = []  # Lista para armazenar os elementos do menu

# Carregando texturas (imagens para os carros e a pista)
player_car_texture = 'player_car.png'  # Substitua pelo caminho da imagem do carro do jogador
enemy_car_texture = 'enemy_car.png'  # Substitua pelo caminho da imagem do carro inimigo
track_texture = 'track_texture.png'  # Substitua pelo caminho da imagem da pista

# Criando um quadrado invisível (colisor) para o carro do jogador
player_car_collider = Entity(model='cube', scale=(0.5, 0.5, 2), position=(0, 0, -10), collider='box', visible=False)

# Adicionando a imagem do carro, vinculada ao quadrado invisível (colisor)
player_car_visual = Entity(model='quad', texture=player_car_texture, scale=(1, 0.5), position=player_car_collider.position)

# Criação de múltiplos segmentos da pista com a imagem
track_segments = []
num_segments = 10
for i in range(num_segments):
    segment = Entity(model='quad', scale=(10, 10), texture=track_texture, position=(0, -0.5, i * 10))
    segment.rotation_x = 90  # Rotacionar para ficar plano
    track_segments.append(segment)

# Lista de carros adversários (com colisores invisíveis)
enemy_cars = []

# Função para verificar se um novo carro está sobreposto com carros existentes
def is_overlapping(car_position, other_cars, min_distance=2):
    for car_collider, _ in other_cars:
        distance = abs(car_position - car_collider.x)
        if distance < min_distance:
            return True
    return False

# Função para criar carros adversários sem sobreposição
def create_enemy_car():
    max_attempts = 10
    for _ in range(max_attempts):
        car_x = random.uniform(-4, 4)  # Gera uma posição X aleatória
        if not is_overlapping(car_x, enemy_cars):
            car_collider = Entity(model='cube', scale=(0.75, 0.5, 2), position=(car_x, 0, random.uniform(5, 20)), collider='box', visible=False)
            car_visual = Entity(model='quad', texture=enemy_car_texture, scale=(1, 0.5), position=car_collider.position)
            enemy_cars.append((car_collider, car_visual))
            break  # Se o carro não está sobreposto, finalize a criação

# Função para recriar carros adversários
def recreate_enemy_cars():
    global enemy_cars
    # Destruir carros adversários anteriores
    for car_collider, car_visual in enemy_cars:
        car_collider.disable()
        car_visual.disable()
    
    enemy_cars = []  # Reiniciar a lista de carros
    for _ in range(7):
        create_enemy_car()

# Criar os carros adversários inicialmente
recreate_enemy_cars()

# Função para mover os segmentos da pista, simulando movimento
def move_track():
    for segment in track_segments:
        segment.z -= 5 * time.dt  # Movimento contínuo para simular movimento da pista
        if segment.z < player_car_collider.z - 10:  # Reposicionar o segmento quando ele sair da visão
            segment.z += num_segments * 10  # Reposiciona o segmento à frente

# Configuração da câmera para terceira pessoa
def update_camera():
    camera.position = (player_car_collider.x, player_car_collider.y + 2, player_car_collider.z - 10)  # Posição da câmera atrás e acima do carro
    camera.look_at(player_car_collider)  # A câmera sempre olha para o carro

# Função para finalizar o jogo, exibindo o menu de "Game Over"
def game_over():
    global score
    player_car_collider.disable()  # Pausar o jogo removendo o carro do jogador
    player_car_visual.disable()

    # Exibir a pontuação
    score_text = Text(f'Sua pontuação: {int(score)}', scale=2, position=(0, 0.2))
    menu_elements.append(score_text)  # Armazenar o elemento para ser destruído mais tarde

    # Botão para jogar novamente
    retry_button = Button('Jogar Novamente', color=color.azure, scale=(0.3, 0.1), position=(0, 0))
    retry_button.on_click = restart_game
    menu_elements.append(retry_button)

    # Botão para sair do jogo
    exit_button = Button('Sair', color=color.red, scale=(0.3, 0.1), position=(0, -0.15))
    exit_button.on_click = application.quit
    menu_elements.append(exit_button)

# Função para reiniciar o jogo
def restart_game():
    global score
    score = 0  # Reinicializar a pontuação
    player_car_collider.position = (0, 0, -10)
    player_car_visual.position = (0, 0, -10)  # Reposicionar a imagem do carro
    player_car_collider.enable()  # Habilitar o carro do jogador de novo
    player_car_visual.enable()  # Habilitar a imagem do carro novamente

    # Resetar a posição dos segmentos da pista
    for i, segment in enumerate(track_segments):
        segment.position = (0, -0.5, i * 10)

    # Recriar os carros adversários
    recreate_enemy_cars()

    # Destruir todos os elementos do menu e reiniciar o jogo
    destroy_all_menu_elements()

def destroy_all_menu_elements():
    # Remove todos os textos e botões do menu para poder continuar o jogo normalmente
    for element in menu_elements:
        destroy(element)
    menu_elements.clear()  # Limpar a lista de elementos

# Criação do texto da pontuação
score_text = Text(text=f'Pontuação: {int(score)}', position=(-0.7, 0.45), scale=1.5, color=color.white)

# Função para atualizar a lógica do jogo a cada frame
def update():
    global score
    if player_car_collider.enabled:  # Só atualizar o jogo se o jogador estiver ativo
        speed = 10  # Velocidade do carro do jogador
        move_speed = time.dt * speed  # Ajusta a velocidade de acordo com o tempo entre frames

        # Movimento do carro do jogador ao longo do eixo Z (em direção à frente)
        player_car_collider.z += move_speed
        player_car_visual.z = player_car_collider.z  # Mover a imagem junto com o colisor
        player_car_visual.x = player_car_collider.x  # Sincronizar a posição horizontal

        score += move_speed  # Aumenta a pontuação à medida que o jogador avança
        score_text.text = f'Pontuação: {int(score)}'  # Atualiza o texto da pontuação

        # Movimento lateral do jogador (esquerda/direita)
        if held_keys['left arrow']:
            player_car_collider.x -= move_speed * 0.5
        if held_keys['right arrow']:
            player_car_collider.x += move_speed * 0.5

        # Limitar o carro dentro da pista
        player_car_collider.x = clamp(player_car_collider.x, -2.5, 2.5)

        # Reposicionar os carros adversários quando o jogador passar por eles
        for car_collider, car_visual in enemy_cars:
            if player_car_collider.z > car_collider.z + 1:  # Quando o jogador passa pelo carro
                car_collider.z += random.uniform(30, 40)  # Reposicionar mais à frente
                car_collider.x = random.uniform(-2.5, 2.5)    # Nova posição lateral aleatória
                car_visual.z = car_collider.z
                car_visual.x = car_collider.x

        # Atualizar a posição da câmera
        update_camera()

        # Mover os segmentos da pista para dar a sensação de movimento
        move_track()

        # Verificar colisão: Se o colisor do carro do jogador colidir com o colisor de um adversário
        for car_collider, car_visual in enemy_cars:
            if player_car_collider.intersects(car_collider).hit:  # Detecta colisão
                game_over()

# Iniciar o jogo
app.run()
