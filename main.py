import pygame
import sys
import math
import numpy as np
from scipy.integrate import odeint


pygame.init()


WIDTH, HEIGHT = 1000, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Ring Road Traffic Flow Simulation with Traffic Waves")

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 200, 0)
GRAY = (200, 200, 200)

#parametry modelu
NUM_VEHICLES = 30  #ilość aut
v0 = 30  #pożadana prędkosć
T = 1.5  #bezpieczny odstęp czasowy do poprzedzajacego auta
a = 0.73  #max przyśpieszenie
b = 1.67  #kofortowe hamowanie
delta = 4  #wykładnik członu odpowiadającego za wpływ predkości
s0 = 2  #minimalny dystans od poprzedzajacego auta
vehicle_length = 5  #długość pojazdu

#parametry drogi pojazdu
ROAD_RADIUS = 300  #promień drogi w pixelach
ROAD_CENTER = (WIDTH // 2, HEIGHT // 2)  #współrzędne środka drogi
ROAD_WIDTH = 30  #szerokosc drogi
SCALE_FACTOR = 2  #wspolczynnik skalowania do zamiany pixeli na metry

CIRCUMFERENCE = 2 * math.pi * ROAD_RADIUS / SCALE_FACTOR
init_positions = np.linspace(0, CIRCUMFERENCE, NUM_VEHICLES, endpoint=False)

#predkosc na astart
init_velocities = np.ones(NUM_VEHICLES) * v0 - np.random.rand(NUM_VEHICLES)

#mozliwosc wprowadzenia zaklocenia w ruchu pojazdow aby wywolac fale wsteczna
# perturb_vehicle = NUM_VEHICLES // 4
# init_velocities[perturb_vehicle] *= 0.3

#Stan symulacji
positions = init_positions.copy()
velocities = init_velocities.copy()

#Istawienia zwiazane z predksocia symulacji
time_step = 0.02
clock = pygame.time.Clock()


#funkcja licząca s_star
def calculate_s_star(v, delta_v):
    return s0 + v * T + (v * delta_v) / (2 * math.sqrt(a * b))


#definicja modelu IDM
def model(state, t):
    # state = [x1, x2, ..., xn, v1, v2, ..., vn]
    n = len(state) // 2
    positions = state[:n]
    velocities = state[n:]

    dxdt = np.zeros(2 * n)

    dxdt[:n] = velocities

    # Liczymy(dv/dt)
    for i in range(n):
        #Pojazd 50 jedzie za 1, a 1 za 2 itd.
        leader_idx = (i + 1) % n

        delta_x = positions[leader_idx] - positions[i]
        if delta_x < 0:
            delta_x += CIRCUMFERENCE

        s = delta_x - vehicle_length

        delta_v = velocities[i] - velocities[leader_idx]

        s_star = calculate_s_star(velocities[i], delta_v)

        term1 = (velocities[i] / v0) ** delta
        term2 = (s_star / max(s, 0.1)) ** 2

        acceleration = a * (1 - term1 - term2)

        dxdt[n + i] = acceleration

    return dxdt


#zamiana pozycji pojazdu na pozycje na ekranie
def position_to_screen(position):
    angle = (position / CIRCUMFERENCE) * 2 * math.pi - math.pi / 2
    x = ROAD_CENTER[0] + ROAD_RADIUS * math.cos(angle)
    y = ROAD_CENTER[1] + ROAD_RADIUS * math.sin(angle)
    return int(x), int(y)


#Rysowanie samochodu
def draw_car(position, velocity, index):
    velocity_ratio = min(velocity / v0, 1.0)
    color = (
        int(255 * (1 - velocity_ratio)),
        int(255 * velocity_ratio),
        0
    )

    car_pos = position_to_screen(position)
    car_radius = 8
    pygame.draw.circle(screen, color, car_pos, car_radius)

    angle = (position / CIRCUMFERENCE) * 2 * math.pi - math.pi / 2
    indicator_length = car_radius * (0.5 + velocity_ratio)
    end_x = car_pos[0] - math.cos(angle) * indicator_length
    end_y = car_pos[1] - math.sin(angle) * indicator_length
    pygame.draw.line(screen, BLACK, car_pos, (end_x, end_y), 2)

    if NUM_VEHICLES <= 20:
        font = pygame.font.SysFont(None, 20)
        text = font.render(f"{index}", True, WHITE)
        text_rect = text.get_rect(center=car_pos)
        screen.blit(text, text_rect)


state = np.concatenate((positions, velocities))

#Petla symulacji
paused = False
show_info = True
running = True
simulation_time = 0
simulation_speed = 1.0
show_wave_analysis = True


wave_detection_threshold = 0.6
previous_slow_vehicle = None
wave_speed = 0

speed_slider_rect = pygame.Rect(WIDTH - 220, HEIGHT - 40, 200, 20)
speed_slider_button_radius = 10
speed_slider_button_pos = WIDTH - 220 + int(simulation_speed * 100)
is_dragging = False

position_history = []
velocity_history = []
history_length = 100
history_sample_rate = 5
frame_count = 0

cnt = 0 #Dodana inicjalizacja licznika

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                paused = not paused
            elif event.key == pygame.K_i:
                show_info = not show_info
            elif event.key == pygame.K_w:
                show_wave_analysis = not show_wave_analysis
            elif event.key == pygame.K_UP:
                simulation_speed = min(simulation_speed + 0.1, 2.0)
            elif event.key == pygame.K_DOWN:
                simulation_speed = max(simulation_speed - 0.1, 0.1)
            #Zakłócenie na życzienie pod klawiszem P
            elif event.key == pygame.K_p:
                random_vehicle = np.random.randint(0, NUM_VEHICLES)
                velocities[random_vehicle] *= 0.3
                state[NUM_VEHICLES + random_vehicle] = velocities[random_vehicle]
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if speed_slider_rect.collidepoint(event.pos):
                is_dragging = True
        elif event.type == pygame.MOUSEBUTTONUP:
            is_dragging = False
        elif event.type == pygame.MOUSEMOTION and is_dragging:
            x = max(speed_slider_rect.left, min(event.pos[0], speed_slider_rect.right))
            speed_slider_button_pos = x
            simulation_speed = (x - speed_slider_rect.left) / 200 * 2.0  # 0.0 to 2.0

    if not paused:
        adjusted_time_step = time_step * simulation_speed

        new_state = odeint(model, state, [0, adjusted_time_step])[1]
        state = new_state

        positions = state[:NUM_VEHICLES]
        velocities = state[NUM_VEHICLES:]

        positions = positions % CIRCUMFERENCE

        frame_count += 1
        if frame_count % history_sample_rate == 0 and len(position_history) < history_length:
            position_history.append(positions.copy())
            velocity_history.append(velocities.copy())

        slowest_idx = np.argmin(velocities)
        slowest_velocity = velocities[slowest_idx]

        if previous_slow_vehicle is not None and slowest_velocity < wave_detection_threshold * v0:
            if slowest_idx != previous_slow_vehicle:
                wave_direction = -1 if (slowest_idx - previous_slow_vehicle) % NUM_VEHICLES < NUM_VEHICLES / 2 else 1
                wave_speed = wave_direction * (
                            positions[slowest_idx] - positions[previous_slow_vehicle]) / adjusted_time_step
                if abs(wave_speed) > CIRCUMFERENCE / 2:
                    wave_speed = wave_direction * (CIRCUMFERENCE - abs(
                        positions[slowest_idx] - positions[previous_slow_vehicle])) / adjusted_time_step

        previous_slow_vehicle = slowest_idx

        simulation_time += adjusted_time_step

    cnt += 1  # Inkrementacja licznika
    if cnt % 10 == 0:  # Rysowanie co 10. przebieg pętli
        screen.fill(WHITE)

        pygame.draw.circle(screen, GRAY, ROAD_CENTER, ROAD_RADIUS + ROAD_WIDTH // 2, ROAD_WIDTH)
        for i in range(NUM_VEHICLES):
            draw_car(positions[i], velocities[i], i + 1)

        pygame.draw.rect(screen, BLACK, speed_slider_rect, 1)
        speed_slider_button_pos = speed_slider_rect.left + int(simulation_speed * 100)
        pygame.draw.circle(screen, BLACK, (speed_slider_button_pos, speed_slider_rect.centery), speed_slider_button_radius)

        font = pygame.font.SysFont(None, 24)
        speed_text = font.render(f"Speed: {simulation_speed:.1f}x", True, BLACK)
        screen.blit(speed_text, (WIDTH - 320, HEIGHT - 45))

        if show_info:
            font = pygame.font.SysFont(None, 24)
            info_lines = [
                f"Time: {simulation_time:.1f} s",
                f"Vehicles: {NUM_VEHICLES}",
                f"Desired speed: {v0} m/s",
                f"Mean speed: {np.mean(velocities):.2f} m/s",
                f"Min speed: {np.min(velocities):.2f} m/s",
                f"Max speed: {np.max(velocities):.2f} m/s",
                f"Road length: {CIRCUMFERENCE:.0f} m",
                f"Vehicle spacing: {CIRCUMFERENCE / NUM_VEHICLES - vehicle_length:.1f} m",
                "Press SPACE to pause/resume",
                "Press I to toggle info",
                "Press P to add perturbation",
                "Press UP/DOWN to change speed"
            ]

            y_pos = 10
            for line in info_lines:
                text = font.render(line, True, BLACK)
                screen.blit(text, (10, y_pos))
                y_pos += 25

            if show_wave_analysis:
                wave_info = [
                    "Traffic Wave Analysis:",
                    f"Slowest vehicle: #{previous_slow_vehicle + 1}",
                    f"Slowest speed: {np.min(velocities):.2f} m/s",
                    f"Approx. wave speed: {-wave_speed:.2f} m/s"
                ]

                for line in wave_info:
                    text = font.render(line, True, BLACK)
                    screen.blit(text, (10, y_pos))
                    y_pos += 25

        pygame.display.flip()

    #clock.tick(60)

pygame.quit()
sys.exit()