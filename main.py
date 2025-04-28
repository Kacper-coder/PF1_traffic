import pygame
import sys
import math

# Inicjalizacja Pygame
pygame.init()

# Wymiary ekranu
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Rzut prostopadły z oporem ośrodka")

# Kolory
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLACK = (0, 0, 0)
GREEN = (0, 200, 0)
BLUE = (0, 0, 255)

# Właściwości pocisku
ball_radius = 10
ball_mass = 1.0  # masa w kg
ball_x = 50
ball_y = 50
initial_speed = 15
angle_degrees = 0  # Kąt wyrzutu w stopniach
angle_radians = math.radians(angle_degrees)
speed_x = initial_speed * math.cos(angle_radians)
speed_y = -initial_speed * math.sin(angle_radians)  # Ujemna wartość, ponieważ w Pygame oś y rośnie w dół

# Parametry fizyczne
gravity = 0.5
drag_coefficient = 0.005  # Współczynnik oporu ośrodka
time_step = 1

# Śledzenie trajektorii
trajectory = []
trajectory_no_drag = []  # Trajektoria bez oporu dla porównania

# Stan symulacji równoległej bez oporu
ball_x_no_drag = ball_x
ball_y_no_drag = ball_y
speed_x_no_drag = speed_x
speed_y_no_drag = speed_y

# Zegar do kontrolowania liczby klatek na sekundę
clock = pygame.time.Clock()

# Zmienne kontrolne
running = True
simulation_active = False
reset_ready = False
show_comparison = True  # Czy pokazywać porównanie z trajektorią bez oporu

# Główna pętla gry
while running:
    # Obsługa zdarzeń
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                if not simulation_active and not reset_ready:
                    simulation_active = True
                elif reset_ready:
                    # Reset symulacji
                    ball_x = 50
                    ball_y = HEIGHT - 50
                    speed_x = initial_speed * math.cos(angle_radians)
                    speed_y = -initial_speed * math.sin(angle_radians)

                    ball_x_no_drag = ball_x
                    ball_y_no_drag = ball_y
                    speed_x_no_drag = speed_x
                    speed_y_no_drag = speed_y

                    trajectory = []
                    trajectory_no_drag = []
                    simulation_active = False
                    reset_ready = False
            elif event.key == pygame.K_c:
                # Przełącz widoczność porównania
                show_comparison = not show_comparison

    # Aktualizacja pozycji pocisku
    if simulation_active:
        # Dodaj aktualną pozycję do trajektorii
        trajectory.append((int(ball_x), int(ball_y)))
        trajectory_no_drag.append((int(ball_x_no_drag), int(ball_y_no_drag)))

        # Oblicz prędkość całkowitą dla oporu
        speed = math.sqrt(speed_x ** 2 + speed_y ** 2)

        # Oblicz siłę oporu
        drag_force_x = -drag_coefficient * speed * speed_x if speed > 0 else 0
        drag_force_y = -drag_coefficient * speed * speed_y if speed > 0 else 0

        # Aktualizacja prędkości i pozycji z oporem
        speed_x += drag_force_x / ball_mass * time_step
        speed_y += (gravity + drag_force_y / ball_mass) * time_step
        ball_x += speed_x * time_step
        ball_y += speed_y * time_step

        # Aktualizacja prędkości i pozycji bez oporu (dla porównania)
        speed_y_no_drag += gravity * time_step
        ball_x_no_drag += speed_x_no_drag * time_step
        ball_y_no_drag += speed_y_no_drag * time_step

        # Sprawdzenie kolizji z ziemią
        if ball_y >= HEIGHT - 50:
            ball_y = HEIGHT - 50
            simulation_active = False
            reset_ready = True

    # Czyszczenie ekranu
    screen.fill(WHITE)

    # Rysowanie podłoża
    pygame.draw.rect(screen, GREEN, (0, HEIGHT - 40, WIDTH, 40))

    # Rysowanie trajektorii bez oporu (jeśli włączone)
    if show_comparison and len(trajectory_no_drag) > 1:
        pygame.draw.lines(screen, BLUE, False, trajectory_no_drag, 2)

    # Rysowanie trajektorii z oporem
    if len(trajectory) > 1:
        pygame.draw.lines(screen, BLACK, False, trajectory, 2)

    # Rysowanie pocisku
    pygame.draw.circle(screen, RED, (int(ball_x), int(ball_y)), ball_radius)

    # Rysowanie pocisku bez oporu (jeśli włączone i wciąż w powietrzu)
    if show_comparison and ball_y_no_drag < HEIGHT - 50:
        pygame.draw.circle(screen, BLUE, (int(ball_x_no_drag), int(ball_y_no_drag)), ball_radius - 2)

    # Informacje tekstowe
    font = pygame.font.SysFont(None, 24)
    if not simulation_active and not reset_ready:
        text = font.render("Naciśnij SPACJĘ, aby rozpocząć symulację", True, BLACK)
    elif reset_ready:
        text = font.render("Naciśnij SPACJĘ, aby zresetować symulację", True, BLACK)
    else:
        text = font.render(f"Kąt: {angle_degrees}°  Prędkość początkowa: {initial_speed}", True, BLACK)

    screen.blit(text, (10, 10))

    # Informacja o współczynniku oporu
    drag_text = font.render(f"Współczynnik oporu: {drag_coefficient}", True, BLACK)
    screen.blit(drag_text, (10, 40))

    # Informacja o klawiszu C
    compare_text = font.render("Naciśnij C, aby przełączyć widok porównania z rzutem bez oporu", True, BLACK)
    screen.blit(compare_text, (10, 70))

    # Legenda
    if show_comparison:
        legend1 = font.render("Czarny - z oporem ośrodka", True, BLACK)
        legend2 = font.render("Niebieski - bez oporu ośrodka", True, BLUE)
        screen.blit(legend1, (WIDTH - 240, 10))
        screen.blit(legend2, (WIDTH - 240, 40))

    # Aktualizacja wyświetlania
    pygame.display.flip()

    # Ograniczenie liczby klatek na sekundę
    clock.tick(60)

# Zakończenie Pygame
pygame.quit()
sys.exit()