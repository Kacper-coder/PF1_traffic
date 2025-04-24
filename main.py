import pygame
import sys
import math
import numpy as np
from scipy.integrate import odeint

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 1000, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Ring Road Traffic Flow Simulation with Traffic Waves")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 200, 0)
GRAY = (200, 200, 200)

# Simulation parameters from the provided model
NUM_VEHICLES = 50  # 50 vehicles as specified
v0 = 30  # Desired velocity (m/s)
T = 1.5  # Safe time headway (s)
a = 0.73  # Maximum acceleration (m/s^2)
b = 1.67  # Comfortable deceleration (m/s^2)
delta = 4  # Acceleration exponent
s0 = 2  # Minimum distance (m)
vehicle_length = 5  # Vehicle length (m)

# Ring road properties
ROAD_RADIUS = 600  # Radius of the ring road (pixels)
ROAD_CENTER = (WIDTH // 2, HEIGHT // 2)  # Center of the ring road
ROAD_WIDTH = 30  # Width of the road (pixels)
SCALE_FACTOR = 2  # Scale factor to convert meters to pixels

# Initial conditions - ADJUSTED TO CREATE HIGHER DENSITY
# Making the road smaller to increase vehicle density
CIRCUMFERENCE = 2 * math.pi * ROAD_RADIUS / SCALE_FACTOR * 0.5  # Reduced road length for higher density
init_positions = np.linspace(0, CIRCUMFERENCE, NUM_VEHICLES, endpoint=False)  # Uniform positions

# Start all vehicles at a more congested speed
init_velocities = np.ones(NUM_VEHICLES) * v0 * 0.6  # Start slower (60% of desired speed)

# Add perturbation to one vehicle to trigger traffic waves
perturb_vehicle = NUM_VEHICLES // 4  # Choose a vehicle to perturb
init_velocities[perturb_vehicle] *= 0.3  # Dramatically slow down one vehicle

# Simulation state
positions = init_positions.copy()
velocities = init_velocities.copy()

# Time settings
time_step = 0.02  # Small time step for accurate simulation
clock = pygame.time.Clock()


# Calculate s*(v, Δv) - the desired distance as per the equation
def calculate_s_star(v, delta_v):
    return s0 + v * T + (v * delta_v) / (2 * math.sqrt(a * b))


# Calculate the derivative for the ODE solver based on the equations in the image
def model(state, t):
    # state = [x1, x2, ..., xn, v1, v2, ..., vn]
    n = len(state) // 2
    positions = state[:n]
    velocities = state[n:]

    # Initialize derivatives
    dxdt = np.zeros(2 * n)

    # Position derivatives are velocities (dx/dt = v)
    dxdt[:n] = velocities

    # Calculate velocity derivatives (dv/dt) based on the model
    for i in range(n):
        # Vehicle 1 follows vehicle 50, vehicle 2 follows 1, etc.
        leader_idx = (i + 1) % n

        # Calculate headway (distance to the leading vehicle)
        delta_x = positions[leader_idx] - positions[i]
        # Adjust for the ring road (if the leader is behind in the array)
        if delta_x < 0:
            delta_x += CIRCUMFERENCE

        # Calculate the actual gap (distance minus vehicle length)
        s = delta_x - vehicle_length

        # Calculate the velocity difference (negative when catching up)
        delta_v = velocities[i] - velocities[leader_idx]

        # Calculate the desired distance s*(v, Δv)
        s_star = calculate_s_star(velocities[i], delta_v)

        # Calculate acceleration using the formula from the image
        term1 = (velocities[i] / v0) ** delta
        term2 = (s_star / max(s, 0.1)) ** 2  # Prevent division by zero

        acceleration = a * (1 - term1 - term2)

        # Update velocity derivative
        dxdt[n + i] = acceleration

    return dxdt


# Function to convert model position to screen coordinates
def position_to_screen(position):
    # Convert model position (in meters) to angle around the ring
    angle = (position / CIRCUMFERENCE) * 2 * math.pi - math.pi / 2  # Start from top
    # Convert polar coordinates to screen coordinates
    x = ROAD_CENTER[0] + ROAD_RADIUS * math.cos(angle)
    y = ROAD_CENTER[1] + ROAD_RADIUS * math.sin(angle)
    return int(x), int(y)


# Function to draw a car at a given position
def draw_car(position, velocity, index):
    # Determine car color based on velocity
    velocity_ratio = min(velocity / v0, 1.0)
    # Gradient from red (slow) to green (desired speed)
    color = (
        int(255 * (1 - velocity_ratio)),
        int(255 * velocity_ratio),
        0
    )

    # Draw car circle
    car_pos = position_to_screen(position)
    car_radius = 8
    pygame.draw.circle(screen, color, car_pos, car_radius)

    # Draw velocity indicator (longer for faster speeds)
    angle = (position / CIRCUMFERENCE) * 2 * math.pi - math.pi / 2
    indicator_length = car_radius * (0.5 + velocity_ratio)
    end_x = car_pos[0] - math.cos(angle) * indicator_length
    end_y = car_pos[1] - math.sin(angle) * indicator_length
    pygame.draw.line(screen, BLACK, car_pos, (end_x, end_y), 2)

    # Optionally show vehicle index
    if NUM_VEHICLES <= 20:
        font = pygame.font.SysFont(None, 20)
        text = font.render(f"{index}", True, WHITE)
        text_rect = text.get_rect(center=car_pos)
        screen.blit(text, text_rect)


# Combine positions and velocities into a single state vector
state = np.concatenate((positions, velocities))

# Main simulation loop
paused = False
show_info = True
running = True
simulation_time = 0
simulation_speed = 1.0  # Simulation speed multiplier
show_wave_analysis = True  # Display traffic wave analysis

# Variables to track traffic wave
wave_detection_threshold = 0.6  # Threshold for detecting slow vehicles (fraction of v0)
previous_slow_vehicle = None
wave_speed = 0

# Add slider for controlling simulation speed
speed_slider_rect = pygame.Rect(WIDTH - 220, HEIGHT - 40, 200, 20)
speed_slider_button_radius = 10
speed_slider_button_pos = WIDTH - 220 + int(simulation_speed * 100)
is_dragging = False

# Create a history of traffic conditions
position_history = []
velocity_history = []
history_length = 100
history_sample_rate = 5
frame_count = 0

while running:
    # Handle events
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
            # Add speed controls
            elif event.key == pygame.K_UP:
                simulation_speed = min(simulation_speed + 0.1, 2.0)
            elif event.key == pygame.K_DOWN:
                simulation_speed = max(simulation_speed - 0.1, 0.1)
            # Add perturbation on demand
            elif event.key == pygame.K_p:
                # Slow down a random vehicle significantly
                random_vehicle = np.random.randint(0, NUM_VEHICLES)
                velocities[random_vehicle] *= 0.3
                state[NUM_VEHICLES + random_vehicle] = velocities[random_vehicle]
        # Handle slider for simulation speed
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if speed_slider_rect.collidepoint(event.pos):
                is_dragging = True
        elif event.type == pygame.MOUSEBUTTONUP:
            is_dragging = False
        elif event.type == pygame.MOUSEMOTION and is_dragging:
            x = max(speed_slider_rect.left, min(event.pos[0], speed_slider_rect.right))
            speed_slider_button_pos = x
            simulation_speed = (x - speed_slider_rect.left) / 200 * 2.0  # 0.0 to 2.0

    # Update simulation if not paused
    if not paused:
        # Use adjusted time step based on simulation speed
        adjusted_time_step = time_step * simulation_speed

        # Use ODE solver to advance the model
        new_state = odeint(model, state, [0, adjusted_time_step])[1]
        state = new_state

        # Extract positions and velocities from state
        positions = state[:NUM_VEHICLES]
        velocities = state[NUM_VEHICLES:]

        # Ensure positions stay within the circumference
        positions = positions % CIRCUMFERENCE

        # Save history periodically
        frame_count += 1
        if frame_count % history_sample_rate == 0 and len(position_history) < history_length:
            position_history.append(positions.copy())
            velocity_history.append(velocities.copy())

        # Detect traffic wave (find the slowest vehicle)
        slowest_idx = np.argmin(velocities)
        slowest_velocity = velocities[slowest_idx]

        # Simple wave detection
        if previous_slow_vehicle is not None and slowest_velocity < wave_detection_threshold * v0:
            # If the slowest vehicle has changed, we might be seeing a wave
            if slowest_idx != previous_slow_vehicle:
                # Calculate direction and approximate speed of the wave
                wave_direction = -1 if (slowest_idx - previous_slow_vehicle) % NUM_VEHICLES < NUM_VEHICLES / 2 else 1
                wave_speed = wave_direction * (
                            positions[slowest_idx] - positions[previous_slow_vehicle]) / adjusted_time_step
                # Adjust for circular road
                if abs(wave_speed) > CIRCUMFERENCE / 2:
                    wave_speed = wave_direction * (CIRCUMFERENCE - abs(
                        positions[slowest_idx] - positions[previous_slow_vehicle])) / adjusted_time_step

        previous_slow_vehicle = slowest_idx

        # Increment simulation time
        simulation_time += adjusted_time_step

    # Clear the screen
    screen.fill(WHITE)

    # Draw the ring road
    pygame.draw.circle(screen, GRAY, ROAD_CENTER, ROAD_RADIUS + ROAD_WIDTH // 2, ROAD_WIDTH)

    # Draw vehicles
    for i in range(NUM_VEHICLES):
        draw_car(positions[i], velocities[i], i + 1)

    # Draw simulation speed slider
    pygame.draw.rect(screen, BLACK, speed_slider_rect, 1)
    speed_slider_button_pos = speed_slider_rect.left + int(simulation_speed * 100)
    pygame.draw.circle(screen, BLACK, (speed_slider_button_pos, speed_slider_rect.centery), speed_slider_button_radius)

    # Draw speed label
    font = pygame.font.SysFont(None, 24)
    speed_text = font.render(f"Speed: {simulation_speed:.1f}x", True, BLACK)
    screen.blit(speed_text, (WIDTH - 320, HEIGHT - 45))

    # Draw information
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

        # Draw traffic wave analysis
        if show_wave_analysis:
            wave_info = [
                "Traffic Wave Analysis:",
                f"Slowest vehicle: #{previous_slow_vehicle + 1}",
                f"Slowest speed: {np.min(velocities):.2f} m/s",
                f"Approx. wave speed: {-wave_speed:.2f} m/s"  # Negative to show backward propagation
            ]

            for line in wave_info:
                text = font.render(line, True, BLACK)
                screen.blit(text, (10, y_pos))
                y_pos += 25

    # Update the display
    pygame.display.flip()

    # Cap the frame rate
    clock.tick(60)

# Quit pygame
pygame.quit()
sys.exit()