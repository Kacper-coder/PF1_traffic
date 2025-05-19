import matplotlib.pyplot as plt
import sys
import math
import numpy as np
from scipy.integrate import odeint

# parametry modelu
NUM_VEHICLES = 30  # ilość aut
v0 = 30  # pożadana prędkosć
T = 1.5  # bezpieczny odstęp czasowy do poprzedzajacego auta
a = 0.73  # max przyśpieszenie
b = 1.67  # kofortowe hamowanie
delta = 4  # wykładnik członu odpowiadającego za wpływ predkości
s0 = 2  # minimalny dystans od poprzedzajacego auta
vehicle_length = 5  # długość pojazdu

# parametry drogi pojazdu
ROAD_RADIUS = 300  # promień drogi w pixelach
# ROAD_CENTER = (500, 400)  #współrzędne środka drogi
# ROAD_WIDTH = 30  #szerokosc drogi
SCALE_FACTOR = 2  # wspolczynnik skalowania do zamiany pixeli na metry
CIRCUMFERENCE = 2 * math.pi * ROAD_RADIUS / SCALE_FACTOR

time_step = 0.02
# simulation_speed = 1.0
MAX_SIMULATION_TIME = 600  # Max simulation time for the fundamental diagram


# funkcja licząca s_star
def calculate_s_star(v, delta_v):
    denominator = 2 * math.sqrt(a * b)
    if denominator == 0:  # check for division by zero error and recover from that
        return s0 + v * T  # handle error
    return s0 + v * T + (v * delta_v) / denominator


# definicja modelu IDM
def model(state, t):
    # state = [x1, x2, ..., xn, v1, v2, ..., vn]
    n = len(state) // 2
    positions = state[:n]
    velocities = state[n:]

    dxdt = np.zeros(2 * n)

    dxdt[:n] = velocities

    # Liczymy(dv/dt)
    for i in range(n):
        # Pojazd 50 jedzie za 1, a 1 za 2 itd.
        leader_idx = (i + 1) % n

        delta_x = positions[leader_idx] - positions[i]
        if delta_x < 0:
            delta_x += CIRCUMFERENCE

        s = delta_x - vehicle_length
        if s <= 0.1:
            s = 0.1

        delta_v = velocities[i] - velocities[leader_idx]

        s_star = calculate_s_star(velocities[i], delta_v)

        term1 = (velocities[i] / v0) ** delta
        term2 = (s_star / max(s, 0.1)) ** 2

        acceleration = a * (1 - term1 - term2)

        dxdt[n + i] = acceleration

    return dxdt


# --- Fundamental Diagram ---
def generate_fundamental_diagram():
    print("Generating fundamental diagram of traffic flow...")
    currents = []
    densities = []
    velocities = []

    for num_vehicles_fd in range(1, 140, 1):  # Iterate for fundamental diagram
        # Initial conditions for current num_vehicles_fd
        init_positions_fd = np.linspace(0, CIRCUMFERENCE, num_vehicles_fd, endpoint=False)
        init_velocities_fd = np.ones(num_vehicles_fd) * v0
        state_fd = np.concatenate((init_positions_fd, init_velocities_fd))

        # Simulate for a short period to reach a stable state for each vehicle count
        t_fd = np.linspace(0, MAX_SIMULATION_TIME, int(MAX_SIMULATION_TIME / time_step))
        sim_data_fd = odeint(model, state_fd, t_fd)
        final_velocities_fd = sim_data_fd[-1, num_vehicles_fd:]

        # Calculate current and density
        car_density = num_vehicles_fd / CIRCUMFERENCE
        average_velocity = np.mean(final_velocities_fd)
        current = car_density * average_velocity

        currents.append(current)
        densities.append(car_density)
        velocities.append(average_velocity)

    # Create a figure with 2 subplots for the fundamental diagram
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Flow (current) vs Density subplot
    ax1.scatter(densities, currents, color='blue')
    ax1.set_xlabel("Car Density (cars/meter)")
    ax1.set_ylabel("Car Current (cars/second)")
    ax1.set_title("Flow-Density Relationship")
    ax1.grid(True)

    # Speed vs Density subplot
    ax2.scatter(densities, velocities, color='red')
    ax2.set_xlabel("Car Density (cars/meter)")
    ax2.set_ylabel("Average Speed (m/s)")
    ax2.set_title("Speed-Density Relationship")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


# --- Function to create one simulation solution that can be used for multiple plots ---
def run_simulation(num_vehicles=30, total_time=2100.0):
    print(f"Running simulation for {total_time} seconds with {num_vehicles} vehicles...")

    # Create initial conditions with a small perturbation to encourage wave formation
    init_positions = np.linspace(0, CIRCUMFERENCE, num_vehicles, endpoint=False)
    init_velocities = np.ones(num_vehicles) * v0

    # Add a perturbation to one vehicle to encourage wave formation
    perturb_idx = num_vehicles // 4
    init_velocities[perturb_idx] = v0 * 0.5  # Slow down one vehicle to 50% speed

    initial_state = np.concatenate((init_positions, init_velocities))

    # Time points for the simulation
    t_full = np.linspace(0, total_time, int(total_time / time_step))

    # Run the simulation
    solution = odeint(model, initial_state, t_full)

    return solution, t_full


# --- Function to plot trajectory for a specific time window ---
def plot_trajectory_window(solution, t_full, num_vehicles, window_start, window_end, window_title):
    # Extract positions of all vehicles over time
    vehicle_positions_over_time = solution[:, :num_vehicles]

    # Create the plot
    plt.figure(figsize=(10, 8))

    # Plot each vehicle's trajectory
    for i in range(num_vehicles):
        # Apply modulo CIRCUMFERENCE to wrap positions for plotting
        wrapped_positions = vehicle_positions_over_time[:, i] % CIRCUMFERENCE
        plt.plot(wrapped_positions, t_full, 'k.', markersize=1)

    plt.xlabel("Car Position on Road (m) (Wrapped)")
    plt.ylabel("Time (s)")
    plt.title(f"{window_title} ({num_vehicles} Vehicles)")
    plt.xlim(0, CIRCUMFERENCE)
    plt.ylim(window_start, window_end)  # Show only the requested time window
    plt.grid(False)  # Disable grid to make the waves more visible

    # Add visual indicators for the axes
    plt.text(-10, window_start, f"t={int(window_start)}", color='red', fontsize=12)
    plt.text(-10, window_end, f"t={int(window_end)}", color='red', fontsize=12)
    plt.text(0, window_end + 20, "x=0", color='blue', fontsize=12)
    plt.text(CIRCUMFERENCE, window_end + 20, f"x={int(CIRCUMFERENCE)}", color='blue', fontsize=12)

    # Add an arrow showing the direction
    plt.annotate("", xy=(CIRCUMFERENCE, window_end + 20), xytext=(CIRCUMFERENCE / 2, window_end + 20),
                 arrowprops=dict(arrowstyle="->", color='blue'))

    plt.tight_layout()
    plt.show()


# --- Generate multiple trajectory plots with different time windows ---
def generate_trajectory_plots():
    # Run one long simulation
    total_time = 2100.0
    solution, t_full = run_simulation(NUM_VEHICLES, total_time)

    # Plot early development of traffic waves (300-500s)
    plot_trajectory_window(
        solution, t_full, NUM_VEHICLES,
        window_start=300.0, window_end=500.0,
        window_title="Early Wave Development"
    )

    # Plot mid-simulation traffic waves (900-1100s)
    plot_trajectory_window(
        solution, t_full, NUM_VEHICLES,
        window_start=900.0, window_end=1100.0,
        window_title="Mid-Simulation Wave Propagation"
    )

    # Plot late-simulation traffic waves (1800-2000s)
    plot_trajectory_window(
        solution, t_full, NUM_VEHICLES,
        window_start=1800.0, window_end=2000.0,
        window_title="Late-Simulation Wave Propagation"
    )


# Run the simulations
if __name__ == "__main__":
    # Generate fundamental diagram (flow-density and speed-density relationships)
    generate_fundamental_diagram()

    # Generate multiple trajectory plots at different time windows
    generate_trajectory_plots()

    sys.exit()