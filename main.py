import matplotlib.pyplot as plt
import sys
import math
import numpy as np
from scipy.integrate import odeint


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
# ROAD_CENTER = (500, 400)  #współrzędne środka drogi
# ROAD_WIDTH = 30  #szerokosc drogi
SCALE_FACTOR = 2  #wspolczynnik skalowania do zamiany pixeli na metry
CIRCUMFERENCE = 2 * math.pi * ROAD_RADIUS / SCALE_FACTOR

time_step = 0.02
# simulation_speed = 1.0
MAX_SIMULATION_TIME = 600

#funkcja licząca s_star
def calculate_s_star(v, delta_v):
    denominator = 2 * math.sqrt(a * b)
    if denominator == 0:  # check for division by zero error and recover from that
        return s0 + v * T  # handle error
    return s0 + v * T + (v * delta_v) / denominator

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
        if s <= 0.1:
            s = 0.1

        delta_v = velocities[i] - velocities[leader_idx]

        s_star = calculate_s_star(velocities[i], delta_v)

        term1 = (velocities[i] / v0) ** delta
        term2 = (s_star / max(s, 0.1)) ** 2

        acceleration = a * (1 - term1 - term2)

        dxdt[n + i] = acceleration

    return dxdt



currents = []
densities = []

for num_vehicles in range(1, 140, 1):  # Iterate from 0 to 1000 vehicles
    # Initial conditions for current num_vehicles
    init_positions = np.linspace(0, CIRCUMFERENCE, num_vehicles, endpoint=False)
    init_velocities = np.ones(
        num_vehicles) * v0  # You can add some small random variation here if needed:  - np.random.rand(NUM_VEHICLES)
    state = np.concatenate((init_positions, init_velocities))

    # Simulate for a short period to reach a stable state for each vehicle count
    t = np.linspace(0, MAX_SIMULATION_TIME,int(MAX_SIMULATION_TIME / time_step))  # Simulate for 50 seconds. Adjust simulation length and/or steps as needed for stable state for all vehicle numbers.
    sim_data = odeint(model, state, t)
    final_velocities = sim_data[-1, num_vehicles:]  # Velocities in the last step of the simulation

    # Calculate current and density
    car_density = num_vehicles / CIRCUMFERENCE
    average_velocity = np.mean(final_velocities)  # Average velocity in the last step
    current = car_density * average_velocity

    currents.append(current)
    densities.append(car_density)

plt.scatter(densities, currents)
plt.xlabel("Car Density (cars/meter)")
plt.ylabel("Car Current (cars/second)")
plt.title("Fundamental Diagram of Traffic Flow")
plt.grid(True)
plt.show()

sys.exit()
