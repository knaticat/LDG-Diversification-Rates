import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

random.seed(1234)

# Constants
GRID_SIZE = 100
NUM_SPECIES = 20
TIMESTEP = 100

#Distribution of rates
mean_ori= 0.0001*GRID_SIZE
std_ori = 0.01*GRID_SIZE
mean_ex = 0.0001*GRID_SIZE
std_ex = 0.01*GRID_SIZE
mean_im = 0.0001*GRID_SIZE
std_im = 0.05*GRID_SIZE

boundaries = [GRID_SIZE/3, GRID_SIZE*2/3]

def normal_gradient_array(array_size, mean=0, std_dev=1):
    gradient_values = np.linspace(-3, 3, array_size) 
    gradient = np.exp(-0.5 * ((gradient_values - mean) / std_dev) ** 2)
    gradient_array = np.outer(np.ones(array_size), gradient)
    

    return gradient_array

def inverse_normal_gradient_array(array_size, mean=0, std_dev=1):
    gradient_values = np.linspace(-3, 3, array_size)  
    gradient = np.exp(-0.5 * ((gradient_values - mean) / std_dev) ** 2)
    inverse_gradient = 1 - gradient
    gradient_array = np.outer(np.ones(array_size), inverse_gradient)


    return gradient_array

# Initialize grid, diversification rates, and species colors
grid = np.zeros((GRID_SIZE, GRID_SIZE))

#Uncomment this to simulate Tropic as Cradle Model
# origination_rates = normal_gradient_array(GRID_SIZE, mean=mean_ori, std_dev=std_ori)
# extinction_rates = np.ones((GRID_SIZE, GRID_SIZE))*0.1
# immigration_rates = np.zeros((GRID_SIZE, GRID_SIZE))

#Uncomment this to simulate Tropic as Museum Model
# origination_rates = np.ones((GRID_SIZE, GRID_SIZE))*0.1
# extinction_rates = inverse_normal_gradient_array(GRID_SIZE, mean=mean_ex, std_dev=std_ex)
# immigration_rates = np.zeros((GRID_SIZE, GRID_SIZE))

#Uncomment for OTT model
origination_rates = normal_gradient_array(GRID_SIZE, mean=mean_ori, std_dev=std_ori)
extinction_rates = inverse_normal_gradient_array(GRID_SIZE, mean=mean_ex, std_dev=std_ex)
immigration_rates = inverse_normal_gradient_array(GRID_SIZE, mean = mean_im, std_dev=std_im)


plt.imshow(origination_rates, cmap='viridis', interpolation='nearest')
for boundary in boundaries:
      plt.axvline(x=boundary, color='black', linestyle='--', linewidth=1)
plt.colorbar()  # Add a colorbar to the plot
plt.title('Origination Rates')
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.show()

plt.imshow(extinction_rates, cmap='viridis', interpolation='nearest')
for boundary in boundaries:
      plt.axvline(x=boundary, color='black', linestyle='--', linewidth=1)
plt.colorbar()  # Add a colorbar to the plot
plt.title('Extinction Rates')
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.show()

plt.imshow(immigration_rates, cmap='viridis', interpolation='nearest')
for boundary in boundaries:
      plt.axvline(x=boundary, color='black', linestyle='--', linewidth=1)
plt.colorbar()  # Add a colorbar to the plot
plt.title('Immigration Rates')
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.show()

# Function to update the plot for each frame
def update(grid):      
    new_grid = grid.copy()
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            if grid[x, y] == 0:
                if np.random.rand() < origination_rates[x, y]:
                    new_grid[x, y] = np.random.randint(1, NUM_SPECIES + 1)
            else:
                if np.random.rand() < extinction_rates[x, y]:
                    new_grid[x, y] = 0
                elif np.random.rand() < immigration_rates[x, y]:
                      new_x = np.random.randint(0, GRID_SIZE)
                      new_y = np.random.randint(0, GRID_SIZE)
                      if new_grid[new_x, new_y] == 0:
                        new_grid[new_x, new_y] = grid[x, y]
                      else:
                        new_grid[x, y] = grid[x, y]

    return new_grid


states = [grid]
species_count = [[0]]
for i in range(TIMESTEP):
    new_state = update(grid=states[i])
    unique_counts = np.apply_along_axis(lambda x: len(np.unique(x)) - 1, axis=0, arr = new_state)
    species_count.append(unique_counts)
    states.append(new_state)



fig = plt.figure()

plt.ion()
for i in range(len(states)):
    plt.clf()
    plt.imshow(states[i], cmap='Set3', interpolation='nearest', aspect='auto')
    for boundary in boundaries:
      plt.axvline(x=boundary, color='black', linestyle='--', linewidth=1)
    #plt.title(f'TimeStep {i+ 1}')
    plt.colorbar()
    plt.show()
    if i in [2, 50, 100]:
        save_path = f'plot_iteration_{i}.png'
        plt.savefig(save_path)
        print(f'Plot saved at iteration {i} as {save_path}')
   # input("Press Enter to continue...")
    plt.pause(0.2)
    plt.clf()
    plt.bar(np.arange(0, GRID_SIZE, step=1), species_count[i], color='skyblue', edgecolor='black')
    plt.title('Distribution of Lineage Frequency across Latitude')
    plt.xlabel('Latitude')
    plt.ylabel('Lineage Frequency')
    plt.show()
    if i in [2, 50, 100]:
        save_path = f'plot_iteration_{i}_dist.png'
        plt.savefig(save_path)
        print(f'Plot saved at iteration {i} as {save_path}')
    plt.pause(1)
