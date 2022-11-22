import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from stack_robots_model import StorageModel

max_exec_time = 1
model = StorageModel(5, 5, 25)

for i in range(100):
    model.step()

data = model.datacollector.get_model_vars_dataframe()

figure, axis = plt.subplots(figsize=(7,7))
axis.set_xticks([])
axis.set_yticks([])
#patch = plt.imshow(data.iloc[0][0], cmap='Greys')
patch = plt.imshow(data.iloc[0][0], interpolation="nearest")
plt.colorbar()

def animate(i):
    patch.set_data(data.iloc[i][0])

anim = animation.FuncAnimation(figure, animate, frames=len(data))

plt.show()
