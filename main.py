import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from stack_robots_model import StorageModel

max_exec_time = 1
model = StorageModel(15, 15, 25)

init_time = time.time()
while model.boxes_available() or time.time() - init_time > max_exec_time:
    model.step()
model.drop_boxes()
end_time = time.time()

exec_time = end_time - init_time

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
