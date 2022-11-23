import mesa
import math
import numpy as np

def get_model_grid(model):
    grid = np.zeros((model.grid.width, model.grid.height))

    for cell in model.grid.coord_iter():
        cell_content, x, y = cell

        for content in cell_content:
            if isinstance(content, RobotAgent):
                grid[x][y] = 11 - content.boxes
            else:
                grid[x][y] = content.boxes
    return grid

class RobotAgent(mesa.Agent):
    def __init__(self, unique_id, model, max_stack):
        super().__init__(unique_id, model)
        self.moves = 0
        self.boxes = 0
        self.max_stack = max_stack
        self.move_flag = True

    def step(self):
        self.move_flag = True
        neighbors = self.model.grid.get_neighbors(
            self.pos,
            moore=False,
            include_center=True
        )

        for neighbor in neighbors:
            if isinstance(neighbor, Tile):
                if self.boxes == self.max_stack and neighbor.boxes == 0:
                    self.boxes = 0
                    neighbor.boxes = self.max_stack
                    self.move_flag = False
                    break
                    #print("Boxes placed")
                elif self.boxes < self.max_stack and neighbor.boxes == 1:
                    neighbor.boxes = 0
                    self.boxes += 1
                    self.move_flag = False
                    break

        if self.move_flag:
            self.move()

    def move(self):
        neighborhood = self.model.grid.get_neighborhood(
            self.pos,
            moore=False,
            include_center=False
        )
        new_position = self.random.choice(neighborhood)
        self.moves += 1
        self.model.grid.move_agent(self, new_position)

    def drop_boxes(self):
        neighbors = self.model.grid.get_neighbors(
            self.pos,
            moore=False,
            include_center=False
        )

        for neighbor in neighbors:
            if isinstance(neighbor, Tile) and neighbor.boxes == 0:
                self.boxes = 0
                neighbor.boxes = self.boxes
                break

class Tile(mesa.Agent):
    def __init__(self, pos, model, boxes=0):
        super().__init__(pos, model)
        self.x, self.y = pos
        self.boxes = boxes

class StorageModel(mesa.Model):
    def __init__(self, width, height, K):
        self.num_agents = 5
        self.grid = mesa.space.MultiGrid(width, height, False)
        self.schedule = mesa.time.SimultaneousActivation(self)
        self.boxes = K

        #Posicionamos las cajas en posiciones aleatorias vacias
        empty_cells = list(self.grid.empties)
        for cell in range(self.boxes):
            empty_cell = self.random.choice(empty_cells)
            tile = Tile(empty_cell, self, 1)
            self.grid.place_agent(tile, empty_cell)
            self.schedule.add(tile)
            empty_cells.remove(empty_cell)
            
        empty_cells = list(self.grid.empties)
        for cell in empty_cells:
            tile = Tile(cell, self)
            self.grid.place_agent(tile, cell)
            self.schedule.add(tile)

        #Posicionamos nuestros agentes en posiciones aleatorias
        for i in range(self.num_agents):
            robot_agent = RobotAgent(i, self, self.max_stack_possible())
            
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)

            self.grid.place_agent(robot_agent, (x, y))
            self.schedule.add(robot_agent)

        self.datacollector = mesa.DataCollector(
            model_reporters={'Grid': get_model_grid},
            agent_reporters={'Moves': lambda a: getattr(a, 'moves', None)}
        )

        self.datacollector.collect(self)

    def step(self):
        self.schedule.step()
        self.datacollector.collect(self)

    def drop_boxes(self):
        for agent in self.schedule.agents:
            if isinstance(agent, RobotAgent):
                agent.drop_boxes()
        self.datacollector.collect(self)

    def max_stack_possible(self):
        max_stack = math.ceil(self.boxes / 5)
        return 5 if max_stack > 5 else max_stack

    def boxes_available(self):
        for cell in self.grid.coord_iter():
            cell_content, x, y = cell

            for content in cell_content:
                if isinstance(content, Tile) and content.boxes == 1:
                    return True

        return False
