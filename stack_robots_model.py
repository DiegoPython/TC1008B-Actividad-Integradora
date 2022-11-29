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
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.moves = 0
        self.boxes = 0
        self.max_stack = 5
        self.move_flag = True
        self.target_pos = None

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
                    neighbor.boxes = self.max_stack
                    self.boxes = 0
                    self.move_flag = False
                    self.target_pos = None
                    break
                    #print("Boxes placed")
                elif self.boxes < self.max_stack and neighbor.boxes == 1:
                    neighbor.boxes = 0
                    self.boxes += 1
                    self.move_flag = False
                    self.target_pos = None
                    break

        if self.move_flag:
            self.move()

    def move(self):
        if self.target_pos == None:
            self.find_boxes()
        else:
            self.move_to_target()

    def random_move(self):
        neighborhood = self.model.grid.get_neighborhood(
            self.pos,
            moore=False,
            include_center=False
        )
        new_position = self.random.choice(neighborhood)
        self.moves += 1
        self.model.grid.move_agent(self, new_position)

    def move_to_target(self):
        mov_x = self.target_pos[0] - self.pos[0]
        mov_y = self.target_pos[1] - self.pos[1]

        next_x = self.pos[0] + (1 * np.sign(mov_x))
        next_y = self.pos[1] + (1 * np.sign(mov_y)) 

        if mov_x != 0 and mov_y != 0:
            self.moves += 1
        self.moves += 1

        if (next_x, next_y) == self.target_pos:
            self.target_pos = None

        self.model.grid.move_agent(self, (next_x, next_y))

    def find_boxes(self):
        neighbors = self.model.grid.get_neighbors(
            self.pos,
            moore=True,
            include_center=True,
            radius=3
        )

        for neighbor in neighbors:
            if isinstance(neighbor, Tile) and neighbor.boxes == 1 and not neighbor.target:
                print(neighbor.pos)
                self.target_pos = neighbor.pos
                neighbor.target = True
                break

        if self.target_pos != None:
            self.move_to_target()
        else:
            print("Target not found")
        #self.random_move()

    def drop_boxes(self):
        neighbors = self.model.grid.get_neighbors(
            self.pos,
            moore=False,
            include_center=False
        )

        for neighbor in neighbors:
            if isinstance(neighbor, Tile) and neighbor.boxes == 0:
                neighbor.boxes = self.boxes
                self.boxes = 0
                break

class Tile(mesa.Agent):
    def __init__(self, pos, model, boxes=0):
        super().__init__(pos, model)
        self.x, self.y = pos
        self.boxes = boxes
        self.target = False

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
            #print(type(empty_cell))
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
            robot_agent = RobotAgent(i, self)
            
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

    def boxes_available(self):
        for cell in self.grid.coord_iter():
            cell_content, x, y = cell

            for content in cell_content:
                if isinstance(content, Tile) and content.boxes == 1:
                    return True

        return False
