import numpy as np

class Scene:
    def __init__(self, filename):        
        self.vertices = []
        self.faces = []

        with open(filename) as f:
            for line in f:
                data = line.split()
                if not data:
                    continue
                if data[0] == "v":
                    self.vertices.append(list(map(float, data[1:4])))

                elif data[0] == "f":
                    self.faces.append([int(data[i].split('/')[0]) - 1 for i in range(1, 4)])

        self.vertices = np.asarray(self.vertices, dtype=np.float64)
        self.faces = np.asarray(self.faces, dtype=np.int64)

        
            


