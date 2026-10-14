import numpy as np
from PIL import Image
from math import floor
import math


def draw_line(img_mat, p1, p2):  
    d_max = max(abs(floor(p2[0])-floor(p1[0])), abs(floor(p2[1])-floor(p1[1])))
    L = d_max + 1
    if L == 1:
        img_mat[floor(p1[1]), floor(p1[0])] = 255
        return

    dx = (p2[0] - p1[0])/(L - 1)
    dy = (p2[1] - p1[1])/(L - 1)

    x, y = p1[:2]


    


    for i in range(L):
        img_mat[floor(y), floor(x)] = 255
        x += dx
        y += dy


img_mat = np.zeros((1000, 1000, 3), dtype = np.uint8)


with open("bunny.obj") as f:

    vertices = []
    faces = []
    for line in f:
        data = line.split()
        if data[0] == "v":
            vertices.append(list(map(float, data[1:4])))

        elif data[0] == "f":
            faces.append([int(data[i].split('/')[0]) for i in range(1, 4)])

scale = 5000
x_offset, y_offset = 500, 800
z_offset = 0
vertices = [[v[0] * scale + x_offset, v[1]* - scale + y_offset, v[2] * scale + z_offset] for v in vertices]

for f in faces:
    for i in range(3):
        draw_line(img_mat, vertices[f[i] - 1], vertices[f[(i+1) % 3] - 1])


    
img = Image.fromarray(img_mat)
img.save('img.png')



