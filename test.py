import numpy as np
from PIL import Image
from math import floor
from math import ceil

width = 1000
height = 1000

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



def bc(p, p0, p1, p2):
    lambda0 = ((p[0] - p2[0]) * (p1[1] - p2[1]) - (p1[0] - p2[0]) * (p[1] - p2[1])) /  ((p0[0] - p2[0]) * (p1[1] - p2[1]) - (p1[0] - p2[0]) * (p0[1] - p2[1])) 
    lambda1 = ((p0[0] - p2[0]) * (p[1] - p2[1]) - (p[0] - p2[0]) * (p0[1] - p2[1])) /  ((p0[0] - p2[0]) * (p1[1] - p2[1]) - (p1[0] - p2[0]) * (p0[1] - p2[1])) 
    lambda2 = 1.0 - lambda0 - lambda1 

    return (lambda0, lambda1, lambda2)

def draw_face(img_mat, z_buffer, p0, p1, p2, color):
    xmin = floor(min(p0[0], p1[0], p2[0]))
    ymin = floor(min(p0[1], p1[1], p2[1]))
    xmax = ceil(max(p0[0], p1[0], p2[0]))
    ymax = ceil(max(p0[1], p1[1], p2[1]))

    if xmin < 0: xmin = 0
    if ymin < 0: ymin = 0
    if xmax > width: xmax = width
    if ymax > height: ymax = height

    for x in range(xmin, xmax):
        for y in range(ymin, ymax):
            l0, l1, l2 = bc([x+0.5, y+0.5], p0, p1, p2)
            if l0 > 0 and l1 > 0 and l2 > 0:
                z = l0*p0[2] + l1*p1[2] + l2*p2[2]
                if z < z_buffer[y, x]:
                    z_buffer[y, x] = z
                    img_mat[y, x] = color


img_mat = np.zeros((width, height, 3), dtype = np.uint8)
z_buffer = np.full((width, height), np.inf, dtype = np.float32)


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
vertices = [[v[0] * scale + x_offset, v[1] * -scale + y_offset, v[2] * scale + z_offset] for v in vertices]

for f in faces:
    p0 = vertices[f[0]-1]
    p1 = vertices[f[1]-1]
    p2 = vertices[f[2]-1]
    v1 = [p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]]
    v2 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]]
    normal = -np.cross(v1, v2)
    light = [0, 0, 1]

    cos = np.dot(normal, light)/(np.linalg.norm(normal) * np.linalg.norm(light))

    color = -255 * cos
    if cos < 0:
        draw_face(img_mat, z_buffer, p0, p1, p2, color)

img = Image.fromarray(img_mat)
img.save('img.png')



