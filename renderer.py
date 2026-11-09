import numpy as np
from math import floor
from math import ceil
from numba import njit


@njit(cache=True)
def draw_line(self, px0, py0, px1, py1):  
    d_max = max(abs(floor(px1)-floor(px0)), abs(floor(py1)-floor(py0)))
    L = d_max + 1
    if L == 1:
        self.img[floor(py0), floor(px0)] = 255
        return

    dx = (px1 - px0)/(L - 1)
    dy = (py1 - py0)/(L - 1)

    x, y = px0, py0

    for i in range(L):
        self.img[floor(y), floor(x)] = 255
        x += dx
        y += dy


# Barycentric coordinates
@njit(cache=True)
def bc(px, py, px0, py0, px1, py1, px2, py2):  
    lambda0 = ((px - px2) * (py1 - py2) - (px1 - px2) * (py - py2)) /  ((px0 - px2) * (py1 - py2) - (px1 - px2) * (py0 - py2)) 
    lambda1 = ((px0 - px2) * (py - py2) - (px - px2) * (py0 - py2)) /  ((px0 - px2) * (py1 - py2) - (px1 - px2) * (py0 - py2)) 
    lambda2 = 1.0 - lambda0 - lambda1 

    return (lambda0, lambda1, lambda2)


@njit(cache=True)
def draw_face(img, z_buf, px0, py0, z0, px1, py1, z1, px2, py2, z2, color):
    H = img.shape[0]
    W = img.shape[1]

    xmin = floor(min(px0, min(px1, px2)))
    ymin = floor(min(py0, min(py1, py2)))
    xmax = ceil(max(px0, max(px1, px2)))
    ymax = ceil(max(py0, max(py1, py2)))

    if xmin < 0: xmin = 0
    if ymin < 0: ymin = 0
    if xmax > W: xmax = W
    if ymax > H: ymax = H

    for x in range(xmin, xmax):
        for y in range(ymin, ymax):
            l0, l1, l2 = bc(x+0.5, y+0.5, px0, py0, px1, py1, px2, py2)
            if l0 > 0 and l1 > 0 and l2 > 0:
                z = l0 * z0 + l1 * z1 + l2 * z2
                if z < z_buf[y, x]:
                    z_buf[y, x] = z
                    img[y, x] = color


def render_model(img, z_buf, scene, scale, x_offset, y_offset):
    img[...] = 0
    z_buf[...] = np.inf

    for f in scene.faces:
        x0, y0, z0 = scene.vertices[f[0]]
        x1, y1, z1 = scene.vertices[f[1]]
        x2, y2, z2 = scene.vertices[f[2]]

        # Side vectors
        ux, uy, uz = x1 - x2, y1 - y2, z1 - z2
        vx, vy, vz = x1 - x0, y1 - y0, z1 - z0

        # Normal to face
        nx = uy * vz - uz * vy
        ny = uz * vx - ux * vz
        nz = ux * vy - uy * vx

        # Light
        lx, ly, lz = 0, 0, 1

        cos = (nx*lx + ny*ly + nz*lz) / (np.sqrt(nx*nx + ny*ny + nz*nz) * np.sqrt(lx*lx + ly*ly + lz*lz))

        px0 = x0 * scale + x_offset
        px1 = x1 * scale + x_offset
        px2 = x2 * scale + x_offset
        py0 = y0 * -scale + y_offset
        py1 = y1 * -scale + y_offset
        py2 = y2 * -scale + y_offset

        color = -255 * cos
        if cos < 0:
            draw_face(img, z_buf, px0, py0, z0, px1, py1, z1, px2, py2, z2, color)


class Renderer:
    def __init__(self, W, H, scale, x_offset, y_offset):
        self.W = W
        self.H = H
        self.scale = scale
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.img = np.zeros((W, H, 3), dtype = np.uint8)
        self.z_buf = np.full((W, H), np.inf, dtype = np.float32)


    def render(self, scene):
        render_model(self.img, self.z_buf, scene, self.scale, self.x_offset, self.y_offset)
