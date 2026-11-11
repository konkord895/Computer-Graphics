import numpy as np
from math import floor
from math import ceil
from numba import njit
from math import cos
from math import sin

@njit(cache=True)
def draw_line(img, px0, py0, px1, py1):  
    d_max = max(abs(floor(px1)-floor(px0)), abs(floor(py1)-floor(py0)))
    L = d_max + 1
    if L == 1:
        img[floor(py0), floor(px0)] = 255
        return

    dx = (px1 - px0)/(L - 1)
    dy = (py1 - py0)/(L - 1)

    x, y = px0, py0

    for i in range(L):
        img[floor(y), floor(x)] = 255
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


@njit(cache=True)
def render_model(img, z_buf, vertices, faces, scale, x_shift, y_shift, z_shift, xr, yr, zr, perspective):
    img[...] = 0
    z_buf[...] = np.inf

    H = img.shape[0]
    W = img.shape[1]

    cx = cos(xr)
    cy = cos(yr)
    cz = cos(zr)
    sx = sin(xr)
    sy = sin(yr)
    sz = sin(zr)
    
    R00 = cy*cz
    R01 = cy*sz
    R02 = sy
    R10 = -cx*sz - sx*sy*cz
    R11 = cx*cz - sx*sy*sz
    R12 = sx*cy
    R20 = sx*sz - cx*sy*cz
    R21 = -sx*cz - cx*sy*sz
    R22 = cx*cy

    for f in faces:
        x0, y0, z0 = vertices[f[0]]
        x1, y1, z1 = vertices[f[1]]
        x2, y2, z2 = vertices[f[2]]

        ox, oy, oz = x0, y0, z0
        x0 = R00*ox + R01*oy + R02*oz + x_shift
        y0 = R10*ox + R11*oy + R12*oz + y_shift
        z0 = R20*ox + R21*oy + R22*oz + z_shift

        ox, oy, oz = x1, y1, z1
        x1 = R00*ox + R01*oy + R02*oz + x_shift
        y1 = R10*ox + R11*oy + R12*oz + y_shift
        z1 = R20*ox + R21*oy + R22*oz + z_shift

        ox, oy, oz = x2, y2, z2
        x2 = R00*ox + R01*oy + R02*oz + x_shift
        y2 = R10*ox + R11*oy + R12*oz + y_shift
        z2 = R20*ox + R21*oy + R22*oz + z_shift

        # Side vectors
        ux, uy, uz = x1 - x2, y1 - y2, z1 - z2
        vx, vy, vz = x1 - x0, y1 - y0, z1 - z0

        # Normal to face
        nx = uy * vz - uz * vy
        ny = uz * vx - ux * vz
        nz = ux * vy - uy * vx

        # Light
        lx, ly, lz = 0, 0, 1

        cosine = (nx*lx + ny*ly + nz*lz) / (np.sqrt(nx*nx + ny*ny + nz*nz) * np.sqrt(lx*lx + ly*ly + lz*lz))

        if perspective:
            px0 = x0/z0 * scale + W/2
            px1 = x1/z1 * scale + W/2
            px2 = x2/z2 * scale + W/2
            py0 = y0/z0 * -scale + H/2
            py1 = y1/z1 * -scale + H/2
            py2 = y2/z2 * -scale + H/2
        else:
            px0 = x0 * scale + W/2
            px1 = x1 * scale + W/2
            px2 = x2 * scale + W/2
            py0 = y0 * -scale + H/2
            py1 = y1 * -scale + H/2
            py2 = y2 * -scale + H/2

        color = -255 * cosine
        if cosine < 0:
            draw_face(img, z_buf, px0, py0, z0, px1, py1, z1, px2, py2, z2, color)


class Renderer:
    def __init__(self, W, H, scale, x_shift, y_shift, z_shift, shift_delta):
        self.W = W
        self.H = H
        self.scale = scale
        self.x_shift = x_shift
        self.y_shift = y_shift
        self.z_shift = z_shift
        self.perspective = True
        self.shift_delta = shift_delta


        self.img = np.zeros((W, H, 3), dtype = np.uint8)
        self.z_buf = np.full((W, H), np.inf, dtype = np.float32)

        self.xr = 0
        self.yr = 0
        self.zr = 0

        


    def render(self, scene):
        render_model(self.img, self.z_buf, scene.vertices, scene.faces,
                      self.scale, self.x_shift, self.y_shift, self.z_shift,
                        self.xr, self.yr, self.zr, self.perspective)
