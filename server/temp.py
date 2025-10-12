# Pyglet 2.1.9 • Modern OpenGL (core profile) • No fixed-function, no GLU.
import math, time, random, ctypes
from dataclasses import dataclass
from typing import List, Tuple

import pyglet
from pyglet.window import key, mouse
from pyglet import gl
from pyglet.graphics.shader import Shader, ShaderProgram

# ------------------------------------------------------------
# Minimal OBJ loader (positions only) with triangle fan for ngons
# ------------------------------------------------------------
def load_obj_positions(obj_path: str, scale=1.0, center_y=0.0):
    verts = []
    faces = []
    with open(obj_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if not line or line.startswith('#'): 
                continue
            parts = line.strip().split()
            if not parts: 
                continue
            tag = parts[0].lower()
            if tag == 'v' and len(parts) >= 4:
                x, y, z = map(float, parts[1:4])
                verts.append((x*scale, y*scale + center_y, z*scale))
            elif tag == 'f' and len(parts) >= 4:
                idx = []
                for p in parts[1:]:
                    a = p.split('/')[0]
                    if a: idx.append(int(a)-1)
                for k in range(1, len(idx)-1):
                    faces.append((idx[0], idx[k], idx[k+1]))

    tri = []
    for a, b, c in faces:
        tri.extend([verts[a], verts[b], verts[c]])

    # Print bounds once to help sanity-check scale
    if verts:
        xs, ys, zs = zip(*verts)
        print(f"OBJ bounds: X[{min(xs):.3f},{max(xs):.3f}] "
              f"Y[{min(ys):.3f},{max(ys):.3f}] "
              f"Z[{min(zs):.3f},{max(zs):.3f}]")
        print(f"Approx size (m): {(max(xs)-min(xs)):.3f} x {(max(ys)-min(ys)):.3f} x {(max(zs)-min(zs)):.3f}")

    color = (0.12, 0.75, 0.90)  # cyan-ish flat color
    cols  = [color] * len(tri)
    return tri, cols

# ------------------------------------------------------------
# Math utils
# ------------------------------------------------------------
def clamp(x, lo, hi): return lo if x < lo else hi if x > hi else x

def mat4_identity():
    return [1,0,0,0,
            0,1,0,0,
            0,0,1,0,
            0,0,0,1]

def mat4_mul(a, b):  # a @ b, column-major
    out = [0]*16
    for c in range(4):
        for r in range(4):
            out[c*4 + r] = (a[0*4 + r]*b[c*4 + 0] +
                            a[1*4 + r]*b[c*4 + 1] +
                            a[2*4 + r]*b[c*4 + 2] +
                            a[3*4 + r]*b[c*4 + 3])
    return out

def mat4_translate(x, y, z):
    m = mat4_identity()
    m[12], m[13], m[14] = x, y, z
    return m

def mat4_rotate_y(theta):
    c, s = math.cos(theta), math.sin(theta)
    return [ c,0, s,0,
             0,1, 0,0,
            -s,0, c,0,
             0,0, 0,1]

def perspective(fovy_deg, aspect, znear, zfar):
    f = 1.0 / math.tan(math.radians(fovy_deg)/2.0)
    nf = 1.0 / (znear - zfar)
    return [f/aspect, 0, 0,                           0,
            0,        f, 0,                           0,
            0,        0, (zfar+znear)*nf,            -1,
            0,        0, (2*znear*zfar)*nf,           0]

def look_at(eye, center, up):
    ex,ey,ez = eye; cx,cy,cz = center; ux,uy,uz = up
    fx,fy,fz = cx-ex, cy-ey, cz-ez
    fl = max(1e-9, math.sqrt(fx*fx+fy*fy+fz*fz))
    fx,fy,fz = fx/fl, fy/fl, fz/fl
    sx,sy,sz = fy*uz - fz*uy, fz*ux - fx*uz, fx*uy - fy*ux
    sl = max(1e-9, math.sqrt(sx*sx+sy*sy+sz*sz))
    sx,sy,sz = sx/sl, sy/sl, sz/sl
    ux,uy,uz = sy*fz - sz*fy, sz*fx - sx*fz, sx*fy - sy*fx
    m = [ sx, ux, -fx, 0,
          sy, uy, -fy, 0,
          sz, uz, -fz, 0,
           0,  0,   0, 1]
    t = mat4_translate(-ex, -ey, -ez)
    return mat4_mul(m, t)

# ------------------------------------------------------------
# Shaders (position + color)
# ------------------------------------------------------------
VERT_SRC = """
#version 330
layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec3 in_col;
uniform mat4 u_mvp;
out vec3 v_col;
void main(){
    v_col = in_col;
    gl_Position = u_mvp * vec4(in_pos, 1.0);
}
"""

FRAG_SRC = """
#version 330
in vec3 v_col;
out vec4 out_col;
void main(){
    out_col = vec4(v_col, 1.0);
}
"""

# ------------------------------------------------------------
# Geometry builders
# ------------------------------------------------------------
def make_box_triangles(lx, ly, lz, color=(0.1, 0.8, 0.3)):
    x0,x1 = -lx*0.5, lx*0.5
    y0,y1 = 0.0, ly
    z0,z1 = -lz*0.5, lz*0.5
    faces = [
        # top
        (x0,y1,z0),(x1,y1,z0),(x1,y1,z1),
        (x0,y1,z0),(x1,y1,z1),(x0,y1,z1),
        # bottom
        (x0,y0,z0),(x1,y0,z1),(x1,y0,z0),
        (x0,y0,z0),(x0,y0,z1),(x1,y0,z1),
        # front (-z)
        (x0,y0,z0),(x1,y1,z0),(x1,y0,z0),
        (x0,y0,z0),(x0,y1,z0),(x1,y1,z0),
        # back (+z)
        (x0,y0,z1),(x1,y0,z1),(x1,y1,z1),
        (x0,y0,z1),(x1,y1,z1),(x0,y1,z1),
        # left
        (x0,y0,z0),(x0,y0,z1),(x0,y1,z1),
        (x0,y0,z0),(x0,y1,z1),(x0,y1,z0),
        # right
        (x1,y0,z0),(x1,y1,z1),(x1,y0,z1),
        (x1,y0,z0),(x1,y1,z0),(x1,y1,z1),
    ]
    cols = [color]*len(faces)
    return faces, cols

def make_polyline(points, color=(1.0, 1.0, 0.2)):
    verts, cols = [], []
    for i in range(len(points)-1):
        verts.append(points[i]);  cols.append(color)
        verts.append(points[i+1]);cols.append(color)
    return verts, cols

def make_grid(size=80, step=2.0, color=(0.25, 0.25, 0.25)):
    s = size
    verts, cols = [], []
    v = -s
    while v <= s:
        verts += [(-s, 0.0, v), (s, 0.0, v)]
        cols  += [color,        color]
        verts += [(v, 0.0, -s), (v, 0.0,  s)]
        cols  += [color,        color]
        v += step
    return verts, cols

# ------------------------------------------------------------
# Scene types
# ------------------------------------------------------------
@dataclass
class Mesh:
    verts: List[Tuple[float,float,float]]
    cols:  List[Tuple[float,float,float]]
    mode: int            # gl.GL_TRIANGLES or gl.GL_LINES
    model: List[float]   # 4x4 column-major

class Ego:
    def __init__(self):
        self.pos  = [0.0, 0.5, 0.0]
        self.yaw  = 0.0
        self.v    = 0.0
        self.length, self.width, self.height = 4.6, 1.9, 1.6
        # bicycle params
        self.wb = 2.8
        self.steer = 0.0
        self.max_steer = math.radians(35.0)
        self.max_steer_rate = math.radians(120.0)
        # longitudinal
        self.max_accel = 3.0
        self.max_brake = 6.0
        self.c_roll = 0.015
        self.c_drag = 0.35
        # fallback box mesh
        v,c = make_box_triangles(self.length, self.height, self.width, (0.1,0.8,0.3))
        self.mesh = Mesh(v,c, gl.GL_TRIANGLES, mat4_identity())

    def update(self, dt, throttle, steer_cmd, brake):
        # steer dynamics
        target = clamp(steer_cmd * self.max_steer, -self.max_steer, self.max_steer)
        ds = clamp(target - self.steer, -self.max_steer_rate*dt, self.max_steer_rate*dt)
        self.steer += ds

        # longitudinal accel
        a_prop  = self.max_accel * clamp(throttle, 0.0, 1.0)
        a_brake = -self.max_brake * clamp(brake, 0.0, 1.0)
        sign_v  = 1.0 if self.v >= 0 else -1.0
        a_loss  = -self.c_roll*sign_v - self.c_drag*self.v*abs(self.v)
        a = a_prop + a_brake + a_loss
        self.v += a * dt
        if abs(self.v) < 0.02 and throttle <= 0.0 and brake <= 0.0:
            self.v = 0.0

        # kinematic bicycle (simple)
        self.pos[0] += self.v * math.sin(self.yaw) * dt
        self.pos[2] += -self.v * math.cos(self.yaw) * dt
        self.yaw    += (self.v / self.wb) * math.tan(self.steer) * dt

        # update fallback mesh transform
        self.mesh.model = mat4_mul(mat4_translate(*self.pos), mat4_rotate_y(self.yaw))

class MovingBox:
    def __init__(self, x, y, z, lx, ly, lz, color=(0.9,0.2,0.2), vel=(0.0,0.0,0.0)):
        v,c = make_box_triangles(lx, ly, lz, color)
        self.mesh = Mesh(v, c, gl.GL_TRIANGLES, mat4_mul(mat4_translate(x,y,z), mat4_identity()))
        self.vx, self.vy, self.vz = vel
        self.pos = [x,y,z]
    def update(self, dt):
        self.pos[0] += self.vx*dt
        self.pos[1] += self.vy*dt
        self.pos[2] += self.vz*dt
        self.mesh.model = mat4_translate(*self.pos)

# ------------------------------------------------------------
# Static-VBO Renderer (fast)
# ------------------------------------------------------------
class Renderer:
    """Builds a VAO/VBO once per Mesh, only updates u_mvp per draw."""
    def __init__(self):
        self.program = ShaderProgram(Shader(VERT_SRC, 'vertex'), Shader(FRAG_SRC, 'fragment'))

    def _build_gpu(self, mesh: Mesh):
        n = len(mesh.verts)
        inter = []
        for i in range(n):
            x,y,z = mesh.verts[i]
            r,g,b = mesh.cols[i]
            inter.extend((x,y,z, r,g,b))
        arr = (gl.GLfloat * (6*n))(*inter)

        vao = gl.GLuint()
        vbo = gl.GLuint()
        gl.glGenVertexArrays(1, ctypes.byref(vao))
        gl.glGenBuffers(1, ctypes.byref(vbo))

        gl.glBindVertexArray(vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, ctypes.sizeof(arr), arr, gl.GL_STATIC_DRAW)

        stride = 6 * ctypes.sizeof(gl.GLfloat)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(3*ctypes.sizeof(gl.GLfloat)))

        mesh._gpu = (vao, vbo, n, mesh.mode)

    def draw_mesh(self, mesh: Mesh, proj_view):
        if not hasattr(mesh, "_gpu"):
            self._build_gpu(mesh)
        vao, _vbo, n, mode = mesh._gpu

        self.program.use()
        mvp = mat4_mul(proj_view, mesh.model)
        self.program['u_mvp'] = mvp

        gl.glBindVertexArray(vao)
        gl.glDrawArrays(mode, 0, n)

# ------------------------------------------------------------
# App
# ------------------------------------------------------------
class AVHMI(pyglet.window.Window):
    def __init__(self, width=1280, height=720, fps=60):
        cfg = gl.Config(double_buffer=True, depth_size=24, major_version=3, minor_version=3)
        super().__init__(width=width, height=height, caption="AV HMI 3D", resizable=True, config=cfg)
        self.fps = fps
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDisable(gl.GL_CULL_FACE)  # avoid invisible OBJ if normals/winding are off
        gl.glClearColor(0.05, 0.06, 0.09, 1.0)

        self.keys = key.KeyStateHandler()
        self.push_handlers(self.keys)
        self.set_exclusive_mouse(False)
        self.mouse_captured = False

        # Ego + car mesh
        self.ego = Ego()
        car_obj_path = "assets/WAutoCar.obj"
        try:
            car_v, car_c = load_obj_positions(car_obj_path, scale=0.025, center_y=0.0)  # <- scale 0.01 as requested
            self.car_mesh = Mesh(
                verts=car_v, cols=car_c, mode=gl.GL_TRIANGLES,
                model=mat4_mul(mat4_translate(*self.ego.pos), mat4_rotate_y(self.ego.yaw))
            )
            print(f"Loaded car OBJ with {len(car_v)//3} triangles ")
        except Exception as e:
            self.car_mesh = None
            print(f"WARNING: failed to load '{car_obj_path}': {e}")

        # Obstacles
        rng = random.Random(42)
        self.obstacles: List[MovingBox] = [
            MovingBox(rng.uniform(-4,4), 0.8, -rng.uniform(10,60), 4.0, 1.6, 1.8, (0.9,0.2,0.2))
            for _ in range(6)
        ]

        # Lanes + grid
        lane_pts = [[(off, 0.01, -float(s)) for s in range(0, 200, 2)] for off in (-1.75, 1.75)]
        grid_v, grid_c = make_grid(100, 2.0)
        self.grid  = Mesh(grid_v,  grid_c,  gl.GL_LINES,     mat4_identity())
        self.lanes = [Mesh(*make_polyline(pts), mode=gl.GL_LINES, model=mat4_identity()) for pts in lane_pts]

        # Camera
        self.cam_yawoff = math.pi / 2.0
        self.cam_pitch  = math.radians(60.0)
        self.cam_dist   = 12.0
        self.cam_height = 4.0

        self.renderer = Renderer()
        self.last_time = time.perf_counter()
        pyglet.clock.schedule_interval(self.update, 1.0/self.fps)

    # Input
    def on_mouse_press(self, x, y, button, modifiers):
        if button == mouse.LEFT:
            self.mouse_captured = not self.mouse_captured
            self.set_exclusive_mouse(self.mouse_captured)

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        if self.mouse_captured:
            self.cam_yawoff -= dx * 0.0022   # inverted yaw as requested
            self.cam_pitch   = clamp(self.cam_pitch - dy*0.0022, math.radians(-5), math.radians(80))

    def on_mouse_motion(self, x, y, dx, dy):
        if self.mouse_captured:
            self.cam_yawoff -= dx * 0.0022
            self.cam_pitch   = clamp(self.cam_pitch - dy*0.0022, math.radians(-5), math.radians(80))

    # Update
    def update(self, _dt):
        now = time.perf_counter()
        dt = clamp(now - self.last_time, 0.0, 1.0/60.0)
        self.last_time = now

        throttle  = 1.0 if self.keys[key.W] else 0.0
        brake     = 1.0 if (self.keys[key.S] or self.keys[key.SPACE]) else 0.0
        steer_cmd = (1.0 if self.keys[key.D] else 0.0) - (1.0 if self.keys[key.A] else 0.0)

        self.ego.update(dt, throttle, steer_cmd, brake)

        for o in self.obstacles:
            o.update(dt)
            if o.pos[2] > 10:
                o.pos[2] = -80.0

    # Draw
    def on_draw(self):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        proj = perspective(60.0, max(1e-6, self.width/float(self.height)), 0.1, 500.0)

        # Camera (mouse-only orbit)
        tx, ty, tz = self.ego.pos[0], self.ego.pos[1] + 0.8, self.ego.pos[2]
        yaw = self.cam_yawoff
        cx = tx - math.cos(yaw) * self.cam_dist
        cy = ty + self.cam_height * math.sin(self.cam_pitch)
        cz = tz + math.sin(yaw) * self.cam_dist
        view = look_at((cx, cy, cz), (tx, ty, tz), (0.0, 1.0, 0.0))
        pv = mat4_mul(proj, view)

        # Draw world
        self.renderer.draw_mesh(self.grid, pv)
        for ln in self.lanes:
            self.renderer.draw_mesh(ln, pv)

        # Draw car (OBJ if loaded; else fallback box)
        if self.car_mesh:
            self.car_mesh.model = mat4_mul(mat4_translate(*self.ego.pos), mat4_rotate_y(self.ego.yaw))
            self.renderer.draw_mesh(self.car_mesh, pv)
        else:
            self.renderer.draw_mesh(self.ego.mesh, pv)

        for o in self.obstacles:
            self.renderer.draw_mesh(o.mesh, pv)

        # HUD
        pyglet.text.Label(
            f"Speed {self.ego.v:4.1f} m/s   Yaw {math.degrees(self.ego.yaw):5.1f} deg",
            font_name="Arial", font_size=12, x=10, y=10
        ).draw()

# ------------------------------------------------------------
if __name__ == "__main__":
    window = AVHMI(1280, 720, 60)
    pyglet.app.run()
