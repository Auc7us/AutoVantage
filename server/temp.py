# Pyglet 2.1.9, modern OpenGL (core profile). No fixed-function, no GLU.
import math, time, random
from dataclasses import dataclass
from typing import List, Tuple
import ctypes
import pyglet
from pyglet.window import key, mouse
from pyglet import gl
from pyglet.graphics.shader import Shader, ShaderProgram

# ========== math utils ==========
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

# ========== shaders ==========
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

# ========== geometry builders ==========
def make_box_triangles(lx, ly, lz, color=(0.1, 0.8, 0.3)):
    # Unit-centered box scaled by lx,ly,lz
    x0,x1 = -lx*0.5, lx*0.5
    y0,y1 = 0.0, ly
    z0,z1 = -lz*0.5, lz*0.5
    # 12 triangles, 36 verts
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
    verts = []
    cols = []
    for i in range(len(points)-1):
        verts.append(points[i])
        verts.append(points[i+1])
        cols.append(color); cols.append(color)
    return verts, cols

def make_grid(size=80, step=2.0, color=(0.25, 0.25, 0.25)):
    s = size
    verts = []
    cols = []
    v = -s
    while v <= s:
        # horizontal line at z = v
        verts += [(-s, 0.0, v), (s, 0.0, v)]
        cols  += [color,        color]
        # vertical line at x = v
        verts += [(v, 0.0, -s), (v, 0.0, s)]
        cols  += [color,        color]
        v += step
    return verts, cols


# ========== scene entities ==========
@dataclass
class Mesh:
    verts: List[Tuple[float,float,float]]
    cols:  List[Tuple[float,float,float]]
    mode: int  # gl.GL_TRIANGLES or gl.GL_LINES
    model: List[float]  # 4x4 column-major

# Replace your Ego class with this version
class Ego:
    def __init__(self):
        self.pos = [0.0, 0.5, 0.0]   # world x, y up, z forward negative
        self.yaw = 0.0               # radians, 0 means facing -Z after our sign choice
        self.v = 0.0                 # m/s forward speed
        # vehicle dims
        self.length, self.width, self.height = 4.6, 1.9, 1.6
        # bicycle model params
        self.wb = 2.8                # wheelbase
        self.steer = 0.0             # current steer angle [rad]
        self.max_steer = math.radians(35.0)
        self.max_steer_rate = math.radians(120.0)  # rad/s
        # longitudinal model
        self.max_accel = 3.0         # m/s^2 full throttle
        self.max_brake = 6.0         # m/s^2 full brake
        self.c_roll = 0.015          # rolling resistance
        self.c_drag = 0.35           # aero-ish
        # mesh
        v,c = make_box_triangles(self.length, self.height, self.width, (0.1,0.8,0.3))
        self.mesh = Mesh(v,c, gl.GL_TRIANGLES, mat4_identity())

    def update(self, dt, throttle, steer_cmd, brake):
        # 1) steer dynamics
        target = clamp(steer_cmd * self.max_steer, -self.max_steer, self.max_steer)
        ds = clamp(target - self.steer, -self.max_steer_rate * dt, self.max_steer_rate * dt)
        self.steer += ds

        # 2) longitudinal accel: throttle positive, brake positive
        a_prop = self.max_accel * clamp(throttle, 0.0, 1.0)
        a_brake = -self.max_brake * clamp(brake, 0.0, 1.0)
        # speed-dependent losses
        sign_v = 1.0 if self.v >= 0 else -1.0
        a_loss = -self.c_roll * sign_v - self.c_drag * self.v * abs(self.v)
        a = a_prop + a_brake + a_loss
        self.v += a * dt
        # prevent tiny jitter
        if abs(self.v) < 0.02 and throttle <= 0.0 and brake <= 0.0:
            self.v = 0.0

        # 3) kinematic bicycle
        beta = 0.0  # simple model
        self.pos[0] += self.v * math.sin(self.yaw + beta) * dt
        self.pos[2] += -self.v * math.cos(self.yaw + beta) * dt  # forward is -Z on screen
        self.yaw += (self.v / self.wb) * math.tan(self.steer) * dt

        # update mesh transform
        self.mesh.model = mat4_mul(mat4_translate(*self.pos), mat4_rotate_y(self.yaw))

class MovingBox:
    def __init__(self, x, y, z, lx, ly, lz, color=(0.9,0.2,0.2), vel=(0.0,0.0,0.0)):
        v,c = make_box_triangles(lx, ly, lz, color)
        self.mesh = Mesh(v,c, gl.GL_TRIANGLES, mat4_mul(mat4_translate(x,y,z), mat4_identity()))
        self.vx, self.vy, self.vz = vel
        self.pos = [x,y,z]
    def update(self, dt):
        self.pos[0] += self.vx*dt
        self.pos[1] += self.vy*dt
        self.pos[2] += self.vz*dt
        self.mesh.model = mat4_translate(*self.pos)

# ========== renderer ==========
class Renderer:
    def __init__(self):
        self.program = ShaderProgram(Shader(VERT_SRC, 'vertex'), Shader(FRAG_SRC, 'fragment'))

        # VAO
        self._vao = gl.GLuint()
        gl.glGenVertexArrays(1, ctypes.byref(self._vao))
        gl.glBindVertexArray(self._vao)

        # VBO
        self._vbo = gl.GLuint()
        gl.glGenBuffers(1, ctypes.byref(self._vbo))

        self.mvp = mat4_identity()

    def draw_mesh(self, mesh: Mesh, proj_view):
        verts = mesh.verts
        cols  = mesh.cols
        n = len(verts)
        if n == 0:
            return

        # Interleave position and color: [x,y,z,r,g,b] * n
        data = []
        for i in range(n):
            x, y, z = verts[i]
            r, g, b = cols[i]
            data.extend((x, y, z, r, g, b))

        arr = (gl.GLfloat * (6 * n))(*data)

        gl.glBindVertexArray(self._vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, ctypes.sizeof(arr), arr, gl.GL_DYNAMIC_DRAW)

        stride = 6 * ctypes.sizeof(gl.GLfloat)

        # position attribute 0
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(0))

        # color attribute 1 (offset 3 floats)
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(3 * ctypes.sizeof(gl.GLfloat)))

        # MVP
        mvp = mat4_mul(proj_view, mesh.model)
        self.program.use()
        self.program['u_mvp'] = mvp

        gl.glDrawArrays(mesh.mode, 0, n)


# ========== app ==========
class AVHMI(pyglet.window.Window):
    def __init__(self, width=1280, height=720, fps=60):
        cfg = gl.Config(double_buffer=True, depth_size=24, major_version=3, minor_version=3)
        super().__init__(width=width, height=height, caption="AV HMI 3D", resizable=True, config=cfg)
        self.fps = fps
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glClearColor(0.03, 0.03, 0.05, 1.0)
        self.keys = key.KeyStateHandler()
        self.push_handlers(self.keys)
        self.set_exclusive_mouse(False)
        self.mouse_captured = False
        # scene
        self.ego = Ego()
        rng = random.Random(42)
        self.obstacles: List[MovingBox] = [
            MovingBox(rng.uniform(-4,4), 0.8, -rng.uniform(10,60), 4.0, 1.6, 1.8, (0.9,0.2,0.2))
            for _ in range(6)
        ]
        # lanes and grid
        lane_pts = [[(off, 0.01, -float(s)) for s in range(0, 200, 2)] for off in (-1.75, 1.75)]
        grid_v, grid_c = make_grid(100, 2.0)
        self.grid = Mesh(grid_v, grid_c, gl.GL_LINES, mat4_identity())
        lanes = [Mesh(*make_polyline(pts), mode=gl.GL_LINES, model=mat4_identity()) for pts in lane_pts]
        self.lanes = lanes
        self.cam_yawoff = 0.0
        self.cam_pitch = math.radians(12.0)
        self.cam_dist = 12.0
        self.cam_height = 4.0
        self.renderer = Renderer()
        self.last_time = time.perf_counter()
        pyglet.clock.schedule_interval(self.update, 1.0/self.fps)

    # input
    def on_mouse_press(self, x, y, button, modifiers):
        if button == mouse.LEFT:
            self.mouse_captured = not self.mouse_captured
            self.set_exclusive_mouse(self.mouse_captured)
    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        if self.mouse_captured:
            self.cam_yawoff -= dx * 0.0022
            self.cam_pitch = clamp(self.cam_pitch - dy*0.0022, math.radians(-5), math.radians(80))
    def on_mouse_motion(self, x, y, dx, dy):
        if self.mouse_captured:
            self.cam_yawoff -= dx * 0.0022
            self.cam_pitch = clamp(self.cam_pitch - dy*0.0022, math.radians(-5), math.radians(80))

    # update
    def update(self, _dt):
        now = time.perf_counter()
        dt = clamp(now - self.last_time, 0.0, 1.0 / 60.0)
        self.last_time = now

        throttle = 1.0 if self.keys[key.W] else 0.0
        brake    = 1.0 if (self.keys[key.S] or self.keys[key.SPACE]) else 0.0
        steer_cmd = (1.0 if self.keys[key.D] else 0.0) - (1.0 if self.keys[key.A] else 0.0)

        self.ego.update(dt, throttle, steer_cmd, brake)

        for o in self.obstacles:
            o.update(dt)
            if o.pos[2] > 10: o.pos[2] = -80.0

    # draw
    def on_draw(self):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        proj = perspective(60.0, max(1e-6, self.width/float(self.height)), 0.1, 500.0)
        # chase camera
        tx, ty, tz = self.ego.pos[0], self.ego.pos[1] + 0.8, self.ego.pos[2]
        yaw = self.cam_yawoff
        cx = tx - math.cos(yaw) * self.cam_dist
        cy = ty + self.cam_height * math.sin(self.cam_pitch)
        cz = tz + math.sin(yaw) * self.cam_dist
        view = look_at((cx, cy, cz), (tx, ty, tz), (0.0, 1.0, 0.0))
        pv = mat4_mul(proj, view)
        # draw stuff
        self.renderer.draw_mesh(self.grid, pv)
        for ln in self.lanes: self.renderer.draw_mesh(ln, pv)
        self.renderer.draw_mesh(self.ego.mesh, pv)
        for o in self.obstacles: self.renderer.draw_mesh(o.mesh, pv)
        # HUD
        pyglet.text.Label(f"Speed {self.ego.v:4.1f} m/s   Yaw {math.degrees(self.ego.yaw):5.1f} deg",
            font_name="Arial", font_size=12, x=10, y=10).draw()

if __name__ == "__main__":
    # If you are on a headless or old Mesa stack, run under Xvfb:
    # xvfb-run -s "-screen 0 1280x720x24" python temp.py
    window = AVHMI(1280, 720, 60)
    pyglet.app.run()
