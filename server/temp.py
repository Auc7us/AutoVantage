# Pyglet 2.1.9 • Modern OpenGL (core profile) • Textured OBJ support
import math, time, random, ctypes, os
from dataclasses import dataclass
from typing import List, Tuple, Optional

import pyglet
from pyglet.window import key, mouse
from pyglet import gl
from pyglet.graphics.shader import Shader, ShaderProgram

# ------------------------------------------------------------
# Minimal MTL parser (grabs first map_Kd path)
# ------------------------------------------------------------
def parse_mtl_for_diffuse_texture(mtl_path: str) -> Optional[str]:
    if not os.path.isfile(mtl_path):
        return None
    tex = None
    with open(mtl_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.lstrip().lower().startswith('map_kd'):
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    candidate = parts[1].strip().strip('"')
                    tex = candidate
                    break
    if tex:
        # Resolve relative to MTL directory
        if not os.path.isabs(tex):
            tex = os.path.join(os.path.dirname(mtl_path), tex)
    return tex if os.path.isfile(tex) else None

# ------------------------------------------------------------
# OBJ loader with UV + MTL (map_Kd). Falls back to color if no UV/texture.
# ------------------------------------------------------------
def load_obj_with_uv_mtl(obj_path: str, scale=1.0, center_y=0.0):
    v_list: List[Tuple[float,float,float]] = []
    vt_list: List[Tuple[float,float]] = []
    faces_v_idx: List[Tuple[int,int,int]] = []
    faces_vt_idx: List[Tuple[Optional[int],Optional[int],Optional[int]]] = []
    mtl_file = None

    with open(obj_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if not line or line.startswith('#'): 
                continue
            parts = line.strip().split()
            if not parts: 
                continue
            tag = parts[0].lower()
            if tag == 'mtllib' and len(parts) >= 2:
                mtl_file = parts[1]
            elif tag == 'v' and len(parts) >= 4:
                x, y, z = map(float, parts[1:4])
                v_list.append((x*scale, y*scale + center_y, z*scale))
            elif tag == 'vt' and len(parts) >= 3:
                u, v = map(float, parts[1:3])
                vt_list.append((u, v))
            elif tag == 'f' and len(parts) >= 4:
                idx_v = []
                idx_vt = []
                for p in parts[1:]:
                    chunks = p.split('/')
                    # v / vt / vn → chunks[0], chunks[1], chunks[2]
                    v_i  = int(chunks[0]) - 1 if chunks[0] else None
                    vt_i = int(chunks[1]) - 1 if len(chunks) > 1 and chunks[1] else None
                    idx_v.append(v_i)
                    idx_vt.append(vt_i)
                # triangulate (fan)
                for k in range(1, len(idx_v)-1):
                    faces_v_idx.append((idx_v[0], idx_v[k], idx_v[k+1]))
                    faces_vt_idx.append((idx_vt[0], idx_vt[k], idx_vt[k+1]))

    # Build triangle arrays
    tri_pos: List[Tuple[float,float,float]] = []
    tri_uv:  List[Tuple[float,float]] = []
    have_uv = len(vt_list) > 0 and any(any(i is not None for i in trip) for trip in faces_vt_idx)

    for (a,b,c), (ta,tb,tc) in zip(faces_v_idx, faces_vt_idx):
        tri_pos.extend([v_list[a], v_list[b], v_list[c]])
        if have_uv:
            # Missing vt index? put 0,0
            uva = vt_list[ta] if (ta is not None and 0 <= ta < len(vt_list)) else (0.0, 0.0)
            uvb = vt_list[tb] if (tb is not None and 0 <= tb < len(vt_list)) else (0.0, 0.0)
            uvc = vt_list[tc] if (tc is not None and 0 <= tc < len(vt_list)) else (0.0, 0.0)
            tri_uv.extend([uva, uvb, uvc])

    # Print bounds
    if v_list:
        xs, ys, zs = zip(*v_list)
        print(f"OBJ bounds: X[{min(xs):.3f},{max(xs):.3f}]  "
              f"Y[{min(ys):.3f},{max(ys):.3f}]  Z[{min(zs):.3f},{max(zs):.3f}]")
        print(f"Approx size (m): {(max(xs)-min(xs)):.3f} x {(max(ys)-min(ys)):.3f} x {(max(zs)-min(zs)):.3f}")

    # MTL → texture path
    tex_path = None
    if mtl_file:
        # Resolve mtl relative to obj
        if not os.path.isabs(mtl_file):
            mtl_path = os.path.join(os.path.dirname(obj_path), mtl_file)
        else:
            mtl_path = mtl_file
        tex_path = parse_mtl_for_diffuse_texture(mtl_path)

    return tri_pos, (tri_uv if have_uv else None), tex_path

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
# Shaders (color and textured)
# ------------------------------------------------------------
VERT_COLOR = """
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

FRAG_COLOR = """
#version 330
in vec3 v_col;
out vec4 out_col;
void main(){
    out_col = vec4(v_col, 1.0);
}
"""

VERT_TEX = """
#version 330
layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec2 in_uv;
uniform mat4 u_mvp;
out vec2 v_uv;
void main(){
    v_uv = in_uv;
    gl_Position = u_mvp * vec4(in_pos, 1.0);
}
"""

FRAG_TEX = """
#version 330
in vec2 v_uv;
uniform sampler2D u_tex;
out vec4 out_col;
void main(){
    out_col = texture(u_tex, v_uv);
}
"""

# ------------------------------------------------------------
# Geometry builders for lines/boxes (color pipeline)
# ------------------------------------------------------------
def make_box_triangles(lx, ly, lz, color=(0.1, 0.8, 0.3)):
    x0,x1 = -lx*0.5, lx*0.5
    y0,y1 = 0.0, ly
    z0,z1 = -lz*0.5, lz*0.5
    faces = [
        (x0,y1,z0),(x1,y1,z0),(x1,y1,z1),
        (x0,y1,z0),(x1,y1,z1),(x0,y1,z1),
        (x0,y0,z0),(x1,y0,z1),(x1,y0,z0),
        (x0,y0,z0),(x0,y0,z1),(x1,y0,z1),
        (x0,y0,z0),(x1,y1,z0),(x1,y0,z0),
        (x0,y0,z0),(x0,y1,z0),(x1,y1,z0),
        (x0,y0,z1),(x1,y0,z1),(x1,y1,z1),
        (x0,y0,z1),(x1,y1,z1),(x0,y1,z1),
        (x0,y0,z0),(x0,y0,z1),(x0,y1,z1),
        (x0,y0,z0),(x0,y1,z1),(x0,y1,z0),
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
    cols:  Optional[List[Tuple[float,float,float]]]
    uvs:   Optional[List[Tuple[float,float]]]
    texture_id: Optional[int]
    mode: int
    model: List[float]
    _tex_obj: Optional[pyglet.image.Texture] = None  # keep alive

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
        # fallback colored box
        v,c = make_box_triangles(self.length, self.height, self.width, (0.1,0.8,0.3))
        self.mesh = Mesh(v, c, None, None, gl.GL_TRIANGLES, mat4_identity())

    def update(self, dt, throttle, steer_cmd, brake):
        target = clamp(steer_cmd * self.max_steer, -self.max_steer, self.max_steer)
        ds = clamp(target - self.steer, -self.max_steer_rate*dt, self.max_steer_rate*dt)
        self.steer += ds

        a_prop  = self.max_accel * clamp(throttle, 0.0, 1.0)
        a_brake = -self.max_brake * clamp(brake, 0.0, 1.0)
        sign_v  = 1.0 if self.v >= 0 else -1.0
        a_loss  = -self.c_roll*sign_v - self.c_drag*self.v*abs(self.v)
        a = a_prop + a_brake + a_loss
        self.v += a * dt
        if abs(self.v) < 0.02 and throttle <= 0.0 and brake <= 0.0:
            self.v = 0.0

        self.pos[0] += self.v * math.sin(self.yaw) * dt
        self.pos[2] += -self.v * math.cos(self.yaw) * dt
        self.yaw    += (self.v / self.wb) * math.tan(self.steer) * dt

        self.mesh.model = mat4_mul(mat4_translate(*self.pos), mat4_rotate_y(self.yaw))

class MovingBox:
    def __init__(self, x, y, z, lx, ly, lz, color=(0.9,0.2,0.2), vel=(0.0,0.0,0.0)):
        v,c = make_box_triangles(lx, ly, lz, color)
        self.mesh = Mesh(v, c, None, None, gl.GL_TRIANGLES, mat4_mul(mat4_translate(x,y,z), mat4_identity()))
        self.vx, self.vy, self.vz = vel
        self.pos = [x,y,z]
    def update(self, dt):
        self.pos[0] += self.vx*dt
        self.pos[1] += self.vy*dt
        self.pos[2] += self.vz*dt
        self.mesh.model = mat4_translate(*self.pos)

# ------------------------------------------------------------
# Static-VBO Renderer (color + textured)
# ------------------------------------------------------------
class Renderer:
    def __init__(self):
        self.prog_color = ShaderProgram(Shader(VERT_COLOR, 'vertex'), Shader(FRAG_COLOR, 'fragment'))
        self.prog_tex   = ShaderProgram(Shader(VERT_TEX, 'vertex'),   Shader(FRAG_TEX,   'fragment'))

    def _build_gpu_color(self, mesh: Mesh):
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

        mesh._gpu = (vao, vbo, n, mesh.mode, 'color')

    def _build_gpu_tex(self, mesh: Mesh):
        n = len(mesh.verts)
        inter = []
        for i in range(n):
            x,y,z = mesh.verts[i]
            u,v = mesh.uvs[i]
            inter.extend((x,y,z, u,v))
        arr = (gl.GLfloat * (5*n))(*inter)

        vao = gl.GLuint()
        vbo = gl.GLuint()
        gl.glGenVertexArrays(1, ctypes.byref(vao))
        gl.glGenBuffers(1, ctypes.byref(vbo))
        gl.glBindVertexArray(vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, ctypes.sizeof(arr), arr, gl.GL_STATIC_DRAW)

        stride = 5 * ctypes.sizeof(gl.GLfloat)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 2, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(3*ctypes.sizeof(gl.GLfloat)))

        mesh._gpu = (vao, vbo, n, mesh.mode, 'tex')

    def draw_mesh(self, mesh: Mesh, proj_view):
        if not hasattr(mesh, "_gpu"):
            if mesh.uvs is not None and mesh.texture_id is not None:
                self._build_gpu_tex(mesh)
            else:
                self._build_gpu_color(mesh)

        vao, _vbo, n, mode, kind = mesh._gpu
        mvp = mat4_mul(proj_view, mesh.model)

        if kind == 'tex':
            self.prog_tex.use()
            self.prog_tex['u_mvp'] = mvp
            gl.glActiveTexture(gl.GL_TEXTURE0)
            gl.glBindTexture(gl.GL_TEXTURE_2D, mesh.texture_id)
            self.prog_tex['u_tex'] = 0  # be explicit
        else:
            self.prog_color.use()
            self.prog_color['u_mvp'] = mvp

        gl.glBindVertexArray(vao)
        gl.glDrawArrays(mode, 0, n)


# ------------------------------------------------------------
# Texture helper
# ------------------------------------------------------------
def create_texture_2d(path: str) -> Optional[pyglet.image.Texture]:
    try:
        img = pyglet.image.load(path)
    except Exception as e:
        print(f"Failed to load texture '{path}': {e}")
        return None
    tex = img.get_texture()  # pyglet.image.Texture
    gl.glBindTexture(gl.GL_TEXTURE_2D, tex.id)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)
    return tex


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
        self.car_mesh: Optional[Mesh] = None
        try:
            tri_pos, tri_uv, tex_path = load_obj_with_uv_mtl(car_obj_path, scale=0.025, center_y=0.0)
            if tri_uv and tex_path:
                tex = create_texture_2d(tex_path)
                if tex:
                    self.car_mesh = Mesh(
                        tri_pos, None, tri_uv, tex.id, gl.GL_TRIANGLES,
                        mat4_mul(mat4_translate(*self.ego.pos), mat4_rotate_y(self.ego.yaw)),
                        _tex_obj=tex,  # keep it alive
                    )
                    print(f"Loaded textured car: {len(tri_pos)//3} tris, tex='{tex_path}'")
            if self.car_mesh is None:
                # Fallback: flat color if no UV/texture
                color = (0.12, 0.75, 0.90)
                cols  = [color] * len(tri_pos)
                self.car_mesh = Mesh(tri_pos, cols, None, None, gl.GL_TRIANGLES,
                                     mat4_mul(mat4_translate(*self.ego.pos), mat4_rotate_y(self.ego.yaw)))
                print(f"Loaded car (no texture): {len(tri_pos)//3} tris")
        except Exception as e:
            print(f"WARNING: failed to load '{car_obj_path}': {e}")

        # Obstacles
        rng = random.Random(42)
        self.obstacles: List[MovingBox] = [
            MovingBox(rng.uniform(-4,4), 0.8, -rng.uniform(10,60), 4.0, 1.6, 1.8, (0.9,0.2,0.2))
            for _ in range(6)
        ]

        # Lanes + grid (colored pipeline)
        lane_pts = [[(off, 0.01, -float(s)) for s in range(0, 200, 2)] for off in (-1.75, 1.75)]
        grid_v, grid_c = make_grid(100, 2.0)
        self.grid  = Mesh(grid_v,  grid_c,  None, None, gl.GL_LINES,     mat4_identity())
        self.lanes = [Mesh(*make_polyline(pts), None, None, gl.GL_LINES, mat4_identity()) for pts in lane_pts]

        # Camera (initialize behind the car)
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
            self.cam_yawoff -= dx * 0.0022   # inverted yaw
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

        # Camera (mouse-only orbit, not tied to ego yaw)
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

        # Draw car
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
