# Pyglet 2.1.9 • Modern OpenGL (core profile) • Textured OBJ support
import math, time, random, ctypes, os
from dataclasses import dataclass
from typing import List, Tuple, Optional

import pyglet
from pyglet.window import key, mouse
from pyglet import gl
from pyglet.graphics.shader import Shader, ShaderProgram

# GStreamer RTSP support for NVIDIA Jetson/GPUs
_GST_AVAILABLE = False
try:
    import gi
    gi.require_version('Gst', '1.0')
    gi.require_version('GstRtspServer', '1.0')
    from gi.repository import Gst, GstRtspServer, GLib
    Gst.init(None)
except Exception:
    _GST_AVAILABLE = False

# ------------------------------------------------------------
# Minimal MTL parser (grabs first map_Kd path)
# ------------------------------------------------------------
def parse_mtl(mtl_path: str):
    if not os.path.isfile(mtl_path):
        return None, {}
    first_tex = None
    colors = {}
    current_mtl = None
    with open(mtl_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'): continue
            parts = line.split()
            if not parts: continue
            tag = parts[0].lower()
            if tag == 'newmtl' and len(parts) >= 2:
                current_mtl = parts[1]
            elif tag == 'kd' and len(parts) >= 4 and current_mtl:
                try:
                    colors[current_mtl] = (float(parts[1]), float(parts[2]), float(parts[3]))
                except: pass
            elif tag == 'map_kd' and len(parts) >= 2:
                tex = parts[1].strip('"')
                if not os.path.isabs(tex):
                    tex = os.path.join(os.path.dirname(mtl_path), tex)
                if os.path.isfile(tex):
                    if first_tex is None: first_tex = tex
    return first_tex, colors

def make_grid_tile(size=40.0, step=2.0, color=(0.25, 0.25, 0.25)):
    # Bounded square grid centered at origin
    s = size * 0.5
    verts, cols = [], []
    v = -s
    while v <= s + 1e-6:
        # lines parallel to X
        verts += [(-s, 0.0, v), (s, 0.0, v)]
        cols  += [color,        color]
        # lines parallel to Z
        verts += [(v, 0.0, -s), (v, 0.0,  s)]
        cols  += [color,        color]
        v += step
    return verts, cols

# ------------------------------------------------------------
# OBJ loader with UV + MTL (map_Kd). Falls back to color if no UV/texture.
# ------------------------------------------------------------
def load_obj_with_uv_mtl(obj_path: str, scale=1.0, center_y=0.0):
    v_list: List[Tuple[float,float,float]] = []
    vt_list: List[Tuple[float,float]] = []
    faces_v_idx = []
    faces_vt_idx = []
    faces_mtl = []
    mtl_file = None
    current_mtl = None

    if not os.path.isfile(obj_path):
        print(f"ERROR: OBJ file not found: {obj_path}")
        return [], None, None, None

    with open(obj_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if not line or line.startswith('#'): 
                continue
            parts = line.split()
            if not parts: 
                continue
            tag = parts[0].lower()
            if tag == 'mtllib' and len(parts) >= 2:
                mtl_file = parts[1]
            elif tag == 'usemtl' and len(parts) >= 2:
                current_mtl = parts[1]
            elif tag == 'v' and len(parts) >= 4:
                v_list.append((float(parts[1])*scale, float(parts[2])*scale + center_y, float(parts[3])*scale))
            elif tag == 'vt' and len(parts) >= 3:
                vt_list.append((float(parts[1]), float(parts[2])))
            elif tag == 'f' and len(parts) >= 4:
                idx_v = []
                idx_vt = []
                for p in parts[1:]:
                    chunks = p.split('/')
                    v_i  = int(chunks[0]) - 1 if chunks[0] else None
                    vt_i = int(chunks[1]) - 1 if len(chunks) > 1 and chunks[1] else None
                    idx_v.append(v_i)
                    idx_vt.append(vt_i)
                for k in range(1, len(idx_v)-1):
                    faces_v_idx.append((idx_v[0], idx_v[k], idx_v[k+1]))
                    faces_vt_idx.append((idx_vt[0], idx_vt[k], idx_vt[k+1]))
                    faces_mtl.append(current_mtl)

    tex_path, mtl_colors = None, {}
    if mtl_file:
        mtl_path = os.path.join(os.path.dirname(obj_path), mtl_file) if not os.path.isabs(mtl_file) else mtl_file
        if not os.path.isfile(mtl_path):
            dir_files = os.listdir(os.path.dirname(obj_path))
            mtl_files = [f for f in dir_files if f.lower().endswith('.mtl')]
            if mtl_files:
                mtl_path = os.path.join(os.path.dirname(obj_path), mtl_files[0])
        
        tex_path, mtl_colors = parse_mtl(mtl_path)

    tri_pos = []
    tri_uv = []
    tri_col = []
    
    have_uv = len(vt_list) > 0 and any(any(i is not None for i in trip) for trip in faces_vt_idx)
    have_col = len(mtl_colors) > 0

    for (a,b,c), (ta,tb,tc), mtl in zip(faces_v_idx, faces_vt_idx, faces_mtl):
        tri_pos.extend([v_list[a], v_list[b], v_list[c]])
        if have_uv:
            tri_uv.extend([
                vt_list[ta] if (ta is not None and 0 <= ta < len(vt_list)) else (0.0, 0.0),
                vt_list[tb] if (tb is not None and 0 <= tb < len(vt_list)) else (0.0, 0.0),
                vt_list[tc] if (tc is not None and 0 <= tc < len(vt_list)) else (0.0, 0.0)
            ])
        if have_col:
            col = mtl_colors.get(mtl, (0.8, 0.8, 0.8))
            tri_col.extend([col, col, col])

    return tri_pos, (tri_uv if have_uv else None), tex_path, (tri_col if have_col else None)


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

def mat4_rotate_x(theta):
    c, s = math.cos(theta), math.sin(theta)
    return [1,0,0,0,
            0,c,-s,0,
            0,s, c,0,
            0,0,0,1]

def mat4_rotate_z(theta):
    c, s = math.cos(theta), math.sin(theta)
    return [ c,-s,0,0,
             s, c,0,0,
             0, 0,1,0,
             0, 0,0,1]

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
# Geometry builders for lines/boxes
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
# Street sign builders
# ------------------------------------------------------------
def make_regular_polygon(n: int, radius: float, thickness: float, color=(1.0,1.0,1.0), angle_offset: float = 0.0):
    import math
    verts = []
    cols  = []
    ring = []
    for i in range(n):
        ang = (2*math.pi * i) / n + angle_offset
        x = radius * math.cos(ang)
        z = radius * math.sin(ang)
        ring.append((x, 0.0, z))

    for i in range(1, n-1):
        a = ring[0]; b = ring[i]; c = ring[i+1]
        verts.extend([a, b, c]); cols.extend([color, color, color])

    back_ring = [(x, -thickness, z) for (x, _y, z) in ring]

    for i in range(1, n-1):
        a = back_ring[0]; b = back_ring[i+1]; c = back_ring[i]
        verts.extend([a, b, c]); cols.extend([color, color, color])

    for i in range(n):
        j = (i+1) % n
        a = ring[i]; b = ring[j]; c = back_ring[i]; d = back_ring[j]
        verts.extend([a, b, d, a, d, c]); cols.extend([color]*6)
    return verts, cols

def make_triangle_sign(size: float, thickness: float, color=(1.0,1.0,1.0)):
    import math
    r = size / math.sqrt(3.0)
    return make_regular_polygon(3, r, thickness, color, angle_offset=-math.pi/2)

def make_octagon_sign(size: float, thickness: float, color=(1.0,0.1,0.1)):
    r = size
    return make_regular_polygon(8, r, thickness, color)

def make_rect_sign(width: float, height: float, thickness: float, color=(1.0,1.0,1.0)):
    return make_box_triangles(width, thickness, height, color)

# ------------------------------------------------------------
# Barricade builder: trapezoidal prism
# ------------------------------------------------------------
def make_trapezoid_prism(base_len: float, base_depth: float, top_len: float, top_depth: float, height: float, color=(1.0, 0.5, 0.0)):

    bx0, bx1 = -base_len*0.5, base_len*0.5
    bz0, bz1 = -base_depth*0.5, base_depth*0.5
    tx0, tx1 = -top_len*0.5, top_len*0.5
    tz0, tz1 = -top_depth*0.5, top_depth*0.5
    y0, y1 = 0.0, height

    verts = []
    cols  = []

    # Top face
    verts += [(tx0,y1,tz0),(tx1,y1,tz0),(tx1,y1,tz1), (tx0,y1,tz0),(tx1,y1,tz1),(tx0,y1,tz1)]
    cols  += [color]*6
    # Bottom face
    verts += [(bx0,y0,bz0),(bx1,y0,bz1),(bx1,y0,bz0), (bx0,y0,bz0),(bx0,y0,bz1),(bx1,y0,bz1)]
    cols  += [color]*6
    # Side faces
    verts += [(bx1,y0,bz0),(bx1,y0,bz1),(tx1,y1,tz1), (bx1,y0,bz0),(tx1,y1,tz1),(tx1,y1,tz0)]
    cols  += [color]*6
    verts += [(bx0,y0,bz0),(tx0,y1,tz0),(tx0,y1,tz1), (bx0,y0,bz0),(tx0,y1,tz1),(bx0,y0,bz1)]
    cols  += [color]*6
    verts += [(bx0,y0,bz1),(tx0,y1,tz1),(tx1,y1,tz1), (bx0,y0,bz1),(tx1,y1,tz1),(bx1,y0,bz1)]
    cols  += [color]*6
    verts += [(bx0,y0,bz0),(bx1,y0,bz0),(tx1,y1,tz0), (bx0,y0,bz0),(tx1,y1,tz0),(tx0,y1,tz0)]
    cols  += [color]*6

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
    _tex_obj: Optional[pyglet.image.Texture] = None 

class Ego:
    def __init__(self):
        self.pos  = [0.0, 0.0, 0.0]
        self.yaw  = 0.0
        self.v    = 0.0
        self.length, self.width, self.height = 4.6, 1.9, 1.6
        # bicycle params
        self.wb = 2.8
        self.steer = 0.0
        self.max_steer = math.radians(35.0)
        self.max_steer_rate = math.radians(120.0)
        # longitudinal
        self.max_accel = 15.0
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

class Barricade:
    def __init__(self, x, y, z):
        # Orange color
        color = (1.0, 0.5, 0.0)
        # Dimensions for a barricade (approx 0.8m wide, 1.0m high, 0.2m deep)
        lx, ly, lz = 0.8, 1.0, 0.2
        v, c = make_box_triangles(lx, ly, lz, color)
        self.mesh = Mesh(v, c, None, None, gl.GL_TRIANGLES, mat4_translate(x, y, z))
        self.pos = [x, y, z]


class MovingBarricade:
    """Orange work zone barricade (trapezoidal prism) that can move like MovingBox."""
    def __init__(self, x, y, z,
                 base_len=1.8, base_depth=0.6,
                 top_len=1.4, top_depth=0.4,
                 height=1.0,
                 color=(1.0, 0.5, 0.0), vel=(0.0, 0.0, 0.0)):
        v, c = make_trapezoid_prism(base_len, base_depth, top_len, top_depth, height, color)
        self.mesh = Mesh(v, c, None, None, gl.GL_TRIANGLES, mat4_mul(mat4_translate(x,y,z), mat4_identity()))

        self.vx, self.vy, self.vz = vel
        self.pos = [x, y, z]
    def update(self, dt):
        self.pos[0] += self.vx*dt
        self.pos[1] += self.vy*dt
        self.pos[2] += self.vz*dt
        self.mesh.model = mat4_translate(*self.pos)


class MovingCharacter:
    """Simple wrapper for a character mesh with a world position.
    Characters are static by default but provide an update() hook for future animation.
    """
    def __init__(self, mesh: Mesh, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.mesh = mesh
        self.pos = [x, y, z]

    def update(self, dt: float):
        self.mesh.model = mat4_translate(*self.pos)


class StaticObject:
    """Wrapper for a static mesh to be included in the actor list."""
    def __init__(self, mesh: Mesh):
        self.mesh = mesh

    def update(self, dt: float):
        pass # Static objects don't need updates


# ------------------------------------------------------------
# Traffic Light
# ------------------------------------------------------------
class TrafficLight:
    def __init__(self, x=0.0, y=3.0, z=0.0):
        # Create light housing
        box_width = 0.5
        box_height = 1.35
        box_depth = 0.25
        box_verts, box_cols = make_box_triangles(box_width, box_height, box_depth, (0.2, 0.2, 0.2))
        
        # Create the three lights
        light_size = 0.25
        light_depth = 0.1
        
        # Calculate even spacing
        spacing = (box_height - 3*light_size) / 4
        
        # Red light (top)
        red_y = box_height - spacing - light_size
        red_verts, red_cols = make_box_triangles(light_size, light_size, light_depth, (0.9, 0.1, 0.1))  # Red
        red_verts = [(x, y + red_y, z + box_depth/2) for x, y, z in red_verts]
        
        # Yellow light (middle)
        yellow_y = box_height - 2*spacing - 2*light_size
        yellow_verts, yellow_cols = make_box_triangles(light_size, light_size, light_depth, (0.9, 0.9, 0.1))  # Yellow
        yellow_verts = [(x, y + yellow_y, z + box_depth/2) for x, y, z in yellow_verts]
        
        # Green light (bottom)
        green_y = box_height - 3*spacing - 3*light_size
        green_verts, green_cols = make_box_triangles(light_size, light_size, light_depth, (0.1, 0.9, 0.1))  # Green
        green_verts = [(x, y + green_y, z + box_depth/2) for x, y, z in green_verts]
        
        # Combine all vertices and colors
        all_verts = box_verts + red_verts + yellow_verts + green_verts
        all_cols = box_cols + red_cols + yellow_cols + green_cols
        
        # Create the mesh
        self.mesh = Mesh(all_verts, all_cols, None, None, gl.GL_TRIANGLES, mat4_translate(x, y, z))
        self.pos = [x, y, z]


    def update(self, dt: float):
        self.mesh.model = mat4_translate(*self.pos)


# ------------------------------------------------------------
# Street Signs
# ------------------------------------------------------------
class StreetSign:
    def __init__(self, kind: str = 'stop', x: float = 0.0, y: float = 0.0, z: float = 0.0, yaw: float = 0.0, y_adjust: float = 0.0, scale: float = 1.0):
        self.kind = kind
        self.pos = [x, y, z]
        self.yaw = yaw
        self.y_adjust = y_adjust
        self.scale = scale
        pole_h = 2.6 * 0.75
        pole_w = 0.08
        pole_d = 0.08
        pv, pc = make_box_triangles(pole_w, pole_h, pole_d, (0.7,0.7,0.7))
        self.pole_mesh = Mesh(pv, pc, None, None, gl.GL_TRIANGLES, mat4_identity())
        thickness = 0.02
        if kind == 'stop':
            panel_size = 0.74 * 0.6
            sv, sc = make_regular_polygon(6, panel_size, thickness, (0.9, 0.1, 0.1))
        elif kind == 'yield':
            panel_size = 0.9
            sv, sc = make_triangle_sign(panel_size * getattr(self, 'scale', 1.0), thickness, (1.0, 1.0, 1.0))
        else:
            width, height = 0.55, 0.75
            sv, sc = make_rect_sign(width, height, thickness, (1.0, 1.0, 1.0))
        panel_y = y + pole_h + thickness*0.5
        self.panel_offset = (0.0, pole_h + thickness*0.5 + self.y_adjust, pole_d*0.5 + thickness*0.5)
        self.panel_mesh = Mesh(sv, sc, None, None, gl.GL_TRIANGLES, mat4_identity())


    def update(self, dt: float):
        M = mat4_translate(*self.pos)
        self.pole_mesh.model = M
        ox, oy, oz = self.panel_offset
        Mp = mat4_mul(M, mat4_translate(ox, oy, oz))
        Mp = mat4_mul(Mp, mat4_rotate_x(math.radians(90.0)))
        if getattr(self, 'yaw', 0.0) != 0.0:
            Mp = mat4_mul(Mp, mat4_rotate_y(self.yaw))
        self.panel_mesh.model = Mp


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
        # Detect if verts is flat list or list of tuples
        is_flat = n > 0 and isinstance(mesh.verts[0], (int, float))
        
        if is_flat:
            # Flat list: iterate in steps of 3
            vertex_count = n // 3
            for i in range(0, n, 3):
                x = mesh.verts[i]
                y = mesh.verts[i+1]
                z = mesh.verts[i+2]
                r, g, b = mesh.cols[i//3] if mesh.cols else (1.0, 1.0, 1.0)
                inter.extend((x, y, z, r, g, b))
        else:
            # List of tuples
            vertex_count = n
            for i in range(n):
                x, y, z = mesh.verts[i]
                r, g, b = mesh.cols[i] if mesh.cols else (1.0, 1.0, 1.0)
                inter.extend((x, y, z, r, g, b))
        
        arr = (gl.GLfloat * (6 * vertex_count))(*inter)

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

        mesh._gpu = (vao, vbo, vertex_count, mesh.mode, 'color')

    def _build_gpu_tex(self, mesh: Mesh):
        n = len(mesh.verts)
        inter = []
        # Detect if verts is flat list or list of tuples
        is_flat = n > 0 and isinstance(mesh.verts[0], (int, float))
        
        # Detect if uvs is flat list or list of tuples
        is_uv_flat = False
        if mesh.uvs and len(mesh.uvs) > 0:
            is_uv_flat = isinstance(mesh.uvs[0], (int, float))

        if is_flat:
            # Flat list: iterate in steps of 3
            vertex_count = n // 3
            for i in range(0, n, 3):
                x = mesh.verts[i]
                y = mesh.verts[i+1]
                z = mesh.verts[i+2]
                
                idx = i // 3
                if mesh.uvs:
                    if is_uv_flat:
                        u = mesh.uvs[2*idx]
                        v = mesh.uvs[2*idx+1]
                    else:
                        u, v = mesh.uvs[idx]
                else:
                    u, v = 0.0, 0.0
                
                inter.extend((x, y, z, u, v))
        else:
            # List of tuples
            vertex_count = n
            for i in range(n):
                x, y, z = mesh.verts[i]
                if mesh.uvs:
                    if is_uv_flat:
                        u = mesh.uvs[2*i]
                        v = mesh.uvs[2*i+1]
                    else:
                        u, v = mesh.uvs[i]
                else:
                    u, v = 0.0, 0.0
                
                inter.extend((x, y, z, u, v))
        
        arr = (gl.GLfloat * (5 * vertex_count))(*inter)

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

        mesh._gpu = (vao, vbo, vertex_count, mesh.mode, 'tex')

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
            self.prog_tex['u_tex'] = 0
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
    tex = img.get_texture()
    gl.glBindTexture(gl.GL_TEXTURE_2D, tex.id)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)
    return tex


# ------------------------------------------------------------
# App
# ------------------------------------------------------------
class RTSPStreamer:
    def __init__(self, width: int, height: int, fps: int = 30, port: int = 8554, mount: str = "/sim"):
        self.width = width
        self.height = height
        self.fps = fps
        self.port = port
        self.mount = mount
        self._loop = None
        self._server = None
        self._factory = None
        self._appsrc = None
        self._pts_ns = 0
        self.is_active = False

    def _choose_encoder(self):
        if not _GST_AVAILABLE:
            return None
        try:
            if Gst.ElementFactory.find("nvv4l2h264enc"):
                return "nvv4l2h264enc insert-sps-pps=true maxperf-enable=true bitrate=4000000"
            if Gst.ElementFactory.find("nvh264enc"):
                return "nvh264enc bitrate=4000"
        except Exception:
            pass
        return "x264enc tune=zerolatency speed-preset=ultrafast bitrate=4000"

    def start(self):
        if not _GST_AVAILABLE:
            print("RTSP disabled: GStreamer not available on this system")
            return False
        enc = self._choose_encoder()
        if enc is None:
            print("RTSP disabled: no H.264 encoder available")
            return False
        launch = (
            f"appsrc name=mysrc is-live=true format=GST_FORMAT_TIME caps=video/x-raw,format=RGB,width={self.width},height={self.height},framerate={self.fps}/1 "
            f"! videoconvert ! {enc} ! rtph264pay name=pay0 pt=96"
        )
        try:
            self._server = GstRtspServer.RTSPServer()
            self._server.props.service = str(self.port)
            self._factory = GstRtspServer.RTSPMediaFactory()
            self._factory.set_launch(launch)
            self._factory.set_latency(100)
            self._factory.set_shared(True)

            def on_configure(factory, media):
                try:
                    pipeline = media.get_element()
                    self._appsrc = pipeline.get_by_name("mysrc")
                    if self._appsrc:
                        self._appsrc.set_property("stream-type", 0)
                        self._appsrc.set_property("format", Gst.Format.TIME)
                        self._appsrc.set_property("is-live", True)
                        self._appsrc.set_property("do-timestamp", True)
                except Exception as e:
                    print(f"RTSP media-configure error: {e}")

            self._factory.connect("media-configure", on_configure)
            mounts = self._server.get_mount_points()
            mounts.add_factory(self.mount, self._factory)
            self._server.attach(None)
            self._loop = GLib.MainLoop()
            import threading
            threading.Thread(target=self._loop.run, daemon=True).start()
            self.is_active = True
            print(f"RTSP streaming at rtsp://localhost:{self.port}{self.mount}")
            return True
        except Exception as e:
            print(f"Failed to start RTSP server: {e}")
            return False

    def stop(self):
        self.is_active = False
        try:
            if self._loop is not None:
                self._loop.quit()
        except Exception:
            pass

    def push_rgb_frame(self, data: bytes):
        if not (self.is_active and _GST_AVAILABLE and self._appsrc is not None):
            return
        try:
            buf = Gst.Buffer.new_allocate(None, len(data), None)
            buf.fill(0, data)
            dur = int(1e9 / max(1, self.fps))
            buf.pts = self._pts_ns
            buf.dts = self._pts_ns
            buf.duration = dur
            self._pts_ns += dur
            self._appsrc.emit("push-buffer", buf)
        except Exception as e:
            print(f"RTSP push error: {e}")

class AVHMI(pyglet.window.Window):
    def __init__(self, width=1280, height=720, fps=60):
        cfg = gl.Config(double_buffer=True, depth_size=24, major_version=3, minor_version=3)
        super().__init__(width=width, height=height, caption="HMI 3D", resizable=True, config=cfg)
        self.fps = fps
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDisable(gl.GL_CULL_FACE)
        gl.glClearColor(0.05, 0.06, 0.09, 1.0)

        self.keys = key.KeyStateHandler()
        self.push_handlers(self.keys)
        self.set_exclusive_mouse(False)
        self.mouse_captured = False
        self.hud = pyglet.text.Label("", font_name="Roboto", font_size=12, x=10, y=10)

        self.ego = Ego()
        car_obj_path = "assets/WAutoCar.obj"
        self.car_mesh: Optional[Mesh] = None
        self.tex_normal = None
        self.tex_brake = None
        try:
            tri_pos, tri_uv, tex_path, tri_col = load_obj_with_uv_mtl(car_obj_path, scale= 1.0, center_y=0.0)
            if tri_uv and tex_path:
                self.tex_normal = create_texture_2d(tex_path)
                if self.tex_normal:
                    self.car_mesh = Mesh(
                        tri_pos, tri_col, tri_uv, self.tex_normal.id, gl.GL_TRIANGLES,
                        mat4_mul(mat4_translate(*self.ego.pos), mat4_rotate_y(self.ego.yaw)),
                        _tex_obj=self.tex_normal,
                    )
                    print(f"Loaded textured car: {len(tri_pos)//3} tris, tex='{tex_path}'")
                base, ext = os.path.splitext(tex_path)
                tex_brake_path = f"{base}_brake{ext}"
                if os.path.isfile(tex_brake_path):
                    self.tex_brake = create_texture_2d(tex_brake_path)
                    print(f"Loaded brake light texture: {tex_brake_path}")
                else:
                    print("No _brake texture found, using default")
            if self.car_mesh is None:
                color = (0.12, 0.75, 0.90)
                cols  = tri_col if tri_col else [color] * len(tri_pos)
                self.car_mesh = Mesh(tri_pos, cols, None, None, gl.GL_TRIANGLES,
                                     mat4_mul(mat4_translate(*self.ego.pos), mat4_rotate_y(self.ego.yaw)))
                print(f"Loaded car (no texture): {len(tri_pos)//3} tris")
        except Exception as e:
            print(f"WARNING: failed to load '{car_obj_path}': {e}")

        # Wheels
        self.wheels = []
        whl_R_obj_path = "assets/whl/whl_R.obj"
        whl_L_obj_path = "assets/whl/whl_L.obj"
        try:
            wpos_r, wuv_r, wtex_r, wcol_r = load_obj_with_uv_mtl(whl_R_obj_path, scale=1.0, center_y=0.0) 
            wpos_l, wuv_l, wtex_l, wcol_l = load_obj_with_uv_mtl(whl_L_obj_path, scale=1.0, center_y=0.0)
            
            if wuv_r and wtex_r:
                wtex_r_obj = create_texture_2d(wtex_r)
                wmesh_r = Mesh(wpos_r, wcol_r, wuv_r, (wtex_r_obj.id if wtex_r_obj else None), gl.GL_TRIANGLES, mat4_identity(), _tex_obj=wtex_r_obj)
            else:
                wcols = wcol_r if wcol_r else [(0.15, 0.15, 0.15)] * len(wpos_r)
                wmesh_r = Mesh(wpos_r, wcols, None, None, gl.GL_TRIANGLES, mat4_identity())

            if wuv_l and wtex_l:
                wtex_l_obj = create_texture_2d(wtex_l)
                wmesh_l = Mesh(wpos_l, wcol_l, wuv_l, (wtex_l_obj.id if wtex_l_obj else None), gl.GL_TRIANGLES, mat4_identity(), _tex_obj=wtex_l_obj)
            else:
                wcols = wcol_l if wcol_l else [(0.15, 0.15, 0.15)] * len(wpos_l)
                wmesh_l = Mesh(wpos_l, wcols, None, None, gl.GL_TRIANGLES, mat4_identity())

            self.wheels.append({
                'mesh': wmesh_r,
                'offset': (0.825, 0.35, -1.6625),
                'steer': True, 
                'radius': 0.35
            })

            self.wheels.append({
                'mesh': wmesh_l,
                'offset': (-0.825, 0.35, -1.6625),
                'steer': True, 
                'radius': 0.35
            })

            self.wheels.append({
                'mesh': wmesh_r,
                'offset': (0.8, 0.35, 1.2225), 
                'steer': False,
                'radius': 0.35
            })

            self.wheels.append({
                'mesh': wmesh_l,
                'offset': (-0.8, 0.35, 1.2225), 
                'steer': False,
                'radius': 0.35
            })

        except Exception as e:
            print(f"WARNING: failed to load wheels: {e}")

        # Rolling state
        self._wheel_roll = 0.0

        # Orange barricade
        bl, bd = 1.8*0.85, 0.6*0.85
        tl, td = 1.4*0.85, 0.4*0.85
        h      = 1.0*0.85
        bx, by, bz = 3.0 + 1.2, 0.0, -10.0
        self.obstacles: List[MovingBarricade] = [
            MovingBarricade(bx, by, bz, base_len=bl, base_depth=bd, top_len=tl, top_depth=td, height=h, color=(1.0, 0.5, 0.0)),
            MovingBarricade(bx + 2.0, by, bz, base_len=bl, base_depth=bd, top_len=tl, top_depth=td, height=h, color=(0.6, 0.6, 0.6))
        ]
        self.show_obstacles = True

        # Human Models
        self.characters: List[MovingCharacter] = []
        human_obj_path = "assets/human/human.obj"
        try:
            hpos, huv, htex, hcol = load_obj_with_uv_mtl(human_obj_path, scale=1.0, center_y=0.0)
            if hpos:
                ys = [p[1] for p in hpos]
                ymin, ymax = min(ys), max(ys)
                model_h = max(1e-6, ymax - ymin)
                desired_h = 1.82 
                s = desired_h / model_h
                hpos_scaled = [(x*s, (y - ymin)*s, z*s) for (x, y, z) in hpos]
            else:
                hpos_scaled = hpos

            if huv and htex:
                tex_obj = create_texture_2d(htex)
                if tex_obj:
                    hmesh = Mesh(hpos_scaled, hcol, huv, tex_obj.id, gl.GL_TRIANGLES, mat4_translate(2.0, 0.0, -8.0), _tex_obj=tex_obj)
                else:
                    cols = hcol if hcol else [(0.8, 0.7, 0.6)] * len(hpos_scaled)
                    hmesh = Mesh(hpos_scaled, cols, None, None, gl.GL_TRIANGLES, mat4_translate(2.0, 0.0, -8.0))
            else:
                cols = hcol if hcol else [(0.8, 0.7, 0.6)] * len(hpos_scaled)
                hmesh = Mesh(hpos_scaled, cols, None, None, gl.GL_TRIANGLES, mat4_translate(2.0, 0.0, -8.0))
            self.characters.append(MovingCharacter(hmesh, 2.0, 0.0, -8.0))
            try:
                tris = len(hpos_scaled)//3
            except Exception:
                tris = 0
            print(f"Loaded human model: {tris} tris (scale {s:.3f} -> height {desired_h} m)")
        except Exception as e:
            print(f"WARNING: failed to load human '{human_obj_path}': {e}")
            v, c = make_box_triangles(0.6, 1.8, 0.4, (0.8, 0.7, 0.6))
            fallback = Mesh(v, c, None, None, gl.GL_TRIANGLES, mat4_translate(2.0, 0.0, -8.0))

            self.characters.append(MovingCharacter(fallback, 2.0, 0.0, -8.0))

        # Barrel Model
        try:
            barrel_obj_path = "assets/barrel/barrel.obj"
            bpos, buv, btex, bcol = load_obj_with_uv_mtl(barrel_obj_path, scale=1.0, center_y=0.0)
            if bpos:
                ys = [p[1] for p in bpos]
                ymin, ymax = min(ys), max(ys)
                model_h = max(1e-6, ymax - ymin)
                desired_h = 1.0
                s = desired_h / model_h
                bpos_scaled = [(x*s, (y - ymin)*s, z*s) for (x, y, z) in bpos]
            else:
                bpos_scaled = bpos

            if buv and btex:
                tex_obj = create_texture_2d(btex)
                if tex_obj:
                    bmesh = Mesh(bpos_scaled, bcol, buv, tex_obj.id, gl.GL_TRIANGLES, mat4_translate(4.0, 0.0, -8.0), _tex_obj=tex_obj)
                else:
                    cols = bcol if bcol else [(0.5, 0.3, 0.1)] * len(bpos_scaled) # Brown color
                    bmesh = Mesh(bpos_scaled, cols, None, None, gl.GL_TRIANGLES, mat4_translate(4.0, 0.0, -8.0))
            else:
                cols = bcol if bcol else [(0.5, 0.3, 0.1)] * len(bpos_scaled) # Brown color
                bmesh = Mesh(bpos_scaled, cols, None, None, gl.GL_TRIANGLES, mat4_translate(4.0, 0.0, -8.0))
            self.obstacles.append(StaticObject(bmesh))
            try:
                tris = len(bpos_scaled)//3
            except Exception:
                tris = 0
            print(f"Loaded barrel model: {tris} tris (scale {s:.3f} -> height {desired_h} m)")
        except Exception as e:
            print(f"WARNING: failed to load barrel '{barrel_obj_path}': {e}")
            v, c = make_box_triangles(0.6, 1.0, 0.6, (0.5, 0.3, 0.1)) # Fallback brown box
            fallback = Mesh(v, c, None, None, gl.GL_TRIANGLES, mat4_translate(4.0, 0.0, -8.0))
            self.obstacles.append(StaticObject(fallback))

        # Traffic Cone Model
        try:
            cone_obj_path = "assets/traffic-cone/cone.obj"
            cpos, cuv, ctex, ccol = load_obj_with_uv_mtl(cone_obj_path, scale=1.0, center_y=0.0)
            if cpos:
                ys = [p[1] for p in cpos]
                ymin, ymax = min(ys), max(ys)
                model_h = max(1e-6, ymax - ymin)
                desired_h =0.75  # Assuming a desired height of 1.0 meter for the cone
                s = desired_h / model_h
                cpos_scaled = [(x*s, (y - ymin)*s, z*s) for (x, y, z) in cpos]
            else:
                cpos_scaled = cpos

            if cuv and ctex:
                tex_obj = create_texture_2d(ctex)
                if tex_obj:
                    cmesh = Mesh(cpos_scaled, ccol, cuv, tex_obj.id, gl.GL_TRIANGLES, mat4_translate(5.0, 0.0, -8.0), _tex_obj=tex_obj)
                else:
                    cols = ccol if ccol else [(0.9, 0.4, 0.1)] * len(cpos_scaled)  # Orange color
                    cmesh = Mesh(cpos_scaled, cols, None, None, gl.GL_TRIANGLES, mat4_translate(5.0, 0.0, -8.0))
            else:
                cols = ccol if ccol else [(0.9, 0.4, 0.1)] * len(cpos_scaled)  # Orange color
                cmesh = Mesh(cpos_scaled, cols, None, None, gl.GL_TRIANGLES, mat4_translate(5.0, 0.0, -8.0))
            self.obstacles.append(StaticObject(cmesh))
            try:
                tris = len(cpos_scaled)//3
            except Exception:
                tris = 0
            print(f"Loaded traffic cone model: {tris} tris (scale {s:.3f} -> height {desired_h} m)")
        except Exception as e:
            print(f"WARNING: failed to load traffic cone '{cone_obj_path}': {e}")
            v, c = make_box_triangles(0.4, 1.0, 0.4, (0.9, 0.4, 0.1))  # Fallback orange box
            fallback = Mesh(v, c, None, None, gl.GL_TRIANGLES, mat4_translate(5.0, 0.0, -8.0))
            self.obstacles.append(StaticObject(fallback))

        # Deer Model
        try:
            deer_obj_path = "assets/deer/deer.obj"
            dpos, duv, dtex, dcol = [], None, None, None
            try:
                dpos, duv, dtex, dcol = load_obj_with_uv_mtl(deer_obj_path, scale=1.0, center_y=0.0)
            except Exception as _e:
                print(f"WARNING: failed to parse deer OBJ '{deer_obj_path}': {_e}")

            if dpos:
                dys = [p[1] for p in dpos]
                dymin, dymax = min(dys), max(dys)
                model_h = max(1e-6, dymax - dymin)
                desired_h = 2
                s = desired_h / model_h
                dpos_scaled = [(x*s, (y - dymin)*s, z*s) for (x, y, z) in dpos]
            else:
                dpos_scaled = dpos
                
            deer_mesh = None
            if duv and dtex:
                dtex_obj = create_texture_2d(dtex)
                if dtex_obj:
                    deer_mesh = Mesh(dpos_scaled, dcol, duv, dtex_obj.id, gl.GL_TRIANGLES, mat4_translate(10.0, 0.0, -8.0), _tex_obj=dtex_obj)
            
            if deer_mesh is None and dpos_scaled:
                cols = dcol if dcol else [(0.60, 0.45, 0.30)] * len(dpos_scaled)
                deer_mesh = Mesh(dpos_scaled, cols, duv, None, gl.GL_TRIANGLES, mat4_translate(10.0, 0.0, -8.0))
            
            if deer_mesh:
                self.characters.append(MovingCharacter(deer_mesh, 10.0, 0.0, -8.0))
                try:
                    tris = len(dpos_scaled)//3
                except Exception:
                    tris = 0
                print(f"Loaded deer model from '{deer_obj_path}': {tris} tris (scale {s:.3f} -> height {desired_h} m) at (10.0, -8.0)")
            else:
                v, c = make_box_triangles(1.2, 0.8, 0.4, (0.6, 0.4, 0.2))
                fb = Mesh(v, c, None, None, gl.GL_TRIANGLES, mat4_translate(10.0, 0.0, -8.0))
                self.characters.append(MovingCharacter(fb, 10.0, 0.0, -8.0))
                print("Deer OBJ present but failed to build mesh — added fallback box at x=10.0")
        except Exception as e:
            print(f"WARNING: failed to create deer actor: {e}")

        # Bike
        try:
            bike_obj_path = "assets/bike/bike.obj"
            bkpos, bkuv, bktex, bkcol = None, None, None, None
            try:
                bkpos, bkuv, bktex, bkcol = load_obj_with_uv_mtl(bike_obj_path, scale=1.0, center_y=0.0)
            except Exception as load_err:
                print(f"DEBUG: Failed to load bike OBJ: {load_err}")
                bkpos = None

            if bkpos:
                bkxs = [p[0] for p in bkpos]
                bkys = [p[1] for p in bkpos]
                bkzs = [p[2] for p in bkpos]
                bkxmin, bkxmax = min(bkxs), max(bkxs)
                bkymin, bkymax = min(bkys), max(bkys)
                bkzmin, bkzmax = min(bkzs), max(bkzs)
                
                model_h = max(1e-6, bkymax - bkymin)
                desired_h = 1.0  # 1 meter
                s = desired_h / model_h
                
                cx = (bkxmin + bkxmax) * 0.5
                cz = (bkzmin + bkzmax) * 0.5
                
                bkpos_scaled = [((x - cx)*s, (y - bkymin)*s, (z - cz)*s) for (x, y, z) in bkpos]

                bike_x, bike_y, bike_z = 12.0, 0.0, -8.0

                if bkuv and bktex and os.path.isfile(bktex):
                    bktex_obj = create_texture_2d(bktex)
                    bike_mesh = Mesh(bkpos_scaled, bkcol, bkuv, (bktex_obj.id if bktex_obj else None), gl.GL_TRIANGLES, mat4_translate(bike_x, bike_y, bike_z), _tex_obj=(bktex_obj if bktex_obj else None))
                else:
                    if bkcol:
                        bike_mesh = Mesh(bkpos_scaled, bkcol, bkuv, None, gl.GL_TRIANGLES, mat4_translate(bike_x, bike_y, bike_z))
                    else:
                        fallback_cols = [(0.3, 0.3, 0.3)] * len(bkpos_scaled) # Grey fallback
                        bike_mesh = Mesh(bkpos_scaled, fallback_cols, bkuv, None, gl.GL_TRIANGLES, mat4_translate(bike_x, bike_y, bike_z))

                self.characters.append(MovingCharacter(bike_mesh, bike_x, bike_y, bike_z))
                bktris = len(bkpos_scaled)//3
                print(f"Loaded bike model from '{bike_obj_path}': {bktris} tris (scale {s:.6f} -> height {desired_h} m) centered at ({bike_x}, {bike_y}, {bike_z})")
        except Exception as e:
            print(f"WARNING: failed to create bike actor: {e}")

        # Bus
        try:
            bus_obj_path = "assets/bus/bus.obj"
            bus_pos, bus_uv, bus_tex, bus_col = None, None, None, None
            try:
                bus_pos, bus_uv, bus_tex, bus_col = load_obj_with_uv_mtl(bus_obj_path, scale=1.0, center_y=0.0)
            except Exception as load_err:
                print(f"DEBUG: Failed to load bus OBJ: {load_err}")
                bus_pos = None

            if bus_pos:
                bxs = [p[0] for p in bus_pos]
                bys = [p[1] for p in bus_pos]
                bzs = [p[2] for p in bus_pos]
                bxmin, bxmax = min(bxs), max(bxs)
                bymin, bymax = min(bys), max(bys)
                bzmin, bzmax = min(bzs), max(bzs)
                
                model_h = max(1e-6, bymax - bymin)
                desired_h = 3.0 # Bus height
                s = desired_h / model_h
                
                # Center
                cx = (bxmin + bxmax) * 0.5
                cz = (bzmin + bzmax) * 0.5
                
                # Scale, Center, and Rotate 90 degrees left (around Y)
                # Rotation: x' = -z, z' = x
                bus_pos_final = []
                for (x, y, z) in bus_pos:
                    # Scale and center
                    sx = (x - cx) * s
                    sy = (y - bymin) * s
                    sz = (z - cz) * s
                    
                    # Rotate 90 deg right (around Y axis)
                    # x' = z
                    # z' = -x
                    rx = sz
                    rz = -sx
                    ry = sy
                    bus_pos_final.append((rx, ry, rz))

                bus_x, bus_y, bus_z = -15.0, 0.0, -8.0

                if bus_uv and bus_tex and os.path.isfile(bus_tex):
                    bus_tex_obj = create_texture_2d(bus_tex)
                    bus_mesh = Mesh(bus_pos_final, bus_col, bus_uv, (bus_tex_obj.id if bus_tex_obj else None), gl.GL_TRIANGLES, mat4_translate(bus_x, bus_y, bus_z), _tex_obj=(bus_tex_obj if bus_tex_obj else None))
                else:
                    if bus_col:
                        bus_mesh = Mesh(bus_pos_final, bus_col, bus_uv, None, gl.GL_TRIANGLES, mat4_translate(bus_x, bus_y, bus_z))
                    else:
                        fallback_cols = [(0.9, 0.8, 0.1)] * len(bus_pos_final) # Yellow fallback
                        bus_mesh = Mesh(bus_pos_final, fallback_cols, bus_uv, None, gl.GL_TRIANGLES, mat4_translate(bus_x, bus_y, bus_z))

                self.characters.append(MovingCharacter(bus_mesh, bus_x, bus_y, bus_z))
                bustris = len(bus_pos_final)//3
                print(f"Loaded bus model from '{bus_obj_path}': {bustris} tris (scale {s:.6f} -> height {desired_h} m) centered at ({bus_x}, {bus_y}, {bus_z})")
        except Exception as e:
            print(f"WARNING: failed to create bus actor: {e}")

        # Surrounding vehicles
        self.vehicles = []
        try:
            car_obj_path = "assets/car/car.obj"
            cpos, cuv, ctex, ccol = None, None, None, None
            try:
                cpos, cuv, ctex, ccol = load_obj_with_uv_mtl(car_obj_path, scale=1.0, center_y=0.0)
            except Exception:
                cpos = None

            if cpos:
                cys = [p[1] for p in cpos]
                cymin, cymax = min(cys), max(cys)
                model_h = max(1e-6, cymax - cymin)
                desired_h = 1.4
                s = desired_h / model_h
                cpos_scaled = [(x*s, (y - cymin)*s, z*s) for (x, y, z) in cpos]

                if cuv and ctex and os.path.isfile(ctex):
                    ctex_obj = create_texture_2d(ctex)
                    car_mesh = Mesh(cpos_scaled, ccol, cuv, (ctex_obj.id if ctex_obj else None), gl.GL_TRIANGLES, mat4_translate(-2.0, 0.0, -8.0), _tex_obj=(ctex_obj if ctex_obj else None))
                else:
                    cols = ccol if ccol else [(0.8, 0.2, 0.2)] * len(cpos_scaled)
                    car_mesh = Mesh(cpos_scaled, cols, cuv, None, gl.GL_TRIANGLES, mat4_translate(-2.0, 0.0, -8.0))
                
                self.vehicles.append(MovingCharacter(car_mesh, -2.0, 0.0, -8.0))
                try:
                    ctris = len(cpos_scaled)//3
                except:
                    ctris = 0
                print(f"Loaded surrounding car model from '{car_obj_path}': {ctris} tris (scale {s:.3f} -> height {desired_h} m) at (-2.0, -8.0)")
        except Exception as e:
            print(f"WARNING: failed to create surrounding car actor: {e}")

        # Truck
        try:
            truck_obj_path = "assets/truck/truck.obj"
            tpos, tuv, ttex, tcol = None, None, None, None
            try:
                tpos, tuv, ttex, tcol = load_obj_with_uv_mtl(truck_obj_path, scale=1.0, center_y=0.0)
            except Exception as load_err:
                print(f"DEBUG: Failed to load truck OBJ: {load_err}")
                tpos = None

            if tpos:
                txs = [p[0] for p in tpos]
                tys = [p[1] for p in tpos]
                tzs = [p[2] for p in tpos]
                txmin, txmax = min(txs), max(txs)
                tymin, tymax = min(tys), max(tys)
                tzmin, tzmax = min(tzs), max(tzs)
                
                model_h = max(1e-6, tymax - tymin)
                desired_h = 2.1
                s = desired_h / model_h
                
                # Center X and Z, set Y bottom to 0
                cx = (txmin + txmax) * 0.5
                cz = (tzmin + tzmax) * 0.5
                
                tpos_scaled = [((x - cx)*s, (y - tymin)*s, (z - cz)*s) for (x, y, z) in tpos]

                if tuv and ttex and os.path.isfile(ttex):
                    ttex_obj = create_texture_2d(ttex)
                    truck_mesh = Mesh(tpos_scaled, tcol, tuv, (ttex_obj.id if ttex_obj else None), gl.GL_TRIANGLES, mat4_translate(-3.0, 0.0, -8.0), _tex_obj=(ttex_obj if ttex_obj else None))
                else:
                    # If no texture but we have material colors, use them
                    if tcol:
                        truck_mesh = Mesh(tpos_scaled, tcol, tuv, None, gl.GL_TRIANGLES, mat4_translate(-3.0, 0.0, -8.0))
                    else:
                        # Fallback color (orange-ish)
                        fallback_cols = [(0.80, 0.40, 0.10)] * len(tpos_scaled)
                        truck_mesh = Mesh(tpos_scaled, fallback_cols, tuv, None, gl.GL_TRIANGLES, mat4_translate(-3.0, 0.0, -8.0))

                self.characters.append(MovingCharacter(truck_mesh, -10.0, 0.0, -8.0))
                ttris = len(tpos_scaled)//3
                print(f"Loaded truck model from '{truck_obj_path}': {ttris} tris (scale {s:.6f} -> height {desired_h} m) centered at (-3.0, -8.0)")
            else:
                print(f"DEBUG: Truck OBJ loaded but returned empty (tpos=None)")
        except Exception as e:
            print(f"WARNING: failed to create truck actor: {e}")

        # Car
        try:
            car_obj_path = "assets/car/car.obj"
            cpos, cuv, ctex, ccol = None, None, None, None
            try:
                cpos, cuv, ctex, ccol = load_obj_with_uv_mtl(car_obj_path, scale=1.0, center_y=0.0)
            except Exception as load_err:
                print(f"DEBUG: Failed to load car OBJ: {load_err}")
                cpos = None

            if cpos:
                cxs = [p[0] for p in cpos]
                cys = [p[1] for p in cpos]
                czs = [p[2] for p in cpos]
                cxmin, cxmax = min(cxs), max(cxs)
                cymin, cymax = min(cys), max(cys)
                czmin, czmax = min(czs), max(czs)
                
                model_h = max(1e-6, cymax - cymin)
                desired_h = 1.5
                s = desired_h / model_h
                
                # Center X and Z, set Y bottom to 0
                cx = (cxmin + cxmax) * 0.5
                cz = (czmin + czmax) * 0.5
                
                cpos_scaled = [((x - cx)*s, (y - cymin)*s, (z - cz)*s) for (x, y, z) in cpos]

                # Position it next to the truck
                car_x, car_y, car_z = -7, 0.0, -8.0

                if cuv and ctex and os.path.isfile(ctex):
                    ctex_obj = create_texture_2d(ctex)
                    car_mesh = Mesh(cpos_scaled, ccol, cuv, (ctex_obj.id if ctex_obj else None), gl.GL_TRIANGLES, mat4_translate(car_x, car_y, car_z), _tex_obj=(ctex_obj if ctex_obj else None))
                else:
                    if ccol:
                        car_mesh = Mesh(cpos_scaled, ccol, cuv, None, gl.GL_TRIANGLES, mat4_translate(car_x, car_y, car_z))
                    else:
                        fallback_cols = [(0.2, 0.5, 0.8)] * len(cpos_scaled) # Blue-ish
                        car_mesh = Mesh(cpos_scaled, fallback_cols, cuv, None, gl.GL_TRIANGLES, mat4_translate(car_x, car_y, car_z))

                self.characters.append(MovingCharacter(car_mesh, car_x, car_y, car_z))
                ctris = len(cpos_scaled)//3
                print(f"Loaded car model from '{car_obj_path}': {ctris} tris (scale {s:.6f} -> height {desired_h} m) centered at ({car_x}, {car_y}, {car_z})")
        except Exception as e:
            print(f"WARNING: failed to create static car actor: {e}")



        # Lanes + grid
        lane_pts = [[(off, 0.01, -float(s)) for s in range(0, 200, 2)] for off in (-1.75, 1.75)]
        
        self.tile_size   = 40.0      # meters per tile
        self.tile_step   = 2.0       # grid line spacing
        self.tile_radius = 3         # tiles to keep around ego in each axis
        
        # Traffic Light
        self.traffic_light = TrafficLight(x=3.0, y=3.0, z=-10.0)

        # Street signs
        self.street_signs: List[StreetSign] = []
        try:
            # Hexagon base position
            stop_x, stop_y, stop_z = 3.5, 0.0, -15.0
            self.sign_stop = StreetSign('stop', stop_x, stop_y, stop_z)
            # Regular triangle 
            scale_tri = 0.85
            self.sign_regular_triangle = StreetSign('yield', stop_x + 1.4, stop_y, stop_z, scale=scale_tri)
            # Inverted triangle
            r = (0.9 * scale_tri) / math.sqrt(3.0)
            self.sign_inverted_triangle = StreetSign('yield', stop_x + 2.8, stop_y, stop_z, yaw=math.pi, y_adjust=-(r*0.5), scale=scale_tri)
            # Box sign
            self.sign_speed = StreetSign('speed', stop_x + 4.2, stop_y, stop_z)

            # Register signs for rendering
            self.street_signs.append(self.sign_stop)
            self.street_signs.append(self.sign_regular_triangle)
            self.street_signs.append(self.sign_inverted_triangle)
            self.street_signs.append(self.sign_speed)
            print(f"Loaded {len(self.street_signs)} street signs")
        except Exception as e:
            print(f"WARNING: failed to build street signs: {e}")

        gv, gc = make_grid_tile(self.tile_size, self.tile_step)
        self.grid_base = Mesh(gv, gc, None, None, gl.GL_LINES, mat4_identity()) 
        self.grid_tiles = {}
        self.lanes = [Mesh(*make_polyline(pts), None, None, gl.GL_LINES, mat4_identity()) for pts in lane_pts]

        # Camera (initialize behind the car)
        self.cam_yawoff = math.pi / 2.0
        self.cam_pitch  = math.radians(60.0)
        self.cam_dist   = 12.0
        self.cam_height = 4.0

        self.renderer = Renderer()
        self.last_time = time.perf_counter()
        pyglet.clock.schedule_interval(self.update, 1.0/self.fps)

        # Start RTSP streaming if available
        try:
            self.streamer = RTSPStreamer(self.width, self.height, self.fps, port=8554, mount="/sim")
            self.streamer.start()
        except Exception as e:
            print(f"RTSP init failed: {e}")



    def _ego_aabb(self, pos):
        """Get ego vehicle AABB in world space."""
        # Fallback to simple AABB
        x,y,z = pos
        lx, ly, lz = self.ego.length, self.ego.height, self.ego.width
        return (x - lx*0.5, y + 0.0, z - lz*0.5), (x + lx*0.5, y + ly, z + lz*0.5)

    def _aabb_overlap(self, amin, amax, bmin, bmax):
        """Check if two AABBs overlap."""
        if amax[0] < bmin[0] or amin[0] > bmax[0]:
            return False
        if amax[1] < bmin[1] or amin[1] > bmax[1]:
            return False
        if amax[2] < bmin[2] or amin[2] > bmax[2]:
            return False
        return True

    def _stream_grid(self):
        ix = int(math.floor(self.ego.pos[0] / self.tile_size))
        iz = int(math.floor(self.ego.pos[2] / self.tile_size))

        needed = set()
        for i in range(ix - self.tile_radius, ix + self.tile_radius + 1):
            for j in range(iz - self.tile_radius, iz + self.tile_radius + 1):
                needed.add((i, j))
                if (i, j) not in self.grid_tiles:
                    mx = i * self.tile_size
                    mz = j * self.tile_size
                    self.grid_tiles[(i, j)] = mat4_translate(mx, 0.0, mz)

        # Drop tiles that are far behind
        to_delete = [key for key in self.grid_tiles.keys() if key not in needed]
        for key in to_delete:
            del self.grid_tiles[key]

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
        self.brake_on = brake > 0.1

        # Store old position for collision rollback
        old_pos = list(self.ego.pos)
        
        self.ego.update(dt, throttle, steer_cmd, brake)
        for w in self.wheels:
            r = max(1e-6, w['radius'])
            self._wheel_roll += (self.ego.v / r) * dt
        self._stream_grid()

        if getattr(self, 'show_obstacles', False):
            for o in self.obstacles:
                o.update(dt)
        # update characters
        for ch in self.characters:
            ch.update(dt)
        # update street signs
        for ss in self.street_signs:
            ss.update(dt)
        # update traffic light
        self.traffic_light.update(dt)



    # Draw
    def on_draw(self):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        proj = perspective(60.0, max(1e-6, self.width/float(self.height)), 0.1, 500.0)

        # Camera mount point
        cam_mount_local_x = 0.0  # centered laterally
        cam_mount_local_y = 1.65  # on top of windshield
        cam_mount_local_z = -0.5  # forward from center (negative Z is forward)
        
        # Transform camera mount to world space based on vehicle rotation
        c = math.cos(self.ego.yaw)
        s = math.sin(self.ego.yaw)
        cam_mount_world_x = self.ego.pos[0] + (c * cam_mount_local_x - s * cam_mount_local_z)
        cam_mount_world_y = self.ego.pos[1] + cam_mount_local_y
        cam_mount_world_z = self.ego.pos[2] + (s * cam_mount_local_x + c * cam_mount_local_z)
        
        # Camera orbits around the windshield mount point
        tx, ty, tz = cam_mount_world_x, cam_mount_world_y, cam_mount_world_z
        yaw = self.cam_yawoff
        cx = tx - math.cos(yaw) * self.cam_dist
        cy = ty + self.cam_height * math.sin(self.cam_pitch)
        cz = tz + math.sin(yaw) * self.cam_dist
        view = look_at((cx, cy, cz), (tx, ty, tz), (0.0, 1.0, 0.0))
        pv = mat4_mul(proj, view)

        # Draw world with streaming grid
        for _key, model in self.grid_tiles.items():
            self.grid_base.model = model
            self.renderer.draw_mesh(self.grid_base, pv)

        for ln in self.lanes:
            self.renderer.draw_mesh(ln, pv)

        # Car model matrix once
        car_T = mat4_translate(*self.ego.pos)
        car_R = mat4_rotate_y(self.ego.yaw)
        car_M = mat4_mul(car_T, car_R)

        # Draw car
        if self.car_mesh:
            self.car_mesh.model = car_M
            active_tex = self.tex_brake if (self.brake_on and self.tex_brake) else self.tex_normal
            if active_tex:
                self.car_mesh.texture_id = active_tex.id
                self.renderer.draw_mesh(self.car_mesh, pv)
        else:
            self.ego.mesh.model = car_M
            self.renderer.draw_mesh(self.ego.mesh, pv)
        


        # Wheels
        steer_angle = self.ego.steer if self.wheels and self.wheels[0]['steer'] else 0.0
        for w in self.wheels:
            ox, oy, oz = w['offset']
            M = car_M
            M = mat4_mul(M, mat4_translate(ox, oy, oz))
            if w['steer']:
                M = mat4_mul(M, mat4_rotate_y(steer_angle))  # steer about local Y
            M = mat4_mul(M, mat4_rotate_x(self._wheel_roll))  # roll about local X
            w['mesh'].model = M
            self.renderer.draw_mesh(w['mesh'], pv)


        if getattr(self, 'show_obstacles', False):
            for o in self.obstacles:
                self.renderer.draw_mesh(o.mesh, pv)

        # Draw traffic light
        self.renderer.draw_mesh(self.traffic_light.mesh, pv)

        # Draw street signs
        for ss in self.street_signs:
            self.renderer.draw_mesh(ss.pole_mesh, pv)
            self.renderer.draw_mesh(ss.panel_mesh, pv)

        # Draw characters
        for ch in self.characters:
            self.renderer.draw_mesh(ch.mesh, pv)

        self.hud.text = f"Speed {self.ego.v:4.1f} m/s   Yaw {math.degrees(self.ego.yaw):5.1f} deg"
        self.hud.draw()

        # Capture and stream current frame via RTSP
        try:
            if getattr(self, 'streamer', None) and self.streamer.is_active:
                img = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
                data = img.get_data('RGB', self.width * 3)
                self.streamer.push_rgb_frame(data)
        except Exception as e:
            pass

# ------------------------------------------------------------
if __name__ == "__main__":
    window = AVHMI(1280, 720, 60)
    pyglet.app.run()
