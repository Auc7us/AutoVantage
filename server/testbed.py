# Pyglet 2.1.9 • Modern OpenGL (core profile) • Textured OBJ support
import math, time, ctypes, os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import pyglet
from pyglet.window import key, mouse
from pyglet import gl
from pyglet.graphics.shader import Shader, ShaderProgram

try:
    import rclpy
    from rclpy.node import Node
    from wauto_perception_msgs.msg import ObjectArray
    ROS2_AVAILABLE = True
except ImportError:
    rclpy = None
    Node = object
    ObjectArray = None
    ROS2_AVAILABLE = False


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(REPO_ROOT, "assets")
DETECTION_YAW_CORRECTION_DEG = -6.0
DETECTION_LATERAL_OFFSET_M = -1.0
VEHICLE_CENTER_CORRECTION_CLASSES = {1, 11, 13}
LANE_CONSENSUS_CLASSES = {1, 11, 13}
VEHICLE_CENTER_HALF_LENGTH_M = 2.2
VEHICLE_CENTER_HALF_WIDTH_M = 0.9
VEHICLE_AXIS_MIN_STEP_M = 0.12
STATIC_REFERENCE_CLASSES = {4, 5, 6, 7, 8, 9, 10}
MOTION_HEADING_CLASSES = {1, 2, 11, 13, 16}
HEADING_SMOOTH_ALPHA = 0.3
HEADING_CONFIRM_FRAMES = 5
HEADING_UPDATE_DEG = 8.0
HEADING_MOTION_MIN_STEP_M = 0.10
HEADING_HISTORY_SIZE = 6
MOTION_HEADING_MESH_OFFSET_DEG = 180.0
LANE_LATERAL_THRESHOLD_M = 2.2
LANE_LONGITUDINAL_LOOKAHEAD_M = 60.0
LANE_CONSENSUS_MIN_NEIGHBORS = 2
LANE_CONSENSUS_BLEND_ALPHA = 0.22
SPEED_ROLLING_WINDOW = 12
MPS_TO_MPH = 2.23693629


def asset_path(*parts: str) -> str:
    return os.path.join(ASSETS_DIR, *parts)


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

def mat4_scale(sx, sy, sz):
    return [sx,0, 0, 0,
            0, sy,0, 0,
            0, 0, sz,0,
            0, 0, 0, 1]

def lerp_angle(current, target, alpha):
    delta = math.atan2(math.sin(target - current), math.cos(target - current))
    return current + delta * alpha

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

def make_box_triangles(lx, ly, lz, color=(0.1, 0.8, 0.3)):
    """Create a box mesh with triangles for rendering."""
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


def clone_mesh(mesh: Mesh) -> Mesh:
    cloned = Mesh(
        mesh.verts,
        mesh.cols,
        mesh.uvs,
        mesh.texture_id,
        mesh.mode,
        mat4_identity(),
        _tex_obj=mesh._tex_obj,
    )
    if hasattr(mesh, "_gpu"):
        cloned._gpu = mesh._gpu
    return cloned


def colorize_barrel_mesh(verts):
    if not verts:
        return []

    ys = [v[1] for v in verts]
    ymin = min(ys)
    ymax = max(ys)
    yrange = max(1e-6, ymax - ymin)

    orange = (0.95, 0.42, 0.08)
    white = (0.96, 0.96, 0.96)
    dark = (0.18, 0.18, 0.18)

    colors = []
    for _x, y, _z in verts:
        t = (y - ymin) / yrange
        if t < 0.08 or t > 0.92:
            colors.append(dark)
        elif 0.22 <= t <= 0.34 or 0.56 <= t <= 0.68:
            colors.append(white)
        else:
            colors.append(orange)
    return colors


def solid_vertex_colors(verts, color):
    return [color] * len(verts)


def recolor_car_mesh_colors(verts, source_colors, body_gray=(0.55, 0.57, 0.60)):
    if not verts:
        return []

    if not source_colors or len(source_colors) != len(verts):
        source_colors = [(0.8, 0.2, 0.2)] * len(verts)

    recolored = []
    for r, g, b in source_colors:
        # Replace the painted red body regions while leaving trim/glass/etc. alone.
        if r > 0.45 and r > g * 1.2 and r > b * 1.2:
            brightness = (r + g + b) / 3.0
            shade = clamp(0.85 + (brightness - 0.4) * 0.6, 0.7, 1.05)
            recolored.append(tuple(clamp(channel * shade, 0.0, 1.0) for channel in body_gray))
        else:
            recolored.append((r, g, b))
    return recolored


def normalize_vec2(x, z):
    mag = math.hypot(x, z)
    if mag < 1e-6:
        return 0.0, 0.0
    return x / mag, z / mag


def median(values):
    if not values:
        return 0.0
    values = sorted(values)
    n = len(values)
    mid = n // 2
    if n % 2:
        return values[mid]
    return 0.5 * (values[mid - 1] + values[mid])


class ObjectDetectionSubscriber(Node):
    def __init__(self, on_message, topic="/perception/objectdetection"):
        super().__init__("wautovantage_testbed_subscriber")
        self._on_message = on_message
        self.subscription = self.create_subscription(
            ObjectArray,
            topic,
            self._listener_callback,
            10,
        )

    def _listener_callback(self, msg):
        objects = []
        for obj in msg.objects:
            objects.append({
                "object_id": int(obj.object_id),
                "obj_class": int(obj.obj_class),
                "custom_classification": int(obj.custom_classification),
                "height": float(obj.height),
                "width": float(obj.width),
                "x": float(obj.x),
                "y": float(obj.y),
                "z": float(obj.z),
            })
        self._on_message(objects)


class ROSObjectDetectionBridge:
    def __init__(self, topic="/perception/objectdetection"):
        self.topic = topic
        self.enabled = False
        self.node = None
        self._latest_stamp = 0.0
        self._latest_objects: List[Dict[str, float]] = []

        if not ROS2_AVAILABLE:
            print("ROS2 Python packages not available; object detection bridge disabled")
            return

        try:
            if not rclpy.ok():
                rclpy.init(args=None)
            self.node = ObjectDetectionSubscriber(self._store_message, topic=topic)
            self.enabled = True
            print(f"Subscribed to ROS2 topic '{topic}'")
        except Exception as exc:
            print(f"Failed to initialize ROS2 bridge for '{topic}': {exc}")
            self.enabled = False

    def _store_message(self, objects):
        self._latest_objects = objects
        self._latest_stamp = time.perf_counter()

    def spin_once(self):
        if not self.enabled or self.node is None:
            return
        try:
            rclpy.spin_once(self.node, timeout_sec=0.0)
        except Exception as exc:
            print(f"ROS2 spin_once failed: {exc}")
            self.enabled = False

    def latest_snapshot(self):
        return self._latest_stamp, list(self._latest_objects)

    def close(self):
        if self.node is not None:
            try:
                self.node.destroy_node()
            except Exception:
                pass
            self.node = None
        if ROS2_AVAILABLE and rclpy and rclpy.ok():
            try:
                rclpy.shutdown()
            except Exception:
                pass

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


class DetectedEntity:
    def __init__(self, detection: Dict[str, float], actor, actor_key):
        self.object_id = int(detection["object_id"])
        self.obj_class = int(detection["obj_class"])
        self.custom_classification = int(detection.get("custom_classification", 0))
        self.actor = actor
        self.actor_key = actor_key
        self.local_pose = [0.0, 0.0, 0.0]
        self.target_local_pose = [0.0, 0.0, 0.0]
        self._pose_initialized = False
        self._raw_local_pose = None
        self._motion_yaw = None
        self._ground_motion_history = []
        self.last_seen = time.perf_counter()

    def update(self, detection: Dict[str, float], actor=None, actor_key=None):
        if actor is not None:
            self.actor = actor
        if actor_key is not None:
            self.actor_key = actor_key
        self.obj_class = int(detection["obj_class"])
        self.custom_classification = int(detection.get("custom_classification", 0))
        self.last_seen = time.perf_counter()

    def set_local_pose(self, x: float, y: float, z: float, camera_origin_local, ego_motion_local):
        raw_pose = [x, y, z]
        self._update_motion_heading(raw_pose, ego_motion_local)

        if self.obj_class in VEHICLE_CENTER_CORRECTION_CLASSES:
            new_pose = self._vehicle_center_from_nearest_point(raw_pose, camera_origin_local)
        else:
            new_pose = raw_pose

        self.target_local_pose = new_pose
        if not self._pose_initialized:
            self.local_pose = list(new_pose)
            self._pose_initialized = True

    def _update_motion_heading(self, raw_pose, ego_motion_local):
        if self._raw_local_pose is not None:
            dx = raw_pose[0] - self._raw_local_pose[0]
            dz = raw_pose[2] - self._raw_local_pose[2]
            ground_dx = dx + ego_motion_local[0]
            ground_dz = dz + ego_motion_local[2]
            if self.obj_class in MOTION_HEADING_CLASSES and math.hypot(ground_dx, ground_dz) >= HEADING_MOTION_MIN_STEP_M:
                self._ground_motion_history.append((ground_dx, ground_dz))
                if len(self._ground_motion_history) > HEADING_HISTORY_SIZE:
                    self._ground_motion_history.pop(0)

                if len(self._ground_motion_history) >= HEADING_CONFIRM_FRAMES:
                    avg_dx = sum(v[0] for v in self._ground_motion_history) / len(self._ground_motion_history)
                    avg_dz = sum(v[1] for v in self._ground_motion_history) / len(self._ground_motion_history)
                    if math.hypot(avg_dx, avg_dz) < HEADING_MOTION_MIN_STEP_M:
                        self._raw_local_pose = list(raw_pose)
                        return

                    target_yaw = math.atan2(avg_dx, -avg_dz)
                    target_yaw += math.radians(MOTION_HEADING_MESH_OFFSET_DEG)
                    if self._motion_yaw is None:
                        self._motion_yaw = target_yaw
                    else:
                        delta_current = abs(math.degrees(math.atan2(
                            math.sin(target_yaw - self._motion_yaw),
                            math.cos(target_yaw - self._motion_yaw),
                        )))
                        if delta_current >= HEADING_UPDATE_DEG:
                            self._motion_yaw = lerp_angle(self._motion_yaw, target_yaw, HEADING_SMOOTH_ALPHA)
            else:
                if self._ground_motion_history:
                    self._ground_motion_history.pop(0)
        self._raw_local_pose = list(raw_pose)

    def _vehicle_center_from_nearest_point(self, raw_pose, camera_origin_local):
        if self._motion_yaw is None:
            return raw_pose

        cam_x, _cam_y, cam_z = camera_origin_local
        los_x, los_z = normalize_vec2(raw_pose[0] - cam_x, raw_pose[2] - cam_z)
        if los_x == 0.0 and los_z == 0.0:
            return raw_pose

        fwd_x = math.sin(self._motion_yaw)
        fwd_z = -math.cos(self._motion_yaw)
        right_x = math.cos(self._motion_yaw)
        right_z = math.sin(self._motion_yaw)

        longitudinal = abs((los_x * fwd_x) + (los_z * fwd_z))
        lateral = abs((los_x * right_x) + (los_z * right_z))
        shift = (VEHICLE_CENTER_HALF_LENGTH_M * longitudinal) + (VEHICLE_CENTER_HALF_WIDTH_M * lateral)

        return [
            raw_pose[0] + (los_x * shift),
            raw_pose[1],
            raw_pose[2] + (los_z * shift),
        ]

    def apply_anchor(self, anchor_model):
        actor = self.actor
        if actor is None:
            return

        alpha = 0.28
        for i in range(3):
            self.local_pose[i] += (self.target_local_pose[i] - self.local_pose[i]) * alpha

        local_model = mat4_translate(*self.local_pose)
        world_model = mat4_mul(anchor_model, local_model)

        if isinstance(actor, StreetSign):
            actor.pole_mesh.model = world_model
            ox, oy, oz = actor.panel_offset
            panel_model = mat4_mul(world_model, mat4_translate(ox, oy, oz))
            panel_model = mat4_mul(panel_model, mat4_rotate_x(math.radians(90.0)))
            if getattr(actor, "yaw", 0.0) != 0.0:
                panel_model = mat4_mul(panel_model, mat4_rotate_y(actor.yaw))
            actor.panel_mesh.model = panel_model
            return

        if isinstance(actor, TrafficLight):
            actor.mesh.model = world_model
            return

        if hasattr(actor, "mesh"):
            if self.obj_class in MOTION_HEADING_CLASSES and self._motion_yaw is not None:
                world_model = mat4_mul(mat4_translate(world_model[12], world_model[13], world_model[14]), mat4_rotate_y(self._motion_yaw))
            actor.mesh.model = world_model
        elif isinstance(actor, Mesh):
            actor.model = world_model

    def average_ground_motion(self):
        if len(self._ground_motion_history) < HEADING_CONFIRM_FRAMES:
            return None
        avg_dx = sum(v[0] for v in self._ground_motion_history) / len(self._ground_motion_history)
        avg_dz = sum(v[1] for v in self._ground_motion_history) / len(self._ground_motion_history)
        if math.hypot(avg_dx, avg_dz) < HEADING_MOTION_MIN_STEP_M:
            return None
        return avg_dx, avg_dz

    def apply_lane_consensus_yaw(self, consensus_yaw):
        consensus_yaw += math.radians(MOTION_HEADING_MESH_OFFSET_DEG)
        if self._motion_yaw is None:
            self._motion_yaw = consensus_yaw
            return
        self._motion_yaw = lerp_angle(self._motion_yaw, consensus_yaw, LANE_CONSENSUS_BLEND_ALPHA)

    def draw(self, renderer: "Renderer", pv):
        actor = self.actor
        if actor is None:
            return

        if isinstance(actor, StreetSign):
            renderer.draw_mesh(actor.pole_mesh, pv)
            renderer.draw_mesh(actor.panel_mesh, pv)
            return

        if isinstance(actor, TrafficLight):
            renderer.draw_mesh(actor.mesh, pv)
            return

        if isinstance(actor, Mesh):
            renderer.draw_mesh(actor, pv)
            return

        renderer.draw_mesh(actor.mesh, pv)


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

    def warm_mesh(self, mesh: Mesh):
        if hasattr(mesh, "_gpu"):
            return
        if mesh.uvs is not None and mesh.texture_id is not None:
            self._build_gpu_tex(mesh)
        else:
            self._build_gpu_color(mesh)


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
        self.speed_hud = pyglet.text.Label(
            "",
            font_name="Roboto",
            font_size=20,
            x=width // 2,
            y=height - 28,
            anchor_x="center",
            anchor_y="center",
        )
        self.camera_origin_local = (0.0, 1.35, 0.0)
        self.detected_entities: Dict[int, DetectedEntity] = {}
        self.detected_mesh_templates: Dict[str, Mesh] = {}
        self._last_detection_stamp = 0.0
        self._ego_motion_local_estimate = [0.0, 0.0, 0.0]
        self._ego_speed_mps_estimate = 0.0
        self._ego_speed_mph_history = []
        self.ros_bridge = ROSObjectDetectionBridge()
        self.detected_car_color = (0.55, 0.57, 0.60)

        self.ego = Ego()
        car_obj_path = asset_path("WAutoCar.obj")
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
        whl_R_obj_path = asset_path("whl", "whl_R.obj")
        whl_L_obj_path = asset_path("whl", "whl_L.obj")
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

        # Barricade Model
        bl, bd = 1.8*0.85, 0.6*0.85
        tl, td = 1.4*0.85, 0.4*0.85
        h      = 1.0*0.85
        bx, by, bz = 3.0 + 1.2, 0.0, -10.0
        self.obstacles: List[MovingBarricade] = [
            MovingBarricade(bx, by, bz, base_len=bl, base_depth=bd, top_len=tl, top_depth=td, height=h, color=(1.0, 0.5, 0.0)),
            MovingBarricade(bx + 2.0, by, bz, base_len=bl, base_depth=bd, top_len=tl, top_depth=td, height=h, color=(0.6, 0.6, 0.6))
        ]
        self.show_obstacles = True

        # Human Model
        self.characters: List[MovingCharacter] = []
        human_obj_path = asset_path("human", "human.obj")
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
            self.detected_mesh_templates["human"] = clone_mesh(hmesh)
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
            self.detected_mesh_templates["human"] = clone_mesh(fallback)

            self.characters.append(MovingCharacter(fallback, 2.0, 0.0, -8.0))

        # Barrel Model
        try:
            barrel_obj_path = asset_path("barrel", "barrel.obj")
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

            cols = colorize_barrel_mesh(bpos_scaled) if bpos_scaled else [(0.95, 0.42, 0.08)]
            bmesh = Mesh(bpos_scaled, cols, None, None, gl.GL_TRIANGLES, mat4_translate(4.0, 0.0, -8.0))
            self.detected_mesh_templates["barrel"] = clone_mesh(bmesh)
            self.obstacles.append(StaticObject(bmesh))
            try:
                tris = len(bpos_scaled)//3
            except Exception:
                tris = 0
            print(f"Loaded barrel model: {tris} tris (scale {s:.3f} -> height {desired_h} m, orange/white striped)")
        except Exception as e:
            print(f"WARNING: failed to load barrel '{barrel_obj_path}': {e}")
            v, _c = make_box_triangles(0.6, 1.0, 0.6, (0.95, 0.42, 0.08))
            c = colorize_barrel_mesh(v)
            fallback = Mesh(v, c, None, None, gl.GL_TRIANGLES, mat4_translate(4.0, 0.0, -8.0))
            self.detected_mesh_templates["barrel"] = clone_mesh(fallback)
            self.obstacles.append(StaticObject(fallback))

        # Traffic Cone Model
        try:
            cone_obj_path = asset_path("traffic-cone", "cone.obj")
            cpos, cuv, ctex, ccol = load_obj_with_uv_mtl(cone_obj_path, scale=1.0, center_y=0.0)
            if cpos:
                ys = [p[1] for p in cpos]
                ymin, ymax = min(ys), max(ys)
                model_h = max(1e-6, ymax - ymin)
                desired_h =0.75
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
            self.detected_mesh_templates["cone"] = clone_mesh(cmesh)
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
            self.detected_mesh_templates["cone"] = clone_mesh(fallback)
            self.obstacles.append(StaticObject(fallback))

        # Deer Model
        try:
            deer_obj_path = asset_path("deer", "deer.obj")
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
                self.detected_mesh_templates["deer"] = clone_mesh(deer_mesh)
                self.characters.append(MovingCharacter(deer_mesh, 10.0, 0.0, -8.0))
                try:
                    tris = len(dpos_scaled)//3
                except Exception:
                    tris = 0
                print(f"Loaded deer model from '{deer_obj_path}': {tris} tris (scale {s:.3f} -> height {desired_h} m) at (10.0, -8.0)")
            else:
                v, c = make_box_triangles(1.2, 0.8, 0.4, (0.6, 0.4, 0.2))
                fb = Mesh(v, c, None, None, gl.GL_TRIANGLES, mat4_translate(10.0, 0.0, -8.0))
                self.detected_mesh_templates["deer"] = clone_mesh(fb)
                self.characters.append(MovingCharacter(fb, 10.0, 0.0, -8.0))
                print("Deer OBJ present but failed to build mesh — added fallback box at x=10.0")
        except Exception as e:
            print(f"WARNING: failed to create deer actor: {e}")

        # Surrounding Car Model
        self.vehicles = []
        try:
            car_obj_path = asset_path("car", "car.obj")
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
                gray_cols = recolor_car_mesh_colors(cpos_scaled, ccol, self.detected_car_color)
                car_mesh = Mesh(cpos_scaled, gray_cols, None, None, gl.GL_TRIANGLES, mat4_translate(-2.0, 0.0, -8.0))
                
                self.detected_mesh_templates["car"] = clone_mesh(car_mesh)
                self.vehicles.append(MovingCharacter(car_mesh, -2.0, 0.0, -8.0))
                try:
                    ctris = len(cpos_scaled)//3
                except:
                    ctris = 0
                print(f"Loaded surrounding car model from '{car_obj_path}': {ctris} tris (scale {s:.3f} -> height {desired_h} m) at (-2.0, -8.0)")
        except Exception as e:
            print(f"WARNING: failed to create surrounding car actor: {e}")

        # Car
        try:
            car_obj_path = asset_path("car", "car.obj")
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
                gray_cols = recolor_car_mesh_colors(cpos_scaled, ccol, self.detected_car_color)
                car_mesh = Mesh(cpos_scaled, gray_cols, None, None, gl.GL_TRIANGLES, mat4_translate(car_x, car_y, car_z))

                if "car" not in self.detected_mesh_templates:
                    self.detected_mesh_templates["car"] = clone_mesh(car_mesh)
                self.characters.append(MovingCharacter(car_mesh, car_x, car_y, car_z))
                ctris = len(cpos_scaled)//3
                print(f"Loaded car model from '{car_obj_path}': {ctris} tris (scale {s:.6f} -> height {desired_h} m) centered at ({car_x}, {car_y}, {car_z})")
        except Exception as e:
            print(f"WARNING: failed to create static car actor: {e}")

        # Lanes + grid
        lane_pts = []
        
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
        self.cam_pitch  = math.radians(48.0)
        self.cam_dist   = 6.5
        self.cam_height = 2.5

        self.renderer = Renderer()
        self._warm_detected_templates()
        self.last_time = time.perf_counter()
        self._disable_demo_detection_actors()
        pyglet.clock.schedule_interval(self.update, 1.0/self.fps)



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

    def _disable_demo_detection_actors(self):
        self.show_obstacles = False
        self.obstacles = []
        self.characters = []
        self.vehicles = []
        self.street_signs = []
        self.traffic_light = None

    def _warm_detected_templates(self):
        for mesh in self.detected_mesh_templates.values():
            self.renderer.warm_mesh(mesh)

    def _valid_dimension(self, value: float) -> bool:
        return isinstance(value, (int, float)) and math.isfinite(value) and 0.01 < value < 1000.0

    def _default_height_for_class(self, obj_class: int) -> float:
        return {
            1: 1.5,
            2: 1.8,
            3: 1.6,
            4: 1.0,
            5: 3.0,
            6: 2.0,
            7: 1.0,
            9: 0.75,
            11: 3.0,
            12: 1.4,
            13: 3.2,
            14: 1.5,
            15: 3.5,
            16: 1.8,
        }.get(obj_class, 1.0)

    def _valid_detection(self, detection: Dict[str, float]) -> bool:
        if int(detection.get("object_id", 255)) == 255:
            return False
        if int(detection.get("obj_class", 255)) == 255:
            return False
        coords = (detection.get("x"), detection.get("y"), detection.get("z"))
        return all(isinstance(v, (int, float)) and math.isfinite(v) and abs(v) < 9000.0 for v in coords)

    def _ros_detection_to_local_pose(self, detection: Dict[str, float]) -> List[float]:
        # ZED is configured as RIGHT_HANDED_Z_UP_X_FWD:
        # x = forward, y = left, z = up
        forward = float(detection["x"])
        left = float(detection["y"])
        up = float(detection["z"])
        cam_x, cam_y, cam_z = self.camera_origin_local

        # Apply a small fixed sensor-frame yaw correction before mapping into the
        # testbed's local coordinates. Negative is clockwise from top-down view.
        right = -left
        theta = math.radians(DETECTION_YAW_CORRECTION_DEG)
        c = math.cos(theta)
        s = math.sin(theta)
        corrected_right = (right * c) - (forward * s)
        corrected_forward = (right * s) + (forward * c)

        obj_height = float(detection.get("height", 0.0))
        if not self._valid_dimension(obj_height):
            obj_height = self._default_height_for_class(int(detection["obj_class"]))

        world_x = cam_x + corrected_right + DETECTION_LATERAL_OFFSET_M
        world_y = max(0.0, cam_y + up - (obj_height * 0.5))
        world_z = cam_z - corrected_forward
        return [world_x, world_y, world_z]

    def _make_mesh_actor(self, mesh: Mesh):
        return MovingCharacter(clone_mesh(mesh), 0.0, 0.0, 0.0)

    def _actor_key_for_detection(self, detection: Dict[str, float]):
        obj_class = int(detection["obj_class"])
        custom = int(detection.get("custom_classification", 0))
        if obj_class == 6:
            if custom == 5:
                return (obj_class, "stop")
            if custom == 9:
                return (obj_class, "yield")
            return (obj_class, "speed")
        return (obj_class,)

    def _build_fallback_actor(self, obj_class: int, detection: Dict[str, float]):
        height = float(detection.get("height", 0.0))
        width = float(detection.get("width", 0.0))
        if not self._valid_dimension(height):
            height = self._default_height_for_class(obj_class)
        if not self._valid_dimension(width):
            width = max(0.4, height * 0.5)
        depth = max(0.4, min(4.0, width * 1.4))
        colors = {
            1: (0.2, 0.5, 0.8),
            4: (1.0, 0.5, 0.0),
            5: (0.3, 0.3, 0.3),
            6: (0.9, 0.9, 0.9),
            7: (0.55, 0.35, 0.15),
            9: (0.95, 0.45, 0.1),
            11: (0.55, 0.55, 0.7),
            12: (0.2, 0.8, 0.2),
            13: (0.9, 0.8, 0.2),
            14: (0.8, 0.2, 0.2),
            15: (0.55, 0.55, 0.55),
            16: (0.8, 0.7, 0.6),
        }
        v, c = make_box_triangles(width, height, depth, colors.get(obj_class, (0.7, 0.7, 0.7)))
        mesh = Mesh(v, c, None, None, gl.GL_TRIANGLES, mat4_identity())
        return MovingCharacter(mesh, 0.0, 0.0, 0.0)

    def _build_detected_actor(self, detection: Dict[str, float]):
        obj_class = int(detection["obj_class"])
        custom = int(detection.get("custom_classification", 0))

        if obj_class == 1 and "car" in self.detected_mesh_templates:
            return self._make_mesh_actor(self.detected_mesh_templates["car"])
        if obj_class in (2, 12, 14, 16) and "human" in self.detected_mesh_templates:
            return self._make_mesh_actor(self.detected_mesh_templates["human"])
        if obj_class == 3 and "deer" in self.detected_mesh_templates:
            return self._make_mesh_actor(self.detected_mesh_templates["deer"])
        if obj_class == 4:
            return MovingBarricade(0.0, 0.0, 0.0)
        if obj_class == 5:
            return TrafficLight(0.0, 0.0, 0.0)
        if obj_class == 6:
            sign_kind = "speed"
            if custom == 5:
                sign_kind = "stop"
            elif custom == 9:
                sign_kind = "yield"
            return StreetSign(sign_kind, 0.0, 0.0, 0.0, yaw=math.pi)
        if obj_class == 7 and "barrel" in self.detected_mesh_templates:
            return self._make_mesh_actor(self.detected_mesh_templates["barrel"])
        if obj_class == 9 and "cone" in self.detected_mesh_templates:
            return self._make_mesh_actor(self.detected_mesh_templates["cone"])
        return self._build_fallback_actor(obj_class, detection)

    def _sync_detected_entities(self, detections: List[Dict[str, float]]):
        detection_dt = None
        if self._last_detection_stamp > 0.0:
            detection_dt = max(1e-3, time.perf_counter() - self._last_detection_stamp)

        ego_motion_local = self._estimate_ego_motion_local(detections, detection_dt)
        active_ids = set()
        for detection in detections:
            if not self._valid_detection(detection):
                continue

            object_id = int(detection["object_id"])
            active_ids.add(object_id)
            local_pose = self._ros_detection_to_local_pose(detection)
            detection = dict(detection)
            detection["local_pose"] = local_pose
            actor_key = self._actor_key_for_detection(detection)

            entity = self.detected_entities.get(object_id)
            actor_changed = (
                entity is None
                or entity.actor_key != actor_key
            )
            actor = self._build_detected_actor(detection) if actor_changed else None

            if entity is None:
                entity = DetectedEntity(detection, actor, actor_key)
                self.detected_entities[object_id] = entity
            else:
                entity.update(detection, actor=actor, actor_key=actor_key)

            entity.set_local_pose(*local_pose, self.camera_origin_local, ego_motion_local)

        self._apply_lane_consensus(active_ids)

        stale_ids = [oid for oid in self.detected_entities.keys() if oid not in active_ids]
        for oid in stale_ids:
            del self.detected_entities[oid]

    def _apply_lane_consensus(self, active_ids):
        vehicle_ids = [
            oid for oid in active_ids
            if oid in self.detected_entities and self.detected_entities[oid].obj_class in LANE_CONSENSUS_CLASSES
        ]

        for oid in vehicle_ids:
            entity = self.detected_entities[oid]
            if not entity._pose_initialized:
                continue

            neighbor_vectors = []
            ex, _, ez = entity.target_local_pose
            for other_id in vehicle_ids:
                other = self.detected_entities[other_id]
                if not other._pose_initialized:
                    continue
                ox, _, oz = other.target_local_pose
                if abs(ox - ex) > LANE_LATERAL_THRESHOLD_M:
                    continue
                if abs(oz - ez) > LANE_LONGITUDINAL_LOOKAHEAD_M:
                    continue

                avg_motion = other.average_ground_motion()
                if avg_motion is None:
                    continue
                neighbor_vectors.append(avg_motion)

            if len(neighbor_vectors) < LANE_CONSENSUS_MIN_NEIGHBORS:
                continue

            avg_dx = sum(v[0] for v in neighbor_vectors) / len(neighbor_vectors)
            avg_dz = sum(v[1] for v in neighbor_vectors) / len(neighbor_vectors)
            if math.hypot(avg_dx, avg_dz) < HEADING_MOTION_MIN_STEP_M:
                continue

            consensus_yaw = math.atan2(avg_dx, -avg_dz)
            entity.apply_lane_consensus_yaw(consensus_yaw)

    def _estimate_ego_motion_local(self, detections: List[Dict[str, float]], detection_dt: Optional[float]):
        motion_samples_x = []
        motion_samples_z = []

        for detection in detections:
            if not self._valid_detection(detection):
                continue
            if int(detection["obj_class"]) not in STATIC_REFERENCE_CLASSES:
                continue

            object_id = int(detection["object_id"])
            entity = self.detected_entities.get(object_id)
            if entity is None or entity._raw_local_pose is None:
                continue

            local_pose = self._ros_detection_to_local_pose(detection)
            dx = local_pose[0] - entity._raw_local_pose[0]
            dz = local_pose[2] - entity._raw_local_pose[2]
            motion_samples_x.append(-dx)
            motion_samples_z.append(-dz)

        if motion_samples_x and motion_samples_z:
            measured = [median(motion_samples_x), 0.0, median(motion_samples_z)]
            alpha = 0.35
            self._ego_motion_local_estimate[0] += (measured[0] - self._ego_motion_local_estimate[0]) * alpha
            self._ego_motion_local_estimate[2] += (measured[2] - self._ego_motion_local_estimate[2]) * alpha
            if detection_dt is not None:
                forward_speed = abs(self._ego_motion_local_estimate[2]) / detection_dt
                self._ego_speed_mps_estimate += (forward_speed - self._ego_speed_mps_estimate) * 0.25
                self._ego_speed_mph_history.append(self._ego_speed_mps_estimate * MPS_TO_MPH)
                if len(self._ego_speed_mph_history) > SPEED_ROLLING_WINDOW:
                    self._ego_speed_mph_history.pop(0)
        else:
            self._ego_motion_local_estimate[0] *= 0.85
            self._ego_motion_local_estimate[2] *= 0.85
            # Keep the last known speed when static references drop out briefly.
            self._ego_speed_mps_estimate *= 0.995

        return list(self._ego_motion_local_estimate)

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

        self.ros_bridge.spin_once()
        detection_stamp, detections = self.ros_bridge.latest_snapshot()
        if detection_stamp > self._last_detection_stamp:
            self._sync_detected_entities(detections)
            self._last_detection_stamp = detection_stamp

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
        if self.traffic_light is not None:
            self.traffic_light.update(dt)



    # Draw
    def on_draw(self):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        proj = perspective(60.0, max(1e-6, self.width/float(self.height)), 0.1, 500.0)

        # Camera mount point
        cam_mount_local_x, cam_mount_local_y, cam_mount_local_z = self.camera_origin_local
        
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
        if self.traffic_light is not None:
            self.renderer.draw_mesh(self.traffic_light.mesh, pv)

        # Draw street signs
        for ss in self.street_signs:
            self.renderer.draw_mesh(ss.pole_mesh, pv)
            self.renderer.draw_mesh(ss.panel_mesh, pv)

        # Draw characters
        for ch in self.characters:
            self.renderer.draw_mesh(ch.mesh, pv)

        for entity in self.detected_entities.values():
            entity.apply_anchor(car_M)
            entity.draw(self.renderer, pv)

        ros_status = "ROS2 OK" if self.ros_bridge.enabled else "ROS2 OFF"
        avg_speed_mph = (
            sum(self._ego_speed_mph_history) / len(self._ego_speed_mph_history)
            if self._ego_speed_mph_history else self._ego_speed_mps_estimate * MPS_TO_MPH
        )
        self.hud.text = f"Sim Speed {self.ego.v:4.1f} m/s   Yaw {math.degrees(self.ego.yaw):5.1f} deg   {ros_status}   Objects {len(self.detected_entities)}"
        self.speed_hud.x = self.width // 2
        self.speed_hud.y = self.height - 28
        self.speed_hud.text = f"Estimated Speed {avg_speed_mph:4.1f} mph"
        self.speed_hud.draw()
        self.hud.draw()
        
        if hasattr(self, 'streaming'):
            self.streaming.push_frame()

    def on_close(self):
        self.ros_bridge.close()
        super().on_close()

# ------------------------------------------------------------
if __name__ == "__main__":
    try:
        from streaming_integration import StreamingIntegration
        streaming = StreamingIntegration(
            width=1280,
            height=720,
            fps=30,
            rtp_host="127.0.0.1",
            rtp_port=5004
        )
    except ImportError:
        print("Streaming module not available - running without streaming")
        streaming = None
    
    # Create and run the main window
    window = AVHMI(1280, 720, 60)
    
    if streaming:
        window.streaming = streaming
    
    try:
        pyglet.app.run()
    finally:
        if streaming:
            streaming.stop()
