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
    from wauto_perception_msgs.msg import ObjectArray, LaneArray
    ROS2_AVAILABLE = True
except ImportError:
    rclpy = None
    Node = object
    ObjectArray = None
    LaneArray = None
    ROS2_AVAILABLE = False


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(REPO_ROOT, "assets")
DETECTION_YAW_CORRECTION_DEG = -6.0
DETECTION_LATERAL_OFFSET_M = -1.0
DETECTION_YAW_CORRECTION_RAD = math.radians(DETECTION_YAW_CORRECTION_DEG)
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
MOTION_HEADING_MESH_OFFSET_RAD = math.radians(MOTION_HEADING_MESH_OFFSET_DEG)
ORIENTATION_RENDER_BLEND_ALPHA = 0.26
LANE_SNAP_RENDER_BLEND_ALPHA = 0.42
LANE_LATERAL_THRESHOLD_M = 2.2
LANE_LONGITUDINAL_LOOKAHEAD_M = 60.0
LANE_CONSENSUS_MIN_NEIGHBORS = 2
LANE_CONSENSUS_BLEND_ALPHA = 0.22
SPEED_ROLLING_WINDOW = 12
MPS_TO_MPH = 2.23693629
TRACKING_DT_DEFAULT_SEC = 0.10
TRACKING_DT_MIN_SEC = 1.0 / 30.0
TRACKING_DT_MAX_SEC = 0.35
POSE_TRACK_ALPHA = 0.42
POSE_TRACK_BETA = 0.16
POSE_PREDICTION_LOOKAHEAD_SEC = 0.18
POSE_TRACK_Y_ALPHA = 0.35
GROUND_VELOCITY_SMOOTH_ALPHA = 0.30
HEADING_MIN_SPEED_MPS = 0.75
LANE_LINE_CONFIDENCE_MIN = 0.30
LANE_LINE_RENDER_LIFT_M = 0.04
LANE_LINE_WIDTH_M = 0.18
LANE_LINE_SMOOTHING_PASSES = 2
LANE_ORIENTATION_CLASSES = {1, 11, 13}
LANE_ORIENTATION_SNAP_DISTANCE_M = 4.5
LANE_ORIENTATION_MIN_SEGMENT_M = 0.35
LANE_ORIENTATION_CELL_SIZE_M = 6.0
ORIENTATION_UPDATE_INTERVAL_FRAMES = 3
GROUND_SPEED_FRESHNESS_SEC = 0.45


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

def wrap_angle(theta):
    return math.atan2(math.sin(theta), math.cos(theta))

def rotate_local_xz(x, z, yaw):
    c = math.cos(yaw)
    s = math.sin(yaw)
    return ((c * x) - (s * z), (s * x) + (c * z))

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


def smooth_polyline(points, passes=1):
    if len(points) < 3 or passes <= 0:
        return list(points)

    smoothed = [tuple(p) for p in points]
    for _ in range(passes):
        if len(smoothed) < 3:
            break
        refined = [smoothed[0]]
        for i in range(len(smoothed) - 1):
            p0 = smoothed[i]
            p1 = smoothed[i + 1]
            q = (
                0.75 * p0[0] + 0.25 * p1[0],
                0.75 * p0[1] + 0.25 * p1[1],
                0.75 * p0[2] + 0.25 * p1[2],
            )
            r = (
                0.25 * p0[0] + 0.75 * p1[0],
                0.25 * p0[1] + 0.75 * p1[1],
                0.25 * p0[2] + 0.75 * p1[2],
            )
            refined.extend((q, r))
        refined.append(smoothed[-1])
        smoothed = refined
    return smoothed


def resample_polyline_for_orientation(points, min_segment_length):
    if len(points) < 2:
        return list(points)

    sampled = [tuple(points[0])]
    anchor = points[0]
    for point in points[1:-1]:
        if math.hypot(point[0] - anchor[0], point[2] - anchor[2]) >= min_segment_length:
            sampled.append(tuple(point))
            anchor = point

    tail = tuple(points[-1])
    if sampled[-1] != tail:
        sampled.append(tail)
    return sampled


def make_polyline_ribbon(points, width=0.15, color=(1.0, 1.0, 0.2)):
    if len(points) < 2:
        return [], []

    half_width = width * 0.5
    left_pts = []
    right_pts = []

    for i in range(len(points)):
        if i == 0:
            tx = points[1][0] - points[0][0]
            tz = points[1][2] - points[0][2]
        elif i == len(points) - 1:
            tx = points[i][0] - points[i - 1][0]
            tz = points[i][2] - points[i - 1][2]
        else:
            tx = points[i + 1][0] - points[i - 1][0]
            tz = points[i + 1][2] - points[i - 1][2]

        tangent_x, tangent_z = normalize_vec2(tx, tz)
        if tangent_x == 0.0 and tangent_z == 0.0:
            tangent_x, tangent_z = 1.0, 0.0

        normal_x = -tangent_z
        normal_z = tangent_x
        px, py, pz = points[i]
        left_pts.append((px + normal_x * half_width, py, pz + normal_z * half_width))
        right_pts.append((px - normal_x * half_width, py, pz - normal_z * half_width))

    verts = []
    cols = []
    for i in range(len(points) - 1):
        a = left_pts[i]
        b = right_pts[i]
        c = left_pts[i + 1]
        d = right_pts[i + 1]
        verts.extend((a, b, d, a, d, c))
        cols.extend((color, color, color, color, color, color))
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


def make_rounded_rect_points(width: float, height: float, radius: float, segments: int = 6):
    half_w = width * 0.5
    half_h = height * 0.5
    radius = max(0.0, min(radius, half_w - 1e-4, half_h - 1e-4))

    centers = [
        (half_w - radius, half_h - radius, 0.0, math.pi * 0.5),
        (-half_w + radius, half_h - radius, math.pi * 0.5, math.pi),
        (-half_w + radius, -half_h + radius, math.pi, math.pi * 1.5),
        (half_w - radius, -half_h + radius, math.pi * 1.5, math.pi * 2.0),
    ]

    points = []
    for cx, cz, a0, a1 in centers:
        for i in range(segments + 1):
            t = i / float(segments)
            ang = a0 + (a1 - a0) * t
            x = cx + radius * math.cos(ang)
            z = cz + radius * math.sin(ang)
            if not points or abs(points[-1][0] - x) > 1e-6 or abs(points[-1][1] - z) > 1e-6:
                points.append((x, z))
    return points


def make_rounded_rect_sign(width: float, height: float, thickness: float, color=(1.0, 1.0, 1.0), radius: float = 0.12, segments: int = 6):
    ring = make_rounded_rect_points(width, height, radius, segments)
    front_y = 0.0
    back_y = -thickness

    verts = []
    cols = []

    for i in range(1, len(ring) - 1):
        a = (ring[0][0], front_y, ring[0][1])
        b = (ring[i][0], front_y, ring[i][1])
        c = (ring[i + 1][0], front_y, ring[i + 1][1])
        verts.extend([a, b, c])
        cols.extend([color, color, color])

    for i in range(1, len(ring) - 1):
        a = (ring[0][0], back_y, ring[0][1])
        b = (ring[i + 1][0], back_y, ring[i + 1][1])
        c = (ring[i][0], back_y, ring[i][1])
        verts.extend([a, b, c])
        cols.extend([color, color, color])

    for i in range(len(ring)):
        j = (i + 1) % len(ring)
        a = (ring[i][0], front_y, ring[i][1])
        b = (ring[j][0], front_y, ring[j][1])
        c = (ring[i][0], back_y, ring[i][1])
        d = (ring[j][0], back_y, ring[j][1])
        verts.extend([a, b, d, a, d, c])
        cols.extend([color] * 6)

    return verts, cols

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


def recolor_human_mesh_colors(verts):
    if not verts:
        return []

    ys = [v[1] for v in verts]
    ymin = min(ys)
    ymax = max(ys)
    yrange = max(1e-6, ymax - ymin)

    skin = (0.78, 0.63, 0.53)
    shirt = (0.14, 0.30, 0.62)
    pants = (0.10, 0.12, 0.18)
    shoes = (0.08, 0.08, 0.08)

    colors = []
    for _x, y, _z in verts:
        t = (y - ymin) / yrange
        if t > 0.84:
            colors.append(skin)
        elif t > 0.48:
            colors.append(shirt)
        elif t > 0.08:
            colors.append(pants)
        else:
            colors.append(shoes)
    return colors


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


def center_ground_mesh_vertices(verts: List[Tuple[float, float, float]]):
    if not verts:
        return []
    xs = [v[0] for v in verts]
    ys = [v[1] for v in verts]
    zs = [v[2] for v in verts]
    cx = (min(xs) + max(xs)) * 0.5
    cz = (min(zs) + max(zs)) * 0.5
    ymin = min(ys)
    return [(x - cx, y - ymin, z - cz) for (x, y, z) in verts]


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


class LaneLineSubscriber(Node):
    def __init__(self, on_message, topic="/perception/lane_lines"):
        super().__init__("wautovantage_lane_testbed_subscriber")
        self._on_message = on_message
        self.subscription = self.create_subscription(
            LaneArray,
            topic,
            self._listener_callback,
            10,
        )

    def _listener_callback(self, msg):
        lanes = []
        for lane in msg.lanes:
            lane_points = []
            for point in lane.points:
                xyzc = [float(v) for v in getattr(point, "xyzc", [])]
                lane_points.append(xyzc)
            lanes.append({
                "lane_id": int(lane.lane_id),
                "line_type": int(lane.line_type),
                "points": lane_points,
            })
        self._on_message(lanes)


class PerceptionSubscriber(Node):
    def __init__(self, on_objects, on_lanes, object_topic="/perception/objectdetection", lane_topic="/perception/lane_lines"):
        super().__init__("wautovantage_perception_testbed_subscriber")
        self._on_objects = on_objects
        self._on_lanes = on_lanes
        self.object_subscription = self.create_subscription(
            ObjectArray,
            object_topic,
            self._object_callback,
            10,
        )
        self.lane_subscription = self.create_subscription(
            LaneArray,
            lane_topic,
            self._lane_callback,
            10,
        )

    def _object_callback(self, msg):
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
        self._on_objects(objects)

    def _lane_callback(self, msg):
        lanes = []
        for lane in msg.lanes:
            lane_points = []
            for point in lane.points:
                lane_points.append([float(v) for v in getattr(point, "xyzc", [])])
            lanes.append({
                "lane_id": int(lane.lane_id),
                "line_type": int(lane.line_type),
                "points": lane_points,
            })
        self._on_lanes(lanes)


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


class ROSLaneLineBridge:
    def __init__(self, topic="/perception/lane_lines"):
        self.topic = topic
        self.enabled = False
        self.node = None
        self._latest_stamp = 0.0
        self._latest_lanes: List[Dict[str, object]] = []

        if not ROS2_AVAILABLE:
            print("ROS2 Python packages not available; lane line bridge disabled")
            return

        try:
            if not rclpy.ok():
                rclpy.init(args=None)
            self.node = LaneLineSubscriber(self._store_message, topic=topic)
            self.enabled = True
            print(f"Subscribed to ROS2 topic '{topic}'")
        except Exception as exc:
            print(f"Failed to initialize ROS2 lane bridge for '{topic}': {exc}")
            self.enabled = False

    def _store_message(self, lanes):
        self._latest_lanes = lanes
        self._latest_stamp = time.perf_counter()

    def spin_once(self):
        if not self.enabled or self.node is None:
            return
        try:
            rclpy.spin_once(self.node, timeout_sec=0.0)
        except Exception as exc:
            print(f"ROS2 lane spin_once failed: {exc}")
            self.enabled = False

    def latest_snapshot(self):
        return self._latest_stamp, list(self._latest_lanes)

    def close(self):
        if self.node is not None:
            try:
                self.node.destroy_node()
            except Exception:
                pass
            self.node = None


class ROSPerceptionBridge:
    def __init__(self, object_topic="/perception/objectdetection", lane_topic="/perception/lane_lines"):
        self.enabled = False
        self.node = None
        self._latest_object_stamp = 0.0
        self._latest_lane_stamp = 0.0
        self._latest_objects: List[Dict[str, float]] = []
        self._latest_lanes: List[Dict[str, object]] = []

        if not ROS2_AVAILABLE:
            print("ROS2 Python packages not available; perception bridge disabled")
            return

        try:
            if not rclpy.ok():
                rclpy.init(args=None)
            self.node = PerceptionSubscriber(
                self._store_objects,
                self._store_lanes,
                object_topic=object_topic,
                lane_topic=lane_topic,
            )
            self.enabled = True
            print(f"Subscribed to ROS2 topics '{object_topic}' and '{lane_topic}'")
        except Exception as exc:
            print(f"Failed to initialize combined perception bridge: {exc}")
            self.enabled = False

    def _store_objects(self, objects):
        self._latest_objects = objects
        self._latest_object_stamp = time.perf_counter()

    def _store_lanes(self, lanes):
        self._latest_lanes = lanes
        self._latest_lane_stamp = time.perf_counter()

    def spin_once(self):
        if not self.enabled or self.node is None:
            return
        try:
            rclpy.spin_once(self.node, timeout_sec=0.0)
        except Exception as exc:
            print(f"ROS2 perception spin_once failed: {exc}")
            self.enabled = False

    def latest_object_snapshot(self):
        return self._latest_object_stamp, list(self._latest_objects)

    def latest_lane_snapshot(self):
        return self._latest_lane_stamp, list(self._latest_lanes)

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
        thickness = 0.02
        width, height = 0.78, 0.92
        border_inset = 0.06
        corner_radius = 0.12

        border_verts, border_cols = make_rounded_rect_sign(
            width,
            height,
            thickness,
            color=(0.05, 0.05, 0.05),
            radius=corner_radius,
        )
        face_verts, face_cols = make_rounded_rect_sign(
            width - (border_inset * 2.0),
            height - (border_inset * 2.0),
            thickness * 0.7,
            color=(1.0, 1.0, 1.0),
            radius=max(0.04, corner_radius - border_inset * 0.65),
        )
        self.pole_mesh = Mesh(border_verts, border_cols, None, None, gl.GL_TRIANGLES, mat4_identity())
        self.panel_offset = (0.0, 1.55 + self.y_adjust, 0.0)
        self.panel_mesh = Mesh(face_verts, face_cols, None, None, gl.GL_TRIANGLES, mat4_identity())


    def update(self, dt: float):
        ox, oy, oz = self.panel_offset
        M = mat4_mul(mat4_translate(*self.pos), mat4_translate(ox, oy, oz))
        M = mat4_mul(M, mat4_rotate_x(math.radians(90.0)))
        if getattr(self, 'yaw', 0.0) != 0.0:
            M = mat4_mul(M, mat4_rotate_y(self.yaw))
        self.pole_mesh.model = M
        inner = mat4_mul(M, mat4_translate(0.0, 0.011, 0.0))
        self.panel_mesh.model = inner


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
        self._tracked_local_pose = None
        self._tracked_local_velocity = [0.0, 0.0, 0.0]
        self._motion_yaw = None
        self._render_yaw = None
        self._ground_velocity_mps = [0.0, 0.0]
        self._ground_motion_history = []
        self._last_motion_pose = None
        self._lane_snap_yaw = None
        self.last_seen = time.perf_counter()

    def update(self, detection: Dict[str, float], actor=None, actor_key=None):
        if actor is not None:
            self.actor = actor
        if actor_key is not None:
            self.actor_key = actor_key
        self.obj_class = int(detection["obj_class"])
        self.custom_classification = int(detection.get("custom_classification", 0))
        self.last_seen = time.perf_counter()

    def set_local_pose(self, x: float, y: float, z: float, camera_origin_local, ego_motion_local, ego_delta_yaw: float, detection_dt: Optional[float]):
        raw_pose = [x, y, z]
        filtered_pose = self._filter_local_pose(raw_pose, detection_dt)
        motion_pose = self._tracked_local_pose if self._tracked_local_pose is not None else filtered_pose
        self._update_motion_heading(motion_pose, ego_motion_local, ego_delta_yaw, detection_dt)
        resolved_yaw = self._refresh_render_yaw()

        if self.obj_class in VEHICLE_CENTER_CORRECTION_CLASSES:
            new_pose = self._vehicle_center_from_nearest_point(filtered_pose, camera_origin_local, motion_yaw=resolved_yaw)
        else:
            new_pose = filtered_pose

        self.target_local_pose = new_pose
        if not self._pose_initialized:
            self.local_pose = list(new_pose)
            self._pose_initialized = True
        self._raw_local_pose = list(raw_pose)

    def _normalized_detection_dt(self, detection_dt: Optional[float]) -> float:
        if detection_dt is None or not math.isfinite(detection_dt):
            return TRACKING_DT_DEFAULT_SEC
        return clamp(detection_dt, TRACKING_DT_MIN_SEC, TRACKING_DT_MAX_SEC)

    def _filter_local_pose(self, raw_pose, detection_dt: Optional[float]):
        if self._tracked_local_pose is None:
            self._tracked_local_pose = list(raw_pose)
            return list(raw_pose)

        dt = self._normalized_detection_dt(detection_dt)
        predicted_x = self._tracked_local_pose[0] + (self._tracked_local_velocity[0] * dt)
        predicted_z = self._tracked_local_pose[2] + (self._tracked_local_velocity[2] * dt)
        residual_x = raw_pose[0] - predicted_x
        residual_z = raw_pose[2] - predicted_z

        tracked_x = predicted_x + (residual_x * POSE_TRACK_ALPHA)
        tracked_y = self._tracked_local_pose[1] + ((raw_pose[1] - self._tracked_local_pose[1]) * POSE_TRACK_Y_ALPHA)
        tracked_z = predicted_z + (residual_z * POSE_TRACK_ALPHA)

        self._tracked_local_velocity[0] += (POSE_TRACK_BETA * residual_x) / dt
        self._tracked_local_velocity[2] += (POSE_TRACK_BETA * residual_z) / dt
        self._tracked_local_velocity[0] *= 0.98
        self._tracked_local_velocity[2] *= 0.98
        self._tracked_local_pose = [tracked_x, tracked_y, tracked_z]

        return [
            tracked_x + (self._tracked_local_velocity[0] * POSE_PREDICTION_LOOKAHEAD_SEC),
            tracked_y,
            tracked_z + (self._tracked_local_velocity[2] * POSE_PREDICTION_LOOKAHEAD_SEC),
        ]

    def _update_motion_heading(self, pose, ego_motion_local, ego_delta_yaw: float, detection_dt: Optional[float]):
        if self._last_motion_pose is not None:
            prev_x, prev_z = rotate_local_xz(self._last_motion_pose[0], self._last_motion_pose[2], -ego_delta_yaw)
            dx = pose[0] - prev_x
            dz = pose[2] - prev_z
            ground_dx = dx + ego_motion_local[0]
            ground_dz = dz + ego_motion_local[2]
            if self.obj_class in MOTION_HEADING_CLASSES and math.hypot(ground_dx, ground_dz) >= HEADING_MOTION_MIN_STEP_M:
                self._ground_motion_history.append((ground_dx, ground_dz))
                if len(self._ground_motion_history) > HEADING_HISTORY_SIZE:
                    self._ground_motion_history.pop(0)

                dt = self._normalized_detection_dt(detection_dt)
                inst_vx = ground_dx / dt
                inst_vz = ground_dz / dt
                self._ground_velocity_mps[0] += (inst_vx - self._ground_velocity_mps[0]) * GROUND_VELOCITY_SMOOTH_ALPHA
                self._ground_velocity_mps[1] += (inst_vz - self._ground_velocity_mps[1]) * GROUND_VELOCITY_SMOOTH_ALPHA

                if len(self._ground_motion_history) >= HEADING_CONFIRM_FRAMES and math.hypot(*self._ground_velocity_mps) >= HEADING_MIN_SPEED_MPS:
                    target_yaw = math.atan2(self._ground_velocity_mps[0], -self._ground_velocity_mps[1])
                    target_yaw += MOTION_HEADING_MESH_OFFSET_RAD
                    if self._motion_yaw is None:
                        self._motion_yaw = target_yaw
                    else:
                        delta_current = abs(math.degrees(math.atan2(
                            math.sin(target_yaw - self._motion_yaw),
                            math.cos(target_yaw - self._motion_yaw),
                        )))
                        blend_alpha = HEADING_SMOOTH_ALPHA if delta_current >= HEADING_UPDATE_DEG else (HEADING_SMOOTH_ALPHA * 0.45)
                        self._motion_yaw = lerp_angle(self._motion_yaw, target_yaw, blend_alpha)
            else:
                if self._ground_motion_history:
                    self._ground_motion_history.pop(0)
                self._ground_velocity_mps[0] *= 0.82
                self._ground_velocity_mps[1] *= 0.82
        self._last_motion_pose = list(pose)

    def _target_render_yaw(self):
        if self._lane_snap_yaw is not None:
            return self._lane_snap_yaw
        return self._motion_yaw

    def _refresh_render_yaw(self):
        target_yaw = self._target_render_yaw()
        if target_yaw is None:
            return self._render_yaw

        if self._render_yaw is None:
            self._render_yaw = target_yaw
            return self._render_yaw

        blend_alpha = LANE_SNAP_RENDER_BLEND_ALPHA if self._lane_snap_yaw is not None else ORIENTATION_RENDER_BLEND_ALPHA
        self._render_yaw = lerp_angle(self._render_yaw, target_yaw, blend_alpha)
        return self._render_yaw

    def _vehicle_center_from_nearest_point(self, raw_pose, camera_origin_local, motion_yaw: Optional[float] = None):
        motion_yaw = self._motion_yaw if motion_yaw is None else motion_yaw
        if motion_yaw is None:
            return raw_pose

        cam_x, _cam_y, cam_z = camera_origin_local
        los_x, los_z = normalize_vec2(raw_pose[0] - cam_x, raw_pose[2] - cam_z)
        if los_x == 0.0 and los_z == 0.0:
            return raw_pose

        fwd_x = math.sin(motion_yaw)
        fwd_z = -math.cos(motion_yaw)
        right_x = math.cos(motion_yaw)
        right_z = math.sin(motion_yaw)

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

        if isinstance(actor, StreetSign):
            world_model = mat4_mul(anchor_model, local_model)
            actor.pole_mesh.model = world_model
            ox, oy, oz = actor.panel_offset
            panel_model = mat4_mul(world_model, mat4_translate(ox, oy, oz))
            panel_model = mat4_mul(panel_model, mat4_rotate_x(math.radians(90.0)))
            if getattr(actor, "yaw", 0.0) != 0.0:
                panel_model = mat4_mul(panel_model, mat4_rotate_y(actor.yaw))
            actor.panel_mesh.model = panel_model
            return

        if isinstance(actor, TrafficLight):
            world_model = mat4_mul(anchor_model, local_model)
            actor.mesh.model = world_model
            return

        if hasattr(actor, "mesh"):
            resolved_yaw = self._refresh_render_yaw()
            if self.obj_class in MOTION_HEADING_CLASSES and resolved_yaw is not None:
                local_model = mat4_mul(local_model, mat4_rotate_y(resolved_yaw))
            world_model = mat4_mul(anchor_model, local_model)
            actor.mesh.model = world_model
        elif isinstance(actor, Mesh):
            world_model = mat4_mul(anchor_model, local_model)
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
        consensus_yaw += MOTION_HEADING_MESH_OFFSET_RAD
        if self._motion_yaw is None:
            self._motion_yaw = consensus_yaw
            return
        self._motion_yaw = lerp_angle(self._motion_yaw, consensus_yaw, LANE_CONSENSUS_BLEND_ALPHA)

    def preferred_lane_reference_yaw(self):
        speed = math.hypot(self._ground_velocity_mps[0], self._ground_velocity_mps[1])
        if speed >= HEADING_MIN_SPEED_MPS:
            return math.atan2(self._ground_velocity_mps[0], -self._ground_velocity_mps[1]) + MOTION_HEADING_MESH_OFFSET_RAD
        if self._render_yaw is not None:
            return self._render_yaw
        if self._lane_snap_yaw is not None:
            return self._lane_snap_yaw
        return self._motion_yaw

    def snap_to_lane_yaw(self, snapped_yaw: Optional[float]):
        self._lane_snap_yaw = snapped_yaw

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
        self._bound_kind = None
        self._bound_texture_id = None
        self._bound_vao = None

    def begin_frame(self):
        self._bound_kind = None
        self._bound_texture_id = None
        self._bound_vao = None

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
            if self._bound_kind != 'tex':
                self.prog_tex.use()
                self._bound_kind = 'tex'
            self.prog_tex['u_mvp'] = mvp
            if self._bound_texture_id != mesh.texture_id:
                gl.glActiveTexture(gl.GL_TEXTURE0)
                gl.glBindTexture(gl.GL_TEXTURE_2D, mesh.texture_id)
                self._bound_texture_id = mesh.texture_id
            self.prog_tex['u_tex'] = 0
        else:
            if self._bound_kind != 'color':
                self.prog_color.use()
                self._bound_kind = 'color'
                self._bound_texture_id = None
            self.prog_color['u_mvp'] = mvp

        if self._bound_vao != vao.value:
            gl.glBindVertexArray(vao)
            self._bound_vao = vao.value
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
        self.camera_origin_local = (0.0, 1.35, 0.0)
        self.detected_entities: Dict[int, DetectedEntity] = {}
        self.detected_mesh_templates: Dict[str, Mesh] = {}
        self._last_detection_stamp = 0.0
        self._last_detection_ego_yaw = 0.0
        self._ego_motion_local_estimate = [0.0, 0.0, 0.0]
        self._ego_speed_mps_estimate = 0.0
        self._ego_forward_speed_mps_estimate = 0.0
        self._ego_speed_mph_history = []
        self.perception_bridge = ROSPerceptionBridge()
        self._last_lane_stamp = 0.0
        self.lane_meshes: List[Mesh] = []
        self._lane_orientation_segments: List[Dict[str, object]] = []
        self._lane_orientation_cells: Dict[Tuple[int, int], List[Dict[str, object]]] = {}
        self._latest_lane_count = 0
        self._update_frame_index = 0
        self._projection_matrix = perspective(60.0, max(1e-6, width / float(height)), 0.1, 500.0)
        self._projection_dirty = False
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

        for wheel in self.wheels:
            wheel['offset_matrix'] = mat4_translate(*wheel['offset'])

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

            cols = recolor_human_mesh_colors(hpos_scaled)
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
                cpos_scaled = center_ground_mesh_vertices([(x*s, (y - cymin)*s, z*s) for (x, y, z) in cpos])
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
        self.lane_meshes = [Mesh(*make_polyline(pts), None, None, gl.GL_LINES, mat4_identity()) for pts in lane_pts]

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

    def _sensor_depth_to_local_pose(self, forward: float, left: float, up: float, lateral_offset: float = DETECTION_LATERAL_OFFSET_M) -> List[float]:
        cam_x, cam_y, cam_z = self.camera_origin_local

        right = -left
        c = math.cos(DETECTION_YAW_CORRECTION_RAD)
        s = math.sin(DETECTION_YAW_CORRECTION_RAD)
        corrected_right = (right * c) - (forward * s)
        corrected_forward = (right * s) + (forward * c)

        world_x = cam_x + corrected_right + lateral_offset
        world_y = cam_y + up
        world_z = cam_z - corrected_forward
        return [world_x, world_y, world_z]

    def _ros_detection_to_local_pose(self, detection: Dict[str, float]) -> List[float]:
        # ZED is configured as RIGHT_HANDED_Z_UP_X_FWD:
        # x = forward, y = left, z = up
        forward = float(detection["x"])
        left = float(detection["y"])
        up = float(detection["z"])
        local_pose = self._sensor_depth_to_local_pose(forward, left, up)

        obj_height = float(detection.get("height", 0.0))
        if not self._valid_dimension(obj_height):
            obj_height = self._default_height_for_class(int(detection["obj_class"]))

        local_pose[1] = max(0.0, local_pose[1] - (obj_height * 0.5))
        return local_pose

    def _lane_color_for_type(self, line_type: int):
        if line_type in (2, 3):
            return (0.95, 0.84, 0.18)
        if line_type == 9:
            return (0.45, 0.45, 0.45)
        return (0.92, 0.92, 0.92)

    def _lane_point_to_local_pose(self, xyzc: List[float]):
        if len(xyzc) < 3:
            return None
        if not all(math.isfinite(float(v)) for v in xyzc[:3]):
            return None
        if len(xyzc) >= 4 and math.isfinite(float(xyzc[3])) and float(xyzc[3]) < LANE_LINE_CONFIDENCE_MIN:
            return None

        local_pose = self._sensor_depth_to_local_pose(float(xyzc[0]), float(xyzc[1]), float(xyzc[2]), lateral_offset=0.0)
        local_pose[1] = max(LANE_LINE_RENDER_LIFT_M, local_pose[1] + LANE_LINE_RENDER_LIFT_M)
        return tuple(local_pose)

    def _closest_lane_segment_distance(self, pose, segment):
        px, pz = pose[0], pose[2]
        ax, az = segment["start"][0], segment["start"][2]
        bx, bz = segment["end"][0], segment["end"][2]
        abx = bx - ax
        abz = bz - az
        denom = (abx * abx) + (abz * abz)
        if denom < 1e-9:
            return math.hypot(px - ax, pz - az)
        t = clamp(((px - ax) * abx + (pz - az) * abz) / denom, 0.0, 1.0)
        closest_x = ax + (abx * t)
        closest_z = az + (abz * t)
        return math.hypot(px - closest_x, pz - closest_z)

    def _lane_orientation_cell_key(self, x: float, z: float):
        cell = LANE_ORIENTATION_CELL_SIZE_M
        return (
            int(math.floor(x / cell)),
            int(math.floor(z / cell)),
        )

    def _candidate_lane_segments(self, pose):
        base_x, base_z = self._lane_orientation_cell_key(pose[0], pose[2])
        candidates = []
        for dx in (-1, 0, 1):
            for dz in (-1, 0, 1):
                candidates.extend(self._lane_orientation_cells.get((base_x + dx, base_z + dz), []))
        return candidates if candidates else self._lane_orientation_segments

    def _resolve_snapped_lane_yaw(self, reference_yaw, lane_yaw):
        parallel_candidates = [lane_yaw, lane_yaw + math.pi]
        if reference_yaw is None:
            return parallel_candidates[0]

        return min(
            parallel_candidates,
            key=lambda yaw: abs(math.atan2(math.sin(yaw - reference_yaw), math.cos(yaw - reference_yaw))),
        )

    def _nearest_lane_snap_yaw(self, pose, entity: DetectedEntity):
        if not self._lane_orientation_segments:
            return None

        best_segment = None
        best_distance = None
        for segment in self._candidate_lane_segments(pose):
            distance = self._closest_lane_segment_distance(pose, segment)
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_segment = segment

        if best_segment is None or best_distance is None or best_distance > LANE_ORIENTATION_SNAP_DISTANCE_M:
            return None

        reference_yaw = entity.preferred_lane_reference_yaw()
        return self._resolve_snapped_lane_yaw(reference_yaw, best_segment["yaw"])

    def _snap_vehicle_orientations_to_lanes(self, active_ids=None):
        entity_ids = active_ids if active_ids is not None else list(self.detected_entities.keys())
        for oid in entity_ids:
            entity = self.detected_entities.get(oid)
            if entity is None or entity.obj_class not in LANE_ORIENTATION_CLASSES or not entity._pose_initialized:
                continue

            if not self._lane_orientation_segments:
                entity.snap_to_lane_yaw(None)
                continue

            pose = entity.target_local_pose if entity.target_local_pose else entity.local_pose
            entity.snap_to_lane_yaw(self._nearest_lane_snap_yaw(pose, entity))

    def _sync_lane_lines(self, lanes: List[Dict[str, object]]):
        meshes = []
        segments = []
        rendered_lane_count = 0

        for lane in sorted(lanes, key=lambda item: int(item.get("lane_id", 255))):
            line_type = int(lane.get("line_type", 9))
            lane_points = []
            for xyzc in lane.get("points", []):
                local_pose = self._lane_point_to_local_pose(xyzc)
                if local_pose is not None:
                    lane_points.append(local_pose)

            if len(lane_points) < 2:
                continue

            render_points = smooth_polyline(lane_points, passes=LANE_LINE_SMOOTHING_PASSES)
            verts, cols = make_polyline_ribbon(
                render_points,
                width=LANE_LINE_WIDTH_M,
                color=self._lane_color_for_type(line_type),
            )
            meshes.append(Mesh(verts, cols, None, None, gl.GL_TRIANGLES, mat4_identity()))
            orientation_points = resample_polyline_for_orientation(render_points, LANE_ORIENTATION_MIN_SEGMENT_M)
            for idx in range(len(orientation_points) - 1):
                start = orientation_points[idx]
                end = orientation_points[idx + 1]
                dx = end[0] - start[0]
                dz = end[2] - start[2]
                if math.hypot(dx, dz) < LANE_ORIENTATION_MIN_SEGMENT_M:
                    continue
                segments.append({
                    "lane_id": int(lane.get("lane_id", 255)),
                    "line_type": line_type,
                    "start": start,
                    "end": end,
                    "yaw": math.atan2(dx, -dz) + MOTION_HEADING_MESH_OFFSET_RAD,
                })
            rendered_lane_count += 1

        self.lane_meshes = meshes
        for mesh in self.lane_meshes:
            self.renderer.warm_mesh(mesh)
        self._lane_orientation_segments = segments
        cells: Dict[Tuple[int, int], List[Dict[str, object]]] = {}
        for segment in segments:
            mid_x = (segment["start"][0] + segment["end"][0]) * 0.5
            mid_z = (segment["start"][2] + segment["end"][2]) * 0.5
            key = self._lane_orientation_cell_key(mid_x, mid_z)
            cells.setdefault(key, []).append(segment)
        self._lane_orientation_cells = cells
        self._latest_lane_count = rendered_lane_count
        if self._should_update_orientation_this_frame():
            self._snap_vehicle_orientations_to_lanes()

    def _make_mesh_actor(self, mesh: Mesh):
        return MovingCharacter(clone_mesh(mesh), 0.0, 0.0, 0.0)

    def _warm_actor_gpu(self, actor):
        if actor is None:
            return
        if isinstance(actor, StreetSign):
            self.renderer.warm_mesh(actor.pole_mesh)
            self.renderer.warm_mesh(actor.panel_mesh)
            return
        if isinstance(actor, TrafficLight):
            self.renderer.warm_mesh(actor.mesh)
            return
        if isinstance(actor, Mesh):
            self.renderer.warm_mesh(actor)
            return
        if hasattr(actor, "mesh"):
            self.renderer.warm_mesh(actor.mesh)

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

        ego_delta_yaw = wrap_angle(self.ego.yaw - self._last_detection_ego_yaw)
        ego_motion_local = self._estimate_ego_motion_local(detections, detection_dt, ego_delta_yaw)
        active_ids = set()
        for detection in detections:
            if not self._valid_detection(detection):
                continue

            object_id = int(detection["object_id"])
            active_ids.add(object_id)
            local_pose = self._ros_detection_to_local_pose(detection)
            actor_key = self._actor_key_for_detection(detection)

            entity = self.detected_entities.get(object_id)
            actor_changed = (
                entity is None
                or entity.actor_key != actor_key
            )
            actor = self._build_detected_actor(detection) if actor_changed else None
            if actor is not None:
                self._warm_actor_gpu(actor)

            if entity is None:
                entity = DetectedEntity(detection, actor, actor_key)
                self.detected_entities[object_id] = entity
            else:
                entity.update(detection, actor=actor, actor_key=actor_key)

            entity.set_local_pose(*local_pose, self.camera_origin_local, ego_motion_local, ego_delta_yaw, detection_dt)

        if self._should_update_orientation_this_frame():
            self._apply_lane_consensus(active_ids)
            self._snap_vehicle_orientations_to_lanes(active_ids)

        stale_ids = [oid for oid in self.detected_entities.keys() if oid not in active_ids]
        for oid in stale_ids:
            del self.detected_entities[oid]

        self._last_detection_ego_yaw = self.ego.yaw

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

    def _estimate_ego_motion_local(self, detections: List[Dict[str, float]], detection_dt: Optional[float], ego_delta_yaw: float):
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
            prev_x, prev_z = rotate_local_xz(entity._raw_local_pose[0], entity._raw_local_pose[2], -ego_delta_yaw)
            dx = local_pose[0] - prev_x
            dz = local_pose[2] - prev_z
            motion_samples_x.append(-dx)
            motion_samples_z.append(-dz)

        if motion_samples_x and motion_samples_z:
            measured = [median(motion_samples_x), 0.0, median(motion_samples_z)]
            alpha = 0.35
            self._ego_motion_local_estimate[0] += (measured[0] - self._ego_motion_local_estimate[0]) * alpha
            self._ego_motion_local_estimate[2] += (measured[2] - self._ego_motion_local_estimate[2]) * alpha
            if detection_dt is not None:
                signed_forward_speed = (-self._ego_motion_local_estimate[2]) / detection_dt
                forward_speed = abs(self._ego_motion_local_estimate[2]) / detection_dt
                self._ego_forward_speed_mps_estimate += (signed_forward_speed - self._ego_forward_speed_mps_estimate) * 0.25
                self._ego_speed_mps_estimate += (forward_speed - self._ego_speed_mps_estimate) * 0.25
                self._ego_speed_mph_history.append(self._ego_speed_mps_estimate * MPS_TO_MPH)
                if len(self._ego_speed_mph_history) > SPEED_ROLLING_WINDOW:
                    self._ego_speed_mph_history.pop(0)
        else:
            self._ego_motion_local_estimate[0] *= 0.85
            self._ego_motion_local_estimate[2] *= 0.85
            # Keep the last known speed when static references drop out briefly.
            self._ego_forward_speed_mps_estimate *= 0.995
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

    def on_resize(self, width, height):
        self._projection_dirty = True
        return super().on_resize(width, height)

    def _projection(self):
        if self._projection_dirty:
            self._projection_matrix = perspective(60.0, max(1e-6, self.width / float(self.height)), 0.1, 500.0)
            self._projection_dirty = False
        return self._projection_matrix

    def _current_ground_speed_mps(self):
        if self._last_detection_stamp > 0.0:
            age = time.perf_counter() - self._last_detection_stamp
            if age <= GROUND_SPEED_FRESHNESS_SEC:
                return self._ego_forward_speed_mps_estimate
        return self.ego.v

    def _should_update_orientation_this_frame(self):
        return (self._update_frame_index % ORIENTATION_UPDATE_INTERVAL_FRAMES) == 0

    # Update
    def update(self, _dt):
        now = time.perf_counter()
        dt = clamp(now - self.last_time, 0.0, 1.0/60.0)
        self.last_time = now
        self._update_frame_index += 1

        self.perception_bridge.spin_once()
        detection_stamp, detections = self.perception_bridge.latest_object_snapshot()
        if detection_stamp > self._last_detection_stamp:
            self._sync_detected_entities(detections)
            self._last_detection_stamp = detection_stamp

        lane_stamp, lanes = self.perception_bridge.latest_lane_snapshot()
        if lane_stamp > self._last_lane_stamp:
            self._sync_lane_lines(lanes)
            self._last_lane_stamp = lane_stamp

        throttle  = 1.0 if self.keys[key.W] else 0.0
        brake     = 1.0 if (self.keys[key.S] or self.keys[key.SPACE]) else 0.0
        steer_cmd = (1.0 if self.keys[key.D] else 0.0) - (1.0 if self.keys[key.A] else 0.0)
        self.brake_on = brake > 0.1

        self.ego.update(dt, throttle, steer_cmd, brake)
        ground_speed_mps = self._current_ground_speed_mps()
        for w in self.wheels:
            r = max(1e-6, w['radius'])
            self._wheel_roll += (ground_speed_mps / r) * dt
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
        self.renderer.begin_frame()

        proj = self._projection()

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

        # Car model matrix once
        car_T = mat4_translate(*self.ego.pos)
        car_R = mat4_rotate_y(self.ego.yaw)
        car_M = mat4_mul(car_T, car_R)

        for ln in self.lane_meshes:
            ln.model = car_M
            self.renderer.draw_mesh(ln, pv)

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
        steer_matrix = mat4_rotate_y(steer_angle) if steer_angle != 0.0 else None
        wheel_roll_matrix = mat4_rotate_x(self._wheel_roll)
        for w in self.wheels:
            M = mat4_mul(car_M, w['offset_matrix'])
            if w['steer'] and steer_matrix is not None:
                M = mat4_mul(M, steer_matrix)
            M = mat4_mul(M, wheel_roll_matrix)
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

        ros_status = "ROS2 OK" if self.perception_bridge.enabled else "ROS2 OFF"
        self.hud.text = (
            f"Sim Speed {self.ego.v:4.1f} m/s   "
            f"Yaw {math.degrees(self.ego.yaw):5.1f} deg   "
            f"{ros_status}   "
            f"Objects {len(self.detected_entities)}   "
            f"Lanes {self._latest_lane_count}"
        )
        self.hud.draw()
        
        if hasattr(self, 'streaming'):
            self.streaming.push_frame()

    def on_close(self):
        self.perception_bridge.close()
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
