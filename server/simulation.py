#!/usr/bin/env python3
"""
WA Vantage Simulation
Consolidated 3D simulation that displays video on the left and 3D environment on the right.
Shows the rendered vehicle with brake lights and wheels in the 3D scene.
Environment is a clean slate based on testbed-obj-det.py.
"""

import math, time, ctypes, os, sys, argparse, json
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from object_detection import ObjectDetectionSystem

try:
    import pyglet
    from pyglet.window import key, mouse
    from pyglet import gl
    from pyglet.graphics.shader import Shader, ShaderProgram
except ImportError:
    print("Error: Pyglet not installed. Please install with: pip install pyglet")
    sys.exit(1)

import cv2
import numpy as np

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--video_path", help="Path to WA vantage.gif video", default="../data/WA_Challenge.gif")
parser.add_argument("--width", help="Window width", type=int, default=1280)
parser.add_argument("--height", help="Window height", type=int, default=720)
parser.add_argument("--fps", help="Simulation FPS", type=int, default=60)
args = parser.parse_args()

# ------------------------------------------------------------
# Minimal MTL parser
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

def create_texture_2d(img_path):
    if not os.path.isfile(img_path): return None
    try:
        img = pyglet.image.load(img_path)
        tex = img.get_texture()
        gl.glBindTexture(gl.GL_TEXTURE_2D, tex.id)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        return tex
    except: return None

def make_box_triangles(w, h, d, color=(1,1,1)):
    x, y, z = w/2, h, d/2
    v = [
        (-x,0,z),(x,0,z),(x,y,z), (-x,0,z),(x,y,z),(-x,y,z), # front
        (-x,0,-z),(-x,y,-z),(x,y,-z), (-x,0,-z),(x,y,-z),(x,0,-z), # back
        (-x,y,z),(x,y,z),(x,y,-z), (-x,y,z),(x,y,-z),(-x,y,-z), # top
        (-x,0,z),(-x,0,-z),(x,0,-z), (-x,0,z),(x,0,-z),(x,0,z), # bottom
        (-x,0,-z),(-x,0,z),(-x,y,z), (-x,0,-z),(-x,y,z),(-x,y,-z), # left
        (x,0,z),(x,0,-z),(x,y,-z), (x,0,z),(x,y,-z),(x,y,z) # right
    ]
    return v, [color]*len(v)

def make_grid_tile(size=40.0, step=2.0, color=(0.25, 0.25, 0.25)):
    s = size * 0.5
    verts, cols = [], []
    v = -s
    while v <= s + 1e-6:
        verts += [(-s, 0.0, v), (s, 0.0, v)]
        cols  += [color,        color]
        verts += [(v, 0.0, -s), (v, 0.0,  s)]
        cols  += [color,        color]
        v += step
    return verts, cols

# ------------------------------------------------------------
# OBJ loader with UV + MTL
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
        # Fallback to absolute assets path if relative fails
        script_dir = os.path.dirname(os.path.abspath(__file__))
        obj_path = os.path.join(script_dir, obj_path)
        if not os.path.isfile(obj_path):
            print(f"ERROR: OBJ file not found: {obj_path}")
            return [], None, None, None

    with open(obj_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if not line or line.startswith('#'): continue
            parts = line.split()
            if not parts: continue
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
            if mtl_files: mtl_path = os.path.join(os.path.dirname(obj_path), mtl_files[0])
        tex_path, mtl_colors = parse_mtl(mtl_path)

    tri_pos, tri_uv, tri_col = [], [], []
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
# Detection Support
# ------------------------------------------------------------
class LogParser:
    def __init__(self, filename="log.txt"):
        self.filename = filename
        self.last_pos = 0
    def parse_latest(self) -> Dict:
        if not os.path.exists(self.filename): return {}
        data = {}
        with open(self.filename, 'r') as f:
            f.seek(self.last_pos)
            lines = f.readlines()
            self.last_pos = f.tell()
            for line in lines:
                if line.startswith("OBJ_DATA:"):
                    try: data.update(json.loads(line[9:].strip()))
                    except: pass
        return data

class DetectedObject:
    def __init__(self, oid, data, assets):
        self.oid = oid
        self.assets = assets
        self.obj_class = data.get("obj_class", 1)
        self.pos = [0,0,0]
        self.mesh = None
        self._init_mesh()
        self.update_from_data(data)

    def _init_mesh(self):
        # Pick asset based on class (must match keys in _load_assets)
        asset_map = {1: "car", 5: "traffic_light", 9: "cone", 11: "truck", 13: "bus"}
        key = asset_map.get(self.obj_class)
        if key and key in self.assets:
            tmpl = self.assets[key]
            self.mesh = Mesh(tmpl.verts, tmpl.cols, tmpl.uvs, tmpl.texture_id, tmpl.mode, mat4_identity(), _tex_obj=tmpl._tex_obj)
        else:
            # Fallback color box
            colors = {1: (0.2,0.5,0.8), 5: (0.3,0.3,0.3), 9: (1,0.5,0)}
            v, c = make_box_triangles(1, 1, 1, colors.get(self.obj_class, (0.7,0.7,0.7)))
            self.mesh = Mesh(v, c, None, None, gl.GL_TRIANGLES, mat4_identity())

    def update_from_data(self, data):
        coords = data.get("obj_cartesian_coords", [0,0,0])
        if coords[0] > 9000: return
        # Mapping from detector (relative to car) to world
        # Detector uses X=lateral, Y=depth
        # We'll treat these as relative to the ego car position/yaw for now
        # But wait, the user said "render around the vehicle"
        # Let's just place them in the world for now using simple mapping
        self.pos = [coords[0], 0.0, -coords[1]]
        if self.obj_class == 5: self.pos[1] = 3.0 # Traffic light height
        # Do NOT set self.mesh.model here; we'll do it in on_draw relative to the car

# ------------------------------------------------------------

# ------------------------------------------------------------
# Math Utils
# ------------------------------------------------------------
def clamp(x, lo, hi): return lo if x < lo else hi if x > hi else x
def mat4_identity(): return [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]
def mat4_mul(a, b):
    o = [0]*16
    for c in range(4):
        for r in range(4):
            o[c*4+r] = sum(a[i*4+r]*b[c*4+i] for i in range(4))
    return o
def mat4_translate(x, y, z):
    m = mat4_identity()
    m[12], m[13], m[14] = x, y, z
    return m
def mat4_rotate_y(theta):
    c, s = math.cos(theta), math.sin(theta)
    return [c,0,s,0, 0,1,0,0, -s,0,c,0, 0,0,0,1]
def mat4_rotate_x(theta):
    c, s = math.cos(theta), math.sin(theta)
    return [1,0,0,0, 0,c,-s,0, 0,s,c,0, 0,0,0,1]
def perspective(fovy, aspect, znear, zfar):
    f = 1.0 / math.tan(math.radians(fovy)/2.0)
    nf = 1.0 / (znear - zfar)
    return [f/aspect,0,0,0, 0,f,0,0, 0,0,(zfar+znear)*nf,-1, 0,0,(2*znear*zfar)*nf,0]

def lerp_vec3(a, b, t):
    return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]

def look_at(eye, center, up):
    ex,ey,ez = eye; cx,cy,cz = center; ux,uy,uz = up
    fx,fy,fz = cx-ex, cy-ey, cz-ez
    fl = max(1e-9, math.sqrt(fx*fx+fy*fy+fz*fz))
    fx,fy,fz = fx/fl, fy/fl, fz/fl
    sx,sy,sz = fy*uz-fz*uy, fz*ux-fx*uz, fx*uy-fy*ux
    sl = max(1e-9, math.sqrt(sx*sx+sy*sy+sz*sz))
    sx,sy,sz = sx/sl, sy/sl, sz/sl
    ux,uy,uz = sy*fz-sz*fy, sz*fx-sx*fz, sx*fy-sy*fx
    m = [sx,ux,-fx,0, sy,uy,-fy,0, sz,uz,-fz,0, 0,0,0,1]
    return mat4_mul(m, mat4_translate(-ex,-ey,-ez))

# ------------------------------------------------------------
# Shaders
# ------------------------------------------------------------
VERT_COLOR = """#version 330
layout(location=0) in vec3 p; layout(location=1) in vec3 c; uniform mat4 u_mvp; out vec3 v_c;
void main(){ v_c=c; gl_Position=u_mvp*vec4(p,1.0); }"""
FRAG_COLOR = """#version 330
in vec3 v_c; out vec4 o; void main(){ o=vec4(v_c,1.0); }"""
VERT_TEX = """#version 330
layout(location=0) in vec3 p; layout(location=1) in vec2 uv; uniform mat4 u_mvp; out vec2 v_uv;
void main(){ v_uv=uv; gl_Position=u_mvp*vec4(p,1.0); }"""
FRAG_TEX = """#version 330
in vec2 v_uv; uniform sampler2D t; out vec4 o; void main(){ o=texture(t,v_uv); }"""

# ------------------------------------------------------------
# Scene Elements
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

class Renderer:
    def __init__(self):
        self.prog_color = ShaderProgram(Shader(VERT_COLOR, 'vertex'), Shader(FRAG_COLOR, 'fragment'))
        self.prog_tex   = ShaderProgram(Shader(VERT_TEX, 'vertex'),   Shader(FRAG_TEX,   'fragment'))
    def _build_gpu(self, mesh: Mesh):
        n = len(mesh.verts); inter = []
        kind = 'tex' if (mesh.uvs is not None and mesh.texture_id is not None) else 'color'
        for i in range(n):
            inter.extend(mesh.verts[i])
            if kind == 'tex': inter.extend(mesh.uvs[i])
            else: inter.extend(mesh.cols[i] if mesh.cols else (1,1,1))
        arr = (gl.GLfloat * (len(inter)))(*inter)
        vao, vbo = gl.GLuint(), gl.GLuint()
        gl.glGenVertexArrays(1, ctypes.byref(vao)); gl.glGenBuffers(1, ctypes.byref(vbo))
        gl.glBindVertexArray(vao); gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, ctypes.sizeof(arr), arr, gl.GL_STATIC_DRAW)
        stride = (5 if kind=='tex' else 6)*ctypes.sizeof(gl.GLfloat)
        gl.glEnableVertexAttribArray(0); gl.glVertexAttribPointer(0,3,gl.GL_FLOAT,gl.GL_FALSE,stride,0)
        gl.glEnableVertexAttribArray(1); gl.glVertexAttribPointer(1,(2 if kind=='tex' else 3),gl.GL_FLOAT,gl.GL_FALSE,stride,3*ctypes.sizeof(gl.GLfloat))
        mesh._gpu = (vao, n, mesh.mode, kind)
    def draw_mesh(self, mesh: Mesh, pv):
        if not hasattr(mesh, "_gpu"): self._build_gpu(mesh)
        vao, n, mode, kind = mesh._gpu
        mvp = mat4_mul(pv, mesh.model)
        prog = self.prog_tex if kind=='tex' else self.prog_color
        prog.use(); prog['u_mvp'] = mvp
        if kind=='tex':
            gl.glActiveTexture(gl.GL_TEXTURE0); gl.glBindTexture(gl.GL_TEXTURE_2D, mesh.texture_id); prog['t'] = 0
        gl.glBindVertexArray(vao); gl.glDrawArrays(mode, 0, n)

class Ego:
    def __init__(self):
        self.pos = [0.0, 0.0, 0.0]; self.yaw = 0.0; self.v = 0.0; self.steer = 0.0
        self.wb = 2.8; self.max_steer = math.radians(35.0); self.max_steer_rate = math.radians(120.0)
        self.max_accel, self.max_brake = 15.0, 10.0
        self.car_mesh = None; self.tex_normal = None; self.tex_brake = None; self.wheels = []; self._wheel_roll = 0.0
        self._load_car()

    def _load_car(self):
        try:
            p, uv, tpath, col = load_obj_with_uv_mtl("../assets/WAutoCar.obj")
            if tpath:
                self.tex_normal = pyglet.image.load(tpath).get_texture()
                self.car_mesh = Mesh(p, col, uv, self.tex_normal.id, gl.GL_TRIANGLES, mat4_identity(), _tex_obj=self.tex_normal)
                base, ext = os.path.splitext(tpath)
                bp = f"{base}_brake{ext}"
                if os.path.isfile(bp): self.tex_brake = pyglet.image.load(bp).get_texture()
            self._load_wheels()
        except Exception as e: print(f"Car load error: {e}")

    def _load_wheels(self):
        try:
            for side, off in [('R', (0.825,0.35,-1.66)), ('L', (-0.825,0.35,-1.66)), ('R', (0.8,0.35,1.22)), ('L', (-0.8,0.35,1.22))]:
                p, uv, tp, col = load_obj_with_uv_mtl(f"../assets/whl/whl_{side}.obj")
                tex = pyglet.image.load(tp).get_texture() if tp else None
                m = Mesh(p, col, uv, tex.id if tex else None, gl.GL_TRIANGLES, mat4_identity(), _tex_obj=tex)
                self.wheels.append({'mesh': m, 'offset': off, 'steer': off[2] < 0, 'radius': 0.35})
        except Exception as e: print(f"Wheel load error: {e}")

    def update(self, dt, throttle, steer_cmd, brake):
        target = clamp(steer_cmd * self.max_steer, -self.max_steer, self.max_steer)
        self.steer += clamp(target - self.steer, -self.max_steer_rate*dt, self.max_steer_rate*dt)
        a = self.max_accel*clamp(throttle,0,1) - self.max_brake*clamp(brake,0,1) - 0.015*self.v - 0.35*self.v*abs(self.v)
        self.v = max(0, self.v + a*dt)
        self.pos[0] += self.v * math.sin(self.yaw) * dt; self.pos[2] += -self.v * math.cos(self.yaw) * dt
        self.yaw += (self.v/self.wb) * math.tan(self.steer) * dt
        self._wheel_roll += (self.v/0.35)*dt
        if self.car_mesh:
            self.car_mesh.model = mat4_mul(mat4_translate(*self.pos), mat4_rotate_y(self.yaw))
            if self.tex_brake and self.tex_normal:
                self.car_mesh.texture_id = self.tex_brake.id if brake > 0.1 else self.tex_normal.id

# ------------------------------------------------------------
# Simulation Window
# ------------------------------------------------------------
class WASimulation(pyglet.window.Window):
    def __init__(self, width=1280, height=720, fps=60):
        super().__init__(width=width, height=height, caption="WA Vantage Simulation", resizable=True, config=gl.Config(double_buffer=True, depth_size=24, major_version=3, minor_version=3))
        self.fps = fps; self.keys = key.KeyStateHandler(); self.push_handlers(self.keys)
        gl.glEnable(gl.GL_DEPTH_TEST); gl.glClearColor(0.05, 0.06, 0.09, 1.0)
        self.renderer = Renderer(); self.ego = Ego()
        gv, gc = make_grid_tile(); self.grid = Mesh(gv, gc, None, None, gl.GL_LINES, mat4_identity())
        self.hud = pyglet.text.Label("", font_name="Roboto", font_size=12, x=10, y=10)
        self.assets = {}
        self._load_assets()
        
        # Detection setup
        if os.path.exists("log.txt"):
            try: os.remove("log.txt")
            except: pass
        self.detector = ObjectDetectionSystem(args.video_path, log_file="log.txt", use_yolo=False) # Use mock for now
        self.log_parser = LogParser("log.txt")
        self.detected_entities: Dict[str, DetectedObject] = {}
        
        # Split screen setup
        self.left_w = int(width * 0.65)
        self.cap = cv2.VideoCapture(args.video_path); self.v_tex = None
        
        # Detect FPS from video
        video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if video_fps < 1 or video_fps > 120: video_fps = fps # Fallback
        self.fps = video_fps
        print(f"Detected video FPS: {self.fps}")

        self.last_time = time.perf_counter(); self.dt = 1.0/self.fps
        
        # Trajectory synchronization
        self.trajectory = []
        self._load_trajectory() # Now cap is ready
        self.auto_drive = True
        
        # Camera smoothing state
        self.cam_eye = [0.0, 3.25, 9.0]
        self.cam_target = [0.0, 0.0, -5.0]
        
        pyglet.clock.schedule_interval(self.update, 1.0/self.fps)

    def _load_assets(self):
        # Load assets for detected objects
        asset_paths = {
            "car": "../assets/car/car.obj",
            "cone": "../assets/traffic-cone/cone.obj",
            "truck": "../assets/bus/bus.obj",          # Use bus as proxy for missing truck
            "bus": "../assets/bus/bus.obj",
            "barrel": "../assets/barrel/barrel.obj"
        }
        for name, path in asset_paths.items():
            try:
                p, uv, tp, col = load_obj_with_uv_mtl(path)
                if not p: continue
                tex = create_texture_2d(tp) if tp else None
                self.assets[name] = Mesh(p, col, uv, tex.id if tex else None, gl.GL_TRIANGLES, mat4_identity(), _tex_obj=tex)
            except Exception as e: print(f"Error loading {name}: {e}")

    def _load_trajectory(self):
        traj_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trajectory.json")
        print(f"DEBUG: Looking for trajectory at {traj_path}")
        if os.path.exists(traj_path):
            try:
                with open(traj_path, 'r') as f:
                    self.trajectory = json.load(f)
                print(f"Loaded {len(self.trajectory)} trajectory frames from {traj_path}")
                self.auto_drive = True
                # Remove the startup seek; play from frame 0
                self.frame_idx = 0
            except Exception as e: 
                print(f"Trajectory load error: {e}")
                self.auto_drive = False
        else:
            print(f"Warning: {traj_path} not found. Auto-drive disabled.")
            self.auto_drive = False

    def on_resize(self, width, height):
        super().on_resize(width, height)

    def update(self, _dt):
        now = time.perf_counter(); self.dt = clamp(now - self.last_time, 0, 1.0/30.0); self.last_time = now
        
        if self.auto_drive and self.trajectory:
            # Find closest frame in trajectory
            best_match = None
            if self.frame_idx < self.trajectory[0]['frame']:
                # If we haven't reached the trajectory start yet, anchor to the first point
                best_match = self.trajectory[0]
            else:
                for entry in self.trajectory:
                    if entry['frame'] <= self.frame_idx:
                        best_match = entry
                    else: break
            
            if best_match:
                # Direct pose injection
                self.ego.pos[0] = best_match['x']
                self.ego.pos[2] = best_match['z']
                self.ego.yaw = best_match['yaw']
                # Determine velocity based on whether we are actually "moving" in the trajectory
                is_moving = self.frame_idx >= self.trajectory[0]['frame']
                self.ego.v = 15.0 if is_moving else 0.0
                if self.frame_idx % 30 == 0:
                    state = "Syncing" if is_moving else "Waiting"
                    print(f"FRAME {self.frame_idx}: {state} to traj frame {best_match['frame']} | pos=({self.ego.pos[0]:.2f}, {self.ego.pos[2]:.2f})")
        
        # Vehicle update (purely state-based now, manual controls removed)
        self.ego.update(self.dt, 0, 0, 0)
        
        # Video update & Detection
        ret, frame = self.cap.read(); self.frame_idx += 1
        if not ret: 
            # Video looped - Reset environment state
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            self.ego.pos = [0.0, 0.0, 0.0]
            self.ego.v = 0.0
            self.frame_idx = 0
            self.detected_entities.clear()
            
        if ret:
            # 1. Process frame with detector (logs to log.txt)
            self.detector.process_frame(frame)
            
            # 2. Parse latest results from log.txt
            new_data = self.log_parser.parse_latest()
            if new_data:
                for oid, data in new_data.items():
                    if oid in self.detected_entities:
                        self.detected_entities[oid].update_from_data(data)
                    else:
                        self.detected_entities[oid] = DetectedObject(oid, data, self.assets)
            
            # Dynamically adjust left_w to match video aspect ratio
            h, w = frame.shape[:2]
            aspect = w / h
            self.left_w = int(self.height * aspect)
            if self.left_w > self.width * 0.8: self.left_w = int(self.width * 0.8)
            
            frame_rgb = cv2.cvtColor(cv2.flip(frame, 0), cv2.COLOR_BGR2RGB)
            self.v_tex = pyglet.image.ImageData(w, h, 'RGB', frame_rgb.tobytes()).get_texture()

        self.hud.text = f"Speed: {self.ego.v:.1f} m/s | Objects: {len(self.detected_entities)}"

    def on_draw(self):
        self.clear()
        
        # 1. Draw Video (Left Side)
        gl.glViewport(0, 0, self.width, self.height)
        if self.v_tex:
            self.v_tex.blit(0, 0, width=self.left_w, height=self.height)
        
        # 2. Draw 3D Environment (Right Side)
        gl.glViewport(self.left_w, 0, self.width - self.left_w, self.height)
        
        # Speed-dependent FOV for "sense of speed"
        base_fov = 60.0
        dynamic_fov = base_fov + clamp(self.ego.v * 0.5, 0.0, 20.0)
        proj = perspective(dynamic_fov, max(0.1, (self.width - self.left_w)/self.height), 0.1, 800.0)
        
        # Target camera position (relative to car)
        # Distance increases slightly with speed
        dist = 9.0 + clamp(self.ego.v * 0.1, 0, 5)
        ideal_eye = [
            self.ego.pos[0] - dist * math.sin(self.ego.yaw),
            3.25 + clamp(self.ego.v * 0.05, 0, 2), # Raise camera slightly at speed
            self.ego.pos[2] + dist * math.cos(self.ego.yaw)
        ]
        ideal_look = [
            self.ego.pos[0] + 10 * math.sin(self.ego.yaw),
            0.5,
            self.ego.pos[2] - 10 * math.cos(self.ego.yaw)
        ]
        
        # Smooth interpolation (Lerp)
        # Low factor = very smooth/laggy, High factor = rigid
        cam_speed = 4.0 * self.dt
        self.cam_eye = lerp_vec3(self.cam_eye, ideal_eye, clamp(cam_speed, 0, 1))
        self.cam_target = lerp_vec3(self.cam_target, ideal_look, clamp(cam_speed * 1.5, 0, 1))
        
        pv = mat4_mul(proj, look_at(self.cam_eye, self.cam_target, (0,1,0)))
        
        # Grid
        for i in range(-2,3):
            for j in range(-2,3):
                self.grid.model = mat4_translate((int(self.ego.pos[0]/40)+i)*40, 0, (int(self.ego.pos[2]/40)+j)*40)
                self.renderer.draw_mesh(self.grid, pv)
        
        # Ego Vehicle
        if self.ego.car_mesh:
            self.renderer.draw_mesh(self.ego.car_mesh, pv)
            for w in self.ego.wheels:
                ox, oy, oz = w['offset']
                M = mat4_mul(mat4_mul(mat4_translate(*self.ego.pos), mat4_rotate_y(self.ego.yaw)), mat4_translate(ox, oy, oz))
                if w['steer']: M = mat4_mul(M, mat4_rotate_y(self.ego.steer))
                M = mat4_mul(M, mat4_rotate_x(self.ego._wheel_roll))
                w['mesh'].model = M
                self.renderer.draw_mesh(w['mesh'], pv)
        
        # Detected Objects
        for obj in self.detected_entities.values():
            if obj.mesh:
                # Position object relative to the moving ego car
                M = mat4_mul(mat4_translate(*self.ego.pos), mat4_rotate_y(self.ego.yaw))
                obj.mesh.model = mat4_mul(M, mat4_translate(*obj.pos))
                self.renderer.draw_mesh(obj.mesh, pv)
        
        # 3. Draw HUD
        gl.glViewport(0, 0, self.width, self.height)
        self.hud.draw()

if __name__ == "__main__":
    WASimulation(int(args.width), int(args.height), args.fps)
    pyglet.app.run()
