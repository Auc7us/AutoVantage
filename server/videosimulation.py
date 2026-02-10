import math, os, time, csv
from typing import List, Tuple, Optional, Dict
import pyglet
from pyglet import gl
import cv2
import numpy as np

# Reuse renderer and geometry helpers from testbed (same folder)
from testbed import (
    Mesh, Renderer, Ego, TrafficLight,
    make_grid_tile, make_polyline,
    load_obj_with_uv_mtl, create_texture_2d,
    mat4_identity, mat4_translate, mat4_rotate_y, mat4_rotate_x,
    perspective, look_at, mat4_mul,
    MovingBox, SurroundingVehicle, Barricade
)

def ortho(left, right, bottom, top, znear, zfar):
    """Orthographic projection matrix"""
    rl = 1.0 / (right - left)
    tb = 1.0 / (top - bottom)
    fn = 1.0 / (zfar - znear)
    return [2*rl,     0,      0,  -(right+left)*rl,
            0,     2*tb,      0,  -(top+bottom)*tb,
            0,        0,  -2*fn,  -(zfar+znear)*fn,
            0,        0,      0,                  1]

class VideoSimulation(pyglet.window.Window):
    """
    3D simulation that mimics the camera video feed.
    - Loads PNG frames from data/dataset/rgb/
    - Renders 3D objects (Traffic Lights, Barricades) positioned using depth data
    """

    def __init__(self, width=1280, height=720, fps=30):
        # Enable alpha blending for overlays if needed
        cfg = gl.Config(double_buffer=True, depth_size=24, major_version=3, minor_version=3)
        super().__init__(width=width, height=height, caption="Video Simulation 3D", resizable=True, config=cfg)
        self.fps = fps
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDisable(gl.GL_CULL_FACE)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glClearColor(0.05, 0.05, 0.08, 1.0)  # Dark background like testbed

        self.renderer = Renderer()
        self.hud = pyglet.text.Label("", font_name="Roboto", font_size=12, x=10, y=10, color=(255, 255, 255, 255))

        # Ego vehicle (no user controls, driven by simulation)
        self.ego = Ego()
        self._wheel_roll = 0.0

        # Load assets
        self._load_assets()

        # Data loading (paths relative to script location)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up one level from server/ to root, then into assets/data/
        root_dir = os.path.dirname(script_dir)
        dataset_dir = os.path.join(root_dir, "data", "dataset")
        
        self.rgb_dir = os.path.join(dataset_dir, "rgb")
        self.xyz_dir = os.path.join(dataset_dir, "xyz")
        self.csv_path = os.path.join(dataset_dir, "bboxes_light.csv")
        self.csv_barricade_path = os.path.join(dataset_dir, "bboxes_barricade.csv")
        
        self.frames_rgb: List[str] = []
        self.frames_xyz: Dict[int, str] = {}
        self.detections: Dict[int, List[Tuple[int, int, int, int]]] = {}
        self.detections_barricade: Dict[int, List[Tuple[int, int, int, int]]] = {}
        
        self._load_data()

        # Simulation state
        self._start_time = time.perf_counter()
        self.current_frame_idx = 0
        self.total_frames = len(self.frames_rgb)
        print(f"DEBUG: Initialized total_frames = {self.total_frames}, frames_rgb length = {len(self.frames_rgb)}")
        
        # Current frame data
        self.current_frame_tex = None
        self.current_depth_data = None
        
        # 3D Objects
        self.traffic_lights: List[TrafficLight] = []
        self.barricades: List[Barricade] = []
        
        # Camera mount parameters (Windshield)
        self.cam_mount_local = (0.0, 1.65, -0.5) # x, y, z relative to ego center
        self.cam_pitch = math.radians(0.0) # Looking straight ahead
        
        # Background quad for video
        self.bg_mesh = self._create_background_quad()

        # Schedule update
        pyglet.clock.schedule_interval(self.update, 1.0 / self.fps)

    def _load_assets(self):
        # Load car mesh (for visualization if needed, though we are INSIDE the car usually)
        self.car_mesh = None
        try:
            tri_pos, tri_uv, tex_path = load_obj_with_uv_mtl("assets/WAutoCar.obj", scale=1.0, center_y=0.0)
            if tri_uv and tex_path:
                tex = create_texture_2d(tex_path)
                if tex:
                    self.car_mesh = Mesh(tri_pos, None, tri_uv, tex.id, gl.GL_TRIANGLES, mat4_identity(), _tex_obj=tex)
        except Exception:
            pass

        # Load wheel meshes
        self.wheels = []
        try:
            wpos_r, wuv_r, wtex_r = load_obj_with_uv_mtl("assets/whl/whl_R.obj", scale=1.0, center_y=0.0)
            wpos_l, wuv_l, wtex_l = load_obj_with_uv_mtl("assets/whl/whl_L.obj", scale=1.0, center_y=0.0)
            wtex_r_obj = create_texture_2d(wtex_r) if (wuv_r and wtex_r) else None
            wtex_l_obj = create_texture_2d(wtex_l) if (wuv_l and wtex_l) else None
            
            if wpos_r:
                wmesh_r = Mesh(wpos_r, None, wuv_r, (wtex_r_obj.id if wtex_r_obj else None), gl.GL_TRIANGLES, mat4_identity(), _tex_obj=wtex_r_obj)
                self.wheels.append({'mesh': wmesh_r, 'offset': (0.825, 0.35, -1.6625), 'steer': True})
                self.wheels.append({'mesh': wmesh_r, 'offset': (0.8, 0.35, 1.2225), 'steer': False})
            if wpos_l:
                wmesh_l = Mesh(wpos_l, None, wuv_l, (wtex_l_obj.id if wtex_l_obj else None), gl.GL_TRIANGLES, mat4_identity(), _tex_obj=wtex_l_obj)
                self.wheels.append({'mesh': wmesh_l, 'offset': (-0.825, 0.35, -1.6625), 'steer': True})
                self.wheels.append({'mesh': wmesh_l, 'offset': (-0.8, 0.35, 1.2225), 'steer': False})
        except Exception:
            pass

    def _load_data(self):
        # 1. Load RGB frames
        if os.path.exists(self.rgb_dir):
            files = sorted([f for f in os.listdir(self.rgb_dir) if f.endswith('.png')])
            self.frames_rgb = [os.path.join(self.rgb_dir, f) for f in files]
            print(f"Found {len(self.frames_rgb)} RGB frames")
        
        # 2. Load XYZ file mapping
        if os.path.exists(self.xyz_dir):
            files = sorted([f for f in os.listdir(self.xyz_dir) if f.endswith('.npz')])
            for f in files:
                # Extract frame number from depthXXXXXX.npz
                try:
                    num_str = f.replace('depth', '').replace('.npz', '')
                    idx = int(num_str)
                    self.frames_xyz[idx] = os.path.join(self.xyz_dir, f)
                except ValueError:
                    pass
            print(f"Found {len(self.frames_xyz)} XYZ files")

        # 3. Load Detections
        if os.path.exists(self.csv_path):
            try:
                with open(self.csv_path, 'r', newline='') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        fr = int(row['frame'])
                        x1, y1 = int(row['x1']), int(row['y1'])
                        x2, y2 = int(row['x2']), int(row['y2'])
                        if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:
                            continue
                        if fr not in self.detections:
                            self.detections[fr] = []
                        self.detections[fr].append((x1, y1, x2, y2))
                print(f"Loaded detections for {len(self.detections)} frames")
                print(f"First 5 frames with detections: {list(self.detections.keys())[:5]}")
            except Exception as e:
                print(f"Error loading CSV: {e}")

        # 4. Load Barricade Detections (optional)
        if os.path.exists(self.csv_barricade_path):
            try:
                with open(self.csv_barricade_path, 'r', newline='') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        fr = int(row['frame'])
                        x1, y1 = int(row['x1']), int(row['y1'])
                        x2, y2 = int(row['x2']), int(row['y2'])
                        if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:
                            continue
                        if fr not in self.detections_barricade:
                            self.detections_barricade[fr] = []
                        self.detections_barricade[fr].append((x1, y1, x2, y2))
                print(f"Loaded barricade detections for {len(self.detections_barricade)} frames")
            except Exception as e:
                print(f"Error loading Barricade CSV: {e}")
        else:
            print(f"Barricade CSV not found at {self.csv_barricade_path}")

    def _create_background_quad(self):
        # Create a full-screen quad for the video background
        # Z = 0.999 (far plane)
        verts = [
            -1.0, -1.0, 0.999,  1.0, -1.0, 0.999,  -1.0, 1.0, 0.999,
            -1.0, 1.0, 0.999,   1.0, -1.0, 0.999,   1.0, 1.0, 0.999
        ]
        uvs = [
            0.0, 1.0,  1.0, 1.0,  0.0, 0.0,
            0.0, 0.0,  1.0, 1.0,  1.0, 0.0
        ]
        return Mesh(verts, None, uvs, None, gl.GL_TRIANGLES, mat4_identity())

    def update(self, dt):
        # Update frame index
        elapsed = time.perf_counter() - self._start_time
        new_frame_idx = int(elapsed * self.fps)
        
        # Debug: Log frame changes
        if new_frame_idx != self.current_frame_idx:
            print(f"Frame changed: {self.current_frame_idx} -> {new_frame_idx}")
        
        self.current_frame_idx = new_frame_idx
        
        if self.current_frame_idx >= self.total_frames:
            # Loop or stop
            print(f"Looping: Frame {self.current_frame_idx} >= {self.total_frames}")
            self._start_time = time.perf_counter()
            self.current_frame_idx = 0

        # Load frame data
        self._update_frame_data(self.current_frame_idx)
        
        # Update objects
        self._update_objects(self.current_frame_idx)
        
        # Update HUD
        self.hud.text = f"Frame: {self.current_frame_idx}/{self.total_frames} | TL: {len(self.traffic_lights)} | B: {len(self.barricades)}"

    def _update_frame_data(self, frame_idx):
        # Load RGB Texture
        if frame_idx < len(self.frames_rgb):
            path = self.frames_rgb[frame_idx]
            # Use cv2 to load, then convert to texture
            img = cv2.imread(path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w, c = img.shape
                img_data_bytes = img.tobytes()
                
                # Use ImageData to create texture directly
                image_data = pyglet.image.ImageData(w, h, 'RGB', img_data_bytes)
                self.current_frame_tex = image_data.get_texture()
                
                # Update background mesh texture
                self.bg_mesh.texture_id = self.current_frame_tex.id
                self.bg_mesh._tex_obj = self.current_frame_tex # Keep ref

        # Load Depth Data
        self.current_depth_data = None
        if frame_idx in self.frames_xyz:
            try:
                data = np.load(self.frames_xyz[frame_idx])
                if 'xyz' in data:
                    self.current_depth_data = data['xyz'] # Shape (H, W, 4)
            except Exception as e:
                print(f"Failed to load XYZ for frame {frame_idx}: {e}")

    def _update_objects(self, frame_idx):
        self.traffic_lights.clear()
        
        # 1. Traffic Lights - Use bottom-center of bbox for ground-level positioning
        if frame_idx in self.detections:
            for (x1, y1, x2, y2) in self.detections[frame_idx]:
                # Use bottom-center of bounding box (where traffic light pole meets ground)
                cx = (x1 + x2) // 2
                cy_bottom = y2  # Bottom of the bounding box
                
                # Get 3D position
                pos_3d = None
                
                # Method 1: Use XYZ data if available
                if self.current_depth_data is not None:
                    try:
                        h, w, _ = self.current_depth_data.shape
                        cx_clamped = max(0, min(cx, w-1))
                        cy_clamped = max(0, min(cy_bottom, h-1))
                        
                        # Sample XYZ at bottom-center and surrounding pixels for robustness
                        points = []
                        for dy in [-2, -1, 0, 1, 2]:
                            for dx in [-2, -1, 0, 1, 2]:
                                sample_x = max(0, min(cx + dx, w-1))
                                sample_y = max(0, min(cy_bottom + dy, h-1))
                                point = self.current_depth_data[sample_y, sample_x]
                                if not np.all(point == 0):
                                    points.append(point)
                        
                        if points:
                            # Average the valid points
                            avg_point = np.mean(points, axis=0)
                            # Convert Camera Space (Y down, Z forward) to GL Space (Y up, Z back)
                            # Traffic light base is at ground level in camera coords
                            pos_3d = (float(avg_point[0]), -float(avg_point[1]), -float(avg_point[2]))
                    except Exception as e:
                        print(f"Error getting XYZ: {e}")

                # Method 2: Fallback heuristic
                if pos_3d is None:
                    # Get actual image dimensions
                    img_w = 1920
                    img_h = 1200
                    
                    # Estimate distance based on bounding box height
                    # Typical traffic light is ~1.0m tall
                    box_h = abs(y2 - y1)
                    assumed_light_height = 1.0
                    fx = img_h  # Approximate focal length in pixels
                    
                    # Distance = (real_height * focal_length) / pixel_height
                    dist = (assumed_light_height * fx) / max(1.0, box_h)
                    
                    # Unproject pixel coordinates to 3D
                    # Center horizontally, bottom of bbox vertically
                    u = cx - img_w/2
                    v = cy_bottom - img_h/2
                    
                    # 3D position in camera space
                    x_cam = u * dist / fx
                    y_cam = v * dist /  fx  # Positive = down in camera space
                    z_cam = dist
                    
                    # Convert to OpenGL space (Y up, Z back)
                    pos_3d = (float(x_cam), -float(y_cam), -float(z_cam))

                # Create Traffic Light Object at the calculated position
                if pos_3d:
                    tl = TrafficLight(pos_3d[0], pos_3d[1], pos_3d[2])
                    self.traffic_lights.append(tl)

        # 2. Barricades
        self.barricades.clear()
        if frame_idx in self.detections_barricade:
            for (x1, y1, x2, y2) in self.detections_barricade[frame_idx]:
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                pos_3d = None
                
                # Method 1: Use XYZ data
                if self.current_depth_data is not None:
                    try:
                        h, w, _ = self.current_depth_data.shape
                        cx_clamped = max(0, min(cx, w-1))
                        cy_clamped = max(0, min(cy, h-1))
                        point = self.current_depth_data[cy_clamped, cx_clamped]
                        if not np.all(point == 0):
                             pos_3d = (float(point[0]), -float(point[1]), -float(point[2]))
                    except Exception:
                        pass

                # Method 2: Fallback
                if pos_3d is None:
                    box_h = abs(y2 - y1)
                    img_h = 1200
                    dist = 500.0 / max(1.0, box_h) * 2.0
                    fx = img_h
                    u = cx - 1920/2
                    v = cy - 1200/2
                    x = u * dist / fx
                    y = -v * dist / fx
                    z = -dist
                    pos_3d = (float(x), float(y), float(z))

                b = Barricade(pos_3d[0], pos_3d[1], pos_3d[2])
                self.barricades.append(b)

    def on_draw(self):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        # 1. Draw Background (Video Frame) using orthographic projection
        gl.glDisable(gl.GL_DEPTH_TEST)
        if self.current_frame_tex is not None:
            # Orthographic projection for screen-aligned quad
            ortho_proj = ortho(-1, 1, -1, 1, -1, 1)
            self.renderer.draw_mesh(self.bg_mesh, ortho_proj)
        gl.glEnable(gl.GL_DEPTH_TEST)

        # 2. Draw 3D Scene (Traffic Lights, Barricades) with perspective
        # Camera setup: We want to match the video perspective.
        proj = perspective(60.0, max(1e-6, self.width/float(self.height)), 0.1, 500.0)
        
        # Camera at origin, looking forward (-Z)
        view = mat4_identity() 
        pv = mat4_mul(proj, view)

        # Draw Traffic Lights
        for tl in self.traffic_lights:
            self.renderer.draw_mesh(tl.mesh, pv)
        
        # Draw Barricades
        for b in self.barricades:
            self.renderer.draw_mesh(b.mesh, pv)

        # Draw HUD
        gl.glDisable(gl.GL_DEPTH_TEST)
        self.hud.draw()
        gl.glEnable(gl.GL_DEPTH_TEST)

if __name__ == "__main__":
    window = VideoSimulation(1280, 720, fps=30)
    pyglet.app.run()