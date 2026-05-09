"""
Integration module for adding H.264 streaming to the testbed.py application
"""

import sys
import os
import time
import threading
import ctypes
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from streaming import H265Streamer
import pyglet.gl as gl


def _read_bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

class StreamingIntegration:
    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30,
                 rtp_host: str = "127.0.0.1", rtp_port: int = 5000,
                 queue_size: int = 1, keyframe_interval: Optional[int] = None,
                 stream_mode: str = "mpegts"):
        self.width = width
        self.height = height
        self.fps = fps
        self.rtp_host = rtp_host
        self.rtp_port = rtp_port
        self.queue_size = queue_size
        self.keyframe_interval = keyframe_interval
        self.stream_mode = stream_mode
        
        self.streamer: Optional[H265Streamer] = None
        self._last_push = time.perf_counter()
        self._frame_interval = 1.0 / fps
        self.async_readback = _read_bool_env("WAUTOVANTAGE_STREAM_ASYNC_READBACK", True)
        self._frame_bytes = self.width * self.height * 3
        self._pbo_ids = None
        self._pbo_index = 0
        self._pbo_has_previous = False
        
        self._init_streaming()
    
    def _init_streaming(self):
        try:
            self.streamer = H265Streamer(
                width=self.width,
                height=self.height,
                fps=self.fps,
                rtp_host=self.rtp_host,
                rtp_port=self.rtp_port,
                queue_size=self.queue_size,
                keyframe_interval=self.keyframe_interval,
                stream_mode=self.stream_mode
            )
            
            if self.streamer.is_active:
                print(f"✓ H.264 streaming initialized successfully")
                if self.stream_mode == "rtp":
                    print(f"  → RTP: {self.rtp_host}:{self.rtp_port}")
                    print(f"  → SDP: stream.sdp")
                else:
                    print(f"  → UDP/MPEG-TS: {self.rtp_host}:{self.rtp_port}")
            else:
                print("✗ H.264 streaming initialization failed")
                print("  → Check FFmpeg and NVENC installation")
                
        except Exception as e:
            print(f"✗ Failed to initialize streaming: {e}")
            self.streamer = None
    
    def push_frame(self):
        if not self.streamer or not self.streamer.is_active:
            return
        if not self._streamer_can_accept_frame():
            return
        
        current_time = time.perf_counter()
        if (current_time - self._last_push) >= (self._frame_interval - 0.002):
            try:
                frame_data = (
                    self._capture_opengl_frame_async()
                    if self.async_readback
                    else self._capture_opengl_frame()
                )
                self._last_push = current_time
                if frame_data:
                    self.streamer.push_frame(frame_data)
            except Exception as e:
                print(f"Error capturing frame: {e}")
                self.async_readback = False

    def _streamer_can_accept_frame(self) -> bool:
        if not self.streamer or not self.streamer.is_active:
            return False
        if hasattr(self.streamer, "can_accept_frame"):
            return self.streamer.can_accept_frame()
        frame_queue = getattr(self.streamer, "frame_queue", None)
        return frame_queue is None or not frame_queue.full()
    
    def _capture_opengl_frame(self) -> Optional[bytes]:
        try:
            buffer = (gl.GLubyte * (self.width * self.height * 3))()
            
            gl.glReadPixels(
                0, 0, 
                self.width, self.height, 
                gl.GL_RGB, 
                gl.GL_UNSIGNED_BYTE, 
                buffer
            )
            
            frame_data = bytes(buffer)
            return frame_data
            
        except Exception as e:
            print(f"OpenGL frame capture failed: {e}")
            return None

    def _ensure_pbos(self):
        if self._pbo_ids is not None:
            return
        pixel_pack_buffer = getattr(gl, "GL_PIXEL_PACK_BUFFER", 0x88EB)
        stream_read = getattr(gl, "GL_STREAM_READ", 0x88E1)
        self._pbo_ids = (gl.GLuint * 2)()
        gl.glGenBuffers(2, self._pbo_ids)
        for pbo in self._pbo_ids:
            gl.glBindBuffer(pixel_pack_buffer, pbo)
            gl.glBufferData(pixel_pack_buffer, self._frame_bytes, None, stream_read)
        gl.glBindBuffer(pixel_pack_buffer, 0)

    def _capture_opengl_frame_async(self) -> Optional[bytes]:
        pixel_pack_buffer = getattr(gl, "GL_PIXEL_PACK_BUFFER", 0x88EB)
        stream_read = getattr(gl, "GL_STREAM_READ", 0x88E1)
        read_only = getattr(gl, "GL_READ_ONLY", 0x88B8)

        try:
            self._ensure_pbos()
            read_index = self._pbo_index
            map_index = 1 - read_index

            gl.glBindBuffer(pixel_pack_buffer, self._pbo_ids[read_index])
            gl.glBufferData(pixel_pack_buffer, self._frame_bytes, None, stream_read)
            gl.glReadPixels(
                0, 0,
                self.width, self.height,
                gl.GL_RGB,
                gl.GL_UNSIGNED_BYTE,
                ctypes.c_void_p(0),
            )

            frame_data = None
            if self._pbo_has_previous:
                gl.glBindBuffer(pixel_pack_buffer, self._pbo_ids[map_index])
                ptr = gl.glMapBuffer(pixel_pack_buffer, read_only)
                if ptr:
                    frame_data = ctypes.string_at(ptr, self._frame_bytes)
                    gl.glUnmapBuffer(pixel_pack_buffer)

            gl.glBindBuffer(pixel_pack_buffer, 0)
            self._pbo_has_previous = True
            self._pbo_index = map_index
            return frame_data
        except Exception as e:
            gl.glBindBuffer(pixel_pack_buffer, 0)
            print(f"OpenGL async frame capture failed: {e}")
            self.async_readback = False
            return self._capture_opengl_frame()
    
    def stop(self):
        if self._pbo_ids is not None:
            try:
                gl.glDeleteBuffers(2, self._pbo_ids)
            except Exception:
                pass
            self._pbo_ids = None
        if self.streamer:
            self.streamer.stop()
            print("H.264 streaming stopped")


def integrate_with_testbed():
    streaming = StreamingIntegration(
        width=1280,
        height=720,
        fps=30,
        rtp_host="127.0.0.1",
        rtp_port=5000,
        stream_mode="mpegts"
    )
    
    return streaming


if __name__ == "__main__":
    print("Testing H.264 Streaming Integration")
    print("=" * 40)
    
    streaming = StreamingIntegration(stream_mode="mpegts")
    
    if streaming.streamer and streaming.streamer.is_active:
        print("Integration test successful - streaming ready")
        
        print("Simulating frame capture for 5 seconds...")
        for i in range(150):  
            streaming.push_frame()
            time.sleep(1.0 / 30)
            
        print("Test completed")
    else:
        print("Integration test failed - check dependencies")
    
    streaming.stop()
