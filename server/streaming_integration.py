"""
Integration module for adding H.265 streaming to the testbed.py application
"""

import sys
import os
import time
import threading
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from streaming import H265Streamer
import pyglet.gl as gl

class StreamingIntegration:
    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30,
                 rtp_host: str = "127.0.0.1", rtp_port: int = 5004):
        self.width = width
        self.height = height
        self.fps = fps
        self.rtp_host = rtp_host
        self.rtp_port = rtp_port
        
        self.streamer: Optional[H265Streamer] = None
        self._last_push = time.perf_counter()
        self._frame_interval = 1.0 / fps
        
        self._init_streaming()
    
    def _init_streaming(self):
        try:
            self.streamer = H265Streamer(
                width=self.width,
                height=self.height,
                fps=self.fps,
                rtp_host=self.rtp_host,
                rtp_port=self.rtp_port
            )
            
            if self.streamer.is_active:
                print(f"✓ H.265 streaming initialized successfully")
                print(f"  → RTP: {self.rtp_host}:{self.rtp_port}")
                print(f"  → SDP: stream.sdp")
            else:
                print("✗ H.265 streaming initialization failed")
                print("  → Check FFmpeg and NVENC installation")
                
        except Exception as e:
            print(f"✗ Failed to initialize streaming: {e}")
            self.streamer = None
    
    def push_frame(self):
        if not self.streamer or not self.streamer.is_active:
            return
        
        current_time = time.perf_counter()
        if (current_time - self._last_push) >= (self._frame_interval - 0.002):
            try:
                frame_data = self._capture_opengl_frame()
                if frame_data:
                    self.streamer.push_frame(frame_data)
                    self._last_push = current_time
            except Exception as e:
                print(f"Error capturing frame: {e}")
    
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
    
    def stop(self):
        if self.streamer:
            self.streamer.stop()
            print("H.265 streaming stopped")


def integrate_with_testbed():
    streaming = StreamingIntegration(
        width=1280,
        height=720,
        fps=30,
        rtp_host="127.0.0.1",
        rtp_port=5004
    )
    
    return streaming


if __name__ == "__main__":
    print("Testing H.265 Streaming Integration")
    print("=" * 40)
    
    streaming = StreamingIntegration()
    
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