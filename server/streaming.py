"""
Hardware-accelerated H.265 streaming module using FFmpeg + NVENC
Designed for NVIDIA Jetson testing with RTP output
"""

import subprocess
import threading
import time
import queue
import logging
import os
from typing import Optional

class H265Streamer:
    
    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30, 
                 rtp_host: str = "127.0.0.1", rtp_port: int = 5004):
        self.width = width
        self.height = height
        self.fps = fps
        self.rtp_host = rtp_host
        self.rtp_port = rtp_port
        
        # Streaming state
        self.is_active = False
        self.proc = None
        self.worker = None
        self.frame_queue = queue.Queue(maxsize=30)
        
        # Frame timing
        self._frame_interval = 1.0 / fps
        self._last_push = time.perf_counter()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Check if FFmpeg is available
        if not self._check_ffmpeg_availability():
            self.logger.error("FFmpeg not available - streaming disabled")
            return

        # Check if NVENC is available  
        if not self._check_nvenc_availability():
            self.logger.error("NVENC hardware encoder not available - streaming disabled")
            return
            
        # Start streaming
        self._start_streaming()
    
    def _check_ffmpeg_availability(self) -> bool:
        try:
            result = subprocess.run(
                ['ffmpeg', '-version'], 
                capture_output=True, 
                check=True,
                text=True
            )
            self.logger.info("FFmpeg found: %s", result.stdout.split('\n')[0])
            return True
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            self.logger.error("FFmpeg not found: %s", e)
            return False
    
    def _check_nvenc_availability(self) -> bool:
        try:
            result = subprocess.run(
                ['ffmpeg', '-encoders'], 
                capture_output=True, 
                check=True,
                text=True
            )
            
            if 'hevc_nvenc' in result.stdout:
                self.logger.info("NVENC H.265 encoder available")
                return True
            else:
                self.logger.error("NVENC H.265 encoder not found in FFmpeg")
                return False
                
        except subprocess.CalledProcessError as e:
            self.logger.error("Failed to check NVENC availability: %s", e)
            return False
    
    def _start_streaming(self):
        ffmpeg_cmd = [
            'ffmpeg',
            '-y',  
            '-re',  
            '-f', 'rawvideo',  
            '-vcodec', 'rawvideo',  
            '-s', f'{self.width}x{self.height}',
            '-pix_fmt', 'rgb24',
            '-r', str(self.fps),
            '-i', '-',
            '-vf', 'vflip',
            '-c:v', 'hevc_nvenc',
            '-preset', 'p1',
            '-tune', 'ull',
            '-zerolatency', '1',
            '-profile:v', 'main',
            '-level', '5.1',
            '-b:v', '5000k',
            '-f', 'rtp',
            f'rtp://{self.rtp_host}:{self.rtp_port}'
        ]
        
        try:
            self.proc = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0
            )
            
            self._create_sdp_file()    
            self.worker = threading.Thread(target=self._write_worker, daemon=True)
            self.worker.start()
            
            self.is_active = True
            self.logger.info(f"Streaming started: RTP to {self.rtp_host}:{self.rtp_port}")
            self.logger.info(f"SDP file created: stream.sdp")
            
        except Exception as e:
            self.logger.error(f"Failed to start streaming: {e}")
            self.is_active = False
    
    def _create_sdp_file(self):
        sdp_content = f"""v=0
            o=- 0 0 IN IP4 {self.rtp_host}
            s=H265_NVENC_Stream
            c=IN IP4 {self.rtp_host}
            t=0 0
            m=video {self.rtp_port} RTP/AVP 96
            a=rtpmap:96 H265/90000
            a=fmtp:96 profile-level-id=64001f;packetization-mode=1
        """
        
        try:
            with open("stream.sdp", "w") as f:
                f.write(sdp_content)
            self.logger.info("SDP file created successfully")
        except Exception as e:
            self.logger.error(f"Failed to create SDP file: {e}")
    
    def _write_worker(self):
        while self.is_active:
            try:
                frame = self.frame_queue.get(timeout=1.0)
                if self.proc and self.proc.stdin:
                    self.proc.stdin.write(frame)
                    self.proc.stdin.flush()
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error writing frame: {e}")
                break
    
    def push_frame(self, frame_data: bytes):
        if not self.is_active:
            return
            
        current_time = time.perf_counter()
        if (current_time - self._last_push) >= (self._frame_interval - 0.002):
            try:
                self.frame_queue.put_nowait(frame_data)
                self._last_push = current_time
            except queue.Full:
                pass
    
    def stop(self):
        self.is_active = False
        
        if self.worker and self.worker.is_alive():
            self.worker.join(timeout=1.0)
        
        if self.proc:
            try:
                if self.proc.stdin:
                    self.proc.stdin.close()
                self.proc.terminate()
                self.proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
            except Exception as e:
                self.logger.error(f"Error stopping FFmpeg: {e}")
        
        self.logger.info("Streaming stopped")


def main():
    print("H.265 NVENC Streaming Module")
    print("=" * 40)
    
    streamer = H265Streamer(
        width=1280,
        height=720,
        fps=30,
        rtp_host="127.0.0.1",
        rtp_port=5004
    )
    
    if not streamer.is_active:
        print("Streaming not available - check FFmpeg and NVENC installation")
        return
    
    print(f"Streaming to RTP: {streamer.rtp_host}:{streamer.rtp_port}")
    print("Press Ctrl+C to stop")
    
    try:
        import time
        frame_size = streamer.width * streamer.height * 3 
        test_frame = bytes([128] * frame_size) 
        
        while True:
            streamer.push_frame(test_frame)
            time.sleep(1.0 / streamer.fps)
            
    except KeyboardInterrupt:
        print("\nStopping streamer...")
    finally:
        streamer.stop()


if __name__ == "__main__":
    main()