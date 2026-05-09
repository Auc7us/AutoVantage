"""
Hardware-accelerated H.264 streaming module using FFmpeg + NVENC
Designed for NVIDIA Jetson testing with RTP output
"""

import base64
import subprocess
import threading
import time
import queue
import logging
import os
from typing import Optional

class H265Streamer:
    
    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30,
                 rtp_host: str = "127.0.0.1", rtp_port: int = 5000,
                 queue_size: int = 1, keyframe_interval: Optional[int] = None,
                 stream_mode: str = "mpegts"):
        self.width = width
        self.height = height
        self.fps = fps
        self.rtp_host = rtp_host
        self.rtp_port = rtp_port
        self.queue_size = max(1, queue_size)
        normalized_mode = stream_mode.strip().lower()
        self.stream_mode = normalized_mode if normalized_mode in {"mpegts", "rtp"} else "mpegts"
        default_keyframe_interval = max(10, fps // 2) if fps > 1 else 10
        if keyframe_interval is None:
            self.keyframe_interval = default_keyframe_interval
        else:
            self.keyframe_interval = max(1, keyframe_interval)
        
        # Streaming state
        self.is_active = False
        self.proc = None
        self.worker = None
        # Keep the queue shallow so motion stays current instead of buffering stale frames.
        self.frame_queue = queue.Queue(maxsize=self.queue_size)
        
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

        # Select the best available encoder
        if self._check_nvenc_availability():
            self.encoder = 'h264_nvenc'
            self.logger.info("Using Hardware Encoder: h264_nvenc")
        else:
            self.encoder = 'libx264'
            self.logger.info("NVENC hardware encoder not found. falling back to Software Encoder: libx264")
            
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
            
            if 'h264_nvenc' not in result.stdout:
                self.logger.error("NVENC H.264 encoder not found in FFmpeg binary")
                return False
        
            test_cmd = [
                'ffmpeg', '-y', '-f', 'lavfi', '-i', 'testsrc=s=64x64:r=1',
                '-t', '1', '-c:v', 'h264_nvenc', '-f', 'null', '-'
            ]
            test_result = subprocess.run(test_cmd, capture_output=True)
            
            if test_result.returncode == 0:
                self.logger.info("NVENC hardware initialization successful")
                return True
            else:
                self.logger.warning("NVENC binary present but hardware initialization failed (check drivers/GPU)")
                return False
                
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            self.logger.error("Failed to check NVENC availability: %s", e)
            return False
    
    def _start_streaming(self):
        if self.stream_mode == "rtp":
            self._cleanup_sdp_files()
            self._create_sdp_file()
        else:
            self._cleanup_sdp_files()

        ffmpeg_cmd = [
            'ffmpeg',
            '-y',  
            '-f', 'rawvideo',  
            '-vcodec', 'rawvideo',  
            '-s', f'{self.width}x{self.height}',
            '-pix_fmt', 'rgb24',
            '-r', str(self.fps),
            '-i', '-',
            '-vf', 'vflip,format=yuv420p',
        ]
        
        ffmpeg_cmd.extend(self._encoder_args())

        if self.stream_mode == "rtp":
            ffmpeg_cmd.extend([
                '-bsf:v', 'dump_extra=freq=keyframe',
                '-f', 'rtp',
                f'rtp://{self.rtp_host}:{self.rtp_port}?pkt_size=1200'
            ])
        else:
            ffmpeg_cmd.extend([
                '-flush_packets', '1',
                '-muxdelay', '0',
                '-muxpreload', '0',
                '-f', 'mpegts',
                f'udp://{self.rtp_host}:{self.rtp_port}?pkt_size=1316'
            ])
        
        try:
            self.proc = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                bufsize=0
            )
            
            self.is_active = True
            
            self.worker = threading.Thread(target=self._write_worker, daemon=True)
            self.worker.start()

            if self.stream_mode == "rtp":
                self.logger.info(f"Streaming started: RTP to {self.rtp_host}:{self.rtp_port}")
                self.logger.info("SDP file created: stream.sdp")
            else:
                self.logger.info(
                    "Streaming started: MPEG-TS over UDP to %s:%s",
                    self.rtp_host,
                    self.rtp_port,
                )
            
        except Exception as e:
            self.logger.error(f"Failed to start streaming: {e}")
            self.is_active = False

    def _encoder_args(self):
        if self.encoder == 'h264_nvenc':
            return [
                '-c:v', 'h264_nvenc',
                '-preset', 'p1',
                '-tune', 'ull',
                '-aud', '1',
                '-zerolatency', '1',
                '-b:v', '5000k',
                '-g', str(self.keyframe_interval),
                '-bf', '0',
                '-forced-idr', '1'
            ]

        return [
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-tune', 'zerolatency',
            '-g', str(self.keyframe_interval),
            '-bf', '0',
            '-x264-params',
            f'keyint={self.keyframe_interval}:min-keyint={self.keyframe_interval}:'
            'scenecut=0:repeat-headers=1:aud=1'
        ]

    def _cleanup_sdp_files(self):
        for path in ("stream.sdp", "stream.sdp.tmp"):
            try:
                os.unlink(path)
            except FileNotFoundError:
                continue
            except OSError as e:
                self.logger.warning("Failed to remove stale %s: %s", path, e)
    
    def _create_sdp_file(self):
        fmtp_parts = ['packetization-mode=1']
        sps, pps = self._probe_h264_parameter_sets()
        if sps and pps:
            if len(sps) >= 4:
                fmtp_parts.append(f'profile-level-id={sps[1:4].hex()}')
            fmtp_parts.append(
                'sprop-parameter-sets='
                f'{base64.b64encode(sps).decode("ascii")},'
                f'{base64.b64encode(pps).decode("ascii")}'
            )

        sdp_content = f"""v=0
o=- 0 0 IN IP4 {self.rtp_host}
s=H264 Streaming
c=IN IP4 {self.rtp_host}
t=0 0
m=video {self.rtp_port} RTP/AVP 96
a=rtpmap:96 H264/90000
a=fmtp:96 {';'.join(fmtp_parts)}
a=framesize:96 {self.width}-{self.height}
"""
        
        try:
            temp_path = "stream.sdp.tmp"
            with open(temp_path, "w") as f:
                f.write(sdp_content)
            os.replace(temp_path, "stream.sdp")
            self.logger.info("SDP file created successfully")
        except Exception as e:
            self.logger.error(f"Failed to create SDP file: {e}")

    def _probe_h264_parameter_sets(self):
        probe_cmd = [
            'ffmpeg',
            '-loglevel', 'error',
            '-y',
            '-f', 'lavfi',
            '-i', f'color=c=black:s={self.width}x{self.height}:r={self.fps}',
            '-frames:v', '1',
            '-vf', 'format=yuv420p',
        ]
        probe_cmd.extend(self._encoder_args())
        probe_cmd.extend([
            '-f', 'h264',
            '-'
        ])

        try:
            result = subprocess.run(probe_cmd, capture_output=True, check=True)
            sps, pps = self._extract_h264_parameter_sets(result.stdout)
            if not sps or not pps:
                self.logger.warning("Failed to extract H.264 SPS/PPS from probe stream")
            return sps, pps
        except subprocess.CalledProcessError as e:
            self.logger.warning("Failed to probe H.264 parameter sets: %s", e.stderr.decode("utf-8", errors="ignore"))
        except Exception as e:
            self.logger.warning("Unexpected error while probing H.264 parameter sets: %s", e)
        return None, None

    def _extract_h264_parameter_sets(self, bitstream: bytes):
        start_codes = []
        i = 0
        while i < len(bitstream) - 3:
            if bitstream[i:i + 4] == b'\x00\x00\x00\x01':
                start_codes.append((i, 4))
                i += 4
                continue
            if bitstream[i:i + 3] == b'\x00\x00\x01':
                start_codes.append((i, 3))
                i += 3
                continue
            i += 1

        sps = None
        pps = None
        for idx, (start, prefix_len) in enumerate(start_codes):
            nal_start = start + prefix_len
            nal_end = start_codes[idx + 1][0] if idx + 1 < len(start_codes) else len(bitstream)
            nal = bitstream[nal_start:nal_end]
            if not nal:
                continue
            nal_type = nal[0] & 0x1F
            if nal_type == 7 and sps is None:
                sps = nal
            elif nal_type == 8 and pps is None:
                pps = nal
            if sps and pps:
                break

        return sps, pps
    
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
                try:
                    self.frame_queue.get_nowait()
                    self.frame_queue.put_nowait(frame_data)
                    self._last_push = current_time
                except queue.Empty:
                    pass

    def can_accept_frame(self) -> bool:
        return self.is_active and not self.frame_queue.full()
    
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
    print("H.264 NVENC Streaming Module")
    print("=" * 40)
    
    streamer = H265Streamer(
        width=1280,
        height=720,
        fps=30,
        rtp_host="127.0.0.1",
        rtp_port=5000,
        stream_mode="mpegts"
    )
    
    if not streamer.is_active:
        print("Streaming not available - check FFmpeg and NVENC installation")
        return
    
    if streamer.stream_mode == "rtp":
        print(f"Streaming to RTP: {streamer.rtp_host}:{streamer.rtp_port}")
    else:
        print(f"Streaming to UDP/MPEG-TS: {streamer.rtp_host}:{streamer.rtp_port}")
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
