# WAutoVantage

Visualizer-first container workflow.

## Build

```bash
docker build -t wautovantage-dev ~/WAutoVantage
```

## Run

If you want the GUI to appear on the machine's local display:

```bash
export DISPLAY=:0
export XAUTHORITY=$HOME/.Xauthority
xhost +si:localuser:wautodrive
```

Then start the container:

```bash
sudo docker run -it --rm \
  --name wautovantage-dev \
  --runtime nvidia \
  --network host \
  --privileged \
  -e DISPLAY=$DISPLAY \
  -e XAUTHORITY=/home/wautodrive/.Xauthority \
  -v /dev:/dev \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v "$HOME/.Xauthority":/home/wautodrive/.Xauthority:ro \
  -v "$HOME/WAutoVantage":/workspace/WAutoVantage \
  -v "$HOME/WAutoVision":/workspace/WAutoVision \
  wautovantage-dev \
  bash -i
```

In a separate host terminal, start MediaMTX:

```bash
~/mediamtx ~/simple.yml
```

## Inside The Container

If you only want to run the visualizer with no ROS topic subscription:

```bash
cd /workspace/WAutoVantage
python3 server/testbed.py
```

If you want the visualizer to subscribe to ROS topics published elsewhere on the network, build only the message package:

```bash
source /opt/ros/humble/install/setup.bash
cd /workspace/WAutoVision
colcon build --packages-select wauto_perception_msgs
source install/setup.bash
cd /workspace/WAutoVantage
python3 server/testbed.py
```

You do not need a full `colcon build` of `WAutoVision` just to subscribe to remote topics. `WAutoVantage` only needs `rclpy` plus `wauto_perception_msgs`.

## Streaming Notes

`testbed.py` now starts the FFmpeg publisher itself and sends MPEG-TS directly to MediaMTX on `127.0.0.1:5000`. You do not need the old `stream.sdp` bridge command anymore.

Optional stream tuning before `python3 server/testbed.py`:

```bash
export WAUTOVANTAGE_STREAM_FPS=24
export WAUTOVANTAGE_STREAM_QUEUE_SIZE=1
export WAUTOVANTAGE_STREAM_GOP=12
```

Optional compatibility fallback to the old RTP/SDP mode:

```bash
export WAUTOVANTAGE_STREAM_MODE=rtp
export WAUTOVANTAGE_STREAM_PORT=5004
```
