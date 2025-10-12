# WAutoVantage: Real-Time visualization for autonomous vehicle scene understanding 

## Overview

This repository contains a real-time web application that combines [WebTransport](https://github.com/aiortc/aioquic) (via `aioquic`) for low-latency bidirectional data communication and [WebRTC](https://github.com/aiortc/aiortc) (via `aiortc`) for real-time video streaming.

The frontend is a vanilla JavaScript app served by a Python backend. It enables real-time in-browser video playback and two-way data exchange.

<img src="Screencast from 04-30-2025 01-38-51 PM.gif" width="700">

### Features

- Real-time video streaming using WebRTC (H.264 enforced)
- Real-time bidirectional messaging via WebTransport
- Bouncing ball simulation with configurable FPS
- Video processing and rudimentary ball tracking
- Client-server error computation and visualization
- Python backend with multithreading for frame generation
- Comprehensive Python unit tests

### How It Works

1. **Connection Setup**
   - The client sends an SDP offer to the server over a WebTransport bidirectional stream.
   - The server processes the offer and replies with an SDP answer on the same stream.

2. **Frame Generation**
   - A worker thread is spawned on the server to generate video frames (bouncing ball simulation).
   - Frames are encoded as `VideoFrames` using `aiortc` and streamed to the client via WebRTC.

3. **Client-Side Processing**
   - The video is displayed in the browser.
   - Each frame is copied to a hidden canvas to extract pixel data.
   - The red channel is analyzed to compute the centroid of the ball.
   - The detected center is sent back to the server via WebTransport.

4. **Error Computation**
   - The server compares the reported centroid with the true center from the generated frame using the L2 norm.
   - The error is sent back to the client and displayed in real time.

### Testing

This repository includes unit tests for all core Python functions used in the simulation.

---

## Getting Started

**Note:**
- Documentation for Kubernetes deployment usind Minikube can be found [here](./deploy/README.md).
- All the decision decisions, my learnings and challenges faced are documented [here](./docs/DECISIONS.md).

### Prerequisites
**Note:** This application was made and tested on an Ubuntu Machine.

### Web App

- Install [Google Chrome](https://www.google.com/chrome/) (or any other chromium based web browser).

- Install [openssl]

- This project requires using WebTransport. HTTP/3 always operates using TLS, meaning that running a WebTransport over
 HTTP/3 server requires a valid TLS certificate. Chrome/Chromium can be instructed to trust a self-signed
 certificate using command-line flags.  
 Here are step-by-step instructions on how to do that:

   1. Generate a certificate and a private key:
        ```bash
         openssl req -newkey rsa:2048 -nodes -keyout server/certificate.key \
                   -x509 -out server/certificate.pem -subj '/CN=Test Certificate' \
                   -addext "subjectAltName = DNS:localhost"
        ```
   2. Compute the fingerprint of the certificate:
      ```bash
         openssl x509 -pubkey -noout -in server/certificate.pem |
                   openssl rsa -pubin -outform der |
                   openssl dgst -sha256 -binary | base64
        ```
      The result should be a base64-encoded blob that looks like this: `"Gi/HIwdiMcPZo2KBjnstF5kQdLI5bPrYJ8i3Vi6Ybck="`

### Server 
The application's server is available both as a python scripts and docker image.

#### Python
- Setup a virtual environment:

    ```bash
    python3 -m venv .wav
    source .wav/bin/activate
    ```
- Install requirements:

    ```bash
    pip install --no-cache-dir -r requirements.txt
    ```

#### Docker

- Install [Docker](https://docs.docker.com/engine/install/ubuntu/)

- build docker image from the dockerfile: 
    ```bash
    docker build -t http3-media-server .
    ```

---

##  How to Use

### Server

#### Python

- For constant velocity (communication latency testing), run:
    ```bash
    python server/http3_server.py -c server/certificate.pem -k server/certificate.key --fps 60 --grav 0 --vel 1000.0 1000.0 --cor 1
    ```
- For ball simulation with gravity and inellastic collisions, run:
    ```bash
    python server/http3_server.py -c server/certificate.pem -k server/certificate.key --fps 60 --grav 980 --vel 1000.0 1000.0 --cor 0.98` 
    ```
- You can find more configurable arguments in the arg parser [here](./server/http3_server.py)

#### Docker 

- To run server docker container: 
    ```bash
    docker run --rm -p 4433:4433/udp --name local-server-test  http3-media-server
    ```
- To stop container, in a new terminal run: 
    ```bash
    docker stop local-server-test
    ```
- To remove the container, stop the container and run:
    ```bash
    docker rm local-server-test
    ```
- For changing arguments passes to the server, edit the [dockerfile](./dockerfile) and rebuild the image.

### Web App 

- To launch the web app, run: 
    ```bash
    google-chrome   --enable-experimental-web-platform-features   --ignore-certificate-errors-spki-list=ggR1vjmsgl5RdfYS3f5C2nYyZ3LRrjfOyD/Va/JLcXQ=   --origin-to-force-quic-on=localhost:4433   https://localhost:4433/
    ```
    _Note:_ If using your own certificate.pem and certificate.key, use the fingerprint generated above after `-spki-list=`
- Finally to connect to the server and start streaming, click the `Connect to Server` button.

 **You should now see an `orange ball` appear and bounce around on your screen** 
 **The `L2 error` is reported under the video in `red`**

**Note:** The status of the connection is also displayed in the same location of the error in red while no error is being received.

## Tests

### Requirments
- Install dependencies in the venv, if not already installed:
    ```bash
    pip install --no-cache-dir -r requirements.txt
    ```
### Run
- To run the tests:
    ```bash
    python server/unit_tests.py
    ```
