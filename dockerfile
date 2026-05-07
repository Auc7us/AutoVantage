FROM dustynv/ros:humble-ros-base-l4t-r35.4.1

ARG USERNAME=wautodrive
ARG UID=1000
ARG GID=1000
ARG VIDEO_GID=44
ARG RENDER_GID=103
ARG DIALOUT_GID=20

ENV ROS_DOMAIN_ID=0
ENV ROS_LOCALHOST_ONLY=0
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

RUN rm -f /etc/apt/sources.list.d/ros2*.list && \
    apt-get update && apt-get install -y curl gnupg && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=arm64 signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu focal main" \
    > /etc/apt/sources.list.d/ros2.list

RUN apt-get update && apt-get install -y \
    ffmpeg \
    python3-pip \
    sudo && \
    pip3 install --no-cache-dir pyglet==2.0.0 && \
    rm -rf /var/lib/apt/lists/*

RUN if ! getent group "${GID}" >/dev/null; then \
        groupadd --gid "${GID}" "${USERNAME}"; \
    fi && \
    if ! getent group "${VIDEO_GID}" >/dev/null; then \
        groupadd --gid "${VIDEO_GID}" video; \
    fi && \
    if ! getent group "${RENDER_GID}" >/dev/null; then \
        groupadd --gid "${RENDER_GID}" render; \
    fi && \
    if ! getent group "${DIALOUT_GID}" >/dev/null; then \
        groupadd --gid "${DIALOUT_GID}" dialout; \
    fi && \
    useradd --uid "${UID}" --gid "${GID}" --create-home --shell /bin/bash "${USERNAME}" && \
    usermod -aG sudo,video,render,dialout "${USERNAME}" && \
    echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > "/etc/sudoers.d/${USERNAME}" && \
    chmod 0440 "/etc/sudoers.d/${USERNAME}" && \
    mkdir -p /workspace/WAutoVantage && \
    chown -R "${UID}:${GID}" "/home/${USERNAME}" /workspace

RUN echo 'if [ -f /opt/ros/humble/install/setup.bash ]; then source /opt/ros/humble/install/setup.bash; fi' >> "/home/${USERNAME}/.bashrc" && \
    echo 'if [ -f /opt/ros/humble/setup.bash ]; then source /opt/ros/humble/setup.bash; fi' >> "/home/${USERNAME}/.bashrc"

USER ${USERNAME}
ENV HOME=/home/${USERNAME}
WORKDIR /workspace/WAutoVantage
