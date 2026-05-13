#!/usr/bin/env bash
# Build and source the ROS 2 message packages needed by WAutoVantage.
#
# Usage:
#   source scripts/setup_ros_msgs.sh
#
# Optional:
#   WAUTO_VISION_WS=/workspace/WAutoVision source scripts/setup_ros_msgs.sh
#   WAUTO_FORCE_MSG_BUILD=1 source scripts/setup_ros_msgs.sh

_wauto_msgs_is_sourced() {
  [[ "${BASH_SOURCE[0]}" != "$0" ]]
}

_wauto_msgs_script_dir() {
  local source_path="${BASH_SOURCE[0]}"
  while [[ -L "$source_path" ]]; do
    local dir
    dir="$(cd -P "$(dirname "$source_path")" >/dev/null 2>&1 && pwd)"
    source_path="$(readlink "$source_path")"
    [[ "$source_path" != /* ]] && source_path="$dir/$source_path"
  done
  cd -P "$(dirname "$source_path")" >/dev/null 2>&1 && pwd
}

_wauto_msgs_find_vision_ws() {
  if [[ -n "${WAUTO_VISION_WS:-}" ]]; then
    printf '%s\n' "$WAUTO_VISION_WS"
    return
  fi

  local script_dir vantage_dir parent_dir
  script_dir="$(_wauto_msgs_script_dir)"
  vantage_dir="$(cd "$script_dir/.." >/dev/null 2>&1 && pwd)"
  parent_dir="$(dirname "$vantage_dir")"

  for candidate in \
    "$parent_dir/WAutoVision" \
    "/workspace/WAutoVision" \
    "/home/wautodrive/WAutoVision"; do
    if [[ -f "$candidate/src/wauto_perception_msgs/package.xml" || -f "$candidate/src/wauto_localization_msgs/package.xml" ]]; then
      printf '%s\n' "$candidate"
      return
    fi
  done

  return 1
}

_wauto_msgs_have_interface() {
  local interface_name="$1"
  ros2 interface show "$interface_name" >/dev/null 2>&1
}

_wauto_msgs_build_needed_packages() {
  local vision_ws="$1"
  local packages=()

  if [[ "${WAUTO_FORCE_MSG_BUILD:-0}" == "1" ]] || ! _wauto_msgs_have_interface "wauto_perception_msgs/msg/ObjectArray"; then
    packages+=("wauto_perception_msgs")
  fi

  if [[ "${WAUTO_FORCE_MSG_BUILD:-0}" == "1" ]] || ! _wauto_msgs_have_interface "wauto_localization_msgs/msg/GPSVelocity"; then
    packages+=("wauto_localization_msgs")
  fi

  if (( ${#packages[@]} == 0 )); then
    echo "WAuto message interfaces already available; skipping colcon build."
    return
  fi

  echo "Building ROS 2 message packages in $vision_ws: ${packages[*]}"
  (
    cd "$vision_ws"
    colcon build --packages-select "${packages[@]}"
  )
}

_wauto_msgs_main() {
  if ! command -v colcon >/dev/null 2>&1; then
    echo "colcon not found. Run this inside the ROS 2 Docker environment." >&2
    return 1
  fi

  local ros_setup="/opt/ros/${ROS_DISTRO:-humble}/setup.bash"
  if [[ -f "$ros_setup" ]]; then
    # shellcheck disable=SC1090
    source "$ros_setup"
  fi

  local vision_ws
  if ! vision_ws="$(_wauto_msgs_find_vision_ws)"; then
    echo "Could not find WAutoVision. Set WAUTO_VISION_WS=/path/to/WAutoVision and source this script again." >&2
    return 1
  fi

  if [[ ! -d "$vision_ws/src" ]]; then
    echo "WAutoVision workspace is missing src/: $vision_ws" >&2
    return 1
  fi

  local install_setup="$vision_ws/install/setup.bash"
  if [[ -f "$install_setup" ]]; then
    # Source the existing overlay before deciding what to build. This avoids
    # rebuilding slow message packages that are already available.
    # shellcheck disable=SC1090
    source "$install_setup"
  fi

  _wauto_msgs_build_needed_packages "$vision_ws" || return 1

  if [[ ! -f "$install_setup" ]]; then
    echo "Build completed, but install/setup.bash was not found at $install_setup" >&2
    return 1
  fi

  # shellcheck disable=SC1090
  source "$install_setup"
  echo "Sourced $install_setup"

  if ! ros2 interface show wauto_localization_msgs/msg/GPSVelocity >/dev/null; then
    echo "wauto_localization_msgs/msg/GPSVelocity is still unavailable after sourcing $install_setup" >&2
    return 1
  fi
  if ! ros2 interface show wauto_perception_msgs/msg/ObjectArray >/dev/null; then
    echo "wauto_perception_msgs/msg/ObjectArray is still unavailable after sourcing $install_setup" >&2
    return 1
  fi
  echo "WAuto ROS 2 message interfaces are ready."
}

if _wauto_msgs_is_sourced; then
  _wauto_msgs_main "$@"
else
  echo "This script must be sourced so it can update your current shell:"
  echo "  source ${BASH_SOURCE[0]}"
  exit 1
fi
