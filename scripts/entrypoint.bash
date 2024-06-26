#!/usr/bin/env bash

# Set the DISPLAY environment variable
export DISPLAY=${DISPLAY}
export QT_X11_NO_MITSHM=1

source /opt/ros/noetic/setup.bash
source /root/catkin_ws/devel/setup.bash
source /root/catkin_ws/setup.bash

rosrun learning_machines learning_robobo_controller.py "$@"
