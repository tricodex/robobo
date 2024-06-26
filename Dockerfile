# FROM ros:noetic

# # Making sure our ROS node has ports to connect through.
# # These are the ports specified in `rospy.init_node()` in hardware.py
# EXPOSE 45100
# EXPOSE 45101

# RUN apt-get update -y && apt-get install -y python3 python3-pip git && rm -rf /var/lib/apt/lists/*

# # Install dependencies.
# # These are package requirements for the dependencies.
# # You should add to these if you add python packages that require c libraries to be installed
# RUN apt-get update -y && apt-get install ffmpeg libsm6 libxext6 ros-noetic-opencv-apps dos2unix x11-apps -y && rm -rf /var/lib/apt/lists/*

# # Install X11 and Mesa for OpenGL support
# RUN apt-get update && apt-get install -y \
#     libgl1-mesa-glx \
#     libgl1-mesa-dri \
#     x11-xserver-utils \
#     && rm -rf /var/lib/apt/lists/*

# # The python3 interpreter is already being shilled by ros:noetic, so no need for a venv.
# COPY ./requirements.txt /requirements.txt
# RUN python3 -m pip install -r /requirements.txt && rm /requirements.txt
# RUN python3 -m pip install matplotlib gym stable_baselines3 shimmy tensorboard

# # This cd's into a new `catkin_ws` directory anyone starting the shell will end up in.
# WORKDIR /root/catkin_ws

# # This copies the local catkin_ws into the docker container.
# COPY ./catkin_ws .

# # Set up the environment to actually run the code
# COPY ./scripts/entrypoint.bash ./entrypoint.bash
# COPY ./scripts/setup.bash ./setup.bash

# # Convert the line endings for the Windows users,
# # calling `dos2unix` on all files ending in `.py` or `.bash`
# RUN find . -type f \( -name '*.py' -o -name '*.bash' \) -exec 'dos2unix' -l -- '{}' \; && apt-get --purge remove -y dos2unix && rm -rf /var/lib/apt/lists/*

# # Compile the catkin_ws.
# RUN bash -c 'source /opt/ros/noetic/setup.bash && catkin_make'

# RUN chmod -R u+x /root/catkin_ws/

# # Uncomment these lines and comment out the last line for debugging
# # RUN echo 'source /opt/ros/noetic/setup.bash' >> /root/.bashrc
# # RUN echo 'source /root/catkin_ws/devel/setup.bash' >> /root/.bashrc
# # RUN echo 'source /root/catkin_ws/setup.bash' >> /root/.bashrc

# # Set the DISPLAY environment variable
# ENV QT_X11_NO_MITSHM=1
# ENV DISPLAY=${DISPLAY:-:0}

# ENTRYPOINT ["./entrypoint.bash"]

FROM ros:noetic

EXPOSE 45100
EXPOSE 45101

RUN apt-get update -y && apt-get install -y python3 python3-pip git ffmpeg libsm6 libxext6 ros-noetic-opencv-apps dos2unix x11-apps && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /requirements.txt
RUN python3 -m pip install -r /requirements.txt && rm /requirements.txt
RUN python3 -m pip install matplotlib gym stable_baselines3 shimmy tensorboard

WORKDIR /root/catkin_ws
COPY ./catkin_ws .

COPY ./scripts/entrypoint.bash ./entrypoint.bash
COPY ./scripts/setup.bash ./setup.bash

RUN find . -type f \( -name '*.py' -o -name '*.bash' \) -exec 'dos2unix' -l -- '{}' \; && apt-get --purge remove -y dos2unix && rm -rf /var/lib/apt/lists/*

RUN bash -c 'source /opt/ros/noetic/setup.bash && catkin_make'

RUN chmod -R u+x /root/catkin_ws/

ENV QT_X11_NO_MITSHM=1
ENV DISPLAY=${DISPLAY:-:0}

ENTRYPOINT ["./entrypoint.bash"]
