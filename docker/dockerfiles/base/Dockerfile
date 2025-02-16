FROM nvcr.io/nvidia/pytorch:21.05-py3

ENV DEBIAN_FRONTEND=noninteractive

# Get rid off conda as it conflicts with
# ROS system python
RUN rm -rf /opt/conda

# setup timezone
RUN echo 'Etc/UTC' > /etc/timezone && \
    ln -s /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    apt-get update && \
    apt-get install -q -y --no-install-recommends tzdata && \
    rm -rf /var/lib/apt/lists/*

# install packages
RUN apt-get update && apt-get install -q -y --no-install-recommends \
    dirmngr \
    gnupg2 \
    python3-pip

# You might think "what's the point of starting with a pytorch images if you reinstall it anyways?"
# The problem is that with other images, installing from pip does not work for some odd reason when using CUDA
# The only way to fix this while keeping compatibility with ROS2 is by reinstalling
RUN pip3 install torch torchvision torchaudio

# setup keys
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

# setup sources.list
RUN echo "deb http://packages.ros.org/ros2/ubuntu focal main" > /etc/apt/sources.list.d/ros2-latest.list

# setup environment
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

ENV ROS_DISTRO foxy



ARG BUILD_GROUP_ID
ARG BUILD_USER_ID

ENV ROS_DISTRO foxy

# Create 'ros' user with sudo powers
ARG USERNAME=ros
RUN groupadd --gid $BUILD_GROUP_ID $USERNAME
RUN useradd -s /bin/bash --uid $BUILD_USER_ID --gid $BUILD_GROUP_ID -m $USERNAME
RUN mkdir -p /etc/sudoers.d
RUN echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME
RUN chmod 0440 /etc/sudoers.d/$USERNAME
RUN echo "source /usr/share/bash-completion/completions/git" >> /home/$USERNAME/.bashrc
RUN echo "if [ -f /opt/ros/${ROS_DISTRO}/setup.bash ]; then source /opt/ros/${ROS_DISTRO}/setup.bash; fi" >> /home/$USERNAME/.bashrc

# install ros2 packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git vim tree net-tools \
    python3-colcon-common-extensions \
    python3-colcon-mixin \
    python3-rosdep \
    python3-vcstool \
    python3-pip \
    ros-foxy-desktop=0.9.2-1* \
    ros-foxy-test-msgs \
    graphicsmagick-libmagick-dev-compat \
    libceres-dev \
    ros-foxy-ompl \
    libzmq3-dev \
    libboost-dev \
    jq \
    ros-foxy-gazebo* \
    wget \
    python3-tk \
    curl \
    libssl-dev \
    libusb-1.0-0-dev \
    pkg-config  \
    libgtk-3-dev \
    freeglut3 \
    freeglut3-dev \
    libglfw3-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev

USER $USERNAME
WORKDIR /home/$USERNAME

RUN sudo mkdir /etc/ros
RUN sudo chown -R ros:ros /etc/ros
RUN sudo chown -R ros:ros /usr/local/lib/python3.8/dist-packages

# bootstrap rosdep
RUN rosdep init
RUN rosdep update --rosdistro $ROS_DISTRO

# setup colcon mixin and metadata
RUN colcon mixin add default \
      https://raw.githubusercontent.com/colcon/colcon-mixin-repository/master/index.yaml && \
    colcon mixin update && \
    colcon metadata add default \
      https://raw.githubusercontent.com/colcon/colcon-metadata-repository/master/index.yaml && \
    colcon metadata update

RUN wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > packages.microsoft.gpg \
  && sudo install -o root -g root -m 644 packages.microsoft.gpg /etc/apt/trusted.gpg.d/ \
  && sudo sh -c 'echo "deb [arch=amd64,arm64,armhf signed-by=/etc/apt/trusted.gpg.d/packages.microsoft.gpg] https://packages.microsoft.com/repos/code stable main" > /etc/apt/sources.list.d/vscode.list' \
  && rm -f packages.microsoft.gpg \
  && sudo apt update \
  && sudo apt install --no-install-recommends -y apt-transport-https  code ninja-build
