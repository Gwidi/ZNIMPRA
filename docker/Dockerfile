FROM ros:noetic

RUN apt-get update && apt-get install -y vim mesa-utils ros-noetic-rviz

RUN echo 'source /opt/ros/noetic/setup.bash' >> /root/.bashrc
WORKDIR /home

ENV ROS_DOMAIN_ID=23