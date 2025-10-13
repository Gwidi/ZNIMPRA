#!/bin/bash

docker run -it \
    --name znimpra_container \
    -v /home/gwidon/Documents/ZNIMPRA:/home/ZNIMPRA \
    --env ROS_DOMAIN_ID=11 \
    --runtime=nvidia \
    --env="DISPLAY" \
    --env="NVIDIA_VISIBLE_DEVICES=all" \
    --env="NVIDIA_DRIVER_CAPABILITIES=all" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:ro" \
    --privileged \
    znimpra_ros bash

