#!/bin/bash

# Delete all stopped containers and unused images
docker system prune

# Delete all images
docker rmi -f $(docker images -aq)