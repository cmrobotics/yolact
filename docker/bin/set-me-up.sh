#!/bin/bash
./bin/run-workspace.sh dt
sleep 5
container_id=$(docker ps -aqf "name=yolact-workspace" | tr -d '\n')
echo "Docker container $container_id"
echo "Write absolute path of the folder on your computer, you want the work files folders ( /home/ros and /usr/local ) on the docker machine to be copied into:"
read destination_folder
mkdir -p $destination_folder/home/ros
mkdir -p $destination_folder/usr/local
docker cp $container_id:/home/ros $destination_folder/home/ros
docker cp $container_id:/usr/local $destination_folder/usr/local
docker stop -t 1 $container_id