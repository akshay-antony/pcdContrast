#!/bin/bash

# Python command to run
PYTHON_COMMAND="python3 trimesh_pcd.py"

# Infinite loop
while true
do
    echo "Starting Python script..."
    $PYTHON_COMMAND
    echo "Python script stopped. Restarting in 2 seconds..."
    sleep 2
done
