#!/bin/bash

ag -l | entr -s 'rsync -r -v -e ssh ./* bruce:~/astar_gpu'

