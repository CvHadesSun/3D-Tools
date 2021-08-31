import argparse
import sys
import os
import socket
#
curr_dir = os.path.dirname(__file__)
root_dir = os.path.join(curr_dir,'../')
#
sys.path.insert(0, os.path.join(root_dir, 'lib'))
sys.path.insert(0, os.path.join(root_dir, 'config'))

from socketer.client import main

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=9999)
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--path', type=str, default='../data/smplx/smpl')
    # parser.add_argument('--smpl', action='store_true')
    parser.add_argument('--smpl', type=bool,default=True)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    main(args)