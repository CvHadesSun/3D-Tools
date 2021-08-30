from .tools import read_smpl
import socket
import time
from .base_client import BaseSocketClient
import os

def send_rand(client):
    import numpy as np
    N_person = 10
    datas = []
    for i in range(N_person):
        transl = (np.random.rand(1, 3) - 0.5) * 3
        kpts = np.random.rand(25, 4)
        kpts[:, :3] += transl
        data = {
            'id': i,
            'keypoints3d': kpts
        }
        datas.append(data)
    for _ in range(1):
        for i in range(N_person):
            move = (np.random.rand(1, 3) - 0.5) * 0.1
            datas[i]['keypoints3d'][:, :3] += move
        client.send(datas)
        time.sleep(0.005)
    client.close()

def send_dir(client, path, step,smpl):
    from os.path import join
    from glob import glob
    from tqdm import tqdm
    from .tools import read_keypoints3d
    results = sorted(glob(join(path, '*.json')))
    for result in tqdm(results[::step]):
        if smpl:
            data = read_smpl(result)
            client.send_smpl(data)
        else:
            data = read_keypoints3d(result)
            client.send(data)
        time.sleep(0.005)

def main(args):
    if args.host == 'auto':
        args.host = socket.gethostname()
    client = BaseSocketClient(args.host, args.port)

    if args.path is not None and os.path.isdir(args.path):
        send_dir(client, args.path, step=args.step,smpl=args.smpl)
    else:
        send_rand(client)

