import importlib
import time
import tabulate
import cv2
import numpy as np
import json
import os
from glob import glob
mkdir = lambda x:os.makedirs(x, exist_ok=True)
mkout = lambda x:mkdir(os.path.dirname(x))

def load_object(module_name, module_args):
    module_path = '.'.join(module_name.split('.')[:-1])
    module = importlib.import_module(module_path)
    name = module_name.split('.')[-1]
    obj = getattr(module, name)(**module_args)
    return obj

class Timer:
    records = {}
    tmp = None

    @classmethod
    def tic(cls):
        cls.tmp = time.time()

    @classmethod
    def toc(cls):
        res = (time.time() - cls.tmp) * 1000
        cls.tmp = None
        return res

    @classmethod
    def report(cls):
        header = ['', 'Time(ms)']
        contents = []
        for key, val in cls.records.items():
            contents.append(['{:20s}'.format(key), '{:.2f}'.format(sum(val) / len(val))])
        print(tabulate.tabulate(contents, header, tablefmt='fancy_grid'))

    def __init__(self, name, silent=False):
        self.name = name
        self.silent = silent
        if name not in Timer.records.keys():
            Timer.records[name] = []

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        end = time.time()
        Timer.records[self.name].append((end - self.start) * 1000)
        if not self.silent:
            t = (end - self.start) * 1000
            if t > 1000:
                print('-> [{:20s}]: {:5.1f}s'.format(self.name, t / 1000))
            elif t > 1e3 * 60 * 60:
                print('-> [{:20s}]: {:5.1f}min'.format(self.name, t / 1e3 / 60))
            else:
                print('-> [{:20s}]: {:5.1f}ms'.format(self.name, (end - self.start) * 1000))


def log(x):
    from datetime import datetime
    time_now = datetime.now().strftime("%m-%d-%H:%M:%S.%f ")
    print(time_now + x)





class BaseCrit:
    def __init__(self, min_conf, min_joints=3) -> None:
        self.min_conf = min_conf
        self.min_joints = min_joints
        self.name = self.__class__.__name__

    def __call__(self, keypoints3d, **kwargs):
        # keypoints3d: (N, 4)
        conf = keypoints3d[..., -1]
        conf[conf < self.min_conf] = 0
        idx = keypoints3d[..., -1] > self.min_conf
        return len(idx) > self.min_joints


class CritWithTorso(BaseCrit):
    def __init__(self, torso_idx, min_conf, **kwargs) -> None:
        super().__init__(min_conf)
        self.idx = torso_idx
        self.min_conf = min_conf

    def __call__(self, keypoints3d, **kwargs) -> bool:
        self.log = '{}'.format(keypoints3d[self.idx, -1])
        return (keypoints3d[self.idx, -1] > self.min_conf).all()


class CritLenTorso(BaseCrit):
    def __init__(self, src, dst, min_torso_length, max_torso_length, min_conf) -> None:
        super().__init__(min_conf)
        self.src = src
        self.dst = dst
        self.min_torso_length = min_torso_length
        self.max_torso_length = max_torso_length

    def __call__(self, keypoints3d, **kwargs):
        """length of torso"""
        # eps = 0.1
        # MIN_TORSO_LENGTH = 0.3
        # MAX_TORSO_LENGTH = 0.8
        if (keypoints3d[[self.src, self.dst], -1] < self.min_conf).all():
            # low confidence, skip
            return True
        length = np.linalg.norm(keypoints3d[self.dst] - keypoints3d[self.src])
        self.log = '{}: {:.3f}'.format(self.name, length)
        if length < self.min_torso_length or length > self.max_torso_length:
            return False
        return True


class CritRange(BaseCrit):
    def __init__(self, minr, maxr, rate_inlier, min_conf) -> None:
        super().__init__(min_conf)
        self.min = minr
        self.max = maxr
        self.rate = rate_inlier

    def __call__(self, keypoints3d, **kwargs):
        idx = keypoints3d[..., -1] > self.min_conf
        k3d = keypoints3d[idx, :3]
        crit = (k3d[:, 0] > self.min[0]) & (k3d[:, 0] < self.max[0]) & \
               (k3d[:, 1] > self.min[1]) & (k3d[:, 1] < self.max[1]) & \
               (k3d[:, 2] > self.min[2]) & (k3d[:, 2] < self.max[2])
        self.log = '{}: {}'.format(self.name, k3d)
        return crit.sum() / crit.shape[0] > self.rate


class CritMinMax(BaseCrit):
    def __init__(self, max_human_length, min_conf) -> None:
        super().__init__(min_conf)
        self.max_human_length = max_human_length

    def __call__(self, keypoints3d, **kwargs):
        idx = keypoints3d[..., -1] > self.min_conf
        k3d = keypoints3d[idx, :3]
        mink = np.min(k3d, axis=0)
        maxk = np.max(k3d, axis=0)
        length = max(np.abs(maxk - mink))
        self.log = '{}: {:.3f}'.format(self.name, length)
        return length < self.max_human_length





def generate_colorbar(N=20, cmap='jet'):
    bar = ((np.arange(N) / (N - 1)) * 255).astype(np.uint8).reshape(-1, 1)
    colorbar = cv2.applyColorMap(bar, cv2.COLORMAP_JET).squeeze()
    if False:
        colorbar = np.clip(colorbar + 64, 0, 255)
    import random
    random.seed(666)
    index = [i for i in range(N)]
    random.shuffle(index)
    rgb = colorbar[index, :]
    rgb = rgb.tolist()
    return rgb


colors_bar_rgb = generate_colorbar(cmap='hsv')

colors_table = {
    'b': [0.65098039, 0.74117647, 0.85882353],
    '_pink': [.9, .7, .7],
    '_mint': [166 / 255., 229 / 255., 204 / 255.],
    '_mint2': [202 / 255., 229 / 255., 223 / 255.],
    '_green': [153 / 255., 216 / 255., 201 / 255.],
    '_green2': [171 / 255., 221 / 255., 164 / 255.],
    'r': [251 / 255., 128 / 255., 114 / 255.],
    '_orange': [253 / 255., 174 / 255., 97 / 255.],
    'y': [250 / 255., 230 / 255., 154 / 255.],
    '_r': [255 / 255, 0, 0],
    'g': [0, 255 / 255, 0],
    '_b': [0, 0, 255 / 255],
    'k': [0, 0, 0],
    '_y': [255 / 255, 255 / 255, 0],
    'purple': [128 / 255, 0, 128 / 255],
    'smap_b': [51 / 255, 153 / 255, 255 / 255],
    'smap_r': [255 / 255, 51 / 255, 153 / 255],
    'smap_b': [51 / 255, 255 / 255, 153 / 255],
}


def get_rgb(index):
    if isinstance(index, int):
        if index == -1:
            return (255, 255, 255)
        if index < -1:
            return (0, 0, 0)
        col = colors_bar_rgb[index % len(colors_bar_rgb)]
    else:
        col = colors_table.get(index, (1, 0, 0))
        col = tuple([int(c * 255) for c in col[::-1]])
    return col


def get_rgb_01(index):
    col = get_rgb(index)
    return [i * 1. / 255 for i in col[:3]]


def plot_point(img, x, y, r, col, pid=-1, font_scale=-1, circle_type=-1):
    cv2.circle(img, (int(x + 0.5), int(y + 0.5)), r, col, circle_type)
    if font_scale == -1:
        font_scale = img.shape[0] / 4000
    if pid != -1:
        cv2.putText(img, '{}'.format(pid), (int(x + 0.5), int(y + 0.5)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, col, 1)


def plot_line(img, pt1, pt2, lw, col):
    cv2.line(img, (int(pt1[0] + 0.5), int(pt1[1] + 0.5)), (int(pt2[0] + 0.5), int(pt2[1] + 0.5)),
             col, lw)


def plot_cross(img, x, y, col, width=-1, lw=-1):
    if lw == -1:
        lw = int(round(img.shape[0] / 1000))
        width = lw * 5
    cv2.line(img, (int(x - width), int(y)), (int(x + width), int(y)), col, lw)
    cv2.line(img, (int(x), int(y - width)), (int(x), int(y + width)), col, lw)


def plot_bbox(img, bbox, pid, vis_id=True):
    # 画bbox: (l, t, r, b)
    x1, y1, x2, y2 = bbox[:4]
    x1 = int(round(x1))
    x2 = int(round(x2))
    y1 = int(round(y1))
    y2 = int(round(y2))
    color = get_rgb(pid)
    lw = max(img.shape[0] // 300, 2)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, lw)
    if vis_id:
        font_scale = img.shape[0] / 1000
        cv2.putText(img, '{}'.format(pid), (x1, y1 + int(25 * font_scale)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color,
                    2)


def plot_keypoints(img, points, pid, config, vis_conf=False, use_limb_color=True, lw=2):
    for ii, (i, j) in enumerate(config['kintree']):
        if i >= len(points) or j >= len(points):
            continue
        pt1, pt2 = points[i], points[j]
        if use_limb_color:
            col = get_rgb(config['colors'][ii])
        else:
            col = get_rgb(pid)
        if pt1[-1] > 0.01 and pt2[-1] > 0.01:
            image = cv2.line(
                img, (int(pt1[0] + 0.5), int(pt1[1] + 0.5)), (int(pt2[0] + 0.5), int(pt2[1] + 0.5)),
                col, lw)
    for i in range(len(points)):
        x, y = points[i][0], points[i][1]
        c = points[i][-1]
        if c > 0.01:
            col = get_rgb(pid)
            cv2.circle(img, (int(x + 0.5), int(y + 0.5)), lw * 2, col, -1)
            if vis_conf:
                cv2.putText(img, '{:.1f}'.format(c), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, col, 2)


def plot_points2d(img, points2d, lines, lw=4, col=(0, 255, 0), putText=True):
    # 将2d点画上去
    if points2d.shape[1] == 2:
        points2d = np.hstack([points2d, np.ones((points2d.shape[0], 1))])
    for i, (x, y, v) in enumerate(points2d):
        if v < 0.01:
            continue
        c = col
        plot_cross(img, x, y, width=10, col=c, lw=lw)
        if putText:
            font_scale = img.shape[0] / 2000
            cv2.putText(img, '{}'.format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, c, 2)
    for i, j in lines:
        if points2d[i][2] < 0.01 or points2d[j][2] < 0.01:
            continue
        plot_line(img, points2d[i], points2d[j], 2, col)


row_col_ = {
    2: (2, 1),
    7: (2, 4),
    8: (2, 4),
    9: (3, 3),
    26: (4, 7)
}


def get_row_col(l):
    if l in row_col_.keys():
        return row_col_[l]
    else:
        from math import sqrt
        row = int(sqrt(l) + 0.5)
        col = int(l / row + 0.5)
        if row * col < l:
            col = col + 1
        if row > col:
            row, col = col, row
        return row, col


def merge(images, row=-1, col=-1, resize=False, ret_range=False, **kwargs):
    if row == -1 and col == -1:
        row, col = get_row_col(len(images))
    height = images[0].shape[0]
    width = images[0].shape[1]
    ret_img = np.zeros((height * row, width * col, images[0].shape[2]), dtype=np.uint8) + 255
    ranges = []
    for i in range(row):
        for j in range(col):
            if i * col + j >= len(images):
                break
            img = images[i * col + j]
            # resize the image size
            img = cv2.resize(img, (width, height))
            ret_img[height * i: height * (i + 1), width * j: width * (j + 1)] = img
            ranges.append((width * j, height * i, width * (j + 1), height * (i + 1)))
    if resize:
        min_height = 3000
        if ret_img.shape[0] > min_height:
            scale = min_height / ret_img.shape[0]
            ret_img = cv2.resize(ret_img, None, fx=scale, fy=scale)
    if ret_range:
        return ret_img, ranges
    return ret_img

def myarray2string(array, separator=', ', fmt='%.3f', indent=8):
    assert len(array.shape) == 2, 'Only support MxN matrix, {}'.format(array.shape)
    blank = ' ' * indent
    res = ['[']
    for i in range(array.shape[0]):
        res.append(blank + '  ' + '[{}]'.format(separator.join([fmt%(d) for d in array[i]])))
        if i != array.shape[0] -1:
            res[-1] += ', '
    res.append(blank + ']')
    return '\r\n'.join(res)

def write_common_results(dumpname=None, results=[], keys=[], fmt='%2.3f'):
    format_out = {'float_kind':lambda x: fmt % x}
    out_text = []
    out_text.append('[\n')
    for idata, data in enumerate(results):
        out_text.append('    {\n')
        output = {}
        output['id'] = data['id']
        for key in keys:
            if key not in data.keys():continue
            # BUG: This function will failed if the rows of the data[key] is too large
            # output[key] = np.array2string(data[key], max_line_width=1000, separator=', ', formatter=format_out)
            output[key] = myarray2string(data[key], separator=', ', fmt=fmt)
        for key in output.keys():
            out_text.append('        \"{}\": {}'.format(key, output[key]))
            if key != keys[-1]:
                out_text.append(',\n')
            else:
                out_text.append('\n')
        out_text.append('    }')
        if idata != len(results) - 1:
            out_text.append(',\n')
        else:
            out_text.append('\n')
    out_text.append(']\n')
    if dumpname is not None:
        mkout(dumpname)
        with open(dumpname, 'w') as f:
            f.writelines(out_text)
    else:
        return ''.join(out_text)
def encode_detect(data):
    res = write_common_results(None, data, ['keypoints3d'])
    res = res.replace('\r', '').replace('\n', '').replace(' ', '')
    return res.encode('ascii')

def encode_smpl(data):
    res = write_common_results(None, data, ['poses', 'shapes', 'expression', 'Rh', 'Th'])
    res = res.replace('\r', '').replace('\n', '').replace(' ', '')
    return res.encode('ascii')

def encode_image(image):
    fourcc = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    #frame을 binary 형태로 변환 jpg로 decoding
    result, img_encode = cv2.imencode('.jpg', image, fourcc)
    data = np.array(img_encode) # numpy array로 안바꿔주면 ERROR
    stringData = data.tostring()
    return stringData

def read_json(path):
    assert os.path.exists(path), path
    with open(path) as f:
        data = json.load(f)
    return data

def read_annot(annotname, mode='body25'):
    data = read_json(annotname)
    if not isinstance(data, list):
        data = data['annots']
    for i in range(len(data)):
        if 'id' not in data[i].keys():
            data[i]['id'] = data[i].pop('personID')
        if 'keypoints2d' in data[i].keys() and 'keypoints' not in data[i].keys():
            data[i]['keypoints'] = data[i].pop('keypoints2d')
        for key in ['bbox', 'keypoints', 'handl2d', 'handr2d', 'face2d']:
            if key not in data[i].keys():continue
            data[i][key] = np.array(data[i][key])
            if key == 'face2d':
                # TODO: Make parameters, 17 is the offset for the eye brows,
                # etc. 51 is the total number of FLAME compatible landmarks
                data[i][key] = data[i][key][17:17+51, :]
        data[i]['bbox'] = data[i]['bbox'][:5]
        if data[i]['bbox'][-1] < 0.001:
            # print('{}/{} bbox conf = 0, may be error'.format(annotname, i))
            data[i]['bbox'][-1] = 1
        if mode == 'body25':
            data[i]['keypoints'] = data[i]['keypoints']
        elif mode == 'body15':
            data[i]['keypoints'] = data[i]['keypoints'][:15, :]
        elif mode in ['handl', 'handr']:
            data[i]['keypoints'] = np.array(data[i][mode+'2d']).astype(np.float32)
            key = 'bbox_'+mode+'2d'
            if key not in data[i].keys():
                data[i]['bbox'] = np.array(get_bbox_from_pose(data[i]['keypoints'])).astype(np.float32)
            else:
                data[i]['bbox'] = data[i]['bbox_'+mode+'2d'][:5]
        elif mode == 'total':
            data[i]['keypoints'] = np.vstack([data[i][key] for key in ['keypoints', 'handl2d', 'handr2d', 'face2d']])
        elif mode == 'bodyhand':
            data[i]['keypoints'] = np.vstack([data[i][key] for key in ['keypoints', 'handl2d', 'handr2d']])
        elif mode == 'bodyhandface':
            data[i]['keypoints'] = np.vstack([data[i][key] for key in ['keypoints', 'handl2d', 'handr2d', 'face2d']])
        conf = data[i]['keypoints'][..., -1]
        conf[conf<0] = 0
    data.sort(key=lambda x:x['id'])
    return data
def get_bbox_from_pose(pose_2d, img=None, rate = 0.1):
    # this function returns bounding box from the 2D pose
    # here use pose_2d[:, -1] instead of pose_2d[:, 2]
    # because when vis reprojection, the result will be (x, y, depth, conf)
    validIdx = pose_2d[:, -1] > 0
    if validIdx.sum() == 0:
        return [0, 0, 100, 100, 0]
    y_min = int(min(pose_2d[validIdx, 1]))
    y_max = int(max(pose_2d[validIdx, 1]))
    x_min = int(min(pose_2d[validIdx, 0]))
    x_max = int(max(pose_2d[validIdx, 0]))
    dx = (x_max - x_min)*rate
    dy = (y_max - y_min)*rate
    # 后面加上类别这些
    bbox = [x_min-dx, y_min-dy, x_max+dx, y_max+dy, 1]
    if img is not None:
        correct_bbox(img, bbox)
    return bbox


def correct_bbox(img, bbox):
    # this function corrects the bbox, which is out of image
    w = img.shape[0]
    h = img.shape[1]
    if bbox[2] <= 0 or bbox[0] >= h or bbox[1] >= w or bbox[3] <= 0:
        bbox[4] = 0
    return bbox

def read_keypoints2d(filename, mode):
    return read_annot(filename, mode)

def read_keypoints3d(filename):
    data = read_json(filename)
    res_ = []
    for d in data:
        pid = d['id'] if 'id' in d.keys() else d['personID']
        pose3d = np.array(d['keypoints3d'])
        if pose3d.shape[0] > 25:
            # 对于有手的情况，把手的根节点赋值成body25上的点
            pose3d[25, :] = pose3d[7, :]
            pose3d[46, :] = pose3d[4, :]
        if pose3d.shape[1] == 3:
            pose3d = np.hstack([pose3d, np.ones((pose3d.shape[0], 1))])
        res_.append({
            'id': pid,
            'keypoints3d': pose3d
        })
    return res_

def read_smpl(filename):
    datas = read_json(filename)
    outputs = []
    for data in datas:
        for key in ['Rh', 'Th', 'poses', 'shapes', 'expression']:
            if key in data.keys():
                data[key] = np.array(data[key])
        # for smplx results
        outputs.append(data)
    return outputs

def read_keypoints3d_a4d(outname):
    res_ = []
    with open(outname, "r") as file:
        lines = file.readlines()
        if len(lines) < 2:
            return res_
        nPerson, nJoints = int(lines[0]), int(lines[1])
        # 只包含每个人的结果
        lines = lines[1:]
        # 每个人的都写了关键点数量
        line_per_person = 1 + 1 + nJoints
        for i in range(nPerson):
            trackId = int(lines[i*line_per_person+1])
            content = ''.join(lines[i*line_per_person+2:i*line_per_person+2+nJoints])
            pose3d = np.fromstring(content, dtype=float, sep=' ').reshape((nJoints, 4))
            # association4d 的关节顺序和正常的定义不一样
            pose3d = pose3d[[4, 1, 5, 9, 13, 6, 10, 14, 0, 2, 7, 11, 3, 8, 12], :]
            res_.append({'id':trackId, 'keypoints3d':np.array(pose3d)})
    return res_

def read_keypoints3d_all(path, key='keypoints3d', pids=[]):
    assert os.path.exists(path), '{} not exists!'.format(path)
    results = {}
    filenames = sorted(glob(os.path.join(path, '*.json')))
    for filename in filenames:
        nf = int(os.path.basename(filename).replace('.json', ''))
        datas = read_keypoints3d(filename)
        for data in datas:
            pid = data['id']
            if len(pids) > 0 and pid not in pids:
                continue
            # 注意 这里没有考虑从哪开始的
            if pid not in results.keys():
                results[pid] = {key: [], 'frames': []}
            results[pid][key].append(data[key])
            results[pid]['frames'].append(nf)
    if key == 'keypoints3d':
        for pid, result in results.items():
            result[key] = np.stack(result[key])
    return results, filenames