'''to test the geometry visualization.
@simple is simple.@'''
import numpy as np
import open3d as o3d
import sys

sys.path.append('../../config')
from tools.cfg import cfg # config dir

from o3d import create_ground,create_bbox,create_coord,create_mesh
from skelon_model import SkelModel
cfg_file = '../../config/v3d.yml'
cfg.merge_from_file(cfg_file)

rotate = True
def o3d_callback_rotate(vis=None):
    global rotate
    rotate = not rotate
    return rotate

# define the vis
vis = o3d.visualization.VisualizerWithKeyCallback()
# vis.register_key_callback(ord('A'), o3d_callback_rotate)
vis.create_window(window_name='Visualizer', width=cfg.width, height=cfg.height)

# # add geometry
#

scene = cfg.scene.items()
scene_dict = {}
for key, args in scene:
    scene_dict[key] = args
coord_args = scene_dict['easymocap.visualize.o3dwrapper.create_coord']
bg_args = scene_dict['easymocap.visualize.o3dwrapper.create_ground']
bbox_args = scene_dict['easymocap.visualize.o3dwrapper.create_bbox']
#
vis.add_geometry(create_ground(**bg_args))
vis.add_geometry(create_bbox(**bbox_args))
vis.add_geometry(create_coord(**coord_args))

#
model_args = cfg.body_model.args

skelon_model = SkelModel(**model_args)
zero_params = skelon_model.init_params(1)
kpts_3d = np.array([
[0.284, 0.198, 1.615, 0.871],
[0.437, 0.291, 1.432, 0.866],
[0.428, 0.464, 1.439, 0.828],
[0.425, 0.704, 1.291, 0.813],
[0.378, 0.964, 1.174, 0.807],
[0.449, 0.122, 1.431, 0.834],
[0.436, -0.122, 1.292, 0.811],
[0.373, -0.382, 1.201, 0.810],
[0.419, 0.308, 0.858, 0.738],
[0.413, 0.418, 0.860, 0.713],
[0.403, 0.438, 0.462, 0.807],
[0.451, 0.459, 0.070, 0.836],
[0.422, 0.193, 0.858, 0.720],
[0.434, 0.158, 0.463, 0.798],
[0.481, 0.128, 0.073, 0.829],
[0.267, 0.234, 1.648, 0.865],
[0.329, 0.192, 1.651, 0.876],
[0.303, 0.336, 1.641, 0.866],
[0.428, 0.223, 1.644, 0.788],
[0.317, 0.096, 0.030, 0.736],
[0.353, 0.058, 0.030, 0.743],
[0.528, 0.141, 0.028, 0.759],
[0.288, 0.492, 0.020, 0.754],
[0.329, 0.531, 0.024, 0.743],
[0.492, 0.444, 0.025, 0.770]])
zero_params['keypoints3d'] = kpts_3d
vertices = skelon_model(return_verts=True, return_tensor=False, **zero_params)[0]
print(vertices.shape)
mesh = create_mesh(vertices=vertices,faces=skelon_model.faces, colors = None)
vis.add_geometry(mesh)


vis.run()