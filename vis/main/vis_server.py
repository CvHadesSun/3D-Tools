import yaml
import sys
import os
#
os.environ['KMP_DUPLICATE_LIB_OK']='True'

curr_dir = os.path.dirname(__file__)
root_dir = os.path.join(curr_dir,'../')
#

sys.path.insert(0, os.path.join(root_dir, 'lib'))
sys.path.insert(0, os.path.join(root_dir, 'config'))
#

from tools.cfg import cfg
from socketer.server import VisOpen3DSocket
#
# ## config
# cfg_file = '../config/v3d.yml'
cfg_file = '../config/smpl_scene.yml'
_cfg_file = yaml.load(cfg_file, Loader=yaml.FullLoader)
cfg.merge_from_file(_cfg_file)

test = VisOpen3DSocket(cfg.host,cfg.port,cfg)

while True:
    test.update()
