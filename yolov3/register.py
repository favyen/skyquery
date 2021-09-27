import json
import os.path
import sys

# Register an instance of the detector with SkyQuery.

name = sys.argv[1]
data_path = sys.argv[2]
config_path = sys.argv[3]
weights_path = sys.argv[4]

detector_cfg = {
    'Name': 'yolov3',
    'ConfigPath': config_path,
    'ModelPath': weights_path,
}
with open(os.path.join(data_path, 'detect', name+'.json'), 'w') as f:
    json.dump(detector_cfg, f)
