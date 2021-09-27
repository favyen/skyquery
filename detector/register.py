import json
import os.path
import sys

# Register an instance of the detector with SkyQuery.

name = sys.argv[1]
data_path = sys.argv[2]
model_path = sys.argv[3]
resize = float(sys.argv[4])

detector_cfg = {
    'Name': 'detector',
    'ModelPath': model_path,
    'Resize': resize,
}
with open(os.path.join(data_path, 'detect', name+'.json'), 'w') as f:
    json.dump(detector_cfg, f)
