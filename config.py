import yaml
import sys
import os

os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'


inFile = sys.argv[1]
#inFile = 'config_1.yaml'

with open(inFile, "r") as f:
    config = yaml.safe_load(f)
