import yaml
import sys

inFile = sys.argv[1]
#inFile = 'config_1.yaml'

with open(inFile, "r") as f:
    config = yaml.safe_load(f)
