from chess import *
import sys
import os
# Add the parent directory to the Python path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

import modelbuilder
import config

builder = modelbuilder.ModelBuilder(input_shape=config.INPUT_SHAPE,output_shape=config.OUTPUT_SHAPE)

model = builder.build_model()
print(f'model is {model}')