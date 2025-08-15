# Agent that plays the chess match

import modelbuilder
import config
from keras.api.keras.models import Model, load_model
import mcts 

import numpy as np
import chess 
import time


class Agent:
    def __init__(self, model_path = None, state = chess.STARTING_FEN):
        self.model = load_model(model_path)
        self.mcts = mcts.MCTS(self, state = state)
    
    def build_model(self):
        model_builder = modelbuilder.ModelBuilder(config.INPUT_SHAPE,config.OUTPUT_SHAPE)
        model = model_builder.build_model()
        return model 
    
    def predict(self,data):
        p,v = self.model(data)
        return p.numpy(),v[0][0]
    
    def save_model(self):
        self.model.save(f"{config.MODEL_FOLDER}/model-{time.time()}.h5")
        