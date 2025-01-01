
import os

# Neural Network Inputs

#equal to black/white 6 pieces, enpassant, which side to move, castling rights for queenside/queenside, repetition
amount_of_input_planes = (2*6+1)+(1+4+1)

#Chess board is 8x8
n = 8

INPUT_SHAPE = (n,n,amount_of_input_planes)




#Neural Network Outputs
# Model will output policy and value
# output_shape[0] is # of possible moves 
#   * 8x8 board = 64 possible actions -> 
# 56 possible queen-like moves, 8 possible knight moves, 9 possible underpromotions
# Total values is 8*8*(56+8+9) = 4672
# output_shape[1] is a scalar value (v)

queen_planes = 56
knight_planes = 8
underpromotion_planes = 9
amount_of_planes = queen_planes+knight_planes+underpromotion_planes

OUTPUT_SHAPE = (8*8*amount_of_planes, 1)


# Neural Network Params

LEARNING_RATE = 0.2
CONVOLUTION_FILTERS = 256
AMOUNT_OF_RESIDUAL_BLOCKS = 19

MODEL_FOLDER = os.environ.get("MODEL_FOLDER",'./models')