import chess
import numpy as np
from chess import Move, Board


class ChessEnv:
    def __init__(self, fen: str = chess.STARTING_FEN):
        self.fen = fen
        self.board = Board(self.fen)


    def step(self, move: Move) -> Board:
        self.board.push(move)
        return self.board
    
    def reset(self):
        self.board = Board(self.fen)


    @staticmethod
    def state_to_input(fen):
        # Converts current board to a input vector understood by the network
        board = Board(fen)
        
        is_white_turn = np.ones((8,8)) if board.turn else np.zeros((8,8))

        #print(f'is_white_turn is {is_white_turn}')
        castling = np.asarray([
            np.ones((8, 8)) if board.has_queenside_castling_rights(
                chess.WHITE) else np.zeros((8, 8)),
            np.ones((8, 8)) if board.has_kingside_castling_rights(
                chess.WHITE) else np.zeros((8, 8)),
            np.ones((8, 8)) if board.has_queenside_castling_rights(
                chess.BLACK) else np.zeros((8, 8)),
            np.ones((8, 8)) if board.has_kingside_castling_rights(
                chess.BLACK) else np.zeros((8, 8)),
        ])
        fifty_move = np.ones((8,8)) if board.can_claim_fifty_moves() else np.zeros((8,8))
        pieces = []
        
        for color in chess.COLORS:
            for piece_type in chess.PIECE_TYPES:
                array = np.zeros((8,8))
                for ind in list(board.pieces(piece_type,color)):
                    array[7-ind//8][ind%8]=True 
                pieces.append(array)
                #print(f'for piece {piece_type} array is {array}')
        pieces = np.asarray(pieces)
        #print(f'final pieces are {pieces}')


        en_passant = np.zeros((8,8))
        if board.has_legal_en_passant():
            en_passant[7-int(board.ep_square/8)][board.ep_square%8]=True 
        
        # there are black/white for 6 types of pieces for 2*6, then is white_turn is 1
        # then castling rights are 4 *8*8, then en_passant and fifty move rule are there too,
        # so there are 19 planes of input for this

        r = np.array([is_white_turn,*castling, fifty_move, *pieces, en_passant]).reshape((1,*(8,8,19)))

        del board 
        return r.astype(bool)
    

    def __str__(self):
        return str(self.board)


