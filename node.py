
from chess import Board, Move
from edge import Edge
class Node:
    def __init__(self, state:str):
        # Node represents some board state inside the Monte_Carlo tree

        self.state = state 
        self.turn = Board(state).turn 

        # Maintain an edge struct to keep track of parameters for each action
        self.edges = []
        #Visit count for node
        self.N = 0

        self.value = 0
    
    def step(self, action:Move):
        #Make a move in the current node board's state

        board = Board(self.state)
        board.push(action)
        # Extract the fen of the new board
        new_state = board.fen()
        # Delete created board for memoyr purposes
        del board 
        #return the new state
        return new_state
    
    def is_game_over(self):
        board = Board(self.state)
        
        ans = False 
        if board.is_game_over():
            ans = True 
        del board 
        return ans
    
    def add_child(self, child, action, prior):
        # Add a child node to current node

        edge = Edge(in_node = self, out_node = child, action = action, prior = prior)
        self.edges.append(edge)

        return edge 

    def get_edge(self, action):
        # Gets the edge between current node and child node with action being action

        for edge in self.edges:
            if edge.action==action:
                return edge 
        return None 
    
    