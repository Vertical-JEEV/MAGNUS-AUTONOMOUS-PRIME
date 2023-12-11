from Chess_engine_module.chess_engine import ChessEngine
from Hardware_module.comunicate_to_robotic_arm import ArduinoComunicator
from Image_processing_module.corner_detection import ChessboardCornerDetection
import cv2

class ChessGame:


    def __init__(self, elo, fen_string, colour, CHESSBOARD_DIMENSIONS):
        self.user_colour = colour
        self.robo_colour = "white" if colour == "black" else "black"
        self.turn = self.get_turn_from_fen_string(fen_string)
        self.mapping_
        self.chess_engine = ChessEngine(elo, fen_string)
        self.chessboard_recognition = ChessboardCornerDetection(CHESSBOARD_DIMENSIONS)
        self.arduino_comunicator = ArduinoComunicator()




    
    @staticmethod
    def get_turn_from_fen_string(fen_string):
        # split the fen string to get each component
        fen_fields = fen_string.split()
        turn = fen_fields[1]
        if turn == "w":
            return "white"
        return "black"
    

    def whos_turn(self):
        if self.turn == "white":
            self.users_turn()
        else:
            self.robots_turn()


    def users_turn(self):
        # get user move
        user_move = self.get_user_move()
        # update board
        self.chess_engine.update_board(user_move)
        # update turn
        self.turn = "black"
        # check if game is over
        if self.chess_engine.check_game_over():
            self.game_over()
        # check if game is draw
        elif self.chess_engine.check_draw():
            self.game_draw()
        # if not game over or draw then its robots turn
        else:
            self.robots_turn()


    def robots_turn(self):
        # get best move from chess engine
        best_move = self.chess_engine.get_best_move()
        # update board
        self.chess_engine.update_board(best_move)
        # update turn
        self.turn = "white"
        # check if game is over
        if self.chess_engine.check_game_over():
            self.game_over()
        # check if game is draw
        elif self.chess_engine.check_draw():
            self.game_draw()
        # if not game over or draw then its users turn
        else:
            self.users_turn()




chess_game = ChessGame(1000, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1", "white", "32x32")

chess_game.chess_engine.__del__()



        
