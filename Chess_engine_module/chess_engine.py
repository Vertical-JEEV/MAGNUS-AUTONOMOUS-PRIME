import chess
import chess.engine


class ChessEngine:

    try:
        STOCKFISH_PATH = r"Chess_engine_module\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe"
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        print("using faster stockfish")

    except:
        STOCKFISH_PATH = r"Chess_engine_module\stockfish-windows-x86-64-modern\stockfish\stockfish-windows-x86-64-modern.exe"
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        print("using modern stockfish")

    


    def __init__(self, elo, fen_string):

        self.skill_level = self.convert_elo_to_skill_level(elo)
        self.fen_string = fen_string if fen_string is not None else "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"  
      
        self.engine.configure({"Skill Level": self.skill_level})
        self.board = chess.Board(self.fen_string)


    @staticmethod
    def convert_elo_to_skill_level(elo):
        if elo < 800:
            return 0
        elif elo < 1200:
            return int((elo - 800) / 100) + 1
        elif elo < 1600:
            return int((elo - 1200) / 100) + 5
        elif elo < 2000:
            return int((elo - 1600) / 100) + 9
        else:
            return 20
        
      


    def get_best_move(self):
        # get best move from stockfish
        result = self.engine.play(self.board, chess.engine.Limit(time=0.1))
        return result.move
    

    def get_board_fen(self):
        # retunr fen string of board
        return self.board.fen()
    

    def check_valid_fen(self):
        # check if fen string is valid
        return chess.Board(self.fen_string).is_valid()
    

    

    def update_board(self, move):
        # check if its a valid move and then update chessboard
        if move in self.board.legal_moves:
            self.board.push(move)
            # update fen string
            return True
        else:
            return False


    def check_for_checkmate(self):
        # check if checkmate
        return self.board.is_checkmate()
    

    def check_for_stalemate(self):
        # check if stalemate
        return self.board.is_stalemate()
    
    def check_for_draw(self):
        # check if draw
        return self.board.is_insufficient_material()
    

    def check_for_threefold_repetition(self):
        # check if threefold repetition
        return self.board.is_repetition(3)
    
    def __del__(self):
        try:
            self.engine.quit()
        except chess.engine.EngineTerminatedError:
            pass

     






    





    


       



    
    
        





        




        



