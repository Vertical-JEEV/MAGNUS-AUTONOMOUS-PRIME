import chess
import chess.engine





class ChessEngine:

 
    STOCKFISH_PATH = r"Chess_engine_module\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe"
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)


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
    

    def get_score(self):
        # get score from stockfish
        info = self.engine.analyse(self.board, chess.engine.Limit(time=0.1))
        if info["score"].relative.score() is not None:
            score = info["score"].relative.score(mate_score=10000) 
        else:
            score = 100 if info["score"].relative.mate() == 1 else 0
        return score
    
    def get_players_scores(self):
        white_score = self.get_score()
        self.board = self.board.mirror() # Flip the board to get the black score
        black_score = self.get_score()
        self.board = self.board.mirror()  # Flip the board back to its original state
        return white_score, black_score
    

def test_chess_engine():
    # Create a ChessEngine instance
    engine = ChessEngine(1700, None)
    print(engine.board)
  
    #Test get_best_move method
    best_move = engine.get_best_move()
    print(f"Best move: {best_move}")

    # Test update_board method
    is_valid = engine.update_board(best_move)
    print(f"Is the move valid? {is_valid}")
    print(engine.board)

    # Test get_board_fen method
    fen = engine.get_board_fen()
    print(f"FEN: {fen}")

    # Test check_valid_fen method
    is_fen_valid = engine.check_valid_fen()
    print(f"Is the FEN valid? {is_fen_valid}")

    # Test check_for_checkmate and check_for_stalemate methods
    is_checkmate = engine.check_for_checkmate()
    is_stalemate = engine.check_for_stalemate()
    print(f"Is it checkmate? {is_checkmate}")
    print(f"Is it stalemate? {is_stalemate}")

    # Test get_score and get_players_scores methods
    
    white_score, black_score = engine.get_players_scores()
   
    print(f"White score: {white_score}, Black score: {black_score}")

test_chess_engine()

    


    


       



    
    
        





        




        



