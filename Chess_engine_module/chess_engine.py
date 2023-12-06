import chess


class ChessEngine:

    def __init__(self, elo, fen_string):

        self.elo = elo
        self.fen_string = fen_string if fen_string is not None else "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"  
        # load stockfish engine for generating moves
        self.engine = chess.engine.SimpleEngine.popen_uci(r"Chess_engine_module\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe")
        # set stockfish elo level
        self.engine.configure({"Skill Level": self.elo})
        # create board
        self.board = chess.Board(self.fen_string)


    def get_best_move(self):
        # get best move from stockfish
        result = self.engine.play(self.board, chess.engine.Limit(time=0.1))
        return result.move
    

    def get_board_fen(self):
        return self.board.fen()
    

    def check_valid_fen(self):
        return chess.Board(self.fen_string).is_valid()
    

    def update_board(self, move):
        # check if its a valid move and then update chessboard
        if move in self.board.legal_moves:
            self.board.push(move)
            return True
        else:
            return False


    def check_for_checkmate(self):
        return self.board.is_checkmate()
    

    def check_for_stalemate(self):
        return self.board.is_stalemate()
    

    def get_score(self):
        # get score from stockfish
        info = self.engine.analyse(self.board, chess.engine.Limit(time=2.0))
        if info["score"].relative.score() is not None:
            score = info["score"].relative.score(mate_score=10000) / 100
        else:
            score = 100 if info["score"].relative.mate() == 1 else 0
        return score
    
    def get_players_scores(self):
        white_score = self.get_score(self.board)
        self.board = self.board.mirror()
        black_score = self.get_score(self.board)
        self.board = self.board.mirror()  # Flip the board back to its original state
        return white_score, black_score
    

    
    


       



    
    
        





        




        



