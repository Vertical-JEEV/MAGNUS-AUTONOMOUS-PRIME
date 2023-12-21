import sqlite3 as sql


class Database:

    # the attributes we need to always pass in are:
    def __init__(self):
        self.db = self.check_for_existing_db() # connects to a table 
        self.cursor = self.db.cursor() # creates a cursor object to execute commands onto the databse
        self.create_table() # creates a table if it doesn't exist


    def check_for_existing_db(self):
        # check if the database exists, if it doesn't create it
        try:
            db = sql.connect(self.database_name)
            print("Connected to database")
            return db
        except sql.Error as e:
            print("Unable to connect to database")
            
            
    def create_table(self):
        # create a table if it doesn't exist
        try:
            # combine key and value pairs into a string so that ie field name next to its data type
            fields_with_type = [f"{field} {field_type}" for field, field_type in self.fields_dict.items()]
            fields_string = ", ".join(fields_with_type)
            # create the table
            self.cursor.execute(f"CREATE TABLE IF NOT EXISTS {self.table_name} (primary_key INTEGER PRIMARY KEY AUTOINCREMENT, {fields_string})")
        except sql.Error as e:
            print("Unable to create table")
       

    def read_record(self, primary_key):
        # read a record from the table using the primary key
        try:
            self.cursor.execute(f"SELECT * FROM {self.table_name} WHERE primary_key = ?", (primary_key,))
            record = self.cursor.fetchone()
            print("Record read")
            return record
        except sql.Error as e:
            print("Unable to get record")
    

    def delete_record(self, primary_key):
        # delete a record from the table using the primary key
        try:
            self.cursor.execute(f"DELETE FROM {self.table_name} WHERE primary_key = ?", (primary_key,))
            self.db.commit()
            print("Record deleted")
        except sql.Error as e:
            print("Unable to delete record")


class GamesDatabase(Database):


    def __init__(self):
        # initialise the attributes
        self.database_name = r"Database_module\saved_games.db"
        self.table_name = "games"
        # create a dictionary for the table with the fields and their data types:
        self.fields_dict = {"game_name": "TEXT", "FEN_string": "TEXT", "ELO": "INTEGER", "user_score": "INTEGER", "robot_score": "INTEGER", "player_colour": "TEXT", "next_move": "TEXT"}
        super().__init__()
        
        

    def check_game_name(self, game_name):
        # check if the game name is valid
        # get all the game names from the database 
        game_name = game_name.strip()
        games = self.cursor.execute(f"SELECT game_name FROM {self.table_name}").fetchall()
        games = [game[0] for game in games]
        # check if the game name is empty or if it already exists
        if game_name == "" or game_name in games:
            return False
        return True


    def insert_record(self, values):
    # insert a record into the table adding corresponding values
        try:
            # create a string of question marks for the values
            values_string = ", ".join(["?" for value in values])
            # check if the game name is valid
            game_name = values[0]
            if not self.check_game_name(game_name):
                # throw an error if the game name is invalid
                raise sql.Error
            # create a string of just the fields
            fields_string = ", ".join(self.fields_dict.keys())
            self.cursor.execute(f"INSERT INTO {self.table_name} ({fields_string}) VALUES ({values_string})", values)
            self.db.commit()
            print("Record inserted")
        except sql.Error:
            print("Unable to insert record")


    def update_record(self, primary_key, value):
        # update a record in the table using the primary key
        try:
            if not self.check_game_name(value):
                # throw an error if the game name is invalid
                raise sql.Error
            self.cursor.execute(f"UPDATE {self.table_name} SET game_name = ? WHERE primary_key = ?", (value, primary_key))
            self.db.commit()
            print("Record updated")
        except sql.Error:
            print("Unable to update record")


class UserParametersDatabase(Database):


    def __init__(self):
        self.database_name = r"Database_module\user_parameters.db"   
        self.table_name = "parameters"
        self.fields_dict = {"chessboard_dimensions": "TEXT", "pawn_height": "REAL", "knight_height": "REAL", "bishop_height": "REAL", "rook_height": "REAL", "queen_height": "REAL", "king_height": "REAL", "intrinsic_parameters": "TEXT", "distortion_parameters": "TEXT", "extrinsic_parameters": "TEXT"}
        super().__init__()
    

    def check_float_values(self, values):
        # check if the values are floats
        try:
            for value in values:
                float(value)
            return True
        except ValueError:
            return False


    def insert_record(self, values):
        # insert a record into the table adding corresponding values
        try:
            # only get the values which are floats
            values_copy = values[1:7]
            # check if the values are floats
            if not self.check_float_values(values_copy):
                raise sql.Error
            # create a string of question marks for the values
            values_string = ", ".join(["?" for value in values])
            # create a string of just the fields
            fields_string = ", ".join(self.fields_dict.keys())
            self.cursor.execute(f"INSERT INTO {self.table_name} ({fields_string}) VALUES ({values_string})", values)
            self.db.commit()
            print("Record inserted")
        except sql.Error as e:
            print("Unable to insert record")




def test_user_parametres_db():
    user_parameters_db = UserParametersDatabase()
    # inserting 2 records
    user_parameters_db.insert_record(["8x8", 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, "intrinsic_parameters", "distortion_parameters", "extrinsic_parameters"])
    user_parameters_db.insert_record(["8x10", 6, 55, 5, 4, 0.555, 0.53434, "intrinsic_parameters2", "distortion_parameters2", "extrinsic_parameters2"])
    # reading both records
    print(user_parameters_db.read_record(1))
    print(user_parameters_db.read_record(2))
    # deleting the first record
    user_parameters_db.delete_record(1)
    #reading both records, now only the second record should be there
    print(user_parameters_db.read_record(1))
    print(user_parameters_db.read_record(2))


def test_games_db():
    games_db = GamesDatabase()
    # inserting 3 records
    games_db.insert_record(["game1", "FEN_string", 1200, 0, 0, "white", ""])
    games_db.insert_record(["game2", "FEN_string", 1200, 0, 0, "white", ""])
    games_db.insert_record(["game3", "FEN_string", 1200, 0, 0, "white", ""])
    # reading the first record
    print(games_db.read_record(1))
    # updating the first record
    games_db.update_record(1, "game4")
    # reading the first record, now it should be updated
    print(games_db.read_record(1))
    # deleting the second record
    games_db.delete_record(2)
    print(games_db.read_record(1))
    print(games_db.read_record(2))
    print(games_db.read_record(3))


# test_user_parametres_db()
# test_games_db()





    






