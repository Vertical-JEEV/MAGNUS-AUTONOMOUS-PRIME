import sqlite3 as sql


class Database:

    # the attributes we need to always pass in are:
    def __init__(self):
       
        self.db = self.check_for_existing_db() # connects to a table 
        self.cursor = self.db.cursor() # creates a cursor object to execute commands onto the databse
        self.create_table() # creates a table if it doesn't exist


    def check_for_existing_db(self):
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
            self.cursor.execute(f"SELECT * FROM {self.table_name} WHERE primary_key = ?", (primary_key))
            record = self.cursor.fetchone()
            print("Record read")
            return record
        except sql.Error as e:
            print("Unable to get record")
    

    def insert_record(self, values):
        # insert a record into the table adding corresponding values
        try:
            # create a string of question marks for the values
            values_string = ", ".join(["?" for value in values])
            # create a string of just the fields
            fields_string = ", ".join(self.fields_dict.keys())

            self.cursor.execute(f"INSERT INTO {self.table_name} ({fields_string}) VALUES ({values_string})", values)
            self.db.commit()
            print("Record inserted")
        except sql.Error as e:
            print("Unable to insert record")


    def delete_record(self, primary_key):
        # delete a record from the table using the primary key
        try:
            self.cursor.execute(f"DELETE FROM {self.table_name} WHERE primary_key = ?", (primary_key))
            print("Record deleted")
            self.db.commit()
        except sql.Error as e:
            print("Unable to delete record")



class GamesDatabase(Database):

    def __init__(self):
        
        super().__init__(self.database_name, self.table_name, self.fields_dict)
        self.database_name = "saved_games.db"
        self.table_name = "games"
        # create a dictionary for the table with the fields and their data types:
        self.fields_dict = {"game_name": "TEXT", "FEN_string": "TEXT", "ELO": "INTEGER", "user_score": "INTEGER", "robot_score": "INTEGER", "player_colour": "TEXT", "next_move": "TEXT"}
        

    def check_game_name(self, game_name):
        # check if the game name is valid
        # get all the game names from the database 
        games = self.cursor.execute(f"SELECT game_name FROM {self.table_name}").fetchall()
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
                raise Exception
            
            # create a string of just the fields
            fields_string = ", ".join(self.fields_dict.keys())

            self.cursor.execute(f"INSERT INTO {self.table_name} ({fields_string}) VALUES ({values_string})", values)
            self.db.commit()
            print("Record inserted")
        except sql.Error or Exception:
            print("Unable to insert record")


    def update_record(self, primary_key, value):
        # update a record in the table using the primary key
        try:
            if not self.check_game_name(value):
                # throw an error if the game name is invalid
                raise Exception
            
            self.cursor.execute(f"UPDATE {self.table_name} SET game_name = ? WHERE primary_key = ?", (value, primary_key))
            self.db.commit()
            print("Record updated")
        except sql.Error or Exception:
            print("Unable to update record")







class UserParametersDatabase(Database):

    def __init__(self, database_name, table_name, fields_dict):
        super().__init__(database_name, table_name, fields_dict)
    

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
            # check if the values are floats
            if not self.check_float_values(values):
                raise Exception
            
            # create a string of question marks for the values
            values_string = ", ".join(["?" for value in values])
            # create a string of just the fields
            fields_string = ", ".join(self.fields_dict.keys())

            self.cursor.execute(f"INSERT INTO {self.table_name} ({fields_string}) VALUES ({values_string})", values)
            self.db.commit()
            print("Record inserted")
        except sql.Error as e:
            print("Unable to insert record")



    














    

      








def main():
    database_name = "saved_games.db"
    table_name = "games"
    # create a dictionary for the table with the fields and their data types:

    fields_dict = {"game_name": "TEXT", "FEN_string": "TEXT", "ELO": "INTEGER", "user_score": "INTEGER", "robot_score": "INTEGER", "player_colour": "TEXT", "next_move": "TEXT"}

    
    games_db = Database(database_name, table_name, fields_dict)
    games_db.insert_record(["game2", "helo ", 1300, 34, 3, "Black", "white"])



    # database_name = "user_parameters.db"
    # table_name = "parameters"
    # # create a columns list for the table with the fields:
    # fields = ["chessboard_dimensions", "pawn_height", "knight_height", "bishop_height", "rook_height", "queen_height", "king_height", "intrinsic_parameters", "distortion_parameters", "extrinsic_parameters"]
    # # create a data type list for the table with the fields:
    # data_types = ["TEXT", "REAL", "REAL", "REAL", "REAL", "REAL", "REAL", "TEXT", "TEXT", "TEXT"]

    # fields_with_type = [f"{field} {field_type}" for field, field_type in zip(fields, data_types)]
    # fields_string = ", ".join(fields_with_type)
    
    # parameters_db = Database(database_name, table_name, fields_string)




main()





