
# module for the interface of the chess game
from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.screenmanager import ScreenManager, Screen, FadeTransition
from kivy.uix.textinput import TextInput
from kivy.uix.popup import Popup
from kivy.utils import get_color_from_hex
from kivy.uix.slider import Slider
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.gridlayout import GridLayout
from kivy.graphics import Color, Rectangle
from kivy.graphics import Line
from kivy.uix.scrollview import ScrollView
from kivy.uix.anchorlayout import AnchorLayout

import re




class MainMenu(Screen):
    def __init__(self, **kwargs):
        super(MainMenu, self).__init__(**kwargs)
        # Create a vertical BoxLayout as the root widget
        self.layout = BoxLayout(orientation='vertical')
        self.add_widget(self.layout)

        # Add title
        self.layout.add_widget(Label(text='MAGNUS AUTONOMOUS PRIME', font_size ='20sp', bold = True, color = get_color_from_hex('#FFFFFF')))
        # Create a horizontal BoxLayout for the start and load buttons
        self.add_buttons_layout()



    def add_buttons_layout(self):
        # Create a horizontal BoxLayout for the start and load buttons
        button_layout = BoxLayout(orientation='horizontal')

        # Add buttons
        start_button = Button(text='Start new game', size_hint_min=(150, 50), color=get_color_from_hex('#FFFFFF'))

        start_button.bind(on_press=self.start_game)
        button_layout.add_widget(start_button)

        load_button = Button(text='Load existing game', size_hint_min=(150, 50), color=get_color_from_hex('#FFFFFF'))
        load_button.bind(on_press=self.load_game)
        button_layout.add_widget(load_button)

        # Add the horizontal BoxLayout to the main layout
        self.layout.add_widget(button_layout)

        calibrate_button = Button(text='Calibrate game', size_hint_min=(150, 50),  color=get_color_from_hex('#FFFFFF'))
        calibrate_button.bind(on_press=self.calibrate_game)
        self.layout.add_widget(calibrate_button)


    def start_game(self, instance):
        # Switch to the start new game menu
        self.manager.current = 'start_new_game_menu'

    def load_game(self, instance):
        # Switch to the load existing game menu
        self.manager.current = 'load_existing_game_menu'

    def calibrate_game(self, instance):
        # Switch to the calibration menu
        self.manager.current = 'calibration_menu'


class StartNewGameMenu(Screen):

    def __init__(self, **kwargs):
        super(StartNewGameMenu, self).__init__(**kwargs)
        self.layout = BoxLayout(orientation='vertical')
        self.add_widget(self.layout)
        # Add title called 'Start new game'
        self.layout.add_widget(Label(text='Start new game', font_size ='20sp', bold = True, color = get_color_from_hex('#FFFFFF')))
        # Create a horizontal BoxLayout
        slider_layout = BoxLayout(orientation='horizontal')
        # Add the slider widget
        self.elo_slider = Slider(min=0, max=3000, value=1500, step=1)
        self.elo_slider.bind(value=self.show_selected_elo_level)
        # Add a label to display the current ELO level
        self.elo_label = Label(text=f'Selected elo level: {int(self.elo_slider.value)}')
        slider_layout.add_widget(self.elo_label)
        # Add the slider to the layout
        slider_layout.add_widget(self.elo_slider)
        # Add the slider layout to the main layout
        self.layout.add_widget(slider_layout)
        # add a toggle button between the slider and the buttons with gaps on either side
        self.toggle = ToggleButton(text='White', size_hint_min=(150, 50), color=get_color_from_hex('#FFFFFF'))
        self.toggle.bind(state=self.set_toggle_state)
        self.layout.add_widget(self.toggle)
        self.add_buttons_layout()


    def show_selected_elo_level(self, instance, value):
        # Update the label text when the slider value is changed
        self.elo_label.text = f'Selected elo level: {int(value)}'

    def set_toggle_state(self,instance, state ):
        # Set the text of the toggle button to either white or black
        if state == 'down':
            instance.text = 'Black'
        else:
            instance.text = 'White'

    
    def add_buttons_layout(self):
        # create a horizontal BoxLayout for the start and load buttons
        button_layout = BoxLayout(orientation='horizontal')

        # Add buttons
        start_button = Button(text='Start', size_hint_min=(150, 50), color=get_color_from_hex('#FFFFFF'))
        start_button.bind(on_press=self.start_game)
        button_layout.add_widget(start_button)


        exit_button = Button(text='Exit', size_hint_min=(150, 50), color=get_color_from_hex('#FFFFFF'))
        exit_button.bind(on_press=self.exit_game)
        button_layout.add_widget(exit_button)


        # Add the horizontal BoxLayout to the main layout
        self.layout.add_widget(button_layout)

    def start_game(self, instance):
        # show slider val and show if white or black
        print(f"elo level is {self.elo_slider.value}")
        if self.toggle.state == 'down':
            print("Black")
        else:
            print("White")

        # Switch to the game window
        self.manager.current = 'game_window'

    def exit_game(self, instance):
        # Switch to the main menu
        self.manager.current = 'main_menu'



class BorderedLabel(Label):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # sets the boardered label to have a white border
        self.border = Line(width=1)
        self.canvas.after.add(Color(1, 1, 1, 1))
        self.canvas.after.add(self.border)
        self.bind(pos=self.update_border, size=self.update_border)

    def update_border(self, *args):
        # updates the border of the label
        self.border.rectangle = (*self.pos, *self.size)

class TableRow(BoxLayout):

    def __init__(self, row_data, menu, selectable=True, **kwargs):
        super(TableRow, self).__init__(**kwargs)
        # Create a background and a rectangle for the row
        self.menu = menu
        self.selectable = selectable
        self.row_data = row_data
        self.orientation = 'horizontal'
        self.background = Color(1, 1, 1, 0)  # Transparent by default
        self.rect = Rectangle(pos=self.pos, size=self.size)
        self.canvas.before.add(self.background)
        self.canvas.before.add(self.rect)
        # Add the data to the row and each row turns to a button
        for item in row_data:
            label = BorderedLabel(text=str(item), halign='left')
            scroll_view = ScrollView()
            scroll_view.add_widget(label)
            self.add_widget(scroll_view)

        self.bind(on_touch_down=self.select_row, pos=self.update_rect, size=self.update_rect)



    def update_rect(self, *args):
        # Update the rectangle when the position or size of the row changes
        self.rect.pos = self.pos
        self.rect.size = self.size

    def select_row(self, instance, touch):
        if self.collide_point(*touch.pos) and self.selectable:
            if self.menu.selected_row:
                self.menu.selected_row.background.a = 0  # Unhighlight the previously selected row
            self.background.a = 0.5  # Highlight the selected row
            self.menu.selected_row = self
            self.menu.update_selected_row_label()  # Update the selected row label
            

class LoadExistingGameMenu(Screen):
    
    def __init__(self, **kwargs):
        super(LoadExistingGameMenu, self).__init__(**kwargs)
        self.layout = BoxLayout(orientation='vertical')
        self.add_widget(self.layout)
        

        # Add title
        self.layout.add_widget(Label(text='Load existing game', font_size ='20sp', bold = True, color = get_color_from_hex('#FFFFFF')))
        self.selected_row_label = Label(text = '')
        self.selected_row = None
        self.layout.add_widget(self.selected_row_label)
        
        # Create a GridLayout for the table with spacing between the cells
        self.table_layout = GridLayout(cols=1)
        self.layout.add_widget(self.table_layout)

        # Add the field names to the table
        field_names = ['Game Name', 'ELO', 'User Score', 'Robot Score', 'User Colour', 'Date']
        self.table_layout.add_widget(TableRow(field_names, self, selectable=False))
        # Fetch the data from the database and add it to the table
        self.populate_table()

        # Create a horizontal BoxLayout for the start and load buttons
        self.add_buttons_layout()

    def select_row(self, row):
        if self.selected_row:
            self.selected_row.background_color = (1, 1, 1, 1)
        row.background_color = (1, 0, 0, 1)
        self.selected_row = row


    def populate_table(self):
        # Fetch the data from the  phony database
      
        data = [['Game 1', 1500, 0, 1, 'White', '01/01/2021'],
                ['Game 2', 1500, 0.5, 0.5, 'White', '01/01/2021'],
                ['Game 3', 1500, 1, 0, 'White', '01/01/2021'],
                ['Game 4', 1500, 0, 1, 'Black', '01/01/2021'],
                ['Game 5', 1500, 0.5, 0.5, 'Black', '01/01/2021'],
                ['Game 6', 1500, 1, 0, 'Black', '01/01/2021']]

        for row in data:
            self.table_layout.add_widget(TableRow(row,self))

    def add_buttons_layout(self):
        # create a horizontal BoxLayout for the start and load buttons
        button_layout = BoxLayout(orientation='horizontal')

        # Add buttons
        load_button = Button(text='Load', size_hint_min=(150, 50), color=get_color_from_hex('#FFFFFF'))
        load_button.bind(on_press=self.load_game)
        button_layout.add_widget(load_button)

        exit_button = Button(text='Exit', size_hint_min=(150, 50), color=get_color_from_hex('#FFFFFF'))
        exit_button.bind(on_press=self.exit_game)
        button_layout.add_widget(exit_button)

        # Add the horizontal BoxLayout to the main layout
        self.layout.add_widget(button_layout)


    def update_selected_row_label(self):
        # Update the selected row label
        if self.selected_row:
            self.selected_row_label.text = f'You selected {self.selected_row.row_data[0]}'

        else:
            self.selected_row_label.text = 'No row selected'

       

    
    def load_game(self, instance):
        # Switch to the game window
        print("game has started")
        if self.selected_row is not None:
            # print  data if the data is not none
            print(f"selected row is {self.selected_row.row_data}")
        
        self.manager.current = 'game_window'

    def exit_game(self, instance):
        self.manager.current = "main_menu"









class CalibrationMenu(Screen):

    def __init__(self, **kwargs):
        super(CalibrationMenu, self).__init__(**kwargs)
        self.layout = BoxLayout(orientation='vertical')
        self.add_widget(self.layout)

        # Add title
        self.layout.add_widget(Label(text='Calibration Menu', font_size ='20sp', bold = True, color = get_color_from_hex('#FFFFFF')))

        # add text boxes for the calibration values
        self.chessboard_dimensions = TextInput(multiline=False, hint_text='Enter Chessboard dimensions')
        self.layout.add_widget(self.chessboard_dimensions)

        self.pawn_height = TextInput(multiline=False, hint_text='Enter Pawn height')
        self.layout.add_widget(self.pawn_height)

        self.rook_height = TextInput(multiline=False, hint_text='Enter Rook height')
        self.layout.add_widget(self.rook_height)

        self.knight_height = TextInput(multiline=False, hint_text='Enter Knight height')
        self.layout.add_widget(self.knight_height)

        self.bishop_height = TextInput(multiline=False, hint_text='Enter Bishop height')
        self.layout.add_widget(self.bishop_height)

        self.queen_height = TextInput(multiline=False, hint_text='Enter Queen height')
        self.layout.add_widget(self.queen_height)

        self.king_height = TextInput(multiline=False, hint_text='Enter King height')
        self.layout.add_widget(self.king_height)
        

        # Create a horizontal BoxLayout for the start and load buttons
        self.add_buttons_layout()


    def add_buttons_layout(self):
        # create a horizontal BoxLayout for the save and exit buttons
        # Add buttons
        save_button = Button(text='Save', size_hint_min=(150, 50), color=get_color_from_hex('#FFFFFF'))
        save_button.bind(on_press=self.save_game)
        self.layout.add_widget(save_button)
        #add a button to exit the calibration menu
        exit_button = Button(text='Exit', size_hint_min=(150, 50), color=get_color_from_hex('#FFFFFF'))
        exit_button.bind(on_press=self.exit_game)
        self.layout.add_widget(exit_button)


    def check_value(self, value):
        # check if the value is a float and greater than 0
        try:
            if float(value) >0:
                return float(value)
        except ValueError:
            return None
        

    def save_game(self, instance):
        # check if the values are valid
        PATTERN = r'^\d+x\d+$'
        chessboard_dimensions = self.chessboard_dimensions.text.strip().lower()
        # makes sure chessboard dimension input has the pattern of a number followed by an x followed by a number
        if not re.match(PATTERN, chessboard_dimensions):
            print('Invalid input')
            width = None
            height = None
        else:

            width, height = chessboard_dimensions.split('x')
            width = self.check_value(width)
            height = self.check_value(height)
        # check if the values are valid
        pawn_height = self.check_value(self.pawn_height.text)
        rook_height = self.check_value(self.rook_height.text)
        knight_height = self.check_value(self.knight_height.text)
        bishop_height = self.check_value(self.bishop_height.text)
        queen_height = self.check_value(self.queen_height.text)
        king_height = self.check_value(self.king_height.text)

        if any(value is None for value in [width, height, pawn_height, rook_height, knight_height, bishop_height, queen_height, king_height]):
            self.show_error_popup("One or more inputs are invalid")
            return

        print(f"chessboard_dimensions are {chessboard_dimensions}")
        print(f"pawn_height is {pawn_height}")
        print(f"rook_height is {rook_height}")
        print(f"knight_height is {knight_height}")
        print(f"bishop_height is {bishop_height}")
        print(f"queen_height is {queen_height}")
        print(f"king_height is {king_height}")

        
        #clear the text boxes, so that they are empty for the next calibration
        self.clear_text_boxes()

    def exit_game(self, instance):
        #clear text boxes
        self.clear_text_boxes()
        # Switch to the main menu
        self.manager.current = 'main_menu'


    def clear_text_boxes(self):
        # clear the text boxes
        self.chessboard_dimensions.text = ''
        self.pawn_height.text = ''
        self.rook_height.text = ''
        self.knight_height.text = ''
        self.bishop_height.text = ''
        self.queen_height.text = ''
        self.king_height.text = ''


    def show_error_popup(self,error_msg):
        # Create a custom layout for the content
        content = BoxLayout(orientation='vertical')
        content.add_widget(Label(text=error_msg))
        button = Button(text='Close')
        content.add_widget(button)
        # Create the pop-up
        popup = Popup(title='Error', content=content,size_hint=(None, None), size=(400, 200), auto_dismiss=False)
        # Bind the on_press event of the button to the dismiss method of the pop-up
        button.bind(on_press=popup.dismiss)
        # Open the pop-up
        popup.open()




class GameWindow(Screen):
    def __init__(self, **kwargs):
        super(GameWindow, self).__init__(**kwargs)
        self.fen_string = None

        # Create an AnchorLayout as the root widget
        self.root_layout = AnchorLayout()
        self.add_widget(self.root_layout)

        # Create a GridLayout for the scores and chessboard with 3 columns
        self.grid_layout = GridLayout(cols=3)
        self.root_layout.add_widget(self.grid_layout)

        

        # Create a BoxLayout for the chessboard
        self.chessboard_layout = BoxLayout(orientation='vertical')
        self.grid_layout.add_widget(self.chessboard_layout)

        # Add the chessboard layout to the chessboard layout
        self.layout = GridLayout(cols=8)
        self.chessboard_layout.add_widget(self.layout)

        #

        # Create a BoxLayout for the buttons at the bottom
        self.button_layout = BoxLayout(orientation='horizontal', size_hint=(1, None), height=40)
        self.root_layout.anchor_y = 'bottom'
        self.root_layout.add_widget(self.button_layout)

        #Clock.schedule_once(lambda _: self.draw_chessboard(self, None))

        # Add the buttons
        #self.add_buttons_layout()

        # Draw the chessboard
        self.layout.bind(size=self.draw_chessboard, pos=self.draw_chessboard)

    def draw_chessboard(self, instance, value):
       # capital letters are White, lower case is black
        piece_img_dict = { 
        'P': r'interface_module\images\white_pieces\wP.png',
        'N': r'interface_module\images\white_pieces\wN.png',
        'B': r'interface_module\images\white_pieces\wB.png',
        'R': r'interface_module\images\white_pieces\wR.png',
        'Q': r'interface_module\images\white_pieces\wQ.png',
        'K': r'interface_module\images\white_pieces\wK.png',
        'p': r'interface_module\images\black_pieces\bP.png',
        'n': r'interface_module\images\black_pieces\bN.png',
        'b': r'interface_module\images\black_pieces\bB.png',
        'r': r'interface_module\images\black_pieces\bR.png',
        'q': r'interface_module\images\black_pieces\bQ.png',
        'k': r'interface_module\images\black_pieces\bK.png',
        }
        self.layout.canvas.clear()
        padding = 0 # adjust this value to change the padding
        board_size = min(self.layout.width, self.layout.height - self.button_layout.height) - 2 * padding
        fen_rows = self.fen_string.split('/') if self.fen_string else "8/8/8/8/8/8/8/8"

        with self.layout.canvas:
            # Draw the squares
            for i in range(8):
                for j in range(8):
                    if (i + j) % 2 == 0:
                        Color(0.44, 0.26, 0.08) # brown
                    else:
                        Color(1, 1, 1) # white  
                    square_size = (board_size / 8, board_size / 8)
                    # Calculate the square position
                    square_pos = ((self.chessboard_layout.width / 2) - 4 * square_size[0] + j * square_size[0], 
                                (self.chessboard_layout.height / 2) - 4 * square_size[1] + i * square_size[1] + self.button_layout.height / 2)



                    Rectangle(pos=square_pos, size=square_size)
                    
                    # Draw the pieces
                    fen_row = fen_rows[7-i]
                    expanded_fen_row = ''
                    # expandthe row if there are any numbers
                    for char in fen_row:
                        if char.isdigit():
                            expanded_fen_row += ' ' * int(char)
                        else:
                            expanded_fen_row += char
                    # if the index is less than the length of the expanded fen row then get the piece
                    if j < len(expanded_fen_row):
                        piece = expanded_fen_row[j]
                        if piece in piece_img_dict:
                            Color(1,1,1)
                            piece_img_path = piece_img_dict[piece]
                            piece_size = (board_size / 8, board_size / 8)
                            piece_pos = ((self.chessboard_layout.width / 2) - 4 * piece_size[0] + j * piece_size[0], (self.chessboard_layout.height / 2) - 4 * piece_size[1] + i * piece_size[1] + self.button_layout.height / 2)
                            Rectangle(source=piece_img_path, pos=piece_pos, size=piece_size)


   

    


    


    

    


  

class MyApp(App):
    def build(self):
        # add windows to the screen manager 
        sm = ScreenManager(transition = FadeTransition())
        sm.add_widget(MainMenu(name='main_menu'))
        sm.add_widget(CalibrationMenu(name='calibration_menu'))
        sm.add_widget(StartNewGameMenu(name='start_new_game_menu'))
        sm.add_widget(LoadExistingGameMenu(name='load_existing_game_menu'))
        sm.add_widget(GameWindow(name='game_window'))

        return sm

if __name__ == '__main__':
    MyApp().run()
