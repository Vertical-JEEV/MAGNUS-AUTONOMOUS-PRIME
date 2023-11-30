# import kivy
from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.screenmanager import ScreenManager, Screen, FadeTransition
from kivy.uix.textinput import TextInput
from kivy.uix.popup import Popup
from kivy.utils import get_color_from_hex
import re





class MainMenu(Screen):
    def __init__(self, **kwargs):
        super(MainMenu, self).__init__(**kwargs)
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
        print('Starting new game')

    def load_game(self, instance):
        print('Loading existing game')

    def calibrate_game(self, instance):
        # Switch to the calibration menu
        self.manager.current = 'calibration_menu'


class StartGameMenu(Screen):
    pass




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
        try:
            return float(value)
        except ValueError:
            return None
        

    def save_game(self, instance):
        PATTERN = r'^\d+x\d+$'
        chessboard_dimensions = self.chessboard_dimensions.text
        if not re.match(PATTERN, chessboard_dimensions.strip()):
            print('Invalid input')
            width = None
            height = None
        else:
            width, height = self.chessboard_dimensions.text.split('x').strip()
            width = self.check_value(width)
            height = self.check_value(height)

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
        self.chessboard_dimensions.text = ''
        self.pawn_height.text = ''
        self.rook_height.text = ''
        self.knight_height.text = ''
        self.bishop_height.text = ''
        self.queen_height.text = ''
        self.king_height.text = ''


    def show_error_popup(self,error_msg):
        popup = Popup(title = "Error", content = Label(text = error_msg), size_hint = (None,None), size = (400,200))
        popup.open()






  

class MyApp(App):
    def build(self):
        sm = ScreenManager(transition = FadeTransition())
        sm.add_widget(MainMenu(name='main_menu'))
        sm.add_widget(CalibrationMenu(name='calibration_menu'))
        return sm

if __name__ == '__main__':
    MyApp().run()



