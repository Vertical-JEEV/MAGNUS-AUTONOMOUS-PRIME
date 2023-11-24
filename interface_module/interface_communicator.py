import sys
from PyQt5.uic import loadUi
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QDialog, QApplication, QWidget
from PyQt5.QtGui import QPixmap

class MainMenu(QDialog):
    def __init__(self):
        super(MainMenu, self).__init__()
        loadUi("interface_module/main_menu.ui", self)
        self.start_button.clicked.connect(self.start_button_clicked)
        self.load_existing_game_button.clicked.connect(self.load_existing_game_button_clicked)
        self.calibration_setting_button.clicked.connect(self.calibration_setting_button_clicked)

        

    def start_button_clicked(self):
        print("Start button clicked")
        #self.hide()
        #self.game = Game()
        #self.game.show()

    def load_existing_game_button_clicked(self):
        print("Load existing game button clicked")
        #self.hide()
        #self.load_game = LoadGame()
        #self.load_game.show()

    def calibration_setting_button_clicked(self):
        print("Calibration setting button clicked")
        #self.hide()
        #self.calibration_setting = CalibrationSetting()
        #self.calibration_setting.show()






app = QApplication(sys.argv)
mainmenu = MainMenu()
widget = QtWidgets.QStackedWidget()
widget.addWidget(mainmenu)
widget.setFixedHeight(600)
widget.setFixedWidth(800)
widget.show()
try:
    sys.exit(app.exec_())
except:
    print("Exiting")
