from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QMainWindow
from PyQt6.QtCore import QSize, Qt 

import sys

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # create button
        self.setWindowTitle("Scara ChessRobot")
        
        



app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec()