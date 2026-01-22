import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton

if __name__ == '__main__':
    app = QApplication(sys.argv)
    print(sys.argv)

    w = QWidget()
    w.setWindowTitle('The first pyqt')

    btn1 = QPushButton("move")
    btn1.setParent(w)
    btn2 = QPushButton("back")
    btn2.setParent(w)
    btn3 = QPushButton("left")
    btn3.setParent(w)
    btn4 = QPushButton("right")
    btn4.setParent(w)
    w.show()
    app.exec_()