import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QRubberBand, QPushButton
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QRect, QSize, pyqtSlot


class ROISelector(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Region of Interest Selector')

        # Load the cat picture and add it to a QLabel widget
        self.cat_label = QLabel(self)
        self.accept_button = QPushButton("Set ROI", self)
        pixmap = QPixmap('cat.jpg')
        self.cat_label.setPixmap(pixmap)
        pixsize = pixmap.size()
        self.cat_label.resize(pixsize)
        self.resize(QSize(pixsize.width(), pixsize.height() + 100))
        self.accept_button.move(
            (pixsize.width() - self.accept_button.size().width())//2, pixsize.height() + 20)
        self.accept_button.clicked.connect(self.accept)

        # Create a QRubberBand object and set its color to red
        self.roi_rubberband = QRubberBand(QRubberBand.Rectangle, self)

    def mousePressEvent(self, event):
        # Set the starting position for the rubber band when the user clicks the mouse
        self.roi_start_pos = event.pos()
        self.roi_rubberband.setGeometry(QRect(self.roi_start_pos, QSize()))
        self.roi_rubberband.show()

    def mouseMoveEvent(self, event):
        # Update the size of the rubber band as the user moves the mouse
        self.roi_rubberband.setGeometry(
            QRect(self.roi_start_pos, event.pos()).normalized())

    def mouseReleaseEvent(self, event):
        # Set the final size of the rubber band when the user releases the mouse button
        self.roi_rubberband.setGeometry(
            QRect(self.roi_start_pos, event.pos()).normalized())

    @pyqtSlot()
    def accept(self):
        print(f"ROI Chosen: {self.roi_rubberband.geometry()}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    roi_selector = ROISelector()
    roi_selector.show()
    sys.exit(app.exec_())
