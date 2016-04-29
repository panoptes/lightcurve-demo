import argparse
import time

from PyQt4.uic import loadUiType
from PyQt4 import QtGui, QtCore

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import (FigureCanvasQTAgg as FigureCanvas)
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection

import seaborn

import numpy as np
import cv2

Ui_MainWindow, QMainWindow = loadUiType('lc-demo.ui')


class Main(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(Main, self).__init__()
        self.setupUi(self)

        self.reset_data()

        # Plot basics
        self.fig1 = Figure()
        self.ax1 = self.fig1.add_subplot(111)

        self.canvas = FigureCanvas(self.fig1)
        self.lcLayout.addWidget(self.canvas)

        # UI callbacks
        self.start_button.clicked.connect(self.start_lightcurve)
        self.clear_button.clicked.connect(self.clear_lightcurve)
        self.quit_button.clicked.connect(self.endCapture)
        self.lc_interval.valueChanged.connect(self.update_interval)
        self._lc_value = self.lc_interval.value()

        self._lc_active = False

        self.fps_box.valueChanged.connect(self.update_webcam)

        # Setup webcam
        self.capture = QtCapture(0, self.radius_slider, self.fps_box)
        self.capture.setParent(self)
        self.capture.setWindowFlags(QtCore.Qt.Tool)

        self.webcamLayout.addWidget(self.capture)

        self.setWindowTitle('LightCurve Demo')
        self.canvas.draw()
        self.show()
        self.startCapture()

    def reset_data(self):
        self._lc_value = self.lc_interval.value()
        self._lc_tick_num = 0
        self._lc_tick = 1000. / self.fps_box.value()
        self._lc_max_tick_num = int((self._lc_value * 1000.) / self._lc_tick)

        self._lc_range = np.arange(0, (self._lc_max_tick_num * self._lc_tick), self._lc_tick) / 1000.
        self._lc_data = np.zeros((3, self._lc_max_tick_num))

    @property
    def getting_lc(self):
        return self._lc_active

    def startCapture(self):
        if not self.capture:
            self.capture = QtCapture(0)
            self.stop_button.clicked.connect(self.capture.stop)
            # self.capture.setFPS(1)
            self.capture.setParent(self)
            self.capture.setWindowFlags(QtCore.Qt.Tool)
        self.capture.show()
        self.start_webcam()

    def update_webcam(self):
        self.stop()
        self.start()

    def update_interval(self):
        if not self.getting_lc:
            # Store
            lc_interval = self.lc_interval.value()
            self._lc_value = lc_interval

            # Update plot
            self.ax1.set_xlim(0, lc_interval)

    def endCapture(self):
        self.capture.deleteLater()
        self.capture = None
        QtCore.QCoreApplication.instance().quit()

    def start_webcam(self):
        self.webcam_timer = QtCore.QTimer()
        self.webcam_timer.timeout.connect(self.webcam_callback)
        self.webcam_timer.start(1000. / self.fps_box.value())

    def stop_webcam(self):
        self.webcam_timer.stop()

    def webcam_callback(self):
        # Capture next webcam frame
        self.img_data = self.capture.get_frame()
        self.plot_values(self.img_data)

    def start_lightcurve(self):
        self._lc_active = True

        self.lc_timer = QtCore.QTimer()
        self.lc_timer.timeout.connect(self.lightcurve_callback)

        self.reset_data()

        self.lc_timer.start(self._lc_tick)

        # UI disable
        self.start_button.setDisabled(True)
        self.lc_interval.setDisabled(True)
        self.fps_box.setDisabled(True)
        self.radius_slider.setDisabled(True)
        self.fps_label.setDisabled(True)
        self.radius_label.setDisabled(True)
        self.seconds_label.setDisabled(True)

    def stop_lightcurve(self):
        self.lc_timer.stop()
        self.clear_button.setEnabled(True)

    def clear_lightcurve(self):
        self._lc_active = False

        # UI enable
        self.start_button.setEnabled(True)
        self.fps_box.setEnabled(True)
        self.radius_slider.setEnabled(True)

        self.clear_button.setDisabled(True)

        self.lc_interval.setEnabled(True)
        self.lc_interval.setValue(self._lc_value)

        self.fps_label.setEnabled(True)
        self.radius_label.setEnabled(True)
        self.seconds_label.setEnabled(True)

        # Clear plot lines
        del self.r_line
        del self.g_line
        del self.b_line

    def lightcurve_callback(self):
        # Which second we are on
        self._lc_sec = np.floor((self._lc_tick_num * self._lc_tick) / 1000.)

        # Show countdown in spinbox
        self.lc_interval.setValue(self._lc_value - self._lc_sec)

        self._lc_tick_num = self._lc_tick_num + 1

        if self._lc_tick_num >= self._lc_max_tick_num:
            self.stop_lightcurve()
        else:
            # Update plot values
            self.plot_values(self.img_data)

    def deleteLater(self):
        self.capture.cap.release()
        super(QtGui.QWidget, self.capture).deleteLater()

    def plot_values(self, masked_data):
        center_r = masked_data[:, :, 0].mean()
        center_g = masked_data[:, :, 1].mean()
        center_b = masked_data[:, :, 2].mean()

        color_max = max(center_r, center_g, center_b)

        r_mean = center_r / color_max * 100.
        g_mean = center_g / color_max * 100.
        b_mean = center_b / color_max * 100.

        if not self.getting_lc:
            self.ax1.clear()
            self._plot_init()

            self.ax1.axhline(r_mean, color='r', ls='dashed')
            self.ax1.axhline(g_mean, color='g', ls='dashed')
            self.ax1.axhline(b_mean, color='b', ls='dashed')
        else:
            if self.lc_timer.isActive():

                self._lc_data[0, self._lc_tick_num] = r_mean
                self._lc_data[1, self._lc_tick_num] = g_mean
                self._lc_data[2, self._lc_tick_num] = b_mean

                if not hasattr(self, 'r_line'):
                    self.r_line, = self.ax1.plot(self._lc_range, self._lc_data[0], color='r')
                    self.g_line, = self.ax1.plot(self._lc_range, self._lc_data[1], color='g')
                    self.b_line, = self.ax1.plot(self._lc_range, self._lc_data[2], color='b')
                else:
                    self.r_line.set_data(self._lc_range, self._lc_data[0])
                    self.g_line.set_data(self._lc_range, self._lc_data[1])
                    self.b_line.set_data(self._lc_range, self._lc_data[2])

        self.fig1.canvas.draw()

    def _plot_init(self):
        # and disable figure-wide autoscale
        # self.ax1.set_autoscale_on(False)

        self.ax1.set_xlim(0, self._lc_value)
        self.ax1.set_ylim(50., 110.)
        self.ax1.set_xlabel("Time [s]")
        self.ax1.set_ylabel("Light [\%]")

        self.fig1.tight_layout()


class QtCapture(QtGui.QWidget):
    def __init__(self, vid_num, radius_slider, fps_box):
        super(QtGui.QWidget, self).__init__()

        self.cap = cv2.VideoCapture(vid_num)

        self.video_frame = QtGui.QLabel()
        lay = QtGui.QVBoxLayout()
        lay.setMargin(0)
        lay.addWidget(self.video_frame)
        self.setLayout(lay)

        ret, frame = self.cap.read()
        self.height, self.width, self.depth = frame.shape

        self._radius_slider = radius_slider
        self._fps_box = fps_box

    @property
    def radius(self):
        return self._radius_slider.value()

    @property
    def fps(self):
        return self._fps_box.value()

    def get_frame(self):
        ret, frame = self.cap.read()

        # My webcam yields frames in BGR format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create circular mask
        circle_img = np.zeros((self.height, self.width), np.uint8)
        cv2.circle(circle_img, (int(self.width / 2), int(self.height / 2)), int(self.radius), 1, -1)

        masked_data = cv2.bitwise_and(frame, frame, mask=circle_img)

        # Show image in window
        show_img = masked_data
        img = QtGui.QImage(show_img, show_img.shape[1], show_img.shape[0], QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(img)
        self.video_frame.setPixmap(pix)

        return masked_data


if __name__ == '__main__':
    import sys

    app = QtGui.QApplication(sys.argv)
    main = Main()
    main.show()
    sys.exit(app.exec_())
