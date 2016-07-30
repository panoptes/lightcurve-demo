import os
import datetime as dt

from PyQt4.uic import loadUiType
from PyQt4 import QtGui, QtCore

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import (FigureCanvasQTAgg as FigureCanvas)


import seaborn

import numpy as np
import cv2

# Make pretty
seaborn.set()

# Load our UI file created in designer
Ui_MainWindow, QMainWindow = loadUiType('lightcurver.ui')


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
        self.quit_button.clicked.connect(self.quit_application)
        self.actionQuit.triggered.connect(self.quit_application)
        self.lc_interval.valueChanged.connect(self.update_interval)
        self._lc_value = self.lc_interval.value()
        self._lc_active = False

        self._normal_factor = [1., 1., 1., ]

        self.fps_box.valueChanged.connect(self.update_webcam)

        # Setup webcam
        self.capture = QtCapture(0, self.radius_slider, self.fps_box, self.actionColors)
        self.capture.setParent(self)
        self.capture.setWindowFlags(QtCore.Qt.Tool)

        self._image_saved = False
        self._image_path = ''

        self.webcamLayout.addWidget(self.capture)

        self.setWindowTitle('LightCurve Demo')
        self.canvas.draw()
        self.show()
        self.start_application()

    def reset_data(self):
        self._lc_value = self.lc_interval.value()
        self._lc_tick_num = 0
        self._lc_tick = 1000. / self.fps_box.value()
        self._lc_max_tick_num = int((self._lc_value * 1000.) / self._lc_tick)

        self._lc_range = np.arange(0, (self._lc_max_tick_num * self._lc_tick), self._lc_tick) / 1000.
        self._lc_data = np.zeros((3, self._lc_max_tick_num))

##################################################################################################
# Properties
##################################################################################################

    @property
    def getting_lc(self):
        return self._lc_active

##################################################################################################
# Timers
##################################################################################################

    def start_application(self):
        if not self.capture:
            self.capture = QtCapture(0)
            self.stop_button.clicked.connect(self.capture.stop)
            self.capture.setParent(self)
            self.capture.setWindowFlags(QtCore.Qt.Tool)
        self.capture.show()
        self.start_webcam()

    def start_webcam(self):
        self.webcam_timer = QtCore.QTimer()
        self.webcam_timer.timeout.connect(self.webcam_callback)
        self.webcam_timer.start(1000. / self.fps_box.value())

    def stop_webcam(self):
        self.webcam_timer.stop()

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

        # If loop mode
        if self.actionLoop_Mode.isChecked():
            # Change start button to stop button
            self.clear_button.setEnabled(True)
            self.clear_button.setText('&Stop')

        self.show()

    def stop_lightcurve(self):
        self.lc_timer.stop()

        if self.actionLoop_Mode.isChecked():
            self.clear_lightcurve()
            self.start_lightcurve()
        else:
            self.clear_button.setEnabled(True)

            # If we have a pic, show it
            if self.actionSave_Pics and os.path.exists(self._image_path):
                self.stop_webcam()
                img_data = cv2.imread(self._image_path)
                img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
                self.capture._set_image(img_data)

                text, ok = QtGui.QInputDialog.getText(self, 'Save Image?', 'Email Address')
                if ok:
                    if text:
                        os.rename(self._image_path, '/var/panoptes/images/webcam/{}.png'.format(text))
                        self.fig1.savefig('/var/panoptes/images/webcam/{}_plot.png'.format(text))
                    else:
                        self.fig1.savefig('{}_plot.png'.format(self._image_path))
                else:
                    # Remove image
                    os.unlink(self._image_path)

                self.clear_lightcurve()

    def clear_lightcurve(self):
        self._lc_active = False

        self._image_saved = False

        self._image_path = ''
        self._lc_sec = 0.

        # Reset timer
        self.lc_interval.setValue(self._lc_value)

        # Clear plot lines
        try:
            del self.gray_line
            del self.r_line
            del self.g_line
            del self.b_line
        except:
            pass

        if not self.actionLoop_Mode.isChecked():
            # UI enable
            self.start_button.setEnabled(True)
            self.fps_box.setEnabled(True)
            self.radius_slider.setEnabled(True)

            self.clear_button.setDisabled(True)

            self.lc_interval.setEnabled(True)

            self.fps_label.setEnabled(True)
            self.radius_label.setEnabled(True)
            self.seconds_label.setEnabled(True)

            self.start_webcam()


##################################################################################################
# Action Methods
##################################################################################################

    def webcam_callback(self):
        if self.actionSave_Pics.isChecked() and \
                not self._image_saved \
                and hasattr(self, '_lc_sec') \
                and self._lc_sec > self._lc_value / 2:
            self._image_path = "/var/panoptes/images/webcam/{}.png".format(dt.datetime.now().isoformat())
            self._image_saved = True

        # Capture next webcam frame
        self.img_data = self.capture.get_frame(save_frame=self._image_path)
        self.plot_values(self.img_data)

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

    def plot_values(self, masked_data):

        if self.actionColors.isChecked():
            self._plot_color(masked_data)
        else:
            self._plot_gray(masked_data)

##################################################################################################
# UI Methods
##################################################################################################

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

    def quit_application(self):
        self._delete_later()
        self.capture = None
        QtCore.QCoreApplication.instance().quit()

##################################################################################################
# Private Methods
##################################################################################################

    def _plot_gray(self, masked_data):

        if not self.getting_lc:
            self.ax1.clear()
            self._plot_init()

            self.ax1.axhline(100., color='k', ls='dashed')

            # Store the normalization
            self._normal_factor[0] = masked_data.sum()
        else:
            if self.lc_timer.isActive():

                light_value = masked_data.sum() / self._normal_factor[0] * 100.

                self._lc_data[0, self._lc_tick_num] = light_value

                if not hasattr(self, 'gray_line'):
                    self.gray_line, = self.ax1.plot(self._lc_range, self._lc_data[0], 'o', color='gray')
                    self.gray_marker, = self.ax1.plot(self._lc_tick_num, color='gray')
                else:
                    self.gray_line.set_data(self._lc_range, self._lc_data[0])
                    self.gray_marker.set_data(self._lc_range, self._lc_data[0])

        self.fig1.canvas.draw()

    def _plot_color(self, masked_data):

        r_sum = masked_data[:, :, 0].sum()
        g_sum = masked_data[:, :, 1].sum()
        b_sum = masked_data[:, :, 2].sum()

        if not self.getting_lc:
            self.ax1.clear()
            self._plot_init()

            self.ax1.axhline(100., color='r', ls='dashed')
            self.ax1.axhline(100., color='g', ls='dashed')
            self.ax1.axhline(100., color='b', ls='dashed')

            self._normal_factor = [r_sum, g_sum, b_sum]
        else:
            if self.lc_timer.isActive():

                r_value = r_sum / self._normal_factor[0] * 100.
                g_value = g_sum / self._normal_factor[1] * 100.
                b_value = b_sum / self._normal_factor[2] * 100.

                self._lc_data[0, self._lc_tick_num] = r_value
                self._lc_data[1, self._lc_tick_num] = g_value
                self._lc_data[2, self._lc_tick_num] = b_value

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
        self.ax1.set_ylim(50., 105.)
        self.ax1.set_xlabel("Time [s]")
        self.ax1.set_ylabel("Light [\%]")

        self.fig1.tight_layout()

    def _delete_later(self):
        self.capture.cap.release()


class QtCapture(QtGui.QWidget):
    def __init__(self, vid_num, radius_slider, fps_box, actionColors):
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
        self.actionColors = actionColors

    @property
    def radius(self):
        return self._radius_slider.value()

    @property
    def fps(self):
        return self._fps_box.value()

    def get_frame(self, save_frame=''):
        ret, frame = self.cap.read()

        if save_frame is not '':
            cv2.imwrite(save_frame, frame)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create circular mask
        circle_img = np.zeros((self.height, self.width), np.uint8)
        cv2.circle(circle_img, (int(self.width / 2), int(self.height / 2)), int(self.radius), 1, -1)

        masked_data = cv2.bitwise_and(frame, frame, mask=circle_img)

        # Show image in window
        self._set_image(masked_data)

        # If grayscale, convert frame before returning
        if not self.actionColors.isChecked():
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            masked_data = cv2.bitwise_and(frame, frame, mask=circle_img)

        return masked_data

    def _set_image(self, img_data):
        img = QtGui.QImage(img_data, img_data.shape[1], img_data.shape[0], QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(img)
        self.video_frame.setPixmap(pix)


if __name__ == '__main__':
    import sys

    app = QtGui.QApplication(sys.argv)
    main = Main()
    main.show()
    sys.exit(app.exec_())
