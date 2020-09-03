import os
import json
import pathlib

from absl import logging
from PyQt5 import QtCore, QtWidgets, QtSvg

import tfne
from .tfnev_welcomewindow_ui import Ui_WelcomeWindow
from .codeepneat import TFNEVCoDeepNEATMainWindow


class TFNEVWelcomeWindow(QtWidgets.QMainWindow, Ui_WelcomeWindow):
    """"""

    def __init__(self, tfne_state_backup_dir_path=None, *args, **kwargs):
        """"""
        super(TFNEVWelcomeWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)

        # Add TFNEV Logo for welcomewindow (placeholder for now)
        self.svg_logo = QtSvg.QSvgWidget(self.centralwidget)
        self.svg_logo.load(os.path.dirname(__file__) + '/tfnev_welcomewindow_logo.svg')
        self.svg_logo.setGeometry(QtCore.QRect(10, 10, 200, 170))

        # Connect signals
        self.button_open_backup_dir.clicked.connect(self.select_tfne_state_backup_folder)

        # Show Window
        self.show()

        # If tfne_state_backup_dir_path parameter already supplied via constructor, open the tfnev mainwindow according
        # to the type of the state backup.
        if tfne_state_backup_dir_path is not None:
            self.open_tfnev_mainwindow(tfne_state_backup_dir_path)

    def select_tfne_state_backup_folder(self):
        """"""
        # Start file dialog to open TFNE Backup directory
        f_dialog = QtWidgets.QFileDialog(self, 'Select TFNE Backup Directory', str(pathlib.Path.home()))
        f_dialog.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, True)
        f_dialog.setFileMode(QtWidgets.QFileDialog.DirectoryOnly)
        if f_dialog.exec_() == QtWidgets.QDialog.Accepted:
            tfne_state_backup_dir_path = f_dialog.selectedFiles()[0]
            logging.debug("Selected TFNE state backup dir path: {}".format(tfne_state_backup_dir_path))
        else:
            logging.debug("Aborted File Dialog")
            return

        # Open the state backup corresponding main window to analyze the backup
        self.open_tfnev_mainwindow(tfne_state_backup_dir_path)

    def open_tfnev_mainwindow(self, tfne_state_backup_dir_path):
        """"""
        # Load and deserialize all populations of the chosen tfne_state_backup_dir_path into the 'tfne_state_backups'
        # dict, associating generation counter to population.
        tfne_state_backups = dict()
        backup_types = set()
        for file_path in os.listdir(tfne_state_backup_dir_path):
            tfne_state_backup_file_path = tfne_state_backup_dir_path + '/' + file_path
            with open(tfne_state_backup_file_path) as tfne_state_backup_file:
                backup_data = json.load(tfne_state_backup_file)
                backup_types.add(backup_data['type'])
                backup_pop = tfne.deserialization.load_population(serialized_population=backup_data['population'])
                backup_generation = backup_pop.generation_counter
                if backup_generation in tfne_state_backups:
                    raise RuntimeError("Selected TFNE state backup contains multiple generation backups of the same"
                                       "generation. Aborting.")
                else:
                    tfne_state_backups[backup_generation] = backup_pop

        # Check if all types of the loaded backups are the same
        if len(backup_types) >= 2:
            raise RuntimeError("Selected TFNE state backup contains backups of 2 or more different kinds of "
                               "algorithms. Aborting.")

        # Open the TFNEV mainwindow corresponding to the type of backup. Pass the loaded and deserialized backup and
        # the parent window as arguments.
        backup_type = backup_types.pop()
        if backup_type == 'CoDeepNEAT':
            self.tfnev_mainwindow = TFNEVCoDeepNEATMainWindow(tfne_state_backups, self)
            self.tfnev_mainwindow.show()
            self.close()
        else:
            raise NotImplementedError("Visualization of the selected TFNE backup type has not yet been implemented. "
                                      "Aborting.")
