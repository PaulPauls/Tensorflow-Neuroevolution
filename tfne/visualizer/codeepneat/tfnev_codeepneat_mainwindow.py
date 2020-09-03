import os
import tempfile
import webbrowser

import matplotlib.pyplot as plt
from PyQt5 import QtCore, QtWidgets, QtGui, QtSvg

from .tfnev_codeepneat_mainwindow_ui import Ui_MainWindow


class TFNEVCoDeepNEATMainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    """"""

    def __init__(self, tfne_state_backups, parent_window, *args, **kwargs):
        """"""
        super(TFNEVCoDeepNEATMainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)

        # Register parameters
        self.tfne_state_backups = tfne_state_backups
        self.parent_window = parent_window

        # Get local temporary directory to save matplotlib created files into
        self.temp_dir = tempfile.gettempdir()

        # Initialize image widgets of all views to refer to them when changing views in order not to create new widgets
        # each time and stack them infinitely
        self.ga_genome_visualization_image = QtSvg.QSvgWidget(self.ga_widget_genome_visualization)
        self.ga_genome_visualization_image.setMaximumHeight(460)
        self.ga_genome_visualization_image.setMaximumWidth(460)
        self.mba_bp_visualization_image = QtSvg.QSvgWidget(self.mba_widget_blueprint_visualization)
        self.mba_bp_visualization_image.setMaximumHeight(460)
        self.mba_bp_visualization_image.setMaximumWidth(460)
        self.svg_best_genome_analysis = QtSvg.QSvgWidget(self.widget_genome_analysis)
        self.svg_best_genome_analysis.setGeometry(QtCore.QRect(10, 0, 440, 320))
        self.svg_mod_spec_fitness_analysis = QtSvg.QSvgWidget(self.widget_mod_species_analysis)
        self.svg_mod_spec_fitness_analysis.setGeometry(QtCore.QRect(10, 0, 920, 660))
        self.svg_bp_spec_fitness_analysis = QtSvg.QSvgWidget(self.widget_bp_species_analysis)
        self.svg_bp_spec_fitness_analysis.setGeometry(QtCore.QRect(10, 0, 920, 660))

        # Declare global variables relevant throughout different events
        self.mba_selected_gen = None

        # Create module species and blueprint species analysis dicts
        generations = sorted(self.tfne_state_backups.keys())
        self.mod_spec_analysis_dict = dict()
        self.bp_spec_analysis_dict = dict()
        for gen in generations:
            for mod_spec_id, spec_fit_hist in self.tfne_state_backups[gen].mod_species_fitness_history.items():
                if mod_spec_id in self.mod_spec_analysis_dict:
                    self.mod_spec_analysis_dict[mod_spec_id]['x'].append(gen)
                    self.mod_spec_analysis_dict[mod_spec_id]['y'].append(spec_fit_hist[gen])
                else:
                    self.mod_spec_analysis_dict[mod_spec_id] = dict()
                    self.mod_spec_analysis_dict[mod_spec_id]['x'] = [gen]
                    self.mod_spec_analysis_dict[mod_spec_id]['y'] = [spec_fit_hist[gen]]
            for bp_spec_id, spec_fit_hist in self.tfne_state_backups[gen].bp_species_fitness_history.items():
                if bp_spec_id in self.bp_spec_analysis_dict:
                    self.bp_spec_analysis_dict[bp_spec_id]['x'].append(gen)
                    self.bp_spec_analysis_dict[bp_spec_id]['y'].append(spec_fit_hist[gen])
                else:
                    self.bp_spec_analysis_dict[bp_spec_id] = dict()
                    self.bp_spec_analysis_dict[bp_spec_id]['x'] = [gen]
                    self.bp_spec_analysis_dict[bp_spec_id]['y'] = [spec_fit_hist[gen]]

        # Set up sidebar buttons to select the type of analysis. Default activate genome analysis mode
        self.svg_btn_genome_analysis = QtSvg.QSvgWidget(self.centralwidget)
        self.svg_btn_mod_bp_analysis = QtSvg.QSvgWidget(self.centralwidget)
        self.svg_btn_mod_spec_analysis = QtSvg.QSvgWidget(self.centralwidget)
        self.svg_btn_bp_spec_analysis = QtSvg.QSvgWidget(self.centralwidget)
        image_base_dir = os.path.dirname(__file__)
        self.svg_btn_genome_analysis.load(image_base_dir + '/genome_analysis_icon.svg')
        self.svg_btn_mod_bp_analysis.load(image_base_dir + '/module_blueprint_analysis_icon.svg')
        self.svg_btn_mod_spec_analysis.load(image_base_dir + '/module_species_analysis_icon.svg')
        self.svg_btn_bp_spec_analysis.load(image_base_dir + '/blueprint_species_analysis_icon.svg')
        self.svg_btn_genome_analysis.setGeometry(QtCore.QRect(0, 0, 60, 170))
        self.svg_btn_mod_bp_analysis.setGeometry(QtCore.QRect(0, 170, 60, 170))
        self.svg_btn_mod_spec_analysis.setGeometry(QtCore.QRect(0, 340, 60, 170))
        self.svg_btn_bp_spec_analysis.setGeometry(QtCore.QRect(0, 510, 60, 170))
        self.event_svg_btn_genome_analysis()

        # Set layouts
        ga_widget_genome_visualization_layout = QtWidgets.QVBoxLayout(self.ga_widget_genome_visualization)
        ga_widget_genome_visualization_layout.setAlignment(QtCore.Qt.AlignCenter)
        ga_widget_genome_visualization_layout.addWidget(self.ga_genome_visualization_image)
        mba_widget_blueprint_visualization_layout = QtWidgets.QVBoxLayout(self.mba_widget_blueprint_visualization)
        mba_widget_blueprint_visualization_layout.setAlignment(QtCore.Qt.AlignCenter)
        mba_widget_blueprint_visualization_layout.addWidget(self.mba_bp_visualization_image)

        # Connect Signals
        self.action_documentation.triggered.connect(self.action_documentation_triggered)
        self.action_close.triggered.connect(self.action_close_triggered)
        self.action_exit.triggered.connect(self.action_exit_triggered)
        self.ga_list_generations.itemClicked.connect(self.click_ga_list_generations)
        self.mba_list_generations.itemClicked.connect(self.click_mba_list_generations)
        self.mba_list_members.itemClicked.connect(self.click_mba_list_members)
        self.svg_btn_genome_analysis.mousePressEvent = self.event_svg_btn_genome_analysis
        self.svg_btn_mod_bp_analysis.mousePressEvent = self.event_svg_btn_module_blueprint_analysis
        self.svg_btn_mod_spec_analysis.mousePressEvent = self.event_svg_btn_module_species_analysis
        self.svg_btn_bp_spec_analysis.mousePressEvent = self.event_svg_btn_blueprint_species_analysis

    def event_svg_btn_genome_analysis(self, *args, **kwargs):
        """"""
        # Set Color focus on Genome Analysis
        svg_btn_genome_analysis_bg = QtGui.QPalette(self.svg_btn_genome_analysis.palette())
        svg_btn_genome_analysis_bg.setColor(QtGui.QPalette.Window, QtGui.QColor('gray'))
        self.svg_btn_genome_analysis.setPalette(svg_btn_genome_analysis_bg)
        self.svg_btn_genome_analysis.setAutoFillBackground(True)
        svg_btn_mod_bp_analysis_bg = QtGui.QPalette(self.svg_btn_mod_bp_analysis.palette())
        svg_btn_mod_bp_analysis_bg.setColor(QtGui.QPalette.Window, QtGui.QColor('darkGray'))
        self.svg_btn_mod_bp_analysis.setPalette(svg_btn_mod_bp_analysis_bg)
        self.svg_btn_mod_bp_analysis.setAutoFillBackground(True)
        svg_btn_mod_spec_analysis_bg = QtGui.QPalette(self.svg_btn_mod_spec_analysis.palette())
        svg_btn_mod_spec_analysis_bg.setColor(QtGui.QPalette.Window, QtGui.QColor('darkGray'))
        self.svg_btn_mod_spec_analysis.setPalette(svg_btn_mod_spec_analysis_bg)
        self.svg_btn_mod_spec_analysis.setAutoFillBackground(True)
        svg_btn_bp_spec_analysis_bg = QtGui.QPalette(self.svg_btn_bp_spec_analysis.palette())
        svg_btn_bp_spec_analysis_bg.setColor(QtGui.QPalette.Window, QtGui.QColor('darkGray'))
        self.svg_btn_bp_spec_analysis.setPalette(svg_btn_bp_spec_analysis_bg)
        self.svg_btn_bp_spec_analysis.setAutoFillBackground(True)

        # Activate genome analysis mode, deactivate other modes
        self.widget_genome_analysis.show()
        self.widget_mod_bp_analysis.close()
        self.widget_mod_species_analysis.close()
        self.widget_bp_species_analysis.close()

        # Create graph showing the best genome fitness over the generations and display it
        x_axis_generations = sorted(self.tfne_state_backups.keys())
        y_axis_fitness = list()
        for gen in x_axis_generations:
            y_axis_fitness.append(self.tfne_state_backups[gen].best_fitness)
        plt.clf()
        plt.plot(x_axis_generations, y_axis_fitness)
        plt.ylabel('best fitness')
        plt.xlabel('generation')
        plt.savefig(self.temp_dir + '/best_genome_fitness_analysis.svg')
        self.svg_best_genome_analysis.load(self.temp_dir + '/best_genome_fitness_analysis.svg')
        self.svg_best_genome_analysis.show()

        # Create strings that are displayed in the list of best genomes
        best_genome_in_gen_list = list()
        for gen in x_axis_generations:
            best_genome_id = self.tfne_state_backups[gen].best_genome.get_id()
            best_genome_in_gen_list.append(f'Generation {gen} - Genome #{best_genome_id}')
        self.ga_list_generations.clear()
        self.ga_list_generations.addItems(best_genome_in_gen_list)

    def event_svg_btn_module_blueprint_analysis(self, *args, **kwargs):
        """"""
        # Set Color focus on Genome Analysis
        svg_btn_genome_analysis_bg = QtGui.QPalette(self.svg_btn_genome_analysis.palette())
        svg_btn_genome_analysis_bg.setColor(QtGui.QPalette.Window, QtGui.QColor('darkGray'))
        self.svg_btn_genome_analysis.setPalette(svg_btn_genome_analysis_bg)
        self.svg_btn_genome_analysis.setAutoFillBackground(True)
        svg_btn_mod_bp_analysis_bg = QtGui.QPalette(self.svg_btn_mod_bp_analysis.palette())
        svg_btn_mod_bp_analysis_bg.setColor(QtGui.QPalette.Window, QtGui.QColor('gray'))
        self.svg_btn_mod_bp_analysis.setPalette(svg_btn_mod_bp_analysis_bg)
        self.svg_btn_mod_bp_analysis.setAutoFillBackground(True)
        svg_btn_mod_spec_analysis_bg = QtGui.QPalette(self.svg_btn_mod_spec_analysis.palette())
        svg_btn_mod_spec_analysis_bg.setColor(QtGui.QPalette.Window, QtGui.QColor('darkGray'))
        self.svg_btn_mod_spec_analysis.setPalette(svg_btn_mod_spec_analysis_bg)
        self.svg_btn_mod_spec_analysis.setAutoFillBackground(True)
        svg_btn_bp_spec_analysis_bg = QtGui.QPalette(self.svg_btn_bp_spec_analysis.palette())
        svg_btn_bp_spec_analysis_bg.setColor(QtGui.QPalette.Window, QtGui.QColor('darkGray'))
        self.svg_btn_bp_spec_analysis.setPalette(svg_btn_bp_spec_analysis_bg)
        self.svg_btn_bp_spec_analysis.setAutoFillBackground(True)

        # Activate genome analysis mode, deactivate other modes
        self.widget_genome_analysis.close()
        self.widget_mod_bp_analysis.show()
        self.widget_mod_species_analysis.close()
        self.widget_bp_species_analysis.close()

        # Close both module and blueprint analysis widgets of the right side, opening them only if specific module or
        # blueprint has been selected
        self.mba_widget_blueprint.close()
        self.mba_widget_module.close()

        # Create strings that are displayed in the list of generations
        generations_list = list()
        for gen in sorted(self.tfne_state_backups.keys()):
            generations_list.append(f'Generation {gen}')
        self.mba_list_generations.clear()
        self.mba_list_generations.addItems(generations_list)

    def event_svg_btn_module_species_analysis(self, *args, **kwargs):
        """"""
        # Set Color focus on Genome Analysis
        svg_btn_genome_analysis_bg = QtGui.QPalette(self.svg_btn_genome_analysis.palette())
        svg_btn_genome_analysis_bg.setColor(QtGui.QPalette.Window, QtGui.QColor('darkGray'))
        self.svg_btn_genome_analysis.setPalette(svg_btn_genome_analysis_bg)
        self.svg_btn_genome_analysis.setAutoFillBackground(True)
        svg_btn_mod_bp_analysis_bg = QtGui.QPalette(self.svg_btn_mod_bp_analysis.palette())
        svg_btn_mod_bp_analysis_bg.setColor(QtGui.QPalette.Window, QtGui.QColor('darkGray'))
        self.svg_btn_mod_bp_analysis.setPalette(svg_btn_mod_bp_analysis_bg)
        self.svg_btn_mod_bp_analysis.setAutoFillBackground(True)
        svg_btn_mod_spec_analysis_bg = QtGui.QPalette(self.svg_btn_mod_spec_analysis.palette())
        svg_btn_mod_spec_analysis_bg.setColor(QtGui.QPalette.Window, QtGui.QColor('gray'))
        self.svg_btn_mod_spec_analysis.setPalette(svg_btn_mod_spec_analysis_bg)
        self.svg_btn_mod_spec_analysis.setAutoFillBackground(True)
        svg_btn_bp_spec_analysis_bg = QtGui.QPalette(self.svg_btn_bp_spec_analysis.palette())
        svg_btn_bp_spec_analysis_bg.setColor(QtGui.QPalette.Window, QtGui.QColor('darkGray'))
        self.svg_btn_bp_spec_analysis.setPalette(svg_btn_bp_spec_analysis_bg)
        self.svg_btn_bp_spec_analysis.setAutoFillBackground(True)

        # Activate genome analysis mode, deactivate other modes
        self.widget_genome_analysis.close()
        self.widget_mod_bp_analysis.close()
        self.widget_mod_species_analysis.show()
        self.widget_bp_species_analysis.close()

        # Create graph showing the average fitness of the module species over the generations and display it
        plt.clf()
        for mod_spec_id, analysis_dict in self.mod_spec_analysis_dict.items():
            plt.plot(analysis_dict['x'], analysis_dict['y'], label=f'Mod Species {mod_spec_id}')
        plt.ylabel('average species fitness')
        plt.xlabel('generation')
        plt.legend()
        plt.savefig(self.temp_dir + '/module_species_fitness_analysis.svg')
        self.svg_mod_spec_fitness_analysis.load(self.temp_dir + '/module_species_fitness_analysis.svg')
        self.svg_mod_spec_fitness_analysis.show()

    def event_svg_btn_blueprint_species_analysis(self, *args, **kwargs):
        """"""
        # Set Color focus on Genome Analysis
        svg_btn_genome_analysis_bg = QtGui.QPalette(self.svg_btn_genome_analysis.palette())
        svg_btn_genome_analysis_bg.setColor(QtGui.QPalette.Window, QtGui.QColor('darkGray'))
        self.svg_btn_genome_analysis.setPalette(svg_btn_genome_analysis_bg)
        self.svg_btn_genome_analysis.setAutoFillBackground(True)
        svg_btn_mod_bp_analysis_bg = QtGui.QPalette(self.svg_btn_mod_bp_analysis.palette())
        svg_btn_mod_bp_analysis_bg.setColor(QtGui.QPalette.Window, QtGui.QColor('darkGray'))
        self.svg_btn_mod_bp_analysis.setPalette(svg_btn_mod_bp_analysis_bg)
        self.svg_btn_mod_bp_analysis.setAutoFillBackground(True)
        svg_btn_mod_spec_analysis_bg = QtGui.QPalette(self.svg_btn_mod_spec_analysis.palette())
        svg_btn_mod_spec_analysis_bg.setColor(QtGui.QPalette.Window, QtGui.QColor('darkGray'))
        self.svg_btn_mod_spec_analysis.setPalette(svg_btn_mod_spec_analysis_bg)
        self.svg_btn_mod_spec_analysis.setAutoFillBackground(True)
        svg_btn_bp_spec_analysis_bg = QtGui.QPalette(self.svg_btn_bp_spec_analysis.palette())
        svg_btn_bp_spec_analysis_bg.setColor(QtGui.QPalette.Window, QtGui.QColor('gray'))
        self.svg_btn_bp_spec_analysis.setPalette(svg_btn_bp_spec_analysis_bg)
        self.svg_btn_bp_spec_analysis.setAutoFillBackground(True)

        # Activate genome analysis mode, deactivate other modes
        self.widget_genome_analysis.close()
        self.widget_mod_bp_analysis.close()
        self.widget_mod_species_analysis.close()
        self.widget_bp_species_analysis.show()

        # Create graph showing the average fitness of the blueprint species over the generations and display it
        plt.clf()
        for bp_spec_id, analysis_dict in self.bp_spec_analysis_dict.items():
            plt.plot(analysis_dict['x'], analysis_dict['y'], label=f'BP Species {bp_spec_id}')
        plt.ylabel('average species fitness')
        plt.xlabel('generation')
        plt.legend()
        plt.savefig(self.temp_dir + '/blueprint_species_fitness_analysis.svg')
        self.svg_bp_spec_fitness_analysis.load(self.temp_dir + '/blueprint_species_fitness_analysis.svg')
        self.svg_bp_spec_fitness_analysis.show()

    def click_ga_list_generations(self, item):
        """"""
        # Determine selected best genome
        item_text = item.text()
        chosen_gen = int(item_text[11:(item_text.find('-', 13) - 1)])
        best_genome = self.tfne_state_backups[chosen_gen].best_genome

        # create visualization of best genome and show in image widget
        best_genome_plot_path = best_genome.visualize(show=False,
                                                      save_dir_path=self.temp_dir)
        self.ga_genome_visualization_image.close()
        self.ga_genome_visualization_image.load(best_genome_plot_path)
        self.ga_genome_visualization_image.show()

        # Update genome info labels to show genome information
        mod_spec_to_mod_id = dict()
        for mod_spec, mod in best_genome.bp_assigned_modules.items():
            mod_spec_to_mod_id[mod_spec] = mod.module_id
        self.ga_lbl_genome_id.setText(f'Genome ID {best_genome.genome_id}')
        self.ga_lbl_genome_fitness_value.setText(str(best_genome.fitness))
        self.ga_lbl_genome_bp_id_value.setText(str(best_genome.blueprint.blueprint_id))
        self.ga_lbl_genome_assign_mod_value.setText(str(mod_spec_to_mod_id))
        self.ga_lbl_genome_out_layers_value.setText(str(best_genome.output_layers))
        self.ga_lbl_genome_input_shape_value.setText(str(best_genome.input_shape))
        self.ga_lbl_genome_dtype_value.setText(str(best_genome.dtype))
        self.ga_lbl_genome_orig_gen_value.setText(str(best_genome.origin_generation))

    def click_mba_list_generations(self, item):
        """"""
        # Determine selected generation and save it as the currently set generation for mod/bp analysis
        chosen_gen = int(item.text()[11:])
        self.mba_selected_gen = chosen_gen

        # Create strings that are displayed in the list of members
        members_list = list()
        module_ids = list(self.tfne_state_backups[chosen_gen].modules.keys())
        for mod_id in module_ids:
            members_list.append(f"Module #{mod_id}")
        blueprint_ids = list(self.tfne_state_backups[chosen_gen].blueprints.keys())
        for bp_id in blueprint_ids:
            members_list.append(f"Blueprint #{bp_id}")
        self.mba_list_members.clear()
        self.mba_list_members.addItems(members_list)

        # Update generation info labels to show generational information
        self.mba_lbl_gen_summary_heading.setText(f'Summary of Generation {chosen_gen}')
        self.mba_lbl_gen_mod_spec_value.setText(str(self.tfne_state_backups[chosen_gen].mod_species))
        self.mba_lbl_gen_mod_spec_repr_value.setText(str(self.tfne_state_backups[chosen_gen].mod_species_repr))
        self.mba_lbl_gen_mod_spec_fit_hist_value.setText(
            str(self.tfne_state_backups[chosen_gen].mod_species_fitness_history))
        self.mba_lbl_gen_bp_spec_value.setText(str(self.tfne_state_backups[chosen_gen].bp_species))
        self.mba_lbl_gen_bp_spec_repr_value.setText(str(self.tfne_state_backups[chosen_gen].bp_species_repr))
        self.mba_lbl_gen_bp_spec_fit_hist_value.setText(
            str(self.tfne_state_backups[chosen_gen].bp_species_fitness_history))

    def click_mba_list_members(self, item):
        """"""
        # Determine if Blueprint or Module selected and which ID
        item_text = item.text()
        if item_text[:6] == 'Module':
            # Activate Module analysis widget
            self.mba_widget_blueprint.close()
            self.mba_widget_module.show()

            # Determine selected Module
            chosen_mod_id = int(item_text[8:])
            chosen_mod = self.tfne_state_backups[self.mba_selected_gen].modules[chosen_mod_id]

            # Update module info labels to show module information
            module_summary_dict = chosen_mod.serialize()
            self.mba_lbl_module_heading.setText(f'Summary of Module ID {chosen_mod_id}')
            module_summary_str = ""
            for mod_param, param_value in module_summary_dict.items():
                param_summary = str(mod_param) + ': ' + str(param_value) + '\n\n'
                module_summary_str += param_summary
            self.mba_lbl_module_summary.setText(module_summary_str)
        else:
            # Activate Blueprint analysis widget
            self.mba_widget_blueprint.show()
            self.mba_widget_module.close()

            # Determine selected Blueprint
            chosen_bp_id = int(item_text[11:])
            chosen_bp = self.tfne_state_backups[self.mba_selected_gen].blueprints[chosen_bp_id]

            # Create visualization of selected blueprint and display in image widget
            bp_plot_path = chosen_bp.visualize(show=False, save_dir_path=self.temp_dir)
            self.mba_bp_visualization_image.close()
            self.mba_bp_visualization_image.load(bp_plot_path)
            self.mba_bp_visualization_image.show()

            # Update blueprint info labels to show blueprint information
            self.mba_lbl_blueprint_heading.setText(f'Blueprint ID {chosen_bp.blueprint_id}')
            self.mba_lbl_bp_parent_mut_value.setText(str(chosen_bp.parent_mutation))
            self.mba_lbl_bp_optimizer_value.setText(str(chosen_bp.optimizer_factory))

    def action_close_triggered(self):
        """"""
        self.parent_window.show()
        self.destroy()

    def action_exit_triggered(self):
        """"""
        self.parent_window.destroy()
        self.destroy()
        exit()

    @staticmethod
    def action_documentation_triggered():
        """"""
        webbrowser.open('https://tfne.readthedocs.io')
