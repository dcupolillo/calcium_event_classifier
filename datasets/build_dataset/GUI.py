""" Created on Thu Sep 12 14:37:00 2024
    @author: dcupolillo """

import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton,
    QVBoxLayout, QHBoxLayout, QWidget, QDialog,
    QListWidget, QFileDialog, QLabel, QMessageBox)
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas)
from matplotlib.figure import Figure
import flammkuchen as fl


class FileSelectionDialog(QDialog):

    def __init__(self, parent=None):

        super().__init__(parent)

        self.selected_zscore_files = []
        self.selected_dff_files = []
        self.init_ui()

    def init_ui(self):

        self.setWindowTitle('Select Data Files')
        self.setGeometry(300, 300, 600, 500)

        layout = QVBoxLayout()

        # Z-score file selection
        label_z = QLabel('Select HDF5 files containing zscore data:')
        layout.addWidget(label_z)
        self.file_list_z = QListWidget()
        layout.addWidget(self.file_list_z)
        btn_z = QPushButton('Add Z-score Files')
        btn_z.clicked.connect(self.add_zscore_files)
        layout.addWidget(btn_z)
        btn_remove_z = QPushButton('Remove Selected Z-score File')
        btn_remove_z.clicked.connect(self.remove_selected_zscore_file)
        layout.addWidget(btn_remove_z)
        btn_clear_z = QPushButton('Clear All Z-score Files')
        btn_clear_z.clicked.connect(self.clear_all_zscore_files)
        layout.addWidget(btn_clear_z)

        # dFF file selection
        label_dff = QLabel('Select HDF5 files containing dFF data:')
        layout.addWidget(label_dff)
        self.file_list_dff = QListWidget()
        layout.addWidget(self.file_list_dff)
        btn_dff = QPushButton('Add dFF Files')
        btn_dff.clicked.connect(self.add_dff_files)
        layout.addWidget(btn_dff)
        btn_remove_dff = QPushButton('Remove Selected dFF File')
        btn_remove_dff.clicked.connect(self.remove_selected_dff_file)
        layout.addWidget(btn_remove_dff)
        btn_clear_dff = QPushButton('Clear All dFF Files')
        btn_clear_dff.clicked.connect(self.clear_all_dff_files)
        layout.addWidget(btn_clear_dff)

        # OK and Cancel buttons
        btn_layout = QHBoxLayout()
        btn_ok = QPushButton('OK')
        btn_ok.clicked.connect(self.accept)
        btn_layout.addWidget(btn_ok)
        btn_cancel = QPushButton('Cancel')
        btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(btn_cancel)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

    def remove_selected_zscore_file(self):
        current_row = self.file_list_z.currentRow()
        if current_row >= 0:
            item = self.file_list_z.takeItem(current_row)
            if item:
                try:
                    self.selected_zscore_files.remove(item.text())
                except ValueError:
                    pass

    def clear_all_zscore_files(self):
        self.file_list_z.clear()
        self.selected_zscore_files.clear()

    def remove_selected_dff_file(self):
        current_row = self.file_list_dff.currentRow()
        if current_row >= 0:
            item = self.file_list_dff.takeItem(current_row)
            if item:
                try:
                    self.selected_dff_files.remove(item.text())
                except ValueError:
                    pass

    def clear_all_dff_files(self):
        self.file_list_dff.clear()
        self.selected_dff_files.clear()

    def accept(self):
        if not self.selected_zscore_files or not self.selected_dff_files:
            QMessageBox.warning(
                self, 'Warning', 'Please select both zscore and dFF files.')
            return
        if len(self.selected_zscore_files) != len(self.selected_dff_files):
            QMessageBox.warning(
                self, 'Warning', 'Number of zscore and dFF files must match.')
            return
        super().accept()

    def add_zscore_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, 'Select Z-score Files', '', 'HDF5 Files (*.h5 *.hdf5);;All Files (*)')
        for f in files:
            if f and f not in self.selected_zscore_files:
                self.selected_zscore_files.append(f)
                self.file_list_z.addItem(f)

    def add_dff_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, 'Select dFF Files', '', 'HDF5 Files (*.h5 *.hdf5);;All Files (*)')
        for f in files:
            if f and f not in self.selected_dff_files:
                self.selected_dff_files.append(f)
                self.file_list_dff.addItem(f)



def load_traces_from_files(
        zscore_files: list,
        dff_files: list
) -> tuple:
    """Load paired traces from selected files."""
    
    all_zscores, all_dffs = [], []
    
    for zf, df in zip(zscore_files, dff_files):
        try:
            z_data = fl.load(zf)
            d_data = fl.load(df)
            # Assuming same structure and order
            for z_spine, d_spine in zip(z_data, d_data):
                for z_sweep, d_sweep in zip(z_spine, d_spine):
                    all_zscores.append(z_sweep)
                    all_dffs.append(d_sweep)
        except Exception as e:
            print(f"Error loading {zf} or {df}: {e}")
    
    return all_zscores, all_dffs


class TraceViewer(QMainWindow):
    
    def __init__(
            self,
            zscores: list,
            dffs: list
    ) -> None:
        
        super().__init__()
        
        self.zscores = np.array(zscores)
        self.dffs = np.array(dffs)
        self.current_index = 0
        self.label = []
        self.decided_zscores = []
        self.decided_dffs = []
        self.last_skipped = False
        self.skipped_count = 0

        widget = QWidget(self)
        main_layout = QVBoxLayout()

        # Top: current traces (zscore and dFF side-by-side)
        current_traces_layout = QHBoxLayout()
        self.figure_current = Figure(figsize=(4, 4))
        self.canvas_current = FigureCanvas(self.figure_current)
        self.canvas_current.setMinimumSize(400, 400)
        self.canvas_current.setMaximumSize(400, 400)
        current_traces_layout.addWidget(self.canvas_current)

        # Buttons to the right of current traces
        buttons_layout = QVBoxLayout()
        self.counter_label = QLabel()
        self.update_counter_display()
        buttons_layout.addWidget(self.counter_label)

        self.exclude_button = QPushButton('Label 0', self)
        self.include_button = QPushButton('Label 1', self)
        self.skip_button = QPushButton('Skip', self)
        self.back_button = QPushButton('Go Back', self)
        self.save_button = QPushButton('Save & Exit', self)

        # Set fixed width for all buttons
        button_width = 140
        self.include_button.setFixedWidth(button_width)
        self.exclude_button.setFixedWidth(button_width)
        self.skip_button.setFixedWidth(button_width)
        self.back_button.setFixedWidth(button_width)
        self.save_button.setFixedWidth(button_width)

        self.include_button.clicked.connect(self.include_trace)
        self.exclude_button.clicked.connect(self.exclude_trace)
        self.skip_button.clicked.connect(self.skip_trace)
        self.back_button.clicked.connect(self.go_back)
        self.save_button.clicked.connect(self.save_and_exit)

        buttons_layout.addWidget(self.exclude_button)
        buttons_layout.addWidget(self.include_button)
        buttons_layout.addWidget(self.skip_button)
        buttons_layout.addWidget(self.back_button)
        buttons_layout.addWidget(self.save_button)

        current_traces_layout.addLayout(buttons_layout)
        main_layout.addLayout(current_traces_layout)

        # Bottom: summary plots (included and excluded side-by-side)
        summary_layout = QHBoxLayout()
        self.figure_included = Figure(figsize=(8, 4))
        self.canvas_included = FigureCanvas(self.figure_included)
        self.figure_excluded = Figure(figsize=(8, 4))
        self.canvas_excluded = FigureCanvas(self.figure_excluded)
        summary_layout.addWidget(self.canvas_excluded)
        summary_layout.addWidget(self.canvas_included)
        main_layout.addLayout(summary_layout)

        widget.setLayout(main_layout)
        self.setCentralWidget(widget)
        self.plot_trace()
        self.setWindowTitle('Calcium Imaging Trace Viewer')
        self.setGeometry(200, 200, 1800, 900)
        self.show()

    def plot_trace(self):
        """Plots the current zscore and dFF trace side-by-side on the canvas."""
        if self.current_index < len(self.zscores):
            # Always reset figure size before plotting
            self.figure_current.clear()
            ax = self.figure_current.add_subplot(111)
            x_data = np.arange(len(self.zscores[self.current_index]))
            # Plot zscore trace
            ax.plot(x_data, self.zscores[self.current_index], color='gray', alpha=0.6, label='Z-score')
            ax.set_xlabel('Data Points')
            ax.set_ylabel('Z-score', color='gray')
            ax.set_ylim(-5, 5)
            ax.set_xticks(np.linspace(0, len(self.zscores[self.current_index]), 6))
            ax.axvline(16, color='red', alpha=0.3)

            # Twin y-axis for dFF
            ax2 = ax.twinx()
            ax2.plot(x_data, self.dffs[self.current_index], color='crimson', linewidth=1.5, label='dFF')
            ax2.set_ylabel('dFF', color='crimson')
            ax2.set_ylim(-5, 5)

            # Title and legend
            ax.set_title(f"Trace {self.current_index + 1}/{len(self.zscores)} (Z-score & dFF)")
            ax.legend(loc='upper left')
            ax2.legend(loc='upper right')

            self.figure_current.tight_layout()
            self.canvas_current.draw()
            self.update_counter_display()
        else:
            self.include_button.setEnabled(False)
            self.exclude_button.setEnabled(False)
            self.skip_button.setEnabled(False)
            self.back_button.setEnabled(False)
            self.save_button.setEnabled(True)
            self.setWindowTitle('All Traces Processed')
            self.update_counter_display()

    def update_summary_plots(self):
        """Update the plots for included and excluded zscore and dFF traces."""
        included_indices = [i for i, label in enumerate(self.label) if label == 1]
        excluded_indices = [i for i, label in enumerate(self.label) if label == 0]

        included_zscores = [self.decided_zscores[i] for i in included_indices]
        included_dffs = [self.decided_dffs[i] for i in included_indices]
        excluded_zscores = [self.decided_zscores[i] for i in excluded_indices]
        excluded_dffs = [self.decided_dffs[i] for i in excluded_indices]

        # Included summary: zscore and dFF side-by-side
        self.figure_included.clear()
        ax_z_incl = self.figure_included.add_subplot(121)
        ax_dff_incl = self.figure_included.add_subplot(122)

        if included_zscores:
            x_data = np.arange(len(included_zscores[0]))
            for trace in included_zscores:
                ax_z_incl.plot(x_data, trace, color='lightgray', alpha=0.8)
            ax_z_incl.plot(x_data, np.mean(included_zscores, axis=0), color='black', linewidth=2)
        ax_z_incl.set_title(f'Label 1 Z-score ({len(included_zscores)})')
        ax_z_incl.set_ylim(-5, 5)
        if included_zscores:
            ax_z_incl.set_xticks(np.linspace(0, len(included_zscores[0]), 6))
            ax_z_incl.axvline(16, color='red', alpha=0.3)
        ax_z_incl.set_xlabel('Data Points')

        if included_dffs:
            x_data_dff = np.arange(len(included_dffs[0]))
            for trace in included_dffs:
                ax_dff_incl.plot(x_data_dff, trace, color='lightgray', alpha=0.8)
            ax_dff_incl.plot(x_data_dff, np.mean(included_dffs, axis=0), color='crimson', linewidth=2)
        ax_dff_incl.set_title(f'Label 1 dFF ({len(included_dffs)})')
        ax_dff_incl.set_ylim(-5, 5)
        if included_dffs:
            ax_dff_incl.set_xticks(np.linspace(0, len(included_dffs[0]), 6))
            ax_dff_incl.axvline(16, color='red', alpha=0.3)
        ax_dff_incl.set_xlabel('Data Points')

        self.figure_included.tight_layout()
        self.canvas_included.draw()

        # Excluded summary: zscore and dFF side-by-side
        self.figure_excluded.clear()
        ax_z_excl = self.figure_excluded.add_subplot(121)
        ax_dff_excl = self.figure_excluded.add_subplot(122)

        if excluded_zscores:
            x_data = np.arange(len(excluded_zscores[0]))
            for trace in excluded_zscores:
                ax_z_excl.plot(x_data, trace, color='lightgray', alpha=0.8)
            ax_z_excl.plot(x_data, np.mean(excluded_zscores, axis=0), color='blue', linewidth=2)
        ax_z_excl.set_title(f'Label 0 Z-score ({len(excluded_zscores)})')
        ax_z_excl.set_ylim(-5, 5)
        if excluded_zscores:
            ax_z_excl.set_xticks(np.linspace(0, len(excluded_zscores[0]), 6))
            ax_z_excl.axvline(16, color='red', alpha=0.3)
        ax_z_excl.set_xlabel('Data Points')

        if excluded_dffs:
            x_data_dff = np.arange(len(excluded_dffs[0]))
            for trace in excluded_dffs:
                ax_dff_excl.plot(x_data_dff, trace, color='lightgray', alpha=0.8)
            ax_dff_excl.plot(x_data_dff, np.mean(excluded_dffs, axis=0), color='green', linewidth=2)
        ax_dff_excl.set_title(f'Label 0 dFF ({len(excluded_dffs)})')
        ax_dff_excl.set_ylim(-5, 5)
        if excluded_dffs:
            ax_dff_excl.set_xticks(np.linspace(0, len(excluded_dffs[0]), 6))
            ax_dff_excl.axvline(16, color='red', alpha=0.3)
        ax_dff_excl.set_xlabel('Data Points')

        self.figure_excluded.tight_layout()
        self.canvas_excluded.draw()

        self.update_counter_display()
    
    def update_counter_display(self):
        """Update the counter display above the buttons."""
        labeled_count = len(self.label)
        analyzed_count = labeled_count + self.skipped_count
        counter_text = (
            f"Labeled: {labeled_count} | "
            f"Skipped: {self.skipped_count} | "
            f"Analyzed: {analyzed_count}")
        self.counter_label.setText(counter_text)

    def make_decision(self, label):
        """Make a decision (include=1 or exclude=0) for the current trace."""
        self.label.append(label)
        self.decided_zscores.append(self.zscores[self.current_index])
        self.decided_dffs.append(self.dffs[self.current_index])
        self.update_summary_plots()
        self.next_trace()

    def include_trace(self):
        """Marks the current trace as included (1)."""
        self.make_decision(1)
        self.last_skipped = False

    def exclude_trace(self):
        """Marks the current trace as excluded (0)."""
        self.make_decision(0)
        self.last_skipped = False

    def skip_trace(self):
        """Skips the current trace without making a decision."""
        self.skipped_count += 1
        self.next_trace()
        self.last_skipped = True

    def go_back(self):
        """Go back to the previous trace and remove the last decision."""
        if self.current_index > 0:
            # Go back one trace
            self.current_index -= 1

            # Remove the last decision and its corresponding data
            if self.last_skipped:
                # If the last action was a skip, decrement skip counter
                self.skipped_count -= 1
            elif self.label:
                # If the last action was a label decision, remove it
                self.label.pop()
                self.decided_zscores.pop()
                self.decided_dffs.pop()
                self.update_summary_plots()

            # Re-plot the previous trace
            self.plot_trace()

    def next_trace(self):
        """Advances to the next trace and updates the plot."""
        self.current_index += 1
        self.plot_trace()

    def save_and_exit(self):
        """Saves the binary decisions and traces in element-wise organization,
        then exits.
        """

        data = {
            'label': np.array(self.label),
            'zscore': np.array(self.decided_zscores),
            'dff': np.array(self.decided_dffs)
        }

        fl.save('trace_data.h5', data)
        num_decisions = len(self.label)
        analyzed_count = num_decisions + self.skipped_count
        print(f"Data saved to 'trace_data.h5' with {num_decisions} "
              f"labeled traces")
        print(f"Labels: {sum(self.label)} included, "
              f"{num_decisions - sum(self.label)} excluded")
        print(f"Skipped: {self.skipped_count} traces")
        print(f"Total analyzed: {analyzed_count} traces")
        sys.exit()


# In run_app(), update usage:
def run_app():

    app = QApplication(sys.argv)
    file_dialog = FileSelectionDialog()

    if file_dialog.exec_() == QDialog.Accepted:
        zscore_files = file_dialog.selected_zscore_files
        dff_files = file_dialog.selected_dff_files

        if zscore_files and dff_files and len(zscore_files) == len(dff_files):
            zscores, dffs = load_traces_from_files(zscore_files, dff_files)

            if zscores and dffs:

                # Shuffle traces to present in random order,
                # but keep dFF-zscore correspondence
                indices = np.arange(len(zscores))

                window_means = [
                    mean_dff_window(dffs[i], start=16, end=19)
                    for i in indices
                ]
                sorted_indices = [
                    i for i in np.argsort(window_means)[::-1]
                ]
                shuffled_zscores = [zscores[i] for i in sorted_indices]
                shuffled_dffs = [dffs[i] for i in sorted_indices]

                # np.random.shuffle(indices)
                # shuffled_zscores = [zscores[i] for i in indices]
                # shuffled_dffs = [dffs[i] for i in indices]

                # Pass both to TraceViewer or modify TraceViewer to accept both
                viewer = TraceViewer(shuffled_zscores, shuffled_dffs)

                return app.exec_()

            else:
                QMessageBox.critical(
                    None, 'Error', 'No valid traces found in selected files.')

                return 1
        else:
            print("Files not properly selected or paired.")
            return 1
    else:
        print("File selection cancelled.")
        return 1


def mean_dff_window(
    dff_trace: np.ndarray,
    start: int = 16,
    end: int = 19
) -> float:
    """Return mean dF/F in the inclusive index window [start, end].

    If the window extends beyond the trace, it is clipped to the
    trace length.
    """
    if dff_trace is None or len(dff_trace) == 0:
        return -np.inf

    end_idx = min(len(dff_trace) - 1, end)
    if start > end_idx:
        return -np.inf

    # end_idx is inclusive, so use end_idx + 1 in slice
    window = dff_trace[start:end_idx + 1]
    return float(np.mean(window))


if __name__ == '__main__':
    sys.exit(run_app())
