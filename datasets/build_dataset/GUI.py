""" Created on Thu Sep 12 14:37:00 2024
    @author: dcupolillo """

from pathlib import Path
import sys
import numpy as np
import itertools
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
        self.selected_files = []
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Select Data Files')
        self.setGeometry(300, 300, 600, 400)

        layout = QVBoxLayout()

        # Label
        label = QLabel('Select HDF5 files containing zscore data:')
        layout.addWidget(label)

        # File list widget
        self.file_list = QListWidget()
        layout.addWidget(self.file_list)

        # Buttons layout
        button_layout = QHBoxLayout()

        # Add file button
        self.add_file_btn = QPushButton('Add Files')
        self.add_file_btn.clicked.connect(self.add_file)
        button_layout.addWidget(self.add_file_btn)

        # Remove file button
        self.remove_file_btn = QPushButton('Remove Selected')
        self.remove_file_btn.clicked.connect(self.remove_file)
        button_layout.addWidget(self.remove_file_btn)

        # Clear all button
        self.clear_all_btn = QPushButton('Clear All')
        self.clear_all_btn.clicked.connect(self.clear_all)
        button_layout.addWidget(self.clear_all_btn)

        layout.addLayout(button_layout)

        # OK and Cancel buttons
        ok_cancel_layout = QHBoxLayout()

        self.ok_btn = QPushButton('OK')
        self.ok_btn.clicked.connect(self.accept)
        ok_cancel_layout.addWidget(self.ok_btn)

        self.cancel_btn = QPushButton('Cancel')
        self.cancel_btn.clicked.connect(self.reject)
        ok_cancel_layout.addWidget(self.cancel_btn)

        layout.addLayout(ok_cancel_layout)

        self.setLayout(layout)

    def add_file(self):
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            'Select HDF5 Files',
            '',
            'HDF5 Files (*.h5 *.hdf5);;All Files (*)'
        )

        for file_path in file_paths:
            if file_path and file_path not in self.selected_files:
                self.selected_files.append(file_path)
                self.file_list.addItem(file_path)

    def remove_file(self):
        current_row = self.file_list.currentRow()
        if current_row >= 0:
            item = self.file_list.takeItem(current_row)
            if item:
                self.selected_files.remove(item.text())

    def clear_all(self):
        self.file_list.clear()
        self.selected_files.clear()

    def accept(self):
        if not self.selected_files:
            QMessageBox.warning(
                self, 'Warning', 'Please select at least one file.')
            return
        super().accept()


def load_traces_from_files(file_paths):
    """Load traces from selected files."""
    all_traces = []

    for file_path in file_paths:
        try:
            # Load the data from each file
            data = fl.load(file_path)

            if isinstance(data, (list, np.ndarray)):
                for spine in data:
                    for sweep in spine:
                        all_traces.append(sweep)
            else:
                print(f"Warning: Unrecognized data structure in {file_path}")

        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    return all_traces


class TraceViewer(QMainWindow):

    def __init__(
            self,
            traces
    ) -> None:

        super().__init__()

        # Data should already be flattened by load_traces_from_files
        # Convert to numpy arrays directly
        self.traces = np.array(traces)

        self.current_index = 0

        # List to store binary decisions (1 for Include, 0 for Exclude)
        self.label = []
        self.decided_traces = []
        self.last_skipped = False

        # Counter for skipped traces
        self.skipped_count = 0

        # Create a main widget and layout
        widget = QWidget(self)
        main_layout = QHBoxLayout()

        # Create buttons layout
        buttons_layout = QVBoxLayout()

        # Create counter label above buttons
        self.counter_label = QLabel()
        self.update_counter_display()
        buttons_layout.addWidget(self.counter_label)

        # Create buttons
        self.include_button = QPushButton('Label 1', self)
        self.exclude_button = QPushButton('Label 0', self)
        self.skip_button = QPushButton('Skip', self)
        self.back_button = QPushButton('Go Back', self)
        self.save_button = QPushButton('Save & Exit', self)

        self.include_button.clicked.connect(self.include_trace)
        self.exclude_button.clicked.connect(self.exclude_trace)
        self.skip_button.clicked.connect(self.skip_trace)
        self.back_button.clicked.connect(self.go_back)
        self.save_button.clicked.connect(self.save_and_exit)

        # Add buttons to layout
        buttons_layout.addWidget(self.include_button)
        buttons_layout.addWidget(self.exclude_button)
        buttons_layout.addWidget(self.skip_button)
        buttons_layout.addWidget(self.back_button)
        buttons_layout.addWidget(self.save_button)

        # Create canvases (current, included, excluded)
        self.figure_current = Figure()
        self.canvas_current = FigureCanvas(self.figure_current)

        self.figure_included = Figure()
        self.canvas_included = FigureCanvas(self.figure_included)

        self.figure_excluded = Figure()
        self.canvas_excluded = FigureCanvas(self.figure_excluded)

        # Add canvases to layout
        main_layout.addWidget(self.canvas_current)
        main_layout.addWidget(self.canvas_included)
        main_layout.addWidget(self.canvas_excluded)

        # Add button layout to the right
        main_layout.addLayout(buttons_layout)

        # Set layout and widget
        widget.setLayout(main_layout)
        self.setCentralWidget(widget)

        # Plot the first trace
        self.plot_trace()

        # Set window title and size
        self.setWindowTitle('Calcium Imaging Trace Viewer')
        self.setGeometry(200, 200, 2000, 300)
        self.show()

    def plot_trace(self):
        """Plots the current trace on the canvas."""
        if self.current_index < len(self.traces):
            # Clear and plot the current trace
            self.figure_current.clear()
            ax_current = self.figure_current.add_subplot(111)

            # Use data point indices as x-axis
            x_data = np.arange(len(self.traces[self.current_index]))
            ax_current.plot(x_data, self.traces[self.current_index])

            ax_current.set_xlabel('Data Points')
            ax_current.set_ylabel('Z-score')
            ax_current.set_ylim(-5, 5)
            ax_current.set_xticks(
                np.linspace(0, len(self.traces[self.current_index]), 6))
            ax_current.axvline(16, color='red', alpha=0.3)  # 20% mark

            # Add title showing progress
            total_traces = len(self.traces)
            ax_current.set_title(
                f"Trace {self.current_index + 1}/{total_traces}")
            self.canvas_current.draw()

            # Update counter display
            self.update_counter_display()
        else:
            # End of traces
            self.include_button.setEnabled(False)
            self.exclude_button.setEnabled(False)
            self.skip_button.setEnabled(False)
            self.back_button.setEnabled(False)
            self.save_button.setEnabled(True)  # Enable save button at the end
            self.setWindowTitle('All Traces Processed')
            # Update counter display
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

    def update_summary_plots(self):
        """Update the plots for included and excluded traces."""
        # Get included and excluded traces based on labels
        included_indices = [
            i for i, label in enumerate(self.label) if label == 1]
        excluded_indices = [
            i for i, label in enumerate(self.label) if label == 0]

        included_traces = [
            self.decided_traces[i] for i in included_indices]
        excluded_traces = [
            self.decided_traces[i] for i in excluded_indices]

        # Update included plot
        self.figure_included.clear()
        ax_included = self.figure_included.add_subplot(111)

        if included_traces:
            # Use data point indices as x-axis
            x_data = np.arange(len(included_traces[0]))

            for trace in included_traces:
                ax_included.plot(x_data, trace, color='lightgray', alpha=0.8)

            ax_included.plot(
                x_data,
                np.mean(included_traces, axis=0),
                color='green',
                linewidth=2)

        ax_included.set_title(f'Included Traces ({len(included_traces)})')
        ax_included.set_ylim(-5, 5)
        if included_traces:
            ax_included.set_xticks(np.linspace(0, len(included_traces[0]), 6))
            ax_included.axvline(16, color='red', alpha=0.3)
        ax_included.set_xlabel('Data Points')
        self.canvas_included.draw()

        # Update excluded plot
        self.figure_excluded.clear()
        ax_excluded = self.figure_excluded.add_subplot(111)

        if excluded_traces:
            # Use data point indices as x-axis
            x_data = np.arange(len(excluded_traces[0]))

            for trace in excluded_traces:
                ax_excluded.plot(x_data, trace, color='lightgray', alpha=0.8)

            ax_excluded.plot(
                x_data,
                np.mean(excluded_traces, axis=0),
                color='red',
                linewidth=2)

        ax_excluded.set_title(f'Excluded Traces ({len(excluded_traces)})')
        ax_excluded.set_ylim(-5, 5)
        if excluded_traces:
            ax_excluded.set_xticks(np.linspace(0, len(excluded_traces[0]), 6))
            ax_excluded.axvline(16, color='red', alpha=0.3)
        ax_excluded.set_xlabel('Data Points')
        self.canvas_excluded.draw()

        # Update counter display
        self.update_counter_display()

    def make_decision(self, label):
        """Make a decision (include=1 or exclude=0) for the current trace."""
        self.label.append(label)
        self.decided_traces.append(self.traces[self.current_index])
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
                self.decided_traces.pop()
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
            'zscore': np.array(self.decided_traces)
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


def run_app():
    """Main application entry point."""
    app = QApplication(sys.argv)

    # Show file selection dialog
    file_dialog = FileSelectionDialog()

    if file_dialog.exec_() == QDialog.Accepted:
        selected_files = file_dialog.selected_files

        if selected_files:
            # Load traces from selected files
            all_traces = load_traces_from_files(selected_files)

            if all_traces:
                # Shuffle the traces
                indices = np.arange(len(all_traces))
                np.random.shuffle(indices)

                shuffled_traces = [all_traces[i] for i in indices]

                # Start the trace viewer
                viewer = TraceViewer(shuffled_traces)
                return app.exec_()
            else:
                QMessageBox.critical(
                    None,
                    'Error',
                    'No valid traces found in selected files.')
                return 1
        else:
            print("No files selected.")
            return 1
    else:
        print("File selection cancelled.")
        return 1


if __name__ == '__main__':
    sys.exit(run_app())
