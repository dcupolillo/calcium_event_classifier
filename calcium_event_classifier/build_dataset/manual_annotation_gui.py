""" Created on March 16, 2026
    @author: dcupolillo
    
    Simple manual annotation GUI for test datasets.
    Allows users to label individual calcium imaging traces.
"""

import sys
import numpy as np
import random
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton,
    QVBoxLayout, QHBoxLayout, QWidget, QFileDialog,
    QLabel, QMessageBox)
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas)
from matplotlib.figure import Figure
import flammkuchen as fl


class ManualAnnotationViewer(QMainWindow):
    """
    GUI for manually annotating calcium imaging traces.
    
    Users can:
    - View individual traces one at a time
    - Label as 0 (no event) or 1 (event)
    - Go back to previous traces
    - Save annotations to HDF5 file
    """
    
    def __init__(
            self,
            data: dict,
            save_path: str = None
    ) -> None:
        """
        Initialize the annotation viewer.
        
        Parameters
        ----------
        data : dict
            Dictionary containing trace data (must have 'dff', 'trace', or similar key)
        save_path : str, optional
            Path where annotations will be saved. Defaults to 'annotated_traces.h5'
        """
        super().__init__()
        
        # Extract traces from data dictionary
        if 'dff' in data:
            self.traces = np.array(data['dff'])
        
        elif 'trace' in data:
            self.traces = np.array(data['trace'])
        
        elif isinstance(data, dict) and len(data) > 0:
            first_key = list(data.keys())[0]
            self.traces = np.array(data[first_key])
            print(f"Using key: '{first_key}'")
        
        else:
            raise ValueError("Data dict does not contain expected keys (dff, trace, etc.)")
        
        self.save_path = Path(save_path) if save_path else Path('annotated_traces.h5')
        
        # Shuffle trace order while maintaining annotation relationships
        self.shuffled_indices = list(range(len(self.traces)))
        random.shuffle(self.shuffled_indices)
        
        self.current_index = 0
        self.labels = []
        self.annotated_traces = []
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface."""
        # Main widget and layout
        widget = QWidget(self)
        main_layout = QVBoxLayout()
        
        # Top: Current trace visualization and control buttons
        current_layout = QHBoxLayout()
        
        # Trace plot
        self.figure_current = Figure(figsize=(6, 4))
        self.canvas_current = FigureCanvas(self.figure_current)
        self.canvas_current.setMinimumSize(600, 400)
        current_layout.addWidget(self.canvas_current)
        
        # Control buttons
        buttons_layout = QVBoxLayout()
        
        # Counter display
        self.counter_label = QLabel()
        self.update_counter_display()
        buttons_layout.addWidget(self.counter_label)
        
        # Annotation buttons
        self.label_0_button = QPushButton('Label 0\n(No Event)', self)
        self.label_1_button = QPushButton('Label 1\n(Event)', self)
        self.back_button = QPushButton('Go Back', self)
        self.save_button = QPushButton('Save & Exit', self)
        
        # Set button properties
        button_width = 120
        for btn in [self.label_0_button, self.label_1_button, 
                    self.back_button, self.save_button]:
            btn.setFixedWidth(button_width)
        
        self.label_0_button.clicked.connect(self.label_0)
        self.label_1_button.clicked.connect(self.label_1)
        self.back_button.clicked.connect(self.go_back)
        self.save_button.clicked.connect(self.save_and_exit)
        
        buttons_layout.addWidget(self.label_0_button)
        buttons_layout.addWidget(self.label_1_button)
        buttons_layout.addWidget(self.back_button)
        buttons_layout.addWidget(self.save_button)
        buttons_layout.addStretch()
        
        current_layout.addLayout(buttons_layout)
        main_layout.addLayout(current_layout)
        
        # Bottom: Summary plots (labeled and unlabeled)
        summary_layout = QHBoxLayout()
        
        self.figure_label_0 = Figure(figsize=(6, 3))
        self.canvas_label_0 = FigureCanvas(self.figure_label_0)
        
        self.figure_label_1 = Figure(figsize=(6, 3))
        self.canvas_label_1 = FigureCanvas(self.figure_label_1)
        
        summary_layout.addWidget(self.canvas_label_0)
        summary_layout.addWidget(self.canvas_label_1)
        main_layout.addLayout(summary_layout)
        
        # Set window properties
        widget.setLayout(main_layout)
        self.setCentralWidget(widget)
        self.setWindowTitle('Manual Trace Annotation')
        self.setGeometry(200, 200, 1600, 900)
        
        # Plot first trace
        self.plot_trace()
        self.show()
    
    def plot_trace(self):
        """Plot the current trace."""
        if self.current_index < len(self.traces):
            self.figure_current.clear()
            ax = self.figure_current.add_subplot(111)
            
            # Access trace using shuffled index
            original_idx = self.shuffled_indices[self.current_index]
            trace = self.traces[original_idx]
            x_data = np.arange(len(trace))
            
            # Plot the trace
            ax.plot(x_data, trace, color='steelblue', linewidth=2)
            ax.fill_between(x_data, trace, alpha=0.3, color='steelblue')
            
            # Mark baseline region (first 16 samples)
            ax.axvline(16, color='red', linestyle='--', linewidth=1.5, 
                      alpha=0.7, label='Baseline end')
            
            # Labels and formatting
            ax.set_xlabel('Time (samples)', fontsize=11)
            ax.set_ylabel('Value', fontsize=11)
            ax.set_title(f"Trace {self.current_index + 1} / {len(self.traces)}", 
                        fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)
            
            self.figure_current.tight_layout()
            self.canvas_current.draw()
            self.update_counter_display()
            
        else:
            # All traces processed
            self.label_0_button.setEnabled(False)
            self.label_1_button.setEnabled(False)
            self.back_button.setEnabled(False)
            self.setWindowTitle('All Traces Processed')
            self.update_counter_display()
    
    def update_summary_plots(self):
        """Update the summary plots showing labeled traces."""
        label_0_indices = [i for i, (lbl, _) in enumerate(self.labels) if lbl == 0]
        label_1_indices = [i for i, (lbl, _) in enumerate(self.labels) if lbl == 1]
        
        label_0_traces = [self.annotated_traces[i] for i in label_0_indices]
        label_1_traces = [self.annotated_traces[i] for i in label_1_indices]
        
        # Plot Label 0 traces
        self.figure_label_0.clear()
        ax0 = self.figure_label_0.add_subplot(111)
        
        if label_0_traces:
            x_data = np.arange(len(label_0_traces[0]))
            for trace in label_0_traces:
                ax0.plot(x_data, trace, color='lightblue', alpha=0.6)
            mean_trace = np.mean(label_0_traces, axis=0)
            ax0.plot(x_data, mean_trace, color='blue', linewidth=2.5, 
                    label='Mean')
            ax0.axvline(16, color='red', linestyle='--', alpha=0.5)
        
        ax0.set_title(f'Label 0 - No Event ({len(label_0_traces)} traces)', 
                     fontsize=11, fontweight='bold')
        ax0.set_ylim(-5, 5)
        ax0.set_xlabel('Time (samples)', fontsize=10)
        ax0.grid(True, alpha=0.3)
        if label_0_traces:
            ax0.legend(fontsize=9)
        
        self.figure_label_0.tight_layout()
        self.canvas_label_0.draw()
        
        # Plot Label 1 traces
        self.figure_label_1.clear()
        ax1 = self.figure_label_1.add_subplot(111)
        
        if label_1_traces:
            x_data = np.arange(len(label_1_traces[0]))
            for trace in label_1_traces:
                ax1.plot(x_data, trace, color='lightcoral', alpha=0.6)
            mean_trace = np.mean(label_1_traces, axis=0)
            ax1.plot(x_data, mean_trace, color='crimson', linewidth=2.5, 
                    label='Mean')
            ax1.axvline(16, color='red', linestyle='--', alpha=0.5)
        
        ax1.set_title(f'Label 1 - Event ({len(label_1_traces)} traces)', 
                     fontsize=11, fontweight='bold')
        ax1.set_ylim(-5, 5)
        ax1.set_xlabel('Time (samples)', fontsize=10)
        ax1.grid(True, alpha=0.3)
        if label_1_traces:
            ax1.legend(fontsize=9)
        
        self.figure_label_1.tight_layout()
        self.canvas_label_1.draw()
    
    def update_counter_display(self):
        """Update the counter display."""
        labeled_count = len(self.labels)
        progress = f"{labeled_count} / {len(self.traces)}"
        counter_text = f"Progress: {progress}\nLabel 0: {sum(1 for lbl, _ in self.labels if lbl == 0)} | Label 1: {sum(1 for lbl, _ in self.labels if lbl == 1)}"
        self.counter_label.setText(counter_text)
    
    def make_decision(self, label: int):
        """Record a label decision and advance to next trace."""
        original_idx = self.shuffled_indices[self.current_index]
        self.labels.append((label, original_idx))
        self.annotated_traces.append(self.traces[original_idx])
        self.update_summary_plots()
        self.next_trace()
    
    def label_0(self):
        """Label current trace as 0 (no event)."""
        self.make_decision(0)
    
    def label_1(self):
        """Label current trace as 1 (event)."""
        self.make_decision(1)
    
    def go_back(self):
        """Go back to the previous trace."""
        if self.current_index > 0:
            self.current_index -= 1
            if self.labels:
                self.labels.pop()
                self.annotated_traces.pop()
                self.update_summary_plots()
            self.plot_trace()
    
    def next_trace(self):
        """Advance to the next trace."""
        self.current_index += 1
        self.plot_trace()
    
    def save_and_exit(self):
        """Save annotations to HDF5 and exit."""
        if len(self.labels) == 0:
            QMessageBox.warning(
                self, 'Warning', 'No traces have been labeled. Cannot save.')
            return
        
        # Reorder labels and traces back to original order
        original_labels = [None] * len(self.labels)
        original_traces = [None] * len(self.annotated_traces)
        
        for i, (label, original_idx) in enumerate(self.labels):
            original_labels[original_idx] = label
            original_traces[original_idx] = self.annotated_traces[i]
        
        data = {
            'label': np.array(original_labels),
            'trace': np.array(original_traces)
        }
        
        # Create parent directory if it doesn't exist
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        
        fl.save(str(self.save_path), data)
        
        num_labels_0 = sum(1 for l in original_labels if l == 0)
        num_labels_1 = sum(1 for l in original_labels if l == 1)
        
        print(f"\n{'='*60}")
        print(f"ANNOTATION COMPLETE")
        print(f"{'='*60}")
        print(f"Data saved to: {self.save_path}")
        print(f"Total labeled: {len(self.labels)} traces")
        print(f"  Label 0 (No Event): {num_labels_0}")
        print(f"  Label 1 (Event):    {num_labels_1}")
        print(f"{'='*60}\n")
        
        sys.exit()


def run_manual_annotation(data: dict = None, save_path: str = None):
    """
    Main application entry point.
    
    Parameters
    ----------
    data : dict, optional
        Dictionary containing trace data. If None, user will be prompted to select a file.
    save_path : str, optional
        Path where annotations will be saved. Defaults to 'annotated_traces.h5'
    
    Returns
    -------
    int
        Exit code (0 for success, 1 for error)
    """
    app = QApplication(sys.argv)
    
    # Load data if not provided
    if data is None:
        # File dialog to select HDF5 file
        file_path, _ = QFileDialog.getOpenFileName(
            None, 
            'Select Test Dataset (HDF5)',
            '',
            'HDF5 Files (*.h5 *.hdf5);;All Files (*)'
        )
        
        if not file_path:
            print("No file selected. Exiting.")
            return 1
        
        try:
            data = fl.load(file_path)
            print(f"Loaded dataset from {file_path}")
        except Exception as e:
            QMessageBox.critical(
                None,
                'Error',
                f'Failed to load dataset:\n{str(e)}'
            )
            print(f"Error: {e}")
            return 1
    
    # Validate data
    if isinstance(data, dict):
        if 'dff' in data:
            traces = np.array(data['dff'])
        elif 'trace' in data:
            traces = np.array(data['trace'])
        elif len(data) > 0:
            first_key = list(data.keys())[0]
            traces = np.array(data[first_key])
        else:
            QMessageBox.critical(
                None,
                'Error',
                'Dataset does not contain expected keys (dff, trace, etc.)'
            )
            return 1
        
        if len(traces) == 0:
            QMessageBox.critical(
                None,
                'Error',
                'No traces found in the dataset.'
            )
            return 1
        
        print(f"Loaded {len(traces)} traces")
    else:
        QMessageBox.critical(
            None,
            'Error',
            'Input must be a dictionary'
        )
        return 1
    
    try:
        # Launch viewer
        viewer = ManualAnnotationViewer(data, save_path=save_path)
        return app.exec_()
        
    except Exception as e:
        QMessageBox.critical(
            None,
            'Error',
            f'Failed to initialize viewer:\n{str(e)}'
        )
        print(f"Error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(run_manual_annotation())
