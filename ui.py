import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog, QDoubleSpinBox, QSizePolicy
)
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt
from PIL import Image
import numpy as np

from config import *
import torch
from src.big_image import process_image
from src.data.utils import get_color_mask, calc_percentage
from src.models.DeepLabv3.model import get_model


model = get_model(SAVE_DIR + 'DeepLabv3_epoch_24.pth')

class ImageProcessorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Field Segmentation")
        self.setGeometry(100, 100, 800, 600)
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        
        left_panel = QVBoxLayout()

        dimensions_layout = QVBoxLayout()

        width_layout = QHBoxLayout()
        width_layout.addWidget(QLabel("Width:"))
        self.width_input = QDoubleSpinBox()
        self.width_input.setRange(0.1, 10000.0)
        self.width_input.setValue(10.0)
        width_layout.addWidget(self.width_input)
        width_layout.addWidget(QLabel("km"))
        dimensions_layout.addLayout(width_layout)
        
        height_layout = QHBoxLayout()
        height_layout.addWidget(QLabel("Height:"))
        self.height_input = QDoubleSpinBox()
        self.height_input.setRange(0.1, 10000.0)
        self.height_input.setValue(5.0)
        height_layout.addWidget(self.height_input)
        height_layout.addWidget(QLabel("km"))
        dimensions_layout.addLayout(height_layout)
        
        image_input_layout = QVBoxLayout()
        self.image_path = QLineEdit()
        browse_btn = QPushButton("Browse Image")
        browse_btn.clicked.connect(self.load_image)
        
        process_btn = QPushButton("Process Image")
        process_btn.clicked.connect(self.process_image)

        self.original_label = QLabel("Original Image")
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        left_panel.addLayout(dimensions_layout)
        left_panel.addWidget(QLabel("Image Path:"))
        left_panel.addWidget(self.image_path)
        left_panel.addWidget(browse_btn)
        left_panel.addWidget(process_btn)
        left_panel.addWidget(self.original_label)
        
        right_panel = QVBoxLayout()
        
        self.processed_label = QLabel("Processed Image")
        self.processed_label.setAlignment(Qt.AlignCenter)
        self.processed_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Compact results display
        results_layout = QHBoxLayout()
        self.error_label = QLabel("NoErrors")
        self.growing_land_label = QLabel("Growing area: -- KM^2")
        self.resting_land_label = QLabel("Resting area: -- KM^2")
        results_layout.addWidget(self.error_label)
        results_layout.addWidget(QLabel("|")) 
        results_layout.addWidget(self.growing_land_label)
        results_layout.addWidget(QLabel("|")) 
        results_layout.addWidget(self.resting_land_label)
        results_layout.addStretch()

        self.save_btn = QPushButton("Save Processed Image")
        self.save_btn.clicked.connect(self.save_image)
        self.save_btn.setEnabled(False)
        
        right_panel.addWidget(self.processed_label)
        right_panel.addLayout(results_layout)
        right_panel.addWidget(self.save_btn)
        
        main_layout.addLayout(left_panel, 40)
        main_layout.addLayout(right_panel, 60)
        
        self.original_image = None
        self.processed_image = None

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg)"
        )
        if path:
            self.image_path.setText(path)
            self.show_image(path, self.original_label)

    def show_image(self, path, label):
        image = QImage(path)
        if not image.isNull():
            scaled = image.scaled(
                label.width(), label.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            label.setPixmap(QPixmap.fromImage(scaled))

    def process_image(self):
        width = self.width_input.value()
        height = self.height_input.value()
        image_path = self.image_path.text()
        
        if not image_path:
            return
        
        try:
            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                mask = process_image(img, (width, height), model.to(device), device)
            
                class_percentage = calc_percentage(mask)
            
                img_np = get_color_mask(mask)
                
                if img_np.dtype != np.uint8: 
                    img_np = img_np.astype(np.uint8)

                if not img_np.flags['C_CONTIGUOUS']:
                    img_np = np.ascontiguousarray(img_np)
 
                height, width, channel = img_np.shape
                bytes_per_line = 3 * width
                processed_qimage = QImage(
                    img_np.data,
                    width,
                    height,
                    bytes_per_line,
                    QImage.Format_RGB888
                )

                growing = width * height * class_percentage[1].item()
                resting = width * height * class_percentage[0].item()
                
                self.processed_image = Image.fromarray(img_np)
                self.show_processed_image(processed_qimage)
                
                self.growing_land_label.setText(f"Growing: {growing:.2f} KM^2")
                self.resting_land_label.setText(f"Resting: {resting:.2f} KM^2")
                self.error_label.setText(f"NoErrors")
                self.save_btn.setEnabled(True)
                
        except Exception as e:
            self.error_label.setText(f"Ooops: {e}")
            print(f"Error processing image: {e}")

    def show_processed_image(self, qimage):
        scaled = qimage.scaled(
            self.processed_label.width(), self.processed_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.processed_label.setPixmap(QPixmap.fromImage(scaled))

    def save_image(self):
        if self.processed_image is not None:
            path, _ = QFileDialog.getSaveFileName(
                self, "Save Image", "", "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg)"
            )
            if path:
                self.processed_image.save(path)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageProcessorWindow()
    window.show()
    sys.exit(app.exec())