import sys
import os
import cv2
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QGraphicsView, QGraphicsScene, 
                             QVBoxLayout, QHBoxLayout, QPushButton, QWidget, QFileDialog, 
                             QLabel, QRadioButton, QButtonGroup, QSlider, QGroupBox, QMessageBox,
                             QSpinBox)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt

class VideoProcessor:
    @staticmethod
    def extract_climax_sequence(video_path, num_frames=120):
        cap = cv2.VideoCapture(video_path)
        frames_gray = []
        while True:
            ret, frame = cap.read()
            if not ret: break
            frames_gray.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        cap.release()

        if len(frames_gray) < 2: return []

        diffs = [cv2.absdiff(frames_gray[i], frames_gray[i-1]) for i in range(1, len(frames_gray))]
        sums = [np.sum(d) for d in diffs]
        peak_idx = np.argmax(sums)

        # Empezamos justo en el clímax (o un frame antes por seguridad) y extraemos hacia el futuro
        start = max(0, peak_idx - 1)
        end = min(len(diffs), start + num_frames)

        return diffs[start:end]

class CanvasView(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.image_item = self.scene.addPixmap(QPixmap())
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        
        # --- FIX ZOOM: Anclar el zoom al puntero del ratón ---
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        
        self.current_diff_array = None
        self.label_array = None 
        self.mask_item = None
        
        self.active_tool = "humo"
        self.brush_size = 20
        self.thresh_humo = 15      
        self.thresh_metralla = 60  
        
        self.drawing = False
        self.last_point = None
        self.undo_stack = []
        
        self._pan_start = None
        self.initial_fit_done = False

    def load_frame(self, diff_array, label_array):
        self.current_diff_array = diff_array
        self.label_array = label_array 
        h, w = diff_array.shape
        
        max_val = np.max(diff_array)
        if max_val > 0:
            factor = 255.0 / max_val
            self.display_array = np.clip(diff_array * factor, 0, 255).astype(np.uint8)
        else:
            self.display_array = diff_array.copy()
            
        qimg = QImage(self.display_array.data, w, h, w, QImage.Format.Format_Grayscale8)
        self.image_item.setPixmap(QPixmap.fromImage(qimg))
        self.scene.setSceneRect(self.image_item.boundingRect())
        
        if not self.initial_fit_done:
            self.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            self.initial_fit_done = True
        
        self.undo_stack.clear()
        
        if self.mask_item: 
            self.scene.removeItem(self.mask_item)
        
        self.mask_item = self.scene.addPixmap(QPixmap())
        self.mask_item.setOpacity(0.5) 
        self.render_mask()

    def render_mask(self):
        h, w = self.label_array.shape
        color_map = np.zeros((3, 4), dtype=np.uint8)
        color_map[0] = [0, 0, 0, 0]       
        color_map[1] = [0, 255, 0, 255]   
        color_map[2] = [0, 0, 255, 255]   
        
        self.rgba_data = color_map[self.label_array]
        qimg = QImage(self.rgba_data.data, w, h, w * 4, QImage.Format.Format_RGBA8888)
        self.mask_item.setPixmap(QPixmap.fromImage(qimg))

    def save_undo_state(self):
        # --- NUEVO: Proteger contra clic sin imagen ---
        if self.label_array is None: 
            return
            
        self.undo_stack.append(self.label_array.copy())
        if len(self.undo_stack) > 20:
            self.undo_stack.pop(0)

    # --- FIX ZOOM: Sobrescribir evento de rueda y aceptarlo para evitar scrolleo ---
    def wheelEvent(self, event):
        zoom_in_factor = 1.15
        zoom_out_factor = 1 / zoom_in_factor
        if event.angleDelta().y() > 0:
            self.scale(zoom_in_factor, zoom_in_factor)
        else:
            self.scale(zoom_out_factor, zoom_out_factor)
        event.accept()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Z:
            if self.undo_stack:
                np.copyto(self.label_array, self.undo_stack.pop())
                self.render_mask()
        super().keyPressEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton:
            self._pan_start = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        elif event.button() == Qt.MouseButton.LeftButton:
            # --- NUEVO: Proteger contra clic sin imagen ---
            if self.label_array is None: 
                return
                
            self.save_undo_state()
            self.drawing = True
            scene_pos = self.mapToScene(event.pos())
            self.last_point = scene_pos
            self.paint_smart_mask(scene_pos, scene_pos)

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.RightButton and self._pan_start is not None:
            delta = event.pos() - self._pan_start
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(delta.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(delta.y()))
            self._pan_start = event.pos()
            
        elif self.drawing and event.buttons() & Qt.MouseButton.LeftButton:
            scene_pos = self.mapToScene(event.pos())
            self.paint_smart_mask(self.last_point, scene_pos)
            self.last_point = scene_pos
            
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton:
            self._pan_start = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
        elif event.button() == Qt.MouseButton.LeftButton:
            self.drawing = False
        super().mouseReleaseEvent(event)

    def paint_smart_mask(self, p1, p2):
        if self.current_diff_array is None: return
        h, w = self.current_diff_array.shape
        
        x1, y1 = int(p1.x()), int(p1.y())
        x2, y2 = int(p2.x()), int(p2.y())
        
        stroke_mask = np.zeros((h, w), dtype=np.uint8)
        if x1 == x2 and y1 == y2:
            cv2.circle(stroke_mask, (x1, y1), self.brush_size // 2, 1, -1)
        else:
            cv2.line(stroke_mask, (x1, y1), (x2, y2), 1, thickness=self.brush_size)
            
        if self.active_tool == "humo":
            valid_pixels = (stroke_mask == 1) & (self.current_diff_array >= self.thresh_humo)
            self.label_array[valid_pixels] = 1
        elif self.active_tool == "metralla":
            valid_pixels = (stroke_mask == 1) & (self.current_diff_array >= self.thresh_metralla)
            self.label_array[valid_pixels] = 2
        elif self.active_tool == "borrador":
            self.label_array[stroke_mask == 1] = 0
            
        self.render_mask()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Detovision Annotator - Modo Producción")
        self.resize(1300, 800)
        self.frames = []
        self.labels_list = [] 
        self.current_idx = 0

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        top_layout = QHBoxLayout()
        
        self.btn_load = QPushButton("Cargar (MP4 / NPZ)")
        self.btn_load.clicked.connect(self.load_file)
        
        # --- NUEVO: Control de cantidad de frames a extraer ---
        self.spin_frames = QSpinBox()
        self.spin_frames.setRange(10, 500)
        self.spin_frames.setValue(120)
        self.spin_frames.setToolTip("Cantidad de frames a extraer tras la detonación")
        
        self.btn_export = QPushButton("Guardar Dataset (.npz)")
        self.btn_export.setStyleSheet("background-color: #2e8b57; color: white; font-weight: bold;")
        self.btn_export.clicked.connect(self.export_dataset)
        
        top_layout.addWidget(self.btn_load)
        top_layout.addWidget(QLabel("Frames post-detonación:"))
        top_layout.addWidget(self.spin_frames)
        top_layout.addWidget(self.btn_export)

        group_tools = QGroupBox("Herramienta")
        tools_layout = QHBoxLayout()
        self.radio_humo = QRadioButton("Humo")
        self.radio_humo.setChecked(True)
        self.radio_metralla = QRadioButton("Metralla")
        self.radio_borrador = QRadioButton("Borrador")
        
        self.tool_group = QButtonGroup()
        self.tool_group.addButton(self.radio_humo)
        self.tool_group.addButton(self.radio_metralla)
        self.tool_group.addButton(self.radio_borrador)
        self.tool_group.buttonClicked.connect(self.change_tool)
        
        tools_layout.addWidget(self.radio_humo)
        tools_layout.addWidget(self.radio_metralla)
        tools_layout.addWidget(self.radio_borrador)
        group_tools.setLayout(tools_layout)
        top_layout.addWidget(group_tools)

        group_settings = QGroupBox("Ajustes")
        settings_layout = QHBoxLayout()
        
        self.slider_size = QSlider(Qt.Orientation.Horizontal)
        self.slider_size.setRange(5, 150)
        self.slider_size.setValue(20)
        self.slider_size.setMinimumWidth(150)
        self.lbl_val_size = QLabel("20")
        self.slider_size.valueChanged.connect(self.change_brush_size)
        
        self.slider_thresh_humo = QSlider(Qt.Orientation.Horizontal)
        self.slider_thresh_humo.setRange(1, 100)
        self.slider_thresh_humo.setValue(15)
        self.slider_thresh_humo.setMinimumWidth(150)
        self.lbl_val_humo = QLabel("15")
        self.slider_thresh_humo.valueChanged.connect(self.change_thresh_humo)

        self.slider_thresh_metralla = QSlider(Qt.Orientation.Horizontal)
        self.slider_thresh_metralla.setRange(10, 200)
        self.slider_thresh_metralla.setValue(60)
        self.slider_thresh_metralla.setMinimumWidth(150)
        self.lbl_val_metralla = QLabel("60")
        self.slider_thresh_metralla.valueChanged.connect(self.change_thresh_metralla)

        settings_layout.addWidget(QLabel("Tamaño:"))
        settings_layout.addWidget(self.slider_size)
        settings_layout.addWidget(self.lbl_val_size)
        
        settings_layout.addWidget(QLabel(" Sens. Humo:"))
        settings_layout.addWidget(self.slider_thresh_humo)
        settings_layout.addWidget(self.lbl_val_humo)
        
        settings_layout.addWidget(QLabel(" Sens. Metralla:"))
        settings_layout.addWidget(self.slider_thresh_metralla)
        settings_layout.addWidget(self.lbl_val_metralla)
        
        group_settings.setLayout(settings_layout)
        top_layout.addWidget(group_settings)
        top_layout.addStretch()

        nav_layout = QHBoxLayout()
        self.btn_prev = QPushButton("< Anterior")
        self.btn_prev.clicked.connect(self.prev_frame)
        self.lbl_status = QLabel("Esperando video...")
        self.btn_next = QPushButton("Siguiente >")
        self.btn_next.clicked.connect(self.next_frame)

        nav_layout.addWidget(self.btn_prev)
        nav_layout.addWidget(self.lbl_status)
        nav_layout.addWidget(self.btn_next)

        layout.addLayout(top_layout)
        self.canvas = CanvasView()
        layout.addWidget(self.canvas)
        layout.addLayout(nav_layout)

    def change_tool(self, button):
        if button == self.radio_humo: self.canvas.active_tool = "humo"
        elif button == self.radio_metralla: self.canvas.active_tool = "metralla"
        else: self.canvas.active_tool = "borrador"

    def change_brush_size(self, value):
        self.canvas.brush_size = value
        self.lbl_val_size.setText(str(value))

    def change_thresh_humo(self, value):
        self.canvas.thresh_humo = value
        self.lbl_val_humo.setText(str(value))

    def change_thresh_metralla(self, value):
        self.canvas.thresh_metralla = value
        self.lbl_val_metralla.setText(str(value))

    def load_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Seleccionar Archivo", 
            "", 
            "Videos y Datasets (*.mp4 *.avi *.npz)"
        )
        
        if file_path:
            self.lbl_status.setText("Procesando archivo...")
            QApplication.processEvents()
            
            # --- Si es un dataset guardado (.npz) ---
            if file_path.endswith('.npz'):
                try:
                    data = np.load(file_path)
                    if 'inputs' not in data or 'targets' not in data:
                        raise ValueError("El archivo .npz no tiene las claves correctas.")
                    
                    # Desempaquetar el tensor 3D de vuelta a una lista de matrices 2D
                    self.frames = list(data['inputs'])
                    self.labels_list = list(data['targets'])
                    
                    self.current_idx = 0
                    self.canvas.initial_fit_done = False 
                    self.update_display()
                    self.lbl_status.setText(f"Dataset NPZ cargado: {len(self.frames)} frames")
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"No se pudo leer el archivo:\n{e}")
                    self.lbl_status.setText("Error al cargar.")
                    
            # --- Si es un video crudo (.mp4, .avi) ---
            else:
                num_frames = self.spin_frames.value()
                self.frames = VideoProcessor.extract_climax_sequence(file_path, num_frames=num_frames)
                if self.frames:
                    self.labels_list = [np.zeros_like(f) for f in self.frames]
                    self.current_idx = 0
                    self.canvas.initial_fit_done = False 
                    self.update_display()
                else:
                    self.lbl_status.setText("No se pudo extraer la secuencia.")

    def update_display(self):
        if not self.frames: return
        self.lbl_status.setText(f"Frame {self.current_idx + 1} / {len(self.frames)}")
        self.canvas.load_frame(self.frames[self.current_idx], self.labels_list[self.current_idx])

    def prev_frame(self):
        if self.frames and self.current_idx > 0:
            self.current_idx -= 1
            self.update_display()

    def next_frame(self):
        if self.frames and self.current_idx < len(self.frames) - 1:
            self.current_idx += 1
            self.update_display()

    def export_dataset(self):
        if not self.frames:
            QMessageBox.warning(self, "Error", "No hay secuencias cargadas.")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(self, "Guardar Dataset", "", "Numpy Zipped (*.npz)")
        if not file_path: return
        
        inputs_tensor = np.stack(self.frames)
        targets_tensor = np.stack(self.labels_list)
        
        np.savez_compressed(file_path, inputs=inputs_tensor, targets=targets_tensor)
        QMessageBox.information(self, "Éxito", f"¡Dataset exportado correctamente en:\n{file_path}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())