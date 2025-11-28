import os
import shutil
import tempfile
import cv2
import numpy as np
try:
    from skimage.metrics import structural_similarity as skimage_ssim
    SKIMAGE_SSIM_AVAILABLE = True
except Exception:
    SKIMAGE_SSIM_AVAILABLE = False
from PyQt5 import QtWidgets, QtCore, QtGui

from colorize_image import colorize_image
import config
import data as data_module
import resnet34_unet as model_module
import datetime
try:
    from Post_Processing import hsl as post_hsl
except Exception:
    post_hsl = None
try:
    from Post_Processing import ops as pp_ops
except Exception:
    pp_ops = None


def safe_ssim(a, b):
    """Compute SSIM robustly between two images. Returns float SSIM or None on error.

    This helper tolerates dtype/shape/channel-order differences and will attempt a channel-swap
    retry if the first attempt fails.
    """
    if not SKIMAGE_SSIM_AVAILABLE:
        return None
    try:
        ia = np.asarray(a)
        ib = np.asarray(b)
        # ensure 3-channel images are uint8
        if ia.ndim == 3 and ia.shape[2] == 3 and ia.dtype != np.uint8:
            ia = np.clip(ia, 0, 255).astype(np.uint8)
        if ib.ndim == 3 and ib.shape[2] == 3 and ib.dtype != np.uint8:
            ib = np.clip(ib, 0, 255).astype(np.uint8)
        # ensure same spatial shape
        if ia.shape[:2] != ib.shape[:2]:
            try:
                ib = cv2.resize(ib, (ia.shape[1], ia.shape[0]))
            except Exception:
                return None
        val = skimage_ssim(ia, ib, data_range=255, multichannel=True)
        return float(val)
    except Exception:
        try:
            # try swapping channels of the first image (BGR <-> RGB) and try again
            if ia.ndim == 3 and ia.shape[2] == 3:
                ia2 = ia[..., ::-1]
                val = skimage_ssim(ia2, ib, data_range=255, multichannel=True)
                return float(val)
        except Exception:
            return None
    return None


class ColorizeThread(QtCore.QThread):
    finished_signal = QtCore.pyqtSignal(str)
    error_signal = QtCore.pyqtSignal(str)

    def __init__(self, in_path, out_path, checkpoint=None, parent=None):
        super().__init__(parent)
        self.in_path = in_path
        self.out_path = out_path
        self.checkpoint = checkpoint
        self.method = 'model'

    def run(self):
        try:
            colorize_image(self.in_path, self.out_path, checkpoint=self.checkpoint, batch=False, method=getattr(self, 'method', 'model'))
            if os.path.exists(self.out_path):
                self.finished_signal.emit(self.out_path)
            else:
                self.error_signal.emit('Colorization finished but output missing')
        except Exception as e:
            self.error_signal.emit(str(e))


class TrainThread(QtCore.QThread):
    progress_signal = QtCore.pyqtSignal(int, int, int, float)  # percent, epoch, batch, loss
    status_signal = QtCore.pyqtSignal(str)
    finished_signal = QtCore.pyqtSignal()
    error_signal = QtCore.pyqtSignal(str)

    def __init__(self, log_path, parent=None):
        super().__init__(parent)
        self.log_path = log_path

    def _progress_cb(self, epoch, batch, total_batches, percent, loss):
        try:
            self.progress_signal.emit(int(percent), int(epoch), int(batch), float(loss))
        except Exception:
            pass

    def run(self):
        try:
            train_data = data_module.DATA(config.TRAIN_DIR)
            net = model_module.MODEL()
            net.build()
            with open(self.log_path, 'w') as log:
                log.write(str(datetime.datetime.now()) + "\n")
                log.write("Training started\n")
                net.train_model(train_data, log, progress_callback=self._progress_cb)
            self.finished_signal.emit()
        except Exception as e:
            self.error_signal.emit(str(e))


def cvimg_to_qpixmap(cv_img, max_size=(512, 512)):
    if cv_img is None:
        return QtGui.QPixmap()
    if len(cv_img.shape) == 2:
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
    elif cv_img.shape[2] == 4:
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGRA2RGB)
    else:
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w = cv_img.shape[:2]
    target_w, target_h = max_size
    scale = min(target_w / w, target_h / h, 1.0)
    new_w, new_h = int(w * scale), int(h * scale)
    rgb = cv2.resize(cv_img, (new_w, new_h))
    qimg = QtGui.QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QtGui.QImage.Format_RGB888)
    return QtGui.QPixmap.fromImage(qimg)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Image Colorizer (PyQt)')
        self.resize(1000, 600)

        self.input_path = None
        self.output_path = None
        self.gt_color_img = None
        self.input_is_gray = False
        self.tempdir = None

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        # main vertical layout: top previews, bottom splitter with controls
        self.main_v_layout = QtWidgets.QVBoxLayout(central)

        left_col = QtWidgets.QVBoxLayout()
        right_col = QtWidgets.QVBoxLayout()

        # Input preview
        self.input_label = QtWidgets.QLabel('Input (none)')
        # use a minimum size so layout can scale on fullscreen
        self.input_label.setMinimumSize(360, 360)
        self.input_label.setStyleSheet('background: #111; border: 1px solid #444;')
        self.input_label.setAlignment(QtCore.Qt.AlignCenter)
        # do not add input_label to left column; it will be placed in the top preview container

        # Buttons row (Open + two Colorize modes)
        btn_row = QtWidgets.QHBoxLayout()
        open_btn = QtWidgets.QPushButton('Open Image')
        open_btn.clicked.connect(self.open_image)
        open_btn.setStyleSheet("QPushButton{background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #4a90e2, stop:1 #357ABD); color: white; padding:8px; border-radius:6px;} QPushButton:hover{background: #5aa0f2}")
        open_btn.setMaximumWidth(140)
        open_btn.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        btn_row.addWidget(open_btn)
        # Load ground-truth color image (optional) for metric comparisons
        gt_btn = QtWidgets.QPushButton('Load Ground Truth')
        gt_btn.clicked.connect(self.load_ground_truth)
        gt_btn.setStyleSheet("QPushButton{background:#666; color: white; padding:6px; border-radius:4px;} QPushButton:hover{background:#777}")
        gt_btn.setMaximumWidth(160)
        gt_btn.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        btn_row.addWidget(gt_btn)
        # Reconstruction colorize
        self.colorize_recon_btn = QtWidgets.QPushButton('Colorize — Reconstruction')
        self.colorize_recon_btn.clicked.connect(lambda: self.start_colorize(method='reconstruct'))
        self.colorize_recon_btn.setEnabled(False)
        self.colorize_recon_btn.setStyleSheet("QPushButton{background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #7bd389, stop:1 #3aa65a); color: white; padding:8px; border-radius:6px;} QPushButton:hover{background: #8beea0}")
        self.colorize_recon_btn.setMaximumWidth(180)
        self.colorize_recon_btn.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        btn_row.addWidget(self.colorize_recon_btn)
        # Grayscale-based colorize
        self.colorize_gray_btn = QtWidgets.QPushButton('Colorize — Grayscale')
        self.colorize_gray_btn.clicked.connect(lambda: self.start_colorize(method='grayscale'))
        self.colorize_gray_btn.setEnabled(False)
        self.colorize_gray_btn.setStyleSheet("QPushButton{background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #f5a623, stop:1 #f08b1a); color: white; padding:8px; border-radius:6px;} QPushButton:hover{background: #f6b942}")
        self.colorize_gray_btn.setMaximumWidth(180)
        self.colorize_gray_btn.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        btn_row.addWidget(self.colorize_gray_btn)
        left_col.addLayout(btn_row)

        # Checkpoint selector: combobox + refresh + manual override
        cp_layout = QtWidgets.QHBoxLayout()
        self.checkpoint_combo = QtWidgets.QComboBox()
        self.checkpoint_combo.setEditable(False)
        self.checkpoint_combo.addItem('Default (use config / RESUME_FROM)')
        cp_layout.addWidget(self.checkpoint_combo)
        refresh_btn = QtWidgets.QPushButton('Refresh')
        refresh_btn.setFixedWidth(80)
        refresh_btn.clicked.connect(lambda: self.populate_checkpoints())
        cp_layout.addWidget(refresh_btn)
        left_col.addLayout(cp_layout)
        self.checkpoint_edit = QtWidgets.QLineEdit()
        self.checkpoint_edit.setPlaceholderText('Or paste a full checkpoint path to override')
        left_col.addWidget(self.checkpoint_edit)

        # populate initial list of checkpoints
        try:
            self.populate_checkpoints()
        except Exception:
            pass

        # Save button (centered, not full-width)
        save_btn = QtWidgets.QPushButton('Save Output')
        save_btn.clicked.connect(self.save_output)
        save_btn.setEnabled(False)
        save_btn.setMaximumWidth(140)
        save_btn.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.save_btn = save_btn
        save_row = QtWidgets.QHBoxLayout()
        save_row.addStretch(1)
        save_row.addWidget(save_btn)
        save_row.addStretch(1)
        left_col.addLayout(save_row)

        # (Preview size policy will be set after both preview widgets are created)

        # Training controls
        train_box = QtWidgets.QGroupBox('Training')
        tb_layout = QtWidgets.QVBoxLayout()
        form = QtWidgets.QFormLayout()
        self.epochs_spin = QtWidgets.QSpinBox()
        self.epochs_spin.setRange(1, 10000)
        try:
            self.epochs_spin.setValue(config.NUM_EPOCHS)
        except Exception:
            self.epochs_spin.setValue(10)
        form.addRow('Epochs:', self.epochs_spin)
        self.batch_spin = QtWidgets.QSpinBox()
        self.batch_spin.setRange(1, 1024)
        try:
            self.batch_spin.setValue(config.BATCH_SIZE)
        except Exception:
            self.batch_spin.setValue(1)
        form.addRow('Batch size:', self.batch_spin)
        tb_layout.addLayout(form)
        self.train_btn = QtWidgets.QPushButton('Train')
        self.train_btn.clicked.connect(self.start_train)
        self.train_btn.setMaximumWidth(160)
        self.train_btn.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        train_btn_row = QtWidgets.QHBoxLayout()
        train_btn_row.addStretch(1)
        train_btn_row.addWidget(self.train_btn)
        train_btn_row.addStretch(1)
        tb_layout.addLayout(train_btn_row)
        self.train_progress = QtWidgets.QProgressBar()
        self.train_progress.setRange(0, 100)
        tb_layout.addWidget(self.train_progress)
        train_box.setLayout(tb_layout)
        left_col.addWidget(train_box)

        # wrap left column layout into a widget so we can put it into a splitter later
        self.left_widget = QtWidgets.QWidget()
        self.left_widget.setLayout(left_col)
        self.left_widget.setMinimumWidth(480)

        # Output preview
        self.output_label = QtWidgets.QLabel('Output (none)')
        # use a minimum size so layout can scale on fullscreen
        self.output_label.setMinimumSize(360, 360)
        self.output_label.setStyleSheet('background: #111; border: 1px solid #444;')
        self.output_label.setAlignment(QtCore.Qt.AlignCenter)
        # do not add output_label to right column; it will be placed in the top preview container

        # We'll place the two preview labels into a top preview area so they stay aligned
        self.preview_container = QtWidgets.QWidget()
        preview_layout = QtWidgets.QHBoxLayout()
        # remove outer margins and spacing so previews touch with no gaps
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(0)
        self.preview_container.setLayout(preview_layout)
        # make labels expand to fill the preview area with no gaps
        self.input_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.output_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        preview_layout.addWidget(self.input_label)
        preview_layout.addWidget(self.output_label)
        # ensure labels will preserve aspect and can be sized by our update routine
        try:
            self.input_label.setScaledContents(True)
            self.output_label.setScaledContents(True)
        except Exception:
            pass
        # add the preview area to the main vertical layout
        try:
            self.main_v_layout.addWidget(self.preview_container)
        except Exception:
            # fallback: if main_v_layout not present, insert at top of central
            central_layout = central.layout() if central is not None else None
            if isinstance(central_layout, QtWidgets.QVBoxLayout):
                central_layout.insertWidget(0, self.preview_container)

        # status
        self.status = QtWidgets.QLabel('Ready')
        right_col.addWidget(self.status)

        # Postprocessing panel
        pp_box = QtWidgets.QGroupBox('Postprocessing (optional)')
        pp_layout = QtWidgets.QVBoxLayout()
        self.pp_enable = QtWidgets.QCheckBox('Enable postprocessing (show warning on sensitive images)')
        pp_layout.addWidget(self.pp_enable)
        # (removed unused pipeline mode dropdown to simplify UI)

        # Manual HLS sliders removed; HSL is now a pipeline op

        btns = QtWidgets.QHBoxLayout()
        self.pp_preview_btn = QtWidgets.QPushButton('Preview')
        self.pp_preview_btn.clicked.connect(self.on_pp_preview)
        btns.addWidget(self.pp_preview_btn)
        self.pp_apply_btn = QtWidgets.QPushButton('Apply')
        self.pp_apply_btn.clicked.connect(self.on_pp_apply)
        btns.addWidget(self.pp_apply_btn)
        self.pp_reset_btn = QtWidgets.QPushButton('Reset')
        self.pp_reset_btn.clicked.connect(self.on_pp_reset)
        btns.addWidget(self.pp_reset_btn)
        pp_layout.addLayout(btns)

        # Pipeline list (for multiple ops)
        self.pp_pipeline_list = QtWidgets.QListWidget()
        self.pp_pipeline_list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.pp_pipeline_list.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        # populate with available ops
        pipeline_ops = ['HSL', 'CLAHE', 'Histogram Stretch', 'Histogram Equalization', 'Histogram Shrink', 'Unsharp Mask', 'Lab Chroma Boost']
        for op in pipeline_ops:
            item = QtWidgets.QListWidgetItem(op)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
            item.setCheckState(QtCore.Qt.Unchecked)
            self.pp_pipeline_list.addItem(item)
        # make the pipeline list a fixed height so parameters remain visible
        self.pp_pipeline_list.setFixedHeight(140)
        pp_layout.addWidget(self.pp_pipeline_list)

        # Parameter stacked widget
        self.pp_param_stack = QtWidgets.QStackedWidget()
        # HSL page (auto or manual parameters)
        hsl_page = QtWidgets.QWidget()
        hf = QtWidgets.QFormLayout()
        self.pp_hsl_mode = QtWidgets.QComboBox()
        self.pp_hsl_mode.addItem('Auto')
        self.pp_hsl_mode.addItem('Manual')
        hf.addRow('Mode:', self.pp_hsl_mode)
        # manual controls
        self.pp_hsl_hue = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.pp_hsl_hue.setRange(-180, 180)
        self.pp_hsl_hue.setValue(0)
        hf.addRow('Hue shift (deg):', self.pp_hsl_hue)
        self.pp_hsl_sat = QtWidgets.QDoubleSpinBox()
        self.pp_hsl_sat.setRange(0.5, 2.0)
        self.pp_hsl_sat.setSingleStep(0.1)
        self.pp_hsl_sat.setValue(1.0)
        hf.addRow('Saturation scale:', self.pp_hsl_sat)
        self.pp_hsl_light = QtWidgets.QDoubleSpinBox()
        self.pp_hsl_light.setRange(0.5, 1.5)
        self.pp_hsl_light.setSingleStep(0.1)
        self.pp_hsl_light.setValue(1.0)
        hf.addRow('Lightness scale:', self.pp_hsl_light)
        hsl_page.setLayout(hf)
        self.pp_param_stack.addWidget(hsl_page)

        # CLAHE page
        clahe_page = QtWidgets.QWidget()
        g = QtWidgets.QFormLayout()
        self.pp_clahe_clip = QtWidgets.QDoubleSpinBox()
        self.pp_clahe_clip.setRange(0.1, 10.0)
        self.pp_clahe_clip.setSingleStep(0.1)
        self.pp_clahe_clip.setValue(2.0)
        g.addRow('Clip limit:', self.pp_clahe_clip)
        self.pp_clahe_tile = QtWidgets.QLineEdit('8,8')
        g.addRow('Tile grid (w,h):', self.pp_clahe_tile)
        clahe_page.setLayout(g)
        self.pp_param_stack.addWidget(clahe_page)
        # Histogram Stretch page
        hs_page = QtWidgets.QWidget()
        f = QtWidgets.QFormLayout()
        self.pp_stretch_low = QtWidgets.QDoubleSpinBox()
        self.pp_stretch_low.setRange(0.0, 50.0)
        self.pp_stretch_low.setValue(2.0)
        f.addRow('Low %:', self.pp_stretch_low)
        self.pp_stretch_high = QtWidgets.QDoubleSpinBox()
        self.pp_stretch_high.setRange(50.0, 100.0)
        self.pp_stretch_high.setValue(98.0)
        f.addRow('High %:', self.pp_stretch_high)
        hs_page.setLayout(f)
        self.pp_param_stack.addWidget(hs_page)
        # Equalize page (no params)
        eql_page = QtWidgets.QWidget()
        eql_page.setLayout(QtWidgets.QVBoxLayout())
        eql_page.layout().addWidget(QtWidgets.QLabel('No parameters'))
        self.pp_param_stack.addWidget(eql_page)
        # Shrink page
        sh_page = QtWidgets.QWidget()
        sf = QtWidgets.QFormLayout()
        self.pp_shrink_factor = QtWidgets.QDoubleSpinBox()
        self.pp_shrink_factor.setRange(0.0, 1.0)
        self.pp_shrink_factor.setSingleStep(0.05)
        self.pp_shrink_factor.setValue(0.8)
        sf.addRow('Shrink factor:', self.pp_shrink_factor)
        sh_page.setLayout(sf)
        self.pp_param_stack.addWidget(sh_page)
        # Unsharp page
        us_page = QtWidgets.QWidget()
        uf = QtWidgets.QFormLayout()
        self.pp_unsharp_amount = QtWidgets.QDoubleSpinBox()
        self.pp_unsharp_amount.setRange(0.0, 3.0)
        self.pp_unsharp_amount.setSingleStep(0.1)
        self.pp_unsharp_amount.setValue(1.0)
        uf.addRow('Amount:', self.pp_unsharp_amount)
        self.pp_unsharp_radius = QtWidgets.QSpinBox()
        self.pp_unsharp_radius.setRange(1, 10)
        self.pp_unsharp_radius.setValue(1)
        uf.addRow('Radius:', self.pp_unsharp_radius)
        us_page.setLayout(uf)
        self.pp_param_stack.addWidget(us_page)
        # Lab chroma page
        lb_page = QtWidgets.QWidget()
        lf = QtWidgets.QFormLayout()
        self.pp_lab_factor = QtWidgets.QDoubleSpinBox()
        self.pp_lab_factor.setRange(0.5, 3.0)
        self.pp_lab_factor.setSingleStep(0.1)
        self.pp_lab_factor.setValue(1.2)
        lf.addRow('Chroma factor:', self.pp_lab_factor)
        lb_page.setLayout(lf)
        self.pp_param_stack.addWidget(lb_page)
        # give the parameter panel a reasonable fixed height so controls are visible
        self.pp_param_stack.setFixedHeight(180)
        pp_layout.addWidget(self.pp_param_stack)

        # connect selection change to show parameter page
        self.pp_pipeline_list.currentRowChanged.connect(self.pp_param_stack.setCurrentIndex)
        # select first op by default so parameters are visible
        if self.pp_pipeline_list.count() > 0:
            self.pp_pipeline_list.setCurrentRow(0)
        # status / warning label for the postprocessing panel
        self.pp_warning = QtWidgets.QLabel('')
        self.pp_warning.setWordWrap(True)
        pp_layout.addWidget(self.pp_warning)

        # finalize postprocessing group and add to the right column wrapped in a scroll area
        pp_box.setLayout(pp_layout)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(pp_box)
        right_col.addWidget(scroll)

        # wrap right column and place both columns inside a horizontal splitter
        self.right_widget = QtWidgets.QWidget()
        self.right_widget.setLayout(right_col)
        self.right_widget.setMinimumWidth(360)

        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.splitter.addWidget(self.left_widget)
        self.splitter.addWidget(self.right_widget)
        # set initial sizes and stretch factors so side panels remain visible when maximized
        try:
                self.splitter.setSizes([700, 360])
                self.splitter.setStretchFactor(0, 2)
                self.splitter.setStretchFactor(1, 1)
        except Exception:
                pass

        # add splitter below the preview area
        try:
            self.main_v_layout.addWidget(self.splitter)
        except Exception:
            # fallback: try to add to the central widget layout if available
            central_layout = central.layout() if central is not None else None
            if isinstance(central_layout, QtWidgets.QVBoxLayout):
                central_layout.addWidget(self.splitter)
            else:
                # last resort: ignore failure without raising (preserve UI startup)
                try:
                    # try again safely (some layouts accept addWidget differently)
                    self.main_v_layout.addWidget(self.splitter)
                except Exception:
                    pass

        # If pipeline backend not available, disable pipeline controls and show clear message
        try:
            if pp_ops is None:
                self.pp_pipeline_list.setEnabled(False)
                self.pp_preview_btn.setEnabled(False)
                self.pp_apply_btn.setEnabled(False)
                self.pp_warning.setText('Pipeline ops module not available — install or check Post_Processing/ops.py')
                self.pp_warning.setStyleSheet('color: #b00')
            # if HSL module not available, disable HSL item but keep pipeline usable
            if post_hsl is None:
                for i in range(self.pp_pipeline_list.count()):
                    it = self.pp_pipeline_list.item(i)
                    if it.text() == 'HSL':
                        it.setFlags(it.flags() & ~QtCore.Qt.ItemIsEnabled)
                        it.setToolTip('HSL module not available — auto mode disabled')
        except Exception:
            pass



    # Note: Move Up / Move Down buttons removed; QListWidget drag-drop handles reordering.

    def open_image(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open image', os.getcwd(), 'Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)')
        if not path:
            return
        self.input_path = path
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        # store original color image if available for metrics; detect grayscale inputs
        try:
            orig_color = cv2.imread(path, cv2.IMREAD_COLOR)
            if orig_color is not None and len(orig_color.shape) == 3 and orig_color.shape[2] == 3:
                # detect whether the loaded image is essentially grayscale (all channels nearly equal)
                try:
                    ch0 = orig_color[..., 0].astype(np.int32)
                    ch1 = orig_color[..., 1].astype(np.int32)
                    ch2 = orig_color[..., 2].astype(np.int32)
                    mean_diff = (np.mean(np.abs(ch0 - ch1)) + np.mean(np.abs(ch1 - ch2)) + np.mean(np.abs(ch0 - ch2))) / 3.0
                    if mean_diff < 2.0:
                        # treat as grayscale input — keep orig for display but don't use it as ground truth
                        self.input_is_gray = True
                        self.orig_color_img = orig_color
                        self.gt_color_img = None
                        self.status.setText('Loaded grayscale-like input; load ground-truth color for metric checks')
                    else:
                        self.input_is_gray = False
                        self.orig_color_img = orig_color
                except Exception:
                    self.input_is_gray = False
                    self.orig_color_img = orig_color
            else:
                self.orig_color_img = None
                self.input_is_gray = False
        except Exception:
            self.orig_color_img = None
            self.input_is_gray = False
        pm = cvimg_to_qpixmap(img, max_size=(460, 460))
        self.input_label.setPixmap(pm)
        self.input_label.setText('')
        self.colorize_recon_btn.setEnabled(True)
        self.colorize_gray_btn.setEnabled(True)

    def load_ground_truth(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Load ground truth color image', os.getcwd(), 'Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)')
        if not path:
            return
        try:
            gt = cv2.imread(path, cv2.IMREAD_COLOR)
            if gt is None:
                self.status.setText('Failed to load ground truth image')
                return
            self.gt_color_img = gt
            # indicate to user that ground-truth is now set
            self.status.setText('Loaded ground truth image for metric comparison')
        except Exception as e:
            self.status.setText('Failed to load ground truth: ' + str(e))

    def populate_checkpoints(self):
        """Scan the MODEL_DIR for .pth files and populate the combobox."""
        self.checkpoint_combo.blockSignals(True)
        self.checkpoint_combo.clear()
        self.checkpoint_combo.addItem('Default (use config / RESUME_FROM)')
        model_dir = getattr(config, 'MODEL_DIR', os.path.join(os.getcwd(), 'MODEL'))
        try:
            files = sorted([f for f in os.listdir(model_dir) if f.lower().endswith('.pth')])
        except Exception:
            files = []
        for f in files:
            self.checkpoint_combo.addItem(os.path.join(model_dir, f))
        self.checkpoint_combo.blockSignals(False)

    def start_colorize(self, method='model'):
        if not self.input_path:
            return
        # prepare temp dir
        self.tempdir = tempfile.mkdtemp(prefix='colorize_pyqt_')
        out_name = os.path.splitext(os.path.basename(self.input_path))[0] + '_color.png'
        out_path = os.path.join(self.tempdir, out_name)
        # priority: manual override textbox > combobox selection (not default) > None
        manual = self.checkpoint_edit.text().strip()
        if manual:
            checkpoint = manual
        else:
            # combobox index 0 is the 'Default' placeholder
            try:
                idx = self.checkpoint_combo.currentIndex()
                if idx > 0:
                    checkpoint = self.checkpoint_combo.currentText()
                else:
                    checkpoint = None
            except Exception:
                checkpoint = None

        # disable UI
        self.colorize_recon_btn.setEnabled(False)
        self.colorize_gray_btn.setEnabled(False)
        self.status.setText('Colorizing... (this may take a while)')
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)

        # start thread
        self.thread = ColorizeThread(self.input_path, out_path, checkpoint=checkpoint, parent=self)
        # store requested method on thread so colorize_image can receive it
        self.thread.method = method
        self.thread.finished_signal.connect(self.on_finished)
        self.thread.error_signal.connect(self.on_error)
        self.thread.start()

    def start_train(self):
        # set config values from UI
        try:
            config.NUM_EPOCHS = int(self.epochs_spin.value())
            config.BATCH_SIZE = int(self.batch_spin.value())
        except Exception:
            pass

        # disable UI
        self.train_btn.setEnabled(False)
        self.status.setText('Starting training...')
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)

        # start train thread
        temp_log_dir = os.path.join(os.getcwd(), 'LOGS', config.DATASET)
        os.makedirs(temp_log_dir, exist_ok=True)
        log_path = os.path.join(temp_log_dir, datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + f"_train_{config.BATCH_SIZE}_{config.NUM_EPOCHS}.txt")
        self.train_thread = TrainThread(log_path)
        self.train_thread.progress_signal.connect(self.on_train_progress)
        self.train_thread.status_signal.connect(self.on_train_status)
        self.train_thread.finished_signal.connect(self.on_train_finished)
        self.train_thread.error_signal.connect(self.on_train_error)
        self.train_thread.start()

    def on_train_progress(self, percent, epoch, batch, loss):
        try:
            self.train_progress.setValue(int(percent))
            self.status.setText(f'Training — epoch {epoch}, batch {batch}, loss {loss:.4f}')
        except Exception:
            pass

    def on_train_status(self, text):
        self.status.setText(text)

    def on_train_finished(self):
        QtWidgets.QApplication.restoreOverrideCursor()
        self.status.setText('Training finished')
        self.train_btn.setEnabled(True)
        # refresh available checkpoints (training likely saved new files)
        try:
            self.populate_checkpoints()
        except Exception:
            pass

    def on_train_error(self, msg):
        QtWidgets.QApplication.restoreOverrideCursor()
        self.status.setText('Training error: ' + msg)
        self.train_btn.setEnabled(True)

    def on_finished(self, out_path):
        QtWidgets.QApplication.restoreOverrideCursor()
        self.output_path = out_path
        img = cv2.imread(out_path, cv2.IMREAD_UNCHANGED)
        pm = cvimg_to_qpixmap(img, max_size=(460, 460))
        self.output_label.setPixmap(pm)
        self.output_label.setText('')
        self.status.setText(f'Done — output saved to {out_path}')
        self.save_btn.setEnabled(True)
        self.colorize_recon_btn.setEnabled(True)
        self.colorize_gray_btn.setEnabled(True)
        # compute metrics if a comparison image is available
        try:
            out_img = cv2.imread(out_path, cv2.IMREAD_COLOR)
            metrics_text = 'Metrics: output unreadable'
            if out_img is not None:
                # choose comparison image: prefer explicit ground-truth if loaded, otherwise use the input image
                comp_img = None
                if hasattr(self, 'gt_color_img') and self.gt_color_img is not None:
                    comp_img = self.gt_color_img
                    comp_label = 'Ground-truth'
                elif hasattr(self, 'orig_color_img') and self.orig_color_img is not None:
                    comp_img = self.orig_color_img
                    comp_label = 'Input'
                else:
                    comp_img = None

                if comp_img is not None:
                    # resize output to match comparison image for fair metrics
                    if comp_img.shape[:2] != out_img.shape[:2]:
                        out_r = cv2.resize(out_img, (comp_img.shape[1], comp_img.shape[0]))
                    else:
                        out_r = out_img
                    try:
                        mse = float(((comp_img.astype('float32') - out_r.astype('float32')) ** 2).mean())
                        psnr = 10.0 * np.log10((255.0 ** 2) / mse) if mse > 0 else float('inf')
                        metrics_text = f'{comp_label} vs Output — MSE: {mse:.2f}   PSNR: {psnr:.2f} dB'
                    except Exception:
                        metrics_text = f'{comp_label} vs Output — metrics error'
                else:
                    metrics_text = 'Metrics: no comparison image available'
            else:
                metrics_text = 'Metrics: output unreadable'
        except Exception as e:
            metrics_text = 'Metrics error: ' + str(e)

        # show metrics in status (and keep previous status)
        self.status.setText(self.status.text() + ' | ' + metrics_text)

        # Only show a modal warning when the user provided an explicit ground-truth image
        try:
            if not (hasattr(self, 'gt_color_img') and self.gt_color_img is not None):
                # no ground-truth loaded: do not show modal warnings automatically
                return

            # compute numeric checks against ground-truth (gt_color_img)
            violation_msgs = []
            if out_img is None:
                return
            gt = self.gt_color_img
            # resize output to match ground-truth
            if gt.shape[:2] != out_img.shape[:2]:
                out_r = cv2.resize(out_img, (gt.shape[1], gt.shape[0]))
            else:
                out_r = out_img

            # compute PSNR
            try:
                mse = float(((gt.astype('float32') - out_r.astype('float32')) ** 2).mean())
                psnr = 10.0 * np.log10((255.0 ** 2) / mse) if mse > 0 else float('inf')
            except Exception:
                psnr = float('inf')

            # compute SSIM if available
            ssim_val = None
            if SKIMAGE_SSIM_AVAILABLE:
                    try:
                        gt_rgb = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
                        out_rgb = cv2.cvtColor(out_r, cv2.COLOR_BGR2RGB)
                        ssim_val = safe_ssim(gt_rgb, out_rgb)
                    except Exception:
                        ssim_val = None

            # Threshold rules (tunable): PSNR < 20 dB, SSIM < 0.50
            try:
                if not np.isinf(psnr) and psnr < 20.0:
                    violation_msgs.append(f'Low PSNR ({psnr:.2f} dB) — image may be poor quality compared to ground-truth.')
            except Exception:
                pass
            if ssim_val is not None:
                try:
                    if ssim_val < 0.50:
                        violation_msgs.append(f'Low SSIM ({ssim_val:.3f}) — perceptual similarity to ground-truth is low.')
                except Exception:
                    pass

            if violation_msgs:
                msg = 'The colorized image triggered one or more quality warnings:\n\n' + '\n'.join(violation_msgs) + '\n\nYou may want to adjust postprocessing or try a different checkpoint.'
                QtWidgets.QMessageBox.warning(self, 'Colorization Warning', msg)
        except Exception:
            pass

    # --- Postprocessing handlers ---
    # Manual HLS sliders removed; HSL auto-enhance is available as a pipeline op

    def on_pp_preview(self):
        if not self.pp_enable.isChecked():
            self.pp_warning.setText('Enable postprocessing first')
            return
        # decide source image: prefer current output if exists, otherwise input
        src_path = self.output_path or self.input_path
        if not src_path:
            self.pp_warning.setText('Open and/or colorize an image first')
            return
        img = cv2.imread(src_path, cv2.IMREAD_COLOR)
        if img is None:
            self.pp_warning.setText('Failed to read source image for preview')
            return
        # Always run pipeline mode (HSL is a pipeline op now)
        out_bgr = img.copy()
        if pp_ops is None:
            self.pp_warning.setText('Pipeline ops module not available')
            return
        # apply enabled ops in the current QListWidget order
        try:
            # iterate through items in visual order
            for i in range(self.pp_pipeline_list.count()):
                item = self.pp_pipeline_list.item(i)
                if item.checkState() != QtCore.Qt.Checked:
                    continue
                op_name = item.text()
                params = {}
                if op_name == 'HSL':
                    # HSL can be Auto (module) or Manual (user sliders)
                    if self.pp_hsl_mode.currentText() == 'Auto':
                        if post_hsl is None:
                            self.pp_warning.setText('HSL module not available')
                            continue
                        try:
                            in_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
                            out_rgb = post_hsl.enhance_hsl(in_rgb)
                            out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
                            continue
                        except Exception as e:
                            self.pp_warning.setText('HSL module error: ' + str(e))
                            continue
                    else:
                        # manual HSL transform based on sliders
                        try:
                            h_deg = int(self.pp_hsl_hue.value())
                            sat_scale = float(self.pp_hsl_sat.value())
                            light_scale = float(self.pp_hsl_light.value())
                            hls = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2HLS).astype(np.float32)
                            h = hls[..., 0]
                            l = hls[..., 1]
                            s = hls[..., 2]
                            h = (h + (h_deg / 2.0)) % 180.0
                            s = np.clip(s * sat_scale, 0, 255)
                            l = np.clip(l * light_scale, 0, 255)
                            hls[..., 0] = h
                            hls[..., 1] = l
                            hls[..., 2] = s
                            out_bgr = cv2.cvtColor(hls.astype(np.uint8), cv2.COLOR_HLS2BGR)
                            continue
                        except Exception:
                            # fallback: skip HSL on error
                            continue
                if op_name == 'CLAHE':
                    params['clip_limit'] = float(self.pp_clahe_clip.value())
                    try:
                        txt = self.pp_clahe_tile.text()
                        p = tuple([int(x.strip()) for x in txt.split(',')])
                        params['tile_grid_size'] = p
                    except Exception:
                        params['tile_grid_size'] = (8, 8)
                elif op_name == 'Histogram Stretch':
                    params['low_perc'] = float(self.pp_stretch_low.value())
                    params['high_perc'] = float(self.pp_stretch_high.value())
                elif op_name == 'Histogram Equalization':
                    pass
                elif op_name == 'Histogram Shrink':
                    params['factor'] = float(self.pp_shrink_factor.value())
                elif op_name == 'Unsharp Mask':
                    params['amount'] = float(self.pp_unsharp_amount.value())
                    params['radius'] = int(self.pp_unsharp_radius.value())
                elif op_name == 'Lab Chroma Boost':
                    params['factor'] = float(self.pp_lab_factor.value())
                out_bgr = pp_ops.apply_op(out_bgr, op_name, params)
        except Exception as e:
            self.pp_warning.setText('Pipeline error: ' + str(e))
            return

        # show preview in output_label (do not overwrite file yet)
        pm = cvimg_to_qpixmap(out_bgr[:, :, ::-1])  # convert BGR->RGB for pixmap helper
        self.output_label.setPixmap(pm)
        self.output_label.setText('')

        # compute metrics if a comparison image is available (prefer explicitly loaded ground-truth)
        try:
            comp_img = None
            comp_label = 'Comparison'
            if hasattr(self, 'gt_color_img') and self.gt_color_img is not None:
                comp_img = self.gt_color_img
                comp_label = 'Ground-truth'
            elif hasattr(self, 'orig_color_img') and self.orig_color_img is not None and not getattr(self, 'input_is_gray', False):
                comp_img = self.orig_color_img
                comp_label = 'Input'

            if comp_img is None:
                self.pp_warning.setText('Preview metrics: no comparison image available (load ground-truth to enable)')
            else:
                preview_rgb = out_bgr[:, :, ::-1]
                if comp_img.shape[:2] != preview_rgb.shape[:2]:
                    preview_rgb = cv2.resize(preview_rgb, (comp_img.shape[1], comp_img.shape[0]))
                mse = float(((comp_img.astype('float32') - preview_rgb.astype('float32')) ** 2).mean())
                psnr = 10.0 * np.log10((255.0 ** 2) / mse) if mse > 0 else float('inf')
                metrics_text = f'{comp_label} vs Preview — MSE: {mse:.2f}   PSNR: {psnr:.2f} dB'

                ssim_val = None
                ssim_text = ''
                if SKIMAGE_SSIM_AVAILABLE:
                    try:
                        ssim_val = safe_ssim(comp_img[..., ::-1], preview_rgb)
                        ssim_text = f'   SSIM: {ssim_val:.3f}' if ssim_val is not None else '   SSIM: error'
                    except Exception:
                        ssim_text = '   SSIM: error'
                else:
                    ssim_text = '   SSIM: (scikit-image not installed)'

                # compare against last saved output (if exists) for relative warning
                warning = ''
                base_img = None
                if self.output_path and os.path.exists(self.output_path):
                    base_img = cv2.imread(self.output_path, cv2.IMREAD_COLOR)
                elif comp_img is not None:
                    base_img = comp_img if comp_img is not None else None

                if base_img is not None:
                    # compare shapes
                    base_cmp = base_img
                    if base_cmp.shape[:2] != preview_rgb.shape[:2]:
                        base_cmp = cv2.resize(base_cmp, (preview_rgb.shape[1], preview_rgb.shape[0]))
                    base_mse = float(((base_cmp.astype('float32') - preview_rgb.astype('float32')) ** 2).mean())
                    base_psnr = 10.0 * np.log10((255.0 ** 2) / base_mse) if base_mse > 0 else float('inf')
                    if base_psnr - psnr > 2.0:
                        warning = 'Warning: preview reduces PSNR by >2 dB — may harm metric-sensitive tasks.'
                    if SKIMAGE_SSIM_AVAILABLE:
                        try:
                            base_rgb = base_cmp[:, :, ::-1]
                            base_ssim = safe_ssim(base_rgb, preview_rgb)
                            if base_ssim is not None and (base_ssim - (ssim_val if ssim_val is not None else 0.0) > 0.05):
                                if warning:
                                    warning += ' '
                                warning += 'Warning: preview SSIM decreased by >0.05 — may harm perceptual quality.'
                        except Exception:
                            pass

                self.pp_warning.setText(metrics_text + ssim_text + (' | ' + warning if warning else ''))
        except Exception as e:
            self.pp_warning.setText('Metrics error: ' + str(e))

    def on_pp_apply(self):
        # apply postprocessing and overwrite output file (or create one)
        if not self.pp_enable.isChecked():
            self.pp_warning.setText('Enable postprocessing first')
            return
        src_path = self.output_path or self.input_path
        if not src_path:
            self.pp_warning.setText('Open and/or colorize an image first')
            return
        img = cv2.imread(src_path, cv2.IMREAD_COLOR)
        if img is None:
            self.pp_warning.setText('Failed to read source image for apply')
            return
        # Always run pipeline mode (HSL is a pipeline op now)
        out_bgr = img.copy()
        if pp_ops is None:
            self.pp_warning.setText('Pipeline ops module not available')
            return
        try:
            for i in range(self.pp_pipeline_list.count()):
                item = self.pp_pipeline_list.item(i)
                if item.checkState() != QtCore.Qt.Checked:
                    continue
                op_name = item.text()
                params = {}
                if op_name == 'HSL':
                    # HSL can be Auto (module) or Manual (user sliders)
                    if self.pp_hsl_mode.currentText() == 'Auto':
                        if post_hsl is None:
                            self.pp_warning.setText('HSL module not available')
                            continue
                        try:
                            in_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
                            out_rgb = post_hsl.enhance_hsl(in_rgb)
                            out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
                            continue
                        except Exception as e:
                            self.pp_warning.setText('HSL module error: ' + str(e))
                            continue
                    else:
                        # manual HSL transform based on sliders
                        try:
                            h_deg = int(self.pp_hsl_hue.value())
                            sat_scale = float(self.pp_hsl_sat.value())
                            light_scale = float(self.pp_hsl_light.value())
                            hls = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2HLS).astype(np.float32)
                            h = hls[..., 0]
                            l = hls[..., 1]
                            s = hls[..., 2]
                            h = (h + (h_deg / 2.0)) % 180.0
                            s = np.clip(s * sat_scale, 0, 255)
                            l = np.clip(l * light_scale, 0, 255)
                            hls[..., 0] = h
                            hls[..., 1] = l
                            hls[..., 2] = s
                            out_bgr = cv2.cvtColor(hls.astype(np.uint8), cv2.COLOR_HLS2BGR)
                            continue
                        except Exception:
                            # fallback: skip HSL on error
                            continue
                elif op_name == 'CLAHE':
                    params['clip_limit'] = float(self.pp_clahe_clip.value())
                    try:
                        txt = self.pp_clahe_tile.text()
                        p = tuple([int(x.strip()) for x in txt.split(',')])
                        params['tile_grid_size'] = p
                    except Exception:
                        params['tile_grid_size'] = (8, 8)
                elif op_name == 'Histogram Stretch':
                    params['low_perc'] = float(self.pp_stretch_low.value())
                    params['high_perc'] = float(self.pp_stretch_high.value())
                elif op_name == 'Histogram Equalization':
                    pass
                elif op_name == 'Histogram Shrink':
                    params['factor'] = float(self.pp_shrink_factor.value())
                elif op_name == 'Unsharp Mask':
                    params['amount'] = float(self.pp_unsharp_amount.value())
                    params['radius'] = int(self.pp_unsharp_radius.value())
                elif op_name == 'Lab Chroma Boost':
                    params['factor'] = float(self.pp_lab_factor.value())
                out_bgr = pp_ops.apply_op(out_bgr, op_name, params)
        except Exception as e:
            self.pp_warning.setText('Pipeline apply error: ' + str(e))
            return

        # write to a new file in tempdir and update output_path
        if not self.tempdir:
            self.tempdir = tempfile.mkdtemp(prefix='colorize_pyqt_')
        out_name = os.path.splitext(os.path.basename(src_path))[0] + '_post.png'
        out_path = os.path.join(self.tempdir, out_name)
        cv2.imwrite(out_path, out_bgr)
        self.output_path = out_path
        pm = cvimg_to_qpixmap(out_bgr[:, :, ::-1])
        self.output_label.setPixmap(pm)
        self.output_label.setText('')
        self.save_btn.setEnabled(True)
        # compute metrics vs original color if available and vs previous base
        try:
            metrics = []
            if hasattr(self, 'orig_color_img') and self.orig_color_img is not None:
                orig = self.orig_color_img
                preview_rgb = out_bgr[:, :, ::-1]
                if orig.shape[:2] != preview_rgb.shape[:2]:
                    preview_rgb = cv2.resize(preview_rgb, (orig.shape[1], orig.shape[0]))
                mse = float(((orig.astype('float32') - preview_rgb.astype('float32')) ** 2).mean())
                psnr = 10.0 * np.log10((255.0 ** 2) / mse) if mse > 0 else float('inf')
                metrics.append(f'MSE: {mse:.2f} PSNR: {psnr:.2f} dB')
                if SKIMAGE_SSIM_AVAILABLE:
                    try:
                        orig_rgb = orig[..., ::-1]
                        ssim_val = safe_ssim(orig_rgb, preview_rgb)
                        if ssim_val is not None:
                            metrics.append(f'SSIM: {ssim_val:.3f}')
                        else:
                            metrics.append('SSIM: error')
                    except Exception:
                        metrics.append('SSIM: error')

            # compare to previous output if exists
            warning = ''
            base_img = None
            if self.output_path and os.path.exists(self.output_path):
                # note: base_img equals the new file we just wrote; look for prior file by name without '_post' if exists
                # attempt to find previous version in tempdir
                prev_name = os.path.splitext(os.path.basename(src_path))[0] + '.png'
                possible_prev = os.path.join(self.tempdir, prev_name) if self.tempdir else None
                if possible_prev and os.path.exists(possible_prev):
                    base_img = cv2.imread(possible_prev, cv2.IMREAD_COLOR)
            if base_img is not None:
                base_rgb = base_img[:, :, ::-1]
                preview_rgb = out_bgr[:, :, ::-1]
                if base_rgb.shape[:2] != preview_rgb.shape[:2]:
                    base_rgb = cv2.resize(base_rgb, (preview_rgb.shape[1], preview_rgb.shape[0]))
                try:
                    base_mse = float(((base_rgb.astype('float32') - preview_rgb.astype('float32')) ** 2).mean())
                    base_psnr = 10.0 * np.log10((255.0 ** 2) / base_mse) if base_mse > 0 else float('inf')
                    # if PSNR dropped >2 dB show warning
                    if base_psnr - psnr > 2.0:
                        warning = 'Warning: applied result reduces PSNR by >2 dB.'
                    if SKIMAGE_SSIM_AVAILABLE:
                        try:
                            base_ssim = safe_ssim(base_rgb, preview_rgb)
                            if base_ssim is not None and (base_ssim - (ssim_val if 'ssim_val' in locals() else 0.0) > 0.05):
                                if warning:
                                    warning += ' '
                                warning += 'Warning: applied result SSIM decreased by >0.05.'
                        except Exception:
                            pass
                except Exception:
                    pass

            msg = 'Applied postprocessing and updated output'
            if metrics:
                msg += ' | ' + ' '.join(metrics)
            if warning:
                msg += ' | ' + warning
            self.pp_warning.setText(msg)
            # show modal warning to the user when postprocessing reduced metrics noticeably
            try:
                if warning:
                    QtWidgets.QMessageBox.warning(self, 'Postprocessing Warning', msg)
            except Exception:
                pass
        except Exception as e:
            self.pp_warning.setText('Applied postprocessing (metrics error): ' + str(e))

    def on_pp_reset(self):
        # reset preview to last saved output or input
        src_path = self.output_path or self.input_path
        if not src_path:
            return
        img = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return
        pm = cvimg_to_qpixmap(img[:, :, ::-1])
        self.output_label.setPixmap(pm)
        self.output_label.setText('')
        self.pp_warning.setText('Reset preview')

    def on_error(self, message):
        QtWidgets.QApplication.restoreOverrideCursor()
        self.status.setText('Error: ' + message)
        self.colorize_recon_btn.setEnabled(True)
        self.colorize_gray_btn.setEnabled(True)

    def save_output(self):
        if not self.output_path or not os.path.exists(self.output_path):
            return
        dest, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save output', os.path.join(os.getcwd(), os.path.basename(self.output_path)), 'PNG Image (*.png);;JPEG Image (*.jpg *.jpeg)')
        if not dest:
            return
        try:
            shutil.copyfile(self.output_path, dest)
            self.status.setText('Saved: ' + dest)
        except Exception as e:
            self.status.setText('Save failed: ' + str(e))

    def update_preview_sizes(self):
        """Make input and output preview labels the same square size and refresh pixmaps."""
        try:
            # compute available sizes from the splitter panes (if present)
            central = self.centralWidget() if hasattr(self, 'centralWidget') else None
            # derive sizes from the preview container so both labels exactly fill it
            try:
                pc = getattr(self, 'preview_container', None)
                if pc is not None:
                    total_w = pc.width()
                    total_h = pc.height()
                else:
                    total_w = central.width() if central is not None else self.width()
                    total_h = central.height() if central is not None else self.height()

                # divide width evenly between the two previews
                half_w = max(120, total_w // 2)

                # cap height to available preview container height so previews stretch vertically
                usable_h = max(120, total_h - 8)
                h = usable_h

                # set both previews to equal width/height so they appear even and fill vertically
                try:
                    self.input_label.setFixedSize(half_w, h)
                    self.output_label.setFixedSize(half_w, h)
                except Exception:
                    self.input_label.setMaximumHeight(h)
                    self.output_label.setMaximumHeight(h)
            except Exception:
                pass

            # refresh displayed pixmaps to fit new size
            try:
                if getattr(self, 'input_path', None):
                    img = cv2.imread(self.input_path, cv2.IMREAD_UNCHANGED)
                    if img is not None:
                        pm = cvimg_to_qpixmap(img, max_size=(self.input_label.width(), self.input_label.height()))
                        self.input_label.setPixmap(pm)
            except Exception:
                pass
            try:
                if getattr(self, 'output_path', None) and os.path.exists(self.output_path):
                    out_img = cv2.imread(self.output_path, cv2.IMREAD_UNCHANGED)
                    if out_img is not None:
                        pm = cvimg_to_qpixmap(out_img, max_size=(self.output_label.width(), self.output_label.height()))
                        self.output_label.setPixmap(pm)
            except Exception:
                pass
        except Exception:
            pass

    def resizeEvent(self, event):
        # keep previews identical size when window is resized
        try:
            self.update_preview_sizes()
        except Exception:
            pass
        return super().resizeEvent(event)


def main():
    import sys
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
