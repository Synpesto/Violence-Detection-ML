# gui/app.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os

from predict_utils import predict_image_cnn_wrapper, predict_image_yolo_wrapper
from analysis_utils import build_stats_dataframe, plot_distribution
from cnn_utils import get_transforms

class App(tk.Tk):
    def __init__(self, base_dir=".", resnet_weights="./checkpoints/resnet50_best.pth", yolo_weights="yolo11n.pt"):
        super().__init__()
        self.title("Violence Detection — GUI")
        self.geometry("1000x700")
        self.resnet_weights = resnet_weights
        self.yolo_weights = yolo_weights
        self.base_dir = base_dir

        nb = ttk.Notebook(self)
        nb.pack(fill='both', expand=True)

        self.tab_cnn = ttk.Frame(nb); nb.add(self.tab_cnn, text="CNN")
        self.tab_yolo = ttk.Frame(nb); nb.add(self.tab_yolo, text="YOLO")
        self.tab_analysis = ttk.Frame(nb); nb.add(self.tab_analysis, text="Analysis")
        self.tab_about = ttk.Frame(nb); nb.add(self.tab_about, text="About")

        self._build_cnn_tab()
        self._build_yolo_tab()
        self._build_analysis_tab()
        self._build_about_tab()

    def _build_cnn_tab(self):
        frame = self.tab_cnn
        ttk.Label(frame, text="Image for CNN prediction:").pack(anchor='nw')
        btn = ttk.Button(frame, text="Choose Image", command=self._choose_image_cnn)
        btn.pack(anchor='nw', pady=6)
        self.cnn_image_label = ttk.Label(frame)
        self.cnn_image_label.pack()
        self.cnn_result = ttk.Label(frame, text="Prediction: -")
        self.cnn_result.pack(pady=6)

    def _choose_image_cnn(self):
        p = filedialog.askopenfilename(title="Select image", filetypes=[("Images","*.jpg *.jpeg *.png")])
        if not p: return
        img = Image.open(p).resize((400,300))
        self.cnn_imgtk = ImageTk.PhotoImage(img)
        self.cnn_image_label.configure(image=self.cnn_imgtk)
        # Run prediction
        try:
            res = predict_image_cnn_wrapper(self.resnet_weights, p)
            self.cnn_result.config(text=f"Pred: {res['predicted_class']}  (conf {res['confidence']:.3f})")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _build_yolo_tab(self):
        frame = self.tab_yolo
        ttk.Label(frame, text="Image for YOLO detection:").pack(anchor='nw')
        btn = ttk.Button(frame, text="Choose Image", command=self._choose_image_yolo)
        btn.pack(anchor='nw', pady=6)
        self.yolo_image_label = ttk.Label(frame)
        self.yolo_image_label.pack()
        self.yolo_result = ttk.Label(frame, text="YOLO: -")
        self.yolo_result.pack(pady=6)

    def _choose_image_yolo(self):
        p = filedialog.askopenfilename(title="Select image", filetypes=[("Images","*.jpg *.jpeg *.png")])
        if not p: return
        img = Image.open(p).resize((400,300))
        self.yolo_imgtk = ImageTk.PhotoImage(img)
        self.yolo_image_label.configure(image=self.yolo_imgtk)
        try:
            res = predict_image_yolo_wrapper(self.yolo_weights, p)
            self.yolo_result.config(text=f"Fight: {res['fight']}  People: {len(res['info'].get('person_boxes',[]))}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _build_analysis_tab(self):
        frame = self.tab_analysis
        ttk.Label(frame, text="Build and view frequency/spatial feature distributions").pack(anchor='nw')
        btn = ttk.Button(frame, text="Build features table", command=self._build_features_table)
        btn.pack(anchor='nw', pady=6)
        self.analysis_status = ttk.Label(frame, text="Status: idle")
        self.analysis_status.pack(anchor='nw')

    def _build_features_table(self):
        from data_utils import build_feature_table
        folder = filedialog.askdirectory(title="Select Frequency_Spectrums root (fighting/not_fighting)")
        if not folder: return
        self.analysis_status.config(text="Computing... (this may take a while)")
        df = build_feature_table(folder, out_csv=os.path.join(folder,"features_table.csv"))
        self.analysis_status.config(text=f"Done. {len(df)} rows saved to features_table.csv")
        plot_distribution(df, metric="mean")

    def _build_about_tab(self):
        frame = self.tab_about
        txt = ("Violence Detection GUI\n\n"
               " - Use CNN (ResNet) and YOLO for detection\n"
               " - Features: spatial + frequency extracted and saved\n"
               " - Sequence models and advanced training scripts included in project\n")
        ttk.Label(frame, text=txt, justify="left").pack(anchor='nw', padx=8, pady=8)

if __name__ == "__main__":
    app = App()
    app.mainloop()
