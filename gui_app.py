import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import torch
import torch.nn as nn
import numpy as np
import os
from torchvision import transforms, models

# --- Module Imports ---
from yolo_utils import load_yolo_model, detect_fight_with_yolo
from cnn_utils import FightDetectionCNN, get_transforms


# --- Model Definitions ---
class FusionModel(nn.Module):
    """
    Fusion Model combining ResNet50 (Spatial) and LSTM/Transformer (Frequency/Sequence).
    """

    def __init__(self, num_classes, model_type='lstm', spec_input_dim=128, hidden_dim=128):
        super(FusionModel, self).__init__()
        self.model_type = model_type.lower()

        # Spatial Branch: ResNet50
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        self.cnn_backbone = nn.Sequential(*list(resnet.children())[:-1])

        for param in self.cnn_backbone.parameters():
            param.requires_grad = False
        self.cnn_out_dim = 2048

        # Frequency Branch: Sequence Model
        if self.model_type == 'lstm':
            self.seq_model = nn.LSTM(input_size=spec_input_dim, hidden_size=hidden_dim, batch_first=True)
        elif self.model_type == 'gru':
            self.seq_model = nn.GRU(input_size=spec_input_dim, hidden_size=hidden_dim, batch_first=True)
        elif self.model_type == 'transformer':
            encoder_layer = nn.TransformerEncoderLayer(d_model=spec_input_dim, nhead=4, batch_first=True)
            self.seq_model = nn.TransformerEncoder(encoder_layer, num_layers=2)
            hidden_dim = spec_input_dim

            # Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(self.cnn_out_dim + hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x_spatial, x_seq):
        # Spatial Pass
        with torch.no_grad():
            # FIX: Must use 'self.cnn_backbone' here too!
            c_out = self.cnn_backbone(x_spatial)
            c_out = c_out.view(c_out.size(0), -1)

            # Sequence Pass
        if self.model_type == 'transformer':
            s_out = self.seq_model(x_seq)
            s_out = s_out.mean(dim=1)
        else:
            out, _ = self.seq_model(x_seq)
            s_out = out[:, -1, :]

            # Concatenation and Classification
        fused = torch.cat((c_out, s_out), dim=1)
        return self.classifier(fused)


class DualImageDatasetHelper:
    """Helper class to compute spectrograms for inference."""

    def __init__(self, spec_size=(128, 128)):
        self.spec_h = spec_size[0]
        self.spec_w = spec_size[1]

    def compute_spectrogram(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None: return np.zeros((self.spec_h, self.spec_w), dtype=np.float32)

        img_resized = cv2.resize(img, (self.spec_w, self.spec_h))
        spec = img_resized.astype(np.float32) / 255.0
        return spec


# --- Main GUI Class ---
class ViolenceDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Violence Detection System")
        self.root.geometry("1000x700")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.current_image_path = None
        self.loaded_models = {}

        # Model Configuration
        self.weights_paths = {
            "YOLOv11 (Heuristic)": "yolo11n.pt",
            "ResNet50 (Transfer Learning)": "checkpoints/resnet50_best.pth",
            "Fusion (CNN+LSTM)": "best_fusion_lstm.pth"
        }

        # UI Layout - Left Panel
        self.left_frame = tk.Frame(root, width=250, bg="#f0f0f0")
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(self.left_frame, text="Control Panel", font=("Arial", 16, "bold"), bg="#f0f0f0").pack(pady=20)

        tk.Label(self.left_frame, text="Select Model:", bg="#f0f0f0").pack(anchor="w", padx=20)
        self.model_var = tk.StringVar(value="ResNet50 (Transfer Learning)")
        self.combo_model = ttk.Combobox(self.left_frame, textvariable=self.model_var, state="readonly")
        self.combo_model['values'] = list(self.weights_paths.keys())
        self.combo_model.pack(fill=tk.X, padx=20, pady=5)

        self.btn_load = tk.Button(self.left_frame, text="📂 Load Image", command=self.load_image, height=2, bg="#ddd")
        self.btn_load.pack(fill=tk.X, padx=20, pady=10)

        self.btn_predict = tk.Button(self.left_frame, text="▶ RUN PREDICTION", command=self.run_prediction, height=3,
                                     bg="#ffcccb", font=("Arial", 10, "bold"))
        self.btn_predict.pack(fill=tk.X, padx=20, pady=10)

        self.lbl_status = tk.Label(self.left_frame, text="Ready", fg="blue", bg="#f0f0f0", wraplength=200)
        self.lbl_status.pack(side=tk.BOTTOM, pady=20)

        # UI Layout - Right Panel
        self.right_frame = tk.Frame(root, bg="white")
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.lbl_result = tk.Label(self.right_frame, text="Result: Waiting...", font=("Arial", 20, "bold"), bg="white",
                                   fg="#333")
        self.lbl_result.pack(pady=10)

        self.canvas_frame = tk.Frame(self.right_frame, bg="gray")
        self.canvas_frame.pack(expand=True, fill=tk.BOTH, padx=20, pady=10)
        self.lbl_image = tk.Label(self.canvas_frame, text="No Image Loaded", bg="#ccc")
        self.lbl_image.pack(expand=True)

    def log(self, message):
        self.lbl_status.config(text=message)
        self.root.update_idletasks()

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png")])
        if file_path:
            self.current_image_path = file_path
            self.display_image(file_path)
            self.lbl_result.config(text="Result: Ready", fg="#333")
            self.log(f"Loaded: {os.path.basename(file_path)}")

    def display_image(self, path, cv_img=None):
        if cv_img is None:
            img = Image.open(path)
        else:
            img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

        w_canvas = 700
        h_canvas = 500
        img.thumbnail((w_canvas, h_canvas))

        self.tk_img = ImageTk.PhotoImage(img)
        self.lbl_image.config(image=self.tk_img, text="")

    # --- Model Loading ---
    def get_yolo_model(self):
        if "yolo" not in self.loaded_models:
            self.log("Loading YOLO model...")
            self.loaded_models["yolo"] = load_yolo_model(self.weights_paths["YOLOv11 (Heuristic)"])
        return self.loaded_models["yolo"]

    def get_resnet_model(self):
        if "resnet" not in self.loaded_models:
            path = self.weights_paths["ResNet50 (Transfer Learning)"]
            if not os.path.exists(path):
                messagebox.showerror("Error", f"Weights not found: {path}")
                return None

            self.log("Loading ResNet50...")
            model = FightDetectionCNN(num_classes=2, pretrained=False)
            model.load_state_dict(torch.load(path, map_location=self.device))
            model.to(self.device).eval()
            self.loaded_models["resnet"] = model
        return self.loaded_models["resnet"]

    def get_fusion_model(self):
        if "fusion" not in self.loaded_models:
            path = self.weights_paths["Fusion (CNN+LSTM)"]
            if not os.path.exists(path):
                messagebox.showerror("Error", f"Weights not found: {path}")
                return None

            self.log("Loading Fusion Model...")
            model = FusionModel(num_classes=2, model_type='lstm', spec_input_dim=128)
            try:
                model.load_state_dict(torch.load(path, map_location=self.device))
            except RuntimeError as e:
                messagebox.showerror("Load Error", f"Model architecture mismatch.\n{e}")
                return None

            model.to(self.device).eval()
            self.loaded_models["fusion"] = model
        return self.loaded_models["fusion"]

    # --- Prediction Logic ---
    def run_prediction(self):
        if not self.current_image_path:
            messagebox.showwarning("Warning", "Please load an image first!")
            return

        model_name = self.model_var.get()
        self.log(f"Running {model_name}...")

        try:
            if "YOLO" in model_name:
                self.predict_yolo()
            elif "ResNet" in model_name:
                self.predict_resnet()
            elif "Fusion" in model_name:
                self.predict_fusion()
        except Exception as e:
            self.log(f"Error: {str(e)}")
            messagebox.showerror("Execution Error", str(e))

    def predict_yolo(self):
        model = self.get_yolo_model()
        is_fight, info = detect_fight_with_yolo(self.current_image_path, model)

        # Visualization
        img = cv2.imread(self.current_image_path)
        results = model(self.current_image_path, verbose=False)[0]
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if int(box.cls[0]) == 0:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = "FIGHT DETECTED!" if is_fight else "Normal / Non-Fight"
        color = "red" if is_fight else "green"

        self.lbl_result.config(text=label, fg=color)
        self.display_image(None, cv_img=img)

        # Extract info
        pair_details = info.get('details') or info.get('pair_details') or []
        count = info.get('person_count', 0)
        self.log(f"Persons: {count} | Pairs: {len(pair_details)}")

    def predict_resnet(self):
        model = self.get_resnet_model()
        if model is None: return

        _, val_tf = get_transforms(img_size=224)
        img_pil = Image.open(self.current_image_path).convert("RGB")
        input_tensor = val_tf(img_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)

        class_idx = predicted.item()
        conf_score = confidence.item() * 100

        if class_idx == 0:
            label = f"FIGHT DETECTED! ({conf_score:.1f}%)"
            color = "red"
        else:
            label = f"Normal / Safe ({conf_score:.1f}%)"
            color = "green"

        self.lbl_result.config(text=label, fg=color)
        self.display_image(self.current_image_path)
        self.log(f"ResNet Probability: {probs.cpu().numpy()}")

    def predict_fusion(self):
        model = self.get_fusion_model()
        if model is None: return

        # Prepare Inputs
        ds_helper = DualImageDatasetHelper()
        spatial_tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        img_pil = Image.open(self.current_image_path).convert("RGB")
        img_tensor = spatial_tf(img_pil).unsqueeze(0).to(self.device)

        spec_matrix = ds_helper.compute_spectrogram(self.current_image_path)
        spec_tensor = torch.tensor(spec_matrix.T, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = model(img_tensor, spec_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)

        class_idx = predicted.item()
        conf_score = confidence.item() * 100

        if class_idx == 0:
            label = f"FIGHT (Fusion) - {conf_score:.1f}%"
            color = "red"
        else:
            label = f"NORMAL (Fusion) - {conf_score:.1f}%"
            color = "green"
        self.lbl_result.config(text=label, fg=color)
        self.display_image(self.current_image_path)
        self.log(f"Fusion Probability: {probs.cpu().numpy()}")


if __name__ == "__main__":
    root = tk.Tk()
    app = ViolenceDetectionApp(root)
    root.mainloop()