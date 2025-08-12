import torch
import csv
from pathlib import Path
from datetime import datetime, timedelta
import torchvision.transforms as transforms
from PIL import Image
import sys
import src.harvestDLEstimator

sys.modules["harvestDLEstimator"] = (
    src.harvestDLEstimator
)  # allows pytorch to load the model


class RegresionModel:
    def __init__(self, models_dir: Path, logger=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(models_dir / "days_to_harvest_best_cnn.pt")
        self.logger = logger
        self.class_mapping = {"verde": 0, "rojo": 1}
        self.transform = transforms.Compose(
            [
                transforms.Resize((48, 48)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def _load_model(self, model_path: Path):
        from src.harvestDLEstimator import HarvestCNN

        model = torch.load(model_path, map_location=self.device)
        model.to(self.device)
        model.eval()
        return model

    def predict_single(self, image_path: Path, cherry_class: str):
        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)

            class_encoded = torch.tensor(
                [[self.class_mapping[cherry_class.lower()]]], dtype=torch.float32
            ).to(self.device)

            with torch.no_grad():
                prediction = self.model(image_tensor, class_encoded)
                days_to_harvest = prediction.item()

            return max(0, days_to_harvest)

        except Exception as e:
            print(f"Error procesando {image_path}: {e}")
            return None

    def predict_directory(self, image_dir: Path, output_dir: Path) -> Path:
        if self.logger:
            self.logger.info("Predicting days to harvest for each cherry...")

        output_file = output_dir / "output.csv"
        output_file.touch()

        with open(output_file, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                ["image_path", "cherry_class", "days_to_harvest", "harvest_date"]
            )

        images_paths = [p for p in image_dir.rglob("*") if p.is_file()]  # SIN PROBAR

        for image_path in images_paths:
            cherry_class = image_path.parent.name
            days_to_harvest = self.predict_single(image_path, cherry_class)
            harvest_date = datetime.now() + timedelta(days=days_to_harvest)

            with open(output_file, "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [image_path, cherry_class, days_to_harvest, harvest_date]
                )

        if self.logger:
            self.logger.info(f"Predictions saved to {output_file}")

        return output_file
