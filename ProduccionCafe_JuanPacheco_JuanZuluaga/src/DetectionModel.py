import cv2
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image
from typing import List, Tuple, Literal

IOU_CLUSTERS = 0.7
CONF_CLUSTERS = 0.7
CONF_CHERRIES = 0.7
IOU_CHERRIES = 0.7


class DetectionModel:
    def __init__(self, models_dir: Path, logger=None):
        self.clusters_model = YOLO(models_dir / "clusters_best.pt")
        self.cherries_model = YOLO(models_dir / "cherries_best.pt")
        self.logger = logger

    def detect_clusters(
        self, image_path: Path, output_dir: Path, iou=IOU_CLUSTERS, conf=CONF_CLUSTERS
    ) -> None:
        if self.logger:
            self.logger.info("Detecting cherry clusters...")

        results = self.clusters_model.predict(
            image_path, iou=iou, conf=conf, verbose=False, save=False
        )

        image = cv2.imread(image_path)
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy().flatten())
            filename = f"{x1}_{y1}.jpg"

            cluster = image[y1:y2, x1:x2]
            cv2.imwrite(output_dir / filename, cluster)

        if self.logger:
            self.logger.info(f"{len(results[0].boxes)} Clusters detected.")
            self.logger.info(f"Clusters images saved to {output_dir}")

    def detect_cherries(
        self, image_dir: Path, output_dir: Path, iou=IOU_CHERRIES, conf=CONF_CHERRIES
    ) -> Tuple[List[Tuple[int, int]], List[Literal["rojo", "verde"]]]:
        """
        Detect cherries in a directory of images and returns a tuple of lists containing the center coordinates and class of each cherry.
        """
        if self.logger:
            self.logger.info("Detecting cherries...")

        images_paths = [p for p in image_dir.iterdir() if p.is_file()]
        results = self.cherries_model.predict(
            images_paths, iou=iou, conf=conf, verbose=False, save=False
        )

        cherry_counter = 0
        for r in results:
            r.save_crop(output_dir)
            cherry_counter += len(r.boxes)

        if self.logger:
            self.logger.info(f"{cherry_counter} Cherries detected.")
            self.logger.info(f"Cherries images saved to {output_dir}")

        centers = []
        classes = []
        for r in results:
            for box in r.boxes:
                abs_x, abs_y = Path(r.path).stem.split("_")
                x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy().flatten())
                center_x = (x1 + x2) // 2 + int(abs_x)
                center_y = (y1 + y2) // 2 + int(abs_y)
                centers.append((center_x, center_y))
                classes.append("verde" if box.cls.item() == 0 else "rojo")

        return centers, classes

    def create_cherries_plot(
        self,
        image_path: Path,
        centers: List[Tuple[int, int]],
        classes: List[Literal["rojo", "verde"]],
        save_dir: Path = Path("."),
    ) -> None:
        """
        Create a plot with marked detections (only cherries), saving with a custom name.
        """
        if self.logger:
            self.logger.info("Creating cherries detections plot...")

        image = Image.open(image_path)
        image_rgb = image.convert("RGB")
        save_path = save_dir / "detection_result.jpg"

        plt.figure(figsize=(16, 12))
        plt.imshow(image_rgb)

        colors = {"verde": "lime", "rojo": "red", "unknown": "yellow"}

        for center, cls in zip(centers, classes):
            center_x, center_y = center
            color = colors.get(cls)

            plt.plot(
                center_x,
                center_y,
                "o",
                color=color,
                markersize=8,
                markeredgecolor="white",
                markeredgewidth=1,
            )

        plt.title(
            f"Detección de Granos de Café\n" f"Granos detectados: {len(classes)}",
            fontsize=16,
            weight="bold",
        )
        plt.axis("off")

        legend_elements = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="lime",
                markersize=8,
                label="Granos Verdes",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="red",
                markersize=8,
                label="Granos Rojos",
            ),
        ]
        plt.legend(
            handles=legend_elements, loc="upper right", bbox_to_anchor=(0.98, 0.98)
        )

        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.show()

        if self.logger:
            self.logger.info(f"Cherries detections plot saved to {save_path}")
