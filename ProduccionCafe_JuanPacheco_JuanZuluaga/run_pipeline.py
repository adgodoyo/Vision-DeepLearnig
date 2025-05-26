"""
Processes a coffee tree image and generates a detailed production report by dates.

Usage:
    python run_pipeline.py <image_path>
    python run_pipeline.py --help
"""

import sys
import logging
import argparse
import tempfile
from PIL import Image
from pathlib import Path
from src.DetectionModel import DetectionModel
from src.RegresionModel import RegresionModel
from src.ProductionReport import ProductionReport


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ImageProcessorError(Exception):
    """Custom exception for image processing errors."""

    pass


def validate_image_file(file_path: Path) -> None:
    if not file_path.exists():
        raise ImageProcessorError(f"File does not exist: {file_path}")

    if not file_path.is_file():
        raise ImageProcessorError(f"Path is not a file: {file_path}")

    valid_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}
    if file_path.suffix.lower() not in valid_extensions:
        raise ImageProcessorError(
            f"Invalid file extension: {file_path.suffix}. "
            f"Supported formats: {', '.join(sorted(valid_extensions))}"
        )

    # try to open
    try:
        with Image.open(file_path) as img:
            img.verify()
    except Exception as e:
        raise ImageProcessorError(f"Invalid or corrupted image file: {e}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Coffee production computer vision pipeline",
        epilog="Example: python %(prog)s /path/to/image.jpg",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "image_path", type=str, help="Path to the image file to process"
    )

    parser.add_argument(
        "target_days", type=int, help="Target days for production forecast"
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )

    return parser.parse_args()


def process_image(image_path: Path, target_days: int) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        models_dir = Path(".") / "data" / "models"
        results_path = Path(".") / "pipeline_output" / image_path.stem
        results_path.mkdir(parents=True, exist_ok=True)

        # !!! NO FUNCIONA PARA RUTA DE MODELO EN COLAB
        detection = DetectionModel(models_dir=models_dir, logger=logger)
        regresion = RegresionModel(models_dir=models_dir, logger=logger)

        clusters_path = temp_path / "clusters"
        clusters_path.mkdir()
        detection.detect_clusters(image_path, clusters_path)

        cherries_path = temp_path / "cherries"
        cherries_path.mkdir()
        centers, classes = detection.detect_cherries(clusters_path, cherries_path)

        prediction_file = regresion.predict_directory(cherries_path, temp_path)

        detection.create_cherries_plot(image_path, centers, classes, results_path)

        report = ProductionReport(
            weight_per_grain=2.0,
            logger=logger,
            prediction_file=prediction_file,
            output_dir=results_path,
        )
        report.create_production_report(target_days=target_days)

        print()
        print("=" * 20 + " RESUMEN FINAL " + "=" * 20)
        print(
            f"Clusters detectados: {len([f for f in clusters_path.rglob('*') if f.is_file()])}"
        )
        print(
            f"Granos detectados: {len([f for f in cherries_path.rglob('*') if f.is_file()])}"
        )

        report.print_report()

        print("=" * 56)


def main() -> int:
    try:
        args = parse_arguments()

        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.debug("Verbose mode enabled")

        image_path = Path(args.image_path).resolve()
        logger.debug(f"Resolved image path: {image_path}")

        logger.info("Validating image file...")
        validate_image_file(image_path)
        logger.info("Image file validation passed")

        process_image(image_path, target_days=args.target_days)

        return 0

    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.debug("Full traceback:", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
