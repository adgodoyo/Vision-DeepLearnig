import matplotlib.pyplot as plt
from typing import Dict
import pandas as pd
from datetime import datetime, timedelta
import logging
from pathlib import Path


class ProductionReport:
    def __init__(
        self,
        weight_per_grain: float,
        logger: logging.Logger,
        prediction_file: Path,
        output_dir: Path,
    ):
        self.weight_per_grain = weight_per_grain
        self.prediction_file = prediction_file
        self.logger = logger
        self.output_dir = output_dir

    def _calculate_production_forecast(self, target_days: int) -> Dict:
        today = datetime.now()
        target_date = today + timedelta(days=target_days)

        predictions_df = pd.read_csv(self.prediction_file)

        ready_grains = predictions_df[
            predictions_df["days_to_harvest"] <= target_days
        ].copy()

        total_grains = len(ready_grains)
        total_weight_kg = (total_grains * self.weight_per_grain) / 1000

        class_stats = (
            ready_grains.groupby("cherry_class")
            .agg({"days_to_harvest": ["count", "mean", "min", "max"]})
            .round(2)
        )

        return {
            "forecast_date": target_date.strftime("%Y-%m-%d"),
            "target_days": target_days,
            "ready_grains": total_grains,
            "estimated_weight_kg": round(total_weight_kg, 3),
            "class_breakdown": class_stats,
            "percentage_ready": (
                round((total_grains / len(predictions_df)) * 100, 1)
                if len(predictions_df) > 0
                else 0
            ),
        }

    def create_production_report(self, target_days: int = 15) -> None:
        self.logger.info("Creating production report...")

        predictions_df = pd.read_csv(self.prediction_file)

        forecast = self._calculate_production_forecast(target_days)

        class_distribution = predictions_df["cherry_class"].value_counts()

        maturation_stats = {
            "avg_days_to_harvest": predictions_df["days_to_harvest"].mean(),
            "min_days_to_harvest": predictions_df["days_to_harvest"].min(),
            "max_days_to_harvest": predictions_df["days_to_harvest"].max(),
            "std_days_to_harvest": predictions_df["days_to_harvest"].std(),
        }

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(
            f"Reporte de Producción - Análisis {target_days} días", fontsize=16
        )

        # class distribution
        axes[0, 0].pie(
            class_distribution.values,
            labels=class_distribution.index,
            autopct="%1.1f%%",
            colors=["red", "green"],
        )
        axes[0, 0].set_title("Distribución por Clase")

        # maturation histogram
        axes[0, 1].hist(
            predictions_df["days_to_harvest"], bins=20, alpha=0.7, color="orange"
        )
        axes[0, 1].axvline(
            target_days,
            color="red",
            linestyle="--",
            label=f"Objetivo: {target_days} días",
        )
        axes[0, 1].set_xlabel("Días hasta cosecha")
        axes[0, 1].set_ylabel("Número de granos")
        axes[0, 1].set_title("Distribución de Maduración")
        axes[0, 1].legend()

        # cumulative ready grains
        days_range = range(
            1, min(121, int(predictions_df["days_to_harvest"].max()) + 1)
        )
        cumulative_ready = [
            len(predictions_df[predictions_df["days_to_harvest"] <= d])
            for d in days_range
        ]

        axes[1, 0].plot(days_range, cumulative_ready, linewidth=2, color="green")
        axes[1, 0].axvline(target_days, color="red", linestyle="--", alpha=0.7)
        axes[1, 0].set_xlabel("Días desde hoy")
        axes[1, 0].set_ylabel("Granos Acumulados")
        axes[1, 0].set_title("Granos Listos (Acumulativo)")
        axes[1, 0].grid(True, alpha=0.3)

        # production by class
        production_by_class = (
            predictions_df.groupby("cherry_class").size() * self.weight_per_grain / 1000
        )
        axes[1, 1].bar(
            production_by_class.index,
            production_by_class.values,
            color=["red", "green"],
            alpha=0.7,
        )
        axes[1, 1].set_ylabel("Peso Total (kg)")
        axes[1, 1].set_title("Producción Total por Clase")

        plt.tight_layout()

        report_path = self.output_dir / "production_analysis.png"

        plt.savefig(report_path, dpi=300, bbox_inches="tight")
        plt.show()

        self.logger.info(f"Production report saved to {report_path}")

        self.report = {
            "fecha_reporte": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_granos_detectados": len(predictions_df),
            "distribucion_clases": class_distribution.to_dict(),
            "estadisticas_maduracion": maturation_stats,
            "forecast_objetivo": forecast,
            "produccion_total_estimada_kg": round(
                (len(predictions_df) * self.weight_per_grain) / 1000, 3
            ),
            "target_days": target_days,
        }

    def print_report(self):
        if not self.report:
            self.logger.info("No se pudo crear el reporte")
            return

        for cls, count in self.report["distribucion_clases"].items():
            print(f"   • {cls.capitalize()}: {count} granos")

        print(f"\n⏰ Análisis para {self.report['target_days']} días:")
        forecast = self.report.get("forecast_objetivo", {})
        print(f"   • Granos listos para cosecha: {forecast.get('ready_grains', 0)}")
        print(f"   • Producción estimada: {forecast.get('estimated_weight_kg', 0)} kg")
        print(f"   • Porcentaje del total: {forecast.get('percentage_ready', 0)}%")

        print(
            f"\n📈 Producción total estimada: {self.report.get('produccion_total_estimada_kg', 0)} kg"
        )
