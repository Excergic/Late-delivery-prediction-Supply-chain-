"""
Drift detection pipeline: compare current window to reference window.

Usage:
    python pipelines/drift_pipeline.py
"""

from zenml import pipeline

from steps.drift_detect import drift_detect


@pipeline(name="supply_chain_drift_pipeline", enable_cache=False)
def drift_pipeline(
    deployment_config_path: str = "configs/deployment_config.yaml",
) -> None:
    _ = drift_detect(config_path=deployment_config_path)


if __name__ == "__main__":
    drift_pipeline()
