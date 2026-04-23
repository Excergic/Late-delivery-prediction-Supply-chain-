"""
Monitoring pipeline: layer-1 health + delayed-label performance checks.

Usage:
    python pipelines/monitoring_pipeline.py
"""

from zenml import pipeline

from steps.monitor import monitor_model


@pipeline(name="supply_chain_monitoring_pipeline", enable_cache=False)
def monitoring_pipeline(
    deployment_config_path: str = "configs/deployment_config.yaml",
) -> None:
    _ = monitor_model(config_path=deployment_config_path)


if __name__ == "__main__":
    monitoring_pipeline()
