import pickle

from autoop.core.ml.pipeline import Pipeline
from autoop.core.ml.artifact import Artifact


def save_pipeline(pipeline: Pipeline, name: str, version: str) -> Artifact:
    """
    Save pipeline.

    Arguments:
        pipeline (Pipeline): pipeline to be saved

    returns:
        the central artifact of the pipeline to be saved.
    """
    pipeline_artifacts = pipeline.artifacts

    artifact_pipeline: Artifact = Artifact(
        type="pipeline",
        name=name,
        version=version,
        data=pickle.dumps(
            {"data": [artifact.id for artifact in pipeline_artifacts]}),
        asset_path=f"{version}{name}")

    return artifact_pipeline
