from typing import TYPE_CHECKING
from app.core.system import AutoMLSystem
import pickle
import streamlit as st

if TYPE_CHECKING:
    from autoop.core.ml.artifact import Artifact

automl: AutoMLSystem = AutoMLSystem.get_instance()

st.set_page_config(page_title="Deployment")

st.title("Deployment")

piplines: list["Artifact"] = automl.registry.list(type="pipeline")
piplines_names: list[str] = [artifact.name for artifact in piplines]

current_pipeline: "Artifact" = st.selectbox(
    "select pipeline",
    piplines_names)

if current_pipeline is not None:
    pipeline: "Artifact" = piplines[piplines_names.index(current_pipeline)]
    pipeline_artifact: list["Artifact"] = [automl.registry.get(id) for id
                                           in pipeline.tags]

    for artifact in pipeline_artifact:
        if artifact.name == "pipeline_config":
            input_feature = pickle.loads(artifact.read)["input_features"]
        if "pipeline_model" in artifact.name:
            model_artifact = artifact

    predict_data_file = st.file_uploader(
        label=f"CSV shouls include as input feature: {input_feature}",
        accept_multiple_files=False,
        type=["csv"])

    # turn model artifact into model

    # input the new csv data in the model

    # get model prediction and display to user.
