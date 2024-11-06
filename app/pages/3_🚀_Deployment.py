from typing import TYPE_CHECKING
from app.core.system import AutoMLSystem
from autoop.core.ml.pipeline import Pipeline
import pickle
import streamlit as st

if TYPE_CHECKING:
    from autoop.core.ml.artifact import Artifact

automl: AutoMLSystem = AutoMLSystem.get_instance

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

    predict_data: list[int | float | str] = []
    for name, type in [(feature.name, feature.type) for
                       feature in input_feature]:
        if type == "numerical":
            predict_data.append(st.number_input(name))
        else:
            predict_data.append(st.text_input(name))
