from typing import TYPE_CHECKING
from app.core.system import AutoMLSystem
from autoop.core.ml.model import Model
import pickle
import streamlit as st
import pandas as pd

if TYPE_CHECKING:
    from autoop.core.ml.artifact import Artifact

automl: AutoMLSystem = AutoMLSystem.get_instance()

st.set_page_config(page_title="Deployment")

st.title("Deployment")

pipelines: list["Artifact"] = automl.registry.list(type="pipeline")
pipelines_names: list[str] = [artifact.name for artifact in pipelines]

current_pipeline: "Artifact" = st.selectbox(
    "select pipeline",
    pipelines_names)

if current_pipeline is not None:
    pipeline: "Artifact" = pipelines[pipelines_names.index(current_pipeline)]
    pipeline_artifact: list["Artifact"] = [automl.registry.get(id) for id
                                           in pipeline.tags]

    for artifact in pipeline_artifact:
        if artifact.name == "pipeline_config":
            input_feature = pickle.loads(artifact.read)["input_features"]
        if "pipeline_model" in artifact.name:
            model_artifact = artifact

    predict_data_file = st.file_uploader(
        label=f"CSV should include as input feature: {input_feature}",
        accept_multiple_files=False,
        type=["csv"])

    if predict_data_file is not None:
        input_data = pd.read_csv(predict_data_file)

        if input_feature not in input_data.columns:
            st.error(f"CSV must include the feature: {input_feature}")
        else:
            model = pickle.loads(model_artifact.read)
            if not isinstance(model, Model):
                st.error("The loaded model is not an instance of the" +
                         "expected Model class.")
            predictions = model.predict(input_data[input_feature])

            st.subheader("Predictions")
            st.write(predictions)
