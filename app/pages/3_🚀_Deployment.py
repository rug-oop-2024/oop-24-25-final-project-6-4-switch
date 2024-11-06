from typing import TYPE_CHECKING
from app.core.system import AutoMLSystem
import streamlit as st

if TYPE_CHECKING:
    from autoop.core.ml.artifact import Artifact

automl: AutoMLSystem = AutoMLSystem.get_instance

st.set_page_config(page_title="Deployment")

st.title("Deployment")

piplines: list["Artifact"] = automl.registry.list(type="pipeline")

current_pipeline = st.selectbox("select pipeline",
                                [Artifact.name for artifact in piplines])

if current_pipeline is not None:
    pass
