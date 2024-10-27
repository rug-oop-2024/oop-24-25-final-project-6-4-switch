import streamlit as st
import pandas as pd

from app.core.system import AutoMLSystem
from app.datasets.list import list_dataset
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
from autoop.core.ml.model.model import Model
from autoop.functional.feature import detect_feature_types
from app.modelling.models import get_models


st.set_page_config(page_title="Modelling", page_icon="📈")


def write_helper_text(text: str):
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)


st.write("# ⚙ Modelling")
write_helper_text(
    "In this section," +
    "you can design a machine learning pipeline to train a model on a dataset."
    )

automl: AutoMLSystem = AutoMLSystem.get_instance()

datasets: list[Dataset] = list_dataset(automl.registry.list(type="dataset"))

# your code here
dataset_names: list[str] = [_.name for _ in datasets]

selected_dataset = st.selectbox("Select dataset to model", dataset_names)

feature_list: list[Feature] = detect_feature_types(
    datasets[dataset_names.index(selected_dataset)]
    )

target_colum: Feature = st.selectbox("select target feature", feature_list)

input_features: list[Feature] = st.multiselect("select inout features",
                                               [feature for feature in
                                                feature_list
                                                if feature != target_colum])

if target_colum is None:
    task_type: str = "(No target selected.)"
else:
    match target_colum.type:
        case "numerical":
            task_type = "regresion"
        case "categorical":
            task_type = "classification"

st.write(f"Detected task type is {task_type}.")

if target_colum is not None:
    model: Model = st.selectbox("select model.", get_models(task_type))
