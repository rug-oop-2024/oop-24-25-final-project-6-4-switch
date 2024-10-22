import streamlit as st
import pandas as pd

from app.core.system import AutoMLSystem
from app.datasets.management.create import create
from autoop.core.ml.dataset import Dataset

automl = AutoMLSystem.get_instance()

datasets = automl.registry.list(type="dataset")
dataset_names = [_.name for _ in datasets]

# your code here

st.set_page_config(page_title="Datasets")

st.title("Dataset")

uploaded_file = st.file_uploader(label="Upload dataset(csv)",
                                 accept_multiple_files=False,
                                 type=["csv"])

if uploaded_file is not None:
    new_dataset = create(uploaded_file)
    st.dataframe(new_dataset.read())
else:
    view_dataset = st.selectbox("select dataset", dataset_names)

    if view_dataset is not None:
        st.dataframe(
            pd.read_csv(datasets[dataset_names.index(view_dataset)].asset_path)
            )
