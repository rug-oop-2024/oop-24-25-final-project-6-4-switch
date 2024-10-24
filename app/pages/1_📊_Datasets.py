import streamlit as st
import pandas as pd

from app.core.system import AutoMLSystem
from app.datasets.management.create import create
from app.datasets.management.save import save
from autoop.core.ml.dataset import Dataset
from copy import deepcopy
from base64 import b64decode

# your code here
automl = AutoMLSystem.get_instance()

datasets = automl.registry.list(type="dataset")
dataset_names = [_.name for _ in datasets]

st.set_page_config(page_title="Datasets")

st.title("Dataset")

uploaded_file = st.file_uploader(label="Upload dataset(csv)",
                                 accept_multiple_files=False,
                                 type=["csv"])

if uploaded_file is not None:
    version = st.text_input("version number of dataset.",
                            help="format is 1.1.1")\

    if (
        st.button("save dataset?") and
            (version == "" or len(version.split(".")) == 3)):
        new_dataset: Dataset = create(deepcopy(uploaded_file), version)
        confirm_save = save(new_dataset)

        if confirm_save:
            st.warning("save complete")

    st.write("cancel upload to go back.")

    st.dataframe(pd.read_csv(deepcopy(uploaded_file)))
else:
    view_dataset = st.selectbox("select dataset", dataset_names)

    if view_dataset is not None:
        st.dataframe(
            pd.read_csv(
                datasets[dataset_names.index(view_dataset)].asset_path)
            )
