import streamlit as st
import pandas as pd

from copy import deepcopy
from typing import IO

from app.datasets.management.create import create
from app.datasets.management.save import save
from app.datasets.list import list_dataset

from app.core.system import AutoMLSystem

# your code here
automl: AutoMLSystem = AutoMLSystem.get_instance()

st.set_page_config(page_title="Datasets")

st.title("Dataset")

uploaded_file: IO = st.file_uploader(label="Upload dataset(csv)",
                                     accept_multiple_files=False,
                                     type=["csv"])

if uploaded_file is not None:
    # save uploaded data mode.
    version = st.text_input("version number of dataset.",
                            help="format is 1.1.1")

    if (
        st.button("save dataset?") and
            (version == "" or len(version.split(".")) == 3)):
        confirm_save: bool = save(create(deepcopy(uploaded_file), version))

        if confirm_save:
            st.warning("save complete")

    st.write("cancel upload to go back.")

    st.dataframe(pd.read_csv(deepcopy(uploaded_file)))
else:
    # view and delete mode.
    # if there are no new data uploaded
    view_dataset = st.selectbox("select dataset to preview.",
                                list_dataset(
                                    automl.registry.list(type="dataset")),
                                index=None)

    if view_dataset is not None:
        st.dataframe(view_dataset.read())

        if st.button("Delete Dataset?"):
            automl.registry.delete(view_dataset.id)
            st.write("Dataset deleted reload.")
