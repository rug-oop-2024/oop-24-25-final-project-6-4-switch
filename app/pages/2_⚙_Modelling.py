import streamlit as st

from app.core.system import AutoMLSystem
from app.datasets.list import list_dataset

from app.modelling.models import get_models
from app.modelling.get_metric import get_metrics
from app.modelling.save import save_pipeline

from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
from autoop.core.ml.pipeline import Pipeline
from autoop.core.ml.metric import Metric
from autoop.core.ml.model.model import Model

from autoop.functional.feature import detect_feature_types


st.set_page_config(page_title="Modelling", page_icon="📈")


def write_helper_text(text: str) -> None:
    """
    Write text.

    Arguments:
        text (str): text to write.

    Returns:
        None
    """
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)


st.write("# ⚙ Modelling")
write_helper_text(
    "In this section, you can design a machine learning pipeline to" +
    "train a model on a dataset.")

automl: AutoMLSystem = AutoMLSystem.get_instance()

datasets: list[Dataset] = list_dataset(automl.registry.list(type="dataset"))

# your code here
# TODO test the classification models.

selected_dataset = st.selectbox("Select dataset to model",
                                datasets)

feature_list: list[Feature] = detect_feature_types(selected_dataset)

target_colum: Feature | None = st.selectbox("select target feature",
                                            feature_list,
                                            index=None)

input_features: list[Feature] | None = st.multiselect(
    "select inout features",
    [feature for feature in feature_list if feature != target_colum],
    default=None)

if target_colum is None:
    task_type: str = "(No target selected.)"
else:
    match target_colum.type:
        case "numerical":
            task_type = "regresion"
        case "categorical":
            task_type = "classification"

st.write(f"Detected task type is {task_type}.")

if target_colum is not None and input_features not in [None, []]:
    split: float = st.slider("Slect split in dataset.",
                             min_value=0.1,
                             max_value=0.9,
                             value=0.8)

    dictionary_models: dict[str, Model] = get_models(task_type)
    model_key: str | None = st.selectbox("select model.",
                                         dictionary_models.keys(),
                                         index=None)
    metrics: list[Metric] | None = st.multiselect("select metrics.",
                                                  get_metrics(task_type),
                                                  default=None)

    if model_key is not None and metrics not in [None, []]:
        instanced_model: None | Model = None
        uninstanced_model: Model = dictionary_models[model_key]

        if st.checkbox("Use custom arguments?"):
            match model_key:
                case "K Nearest Neighbors":
                    instanced_model = uninstanced_model(
                        st.number_input(
                            "k value - the amount of neighbors chosen.",
                            min_value=3,
                            step=1,
                            value=3
                        ))

                case "Logistic Regression":
                    instanced_model = uninstanced_model(
                        st.number_input(
                            "Learning rate for gradient descent.",
                            min_value=0.001,
                            step=0.001,
                            value=0.01),
                        st.number_input(
                            "Number of iterations for gradient descent.",
                            value=1000,
                            min_value=1,
                            step=1
                        ))

                case "Decision Tree":
                    instanced_model = uninstanced_model(
                        st.number_input(
                            "Maximum depth of the tree.",
                            min_value=1
                        ))

                case "Multiple Linear Regression":
                    instanced_model = uninstanced_model(
                        st.number_input(
                            "Regularisation strength.",
                            value=0.0,
                            min_value=0.0
                        ))

                case "Random Forest":
                    instanced_model = uninstanced_model(
                        st.number_input(
                            "Number of trees in the forest.",
                            value=10,
                            min_value=10,
                            step=1),
                        st.number_input(
                            "Maximum depth per tree.",
                            min_value=1,
                            step=1
                        ))

                case _:
                    st.write("No arguments to customize.")
                    instanced_model = uninstanced_model()
        else:
            instanced_model = uninstanced_model()

        if instanced_model is not None:
            pipeline: Pipeline = Pipeline(
                metrics,
                selected_dataset,
                instanced_model,
                input_features,
                target_colum,
                split)

            st.write(pipeline)

            if st.checkbox("auto train train."):
                pipeline_result: dict = pipeline.execute()

                pipeline_result_keys = list(pipeline_result.keys())

                st.write("metrics of the pipeline:")
                st.dataframe(pipeline_result[pipeline_result_keys[0]])

                st.write("metric results of pipeline:")
                st.dataframe(pipeline_result[pipeline_result_keys[1]])

                st.write("predictionss of the pipeline:")
                st.dataframe(pipeline_result[pipeline_result_keys[2]])

                version = st.text_input("version number of dataset.",
                                        help="format is 1.1.1")
                pipeline_name = st.text_input("name of the pipeline")

                if (st.button("save Pipeline?") and
                    (version == "" or len(version.split(".")) == 3) and
                        pipeline_name is not None):
                    central_pipeline_artifact = save_pipeline(pipeline,
                                                              version,
                                                              pipeline_name)

                    automl.registry.register(central_pipeline_artifact)
                    for artifact in pipeline.artifacts:
                        automl.registry.register(artifact)

                    st.write('Pipeline saved')
