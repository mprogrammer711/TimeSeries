import pandas as pd
import numpy as np
import ipywidgets as widgets
from IPython.display import display
import json

# Global variables to store selected values
selected_features = []
selected_temporal_features = []
selected_spot_features = []
selected_macro_features = []
selected_hyperparameters = {}
selected_models = []

def save_selected_parameters():
    global selected_features, selected_temporal_features, selected_spot_features, selected_macro_features, selected_hyperparameters, selected_models
    parameters = {
        'selected_features': selected_features,
        'selected_temporal_features': selected_temporal_features,
        'selected_spot_features': selected_spot_features,
        'selected_macro_features': selected_macro_features,
        'selected_hyperparameters': selected_hyperparameters,
        'selected_models': selected_models
    }
    with open('selected_parameters.json', 'w') as f:
        json.dump(parameters, f)

def interactive_selection(features_csv='features.csv', temporal_features_csv='temporal_features.csv', spot_features_csv='spot_features.csv', macro_features_csv='macro_features.csv'):
    global selected_features, selected_temporal_features, selected_spot_features, selected_macro_features, selected_hyperparameters, selected_models

    # Read features from CSV files
    features_df = pd.read_csv(features_csv)
    temporal_features_df = pd.read_csv(temporal_features_csv)
    spot_features_df = pd.read_csv(spot_features_csv)
    macro_features_df = pd.read_csv(macro_features_csv)

    all_features = features_df['feature_name'].tolist()
    all_temporal_features = temporal_features_df['feature_name'].tolist()
    all_spot_features = spot_features_df['feature_name'].tolist()
    all_macro_features = macro_features_df['feature_name'].tolist()

    # Function to split list into chunks
    def chunk_list(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    # Split the features into 3 columns
    num_columns = 3
    chunks = list(chunk_list(all_features, len(all_features) // num_columns + 1))
    temporal_chunks = list(chunk_list(all_temporal_features, len(all_temporal_features) // num_columns + 1))
    spot_chunks = list(chunk_list(all_spot_features, len(all_spot_features) // num_columns + 1))
    macro_chunks = list(chunk_list(all_macro_features, len(all_macro_features) // num_columns + 1))

    # Create checkboxes for each feature
    checkboxes = [[widgets.Checkbox(value=True, description=feature) for feature in chunk] for chunk in chunks]
    temporal_checkboxes = [[widgets.Checkbox(value=True, description=feature) for feature in chunk] for chunk in temporal_chunks]
    spot_checkboxes = [[widgets.Checkbox(value=True, description=feature) for feature in chunk] for chunk in spot_chunks]
    macro_checkboxes = [[widgets.Checkbox(value=True, description=feature) for feature in chunk] for chunk in macro_chunks]

    # Create a grid layout for the checkboxes
    grid = widgets.GridBox([item for sublist in checkboxes for item in sublist],
                           layout=widgets.Layout(grid_template_columns=f'repeat({num_columns}, 200px)'))
    temporal_grid = widgets.GridBox([item for sublist in temporal_checkboxes for item in sublist],
                                    layout=widgets.Layout(grid_template_columns=f'repeat({num_columns}, 200px)'))
    spot_grid = widgets.GridBox([item for sublist in spot_checkboxes for item in sublist],
                                layout=widgets.Layout(grid_template_columns=f'repeat({num_columns}, 200px)'))
    macro_grid = widgets.GridBox([item for sublist in macro_checkboxes for item in sublist],
                                 layout=widgets.Layout(grid_template_columns=f'repeat({num_columns}, 200px)'))

    # Function to get selected features
    def get_selected_features(checkboxes):
        return [checkbox.description for sublist in checkboxes for checkbox in sublist if checkbox.value]

    # Add buttons to select/deselect all features
    button_select_all_features = widgets.Button(description="Select All Features")
    button_deselect_all_features = widgets.Button(description="Deselect All Features")

    def on_button_select_all_features(b):
        for sublist in checkboxes:
            for checkbox in sublist:
                checkbox.value = True

    def on_button_deselect_all_features(b):
        for sublist in checkboxes:
            for checkbox in sublist:
                checkbox.value = False

    button_select_all_features.on_click(on_button_select_all_features)
    button_deselect_all_features.on_click(on_button_deselect_all_features)

    # Add a button to confirm feature selection
    button_features = widgets.Button(description="Confirm Features")
    output_features = widgets.Output()

    def on_button_clicked_features(b):
        global selected_features
        with output_features:
            selected_features = get_selected_features(checkboxes)
            output_features.clear_output()
            print("Selected features:", selected_features)
            save_selected_parameters()

    button_features.on_click(on_button_clicked_features)

    # Add buttons to select/deselect all temporal features
    button_select_all_temporal_features = widgets.Button(description="Select All Temporal Features")
    button_deselect_all_temporal_features = widgets.Button(description="Deselect All Temporal Features")

    def on_button_select_all_temporal_features(b):
        for sublist in temporal_checkboxes:
            for checkbox in sublist:
                checkbox.value = True

    def on_button_deselect_all_temporal_features(b):
        for sublist in temporal_checkboxes:
            for checkbox in sublist:
                checkbox.value = False

    button_select_all_temporal_features.on_click(on_button_select_all_temporal_features)
    button_deselect_all_temporal_features.on_click(on_button_deselect_all_temporal_features)

    # Add a button to confirm temporal feature selection
    button_temporal_features = widgets.Button(description="Confirm Temporal Features")
    output_temporal_features = widgets.Output()

    def on_button_clicked_temporal_features(b):
        global selected_temporal_features
        with output_temporal_features:
            selected_temporal_features = get_selected_features(temporal_checkboxes)
            output_temporal_features.clear_output()
            print("Selected temporal features:", selected_temporal_features)
            save_selected_parameters()

    button_temporal_features.on_click(on_button_clicked_temporal_features)

    # Add buttons to select/deselect all spot features
    button_select_all_spot_features = widgets.Button(description="Select All Spot Features")
    button_deselect_all_spot_features = widgets.Button(description="Deselect All Spot Features")

    def on_button_select_all_spot_features(b):
        for sublist in spot_checkboxes:
            for checkbox in sublist:
                checkbox.value = True

    def on_button_deselect_all_spot_features(b):
        for sublist in spot_checkboxes:
            for checkbox in sublist:
                checkbox.value = False

    button_select_all_spot_features.on_click(on_button_select_all_spot_features)
    button_deselect_all_spot_features.on_click(on_button_deselect_all_spot_features)

    # Add a button to confirm spot feature selection
    button_spot_features = widgets.Button(description="Confirm Spot Features")
    output_spot_features = widgets.Output()

    def on_button_clicked_spot_features(b):
        global selected_spot_features
        with output_spot_features:
            selected_spot_features = get_selected_features(spot_checkboxes)
            output_spot_features.clear_output()
            print("Selected spot features:", selected_spot_features)
            save_selected_parameters()

    button_spot_features.on_click(on_button_clicked_spot_features)

    # Add buttons to select/deselect all macro features
    button_select_all_macro_features = widgets.Button(description="Select All Macro Features")
    button_deselect_all_macro_features = widgets.Button(description="Deselect All Macro Features")

    def on_button_select_all_macro_features(b):
        for sublist in macro_checkboxes:
            for checkbox in sublist:
                checkbox.value = True

    def on_button_deselect_all_macro_features(b):
        for sublist in macro_checkboxes:
            for checkbox in sublist:
                checkbox.value = False

    button_select_all_macro_features.on_click(on_button_select_all_macro_features)
    button_deselect_all_macro_features.on_click(on_button_deselect_all_macro_features)

    # Add a button to confirm macro feature selection
    button_macro_features = widgets.Button(description="Confirm Macro Features")
    output_macro_features = widgets.Output()

    def on_button_clicked_macro_features(b):
        global selected_macro_features
        with output_macro_features:
            selected_macro_features = get_selected_features(macro_checkboxes)
            output_macro_features.clear_output()
            print("Selected macro features:", selected_macro_features)
            save_selected_parameters()

    button_macro_features.on_click(on_button_clicked_macro_features)

    # Create widgets for hyperparameters with adjusted layout
    layout = widgets.Layout(width='400px')  # Adjust the width of the description area

    learning_rate_slider = widgets.FloatSlider(
        value=0.001,
        min=0.0001,
        max=0.01,
        step=0.00001,  # Increase precision
        description='Learning Rate:',
        continuous_update=False,
        readout_format='.5f',  # Display more decimal places
        style={'description_width': 'initial'},
        layout=layout
    )

    n_epochs_slider = widgets.IntSlider(
        value=50,
        min=10,
        max=200,
        step=10,
        description='Epochs:',
        continuous_update=False,
        style={'description_width': 'initial'},
        layout=layout
    )

    hidden_size_slider = widgets.IntSlider(
        value=128,
        min=32,
        max=512,
        step=32,
        description='Hidden Size:',
        continuous_update=False,
        style={'description_width': 'initial'},
        layout=layout
    )

    dropout_slider = widgets.FloatSlider(
        value=0.3,
        min=0.0,
        max=0.5,
        step=0.1,
        description='Dropout Rate:',
        continuous_update=False,
        style={'description_width': 'initial'},
        layout=layout
    )

    # Display the widgets for hyperparameters
    hyperparameter_selection = widgets.VBox([
        learning_rate_slider,
        n_epochs_slider,
        hidden_size_slider,
        dropout_slider
    ])

    # Add a button to confirm hyperparameters
    button_hyperparams = widgets.Button(description="Confirm Hyperparameters")
    output_hyperparams = widgets.Output()

    def on_button_clicked_hyperparams(b):
        global selected_hyperparameters
        with output_hyperparams:
            selected_hyperparameters = {
                'learning_rate': learning_rate_slider.value,
                'n_epochs': n_epochs_slider.value,
                'hidden_size': hidden_size_slider.value,
                'dropout': dropout_slider.value
            }
            output_hyperparams.clear_output()
            print(f"Selected Hyperparameters:\nLearning Rate: {selected_hyperparameters['learning_rate']:.5f}\nEpochs: {selected_hyperparameters['n_epochs']}\nHidden Size: {selected_hyperparameters['hidden_size']}\nDropout Rate: {selected_hyperparameters['dropout']}")
            save_selected_parameters()

    button_hyperparams.on_click(on_button_clicked_hyperparams)

    # Create checkboxes for model selection
    lstm_checkbox = widgets.Checkbox(value=False, description='LSTM', style={'description_width': 'initial'}, layout=layout)
    momentum_checkbox = widgets.Checkbox(value=False, description='Momentum Strategy', style={'description_width': 'initial'}, layout=layout)
    moving_average_checkbox = widgets.Checkbox(value=False, description='Moving Average', style={'description_width': 'initial'}, layout=layout)

    # Create a list of model checkboxes
    model_checkboxes = [lstm_checkbox, momentum_checkbox, moving_average_checkbox]

    # Add buttons to select/deselect all models
    button_select_all_models = widgets.Button(description="Select All Models")
    button_deselect_all_models = widgets.Button(description="Deselect All Models")

    def on_button_select_all_models(b):
        for checkbox in model_checkboxes:
            checkbox.value = True

    def on_button_deselect_all_models(b):
        for checkbox in model_checkboxes:
            checkbox.value = False

    button_select_all_models.on_click(on_button_select_all_models)
    button_deselect_all_models.on_click(on_button_deselect_all_models)

    # Display the model selection widgets
    model_selection = widgets.VBox(model_checkboxes)

    # Add a button to confirm model selection
    button_model = widgets.Button(description="Confirm Model")
    output_model = widgets.Output()

    def on_button_clicked_model(b):
        global selected_models
        with output_model:
            output_model.clear_output()
            selected_models = [checkbox.description for checkbox in model_checkboxes if checkbox.value]
            print(f"Selected Models: {selected_models}")
            save_selected_parameters()

    button_model.on_click(on_button_clicked_model)

    # Display all sections in a structured layout
    display(widgets.VBox([
        widgets.HTML("<h2>Feature Selection</h2>"),
        grid,
        widgets.HBox([button_select_all_features, button_deselect_all_features]),
        button_features,
        output_features,
        widgets.HTML("<h2>Temporal Feature Selection</h2>"),
        temporal_grid,
        widgets.HBox([button_select_all_temporal_features, button_deselect_all_temporal_features]),
        button_temporal_features,
        output_temporal_features,
        widgets.HTML("<h2>Spot Feature Selection</h2>"),
        spot_grid,
        widgets.HBox([button_select_all_spot_features, button_deselect_all_spot_features]),
        button_spot_features,
        output_spot_features,
        widgets.HTML("<h2>Macro Feature Selection</h2>"),
        macro_grid,
        widgets.HBox([button_select_all_macro_features, button_deselect_all_macro_features]),
        button_macro_features,
        output_macro_features,
        widgets.HTML("<h2>Hyperparameter Selection</h2>"),
        hyperparameter_selection,
        button_hyperparams,
        output_hyperparams,
        widgets.HTML("<h2>Model Selection</h2>"),
        model_selection,
        widgets.HBox([button_select_all_models, button_deselect_all_models]),
        button_model,
        output_model
    ]))

def load_selected_parameters():
    global selected_features, selected_temporal_features, selected_spot_features, selected_macro_features, selected_hyperparameters, selected_models
    with open('selected_parameters.json', 'r') as f:
        parameters = json.load(f)
        selected_features = parameters['selected_features']
        selected_temporal_features = parameters['selected_temporal_features']
        selected_spot_features = parameters['selected_spot_features']
        selected_macro_features = parameters['selected_macro_features']
        selected_hyperparameters = parameters['selected_hyperparameters']
        selected_models = parameters['selected_models']