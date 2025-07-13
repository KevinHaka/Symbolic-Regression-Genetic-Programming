# Libraries and modules
import numpy as np
import pandas as pd

from joblib import Parallel, delayed
from typing import Callable, Optional
from inspect import signature
from tqdm import tqdm

from pysr import PySRRegressor
from pysr_utils import best_equation, nrmse_loss
from pysr_utils import train_val_test_split, fit_and_evaluate_best_equation, cumulative_lambda
from fs_techniques import cmi_feature_selection, shap_feature_selection

def gp(
    train_val_test_sets_list: tuple,
    loss_function: Callable = nrmse_loss,
    record_interval: int = 1,
    pysr_params: Optional[dict] = None
) -> tuple[dict, list, list]:
    
    if pysr_params is None: pysr_params = {}
    method_name = "GP"

    # Determine the total number of iterations and how many times to record losses
    niterations = pysr_params.get("niterations", signature(PySRRegressor).parameters['niterations'].default)
    n_records = niterations // record_interval

    # Initialize variables
    n_runs = len(train_val_test_sets_list)
    results = {
        "training_loss": np.empty((n_runs, n_records)),
        "validation_loss": np.empty((n_runs, n_records)),
        "test_loss": np.empty((n_runs, n_records))
    }
    features = train_val_test_sets_list[0][0].columns.tolist()
    best_eqs = []

    print(f"{method_name+' ':-<20}")

    # Use Parallel processing to run the function in parallel
    outputs = Parallel(n_jobs=-1)(
        delayed(fit_and_evaluate_best_equation)(
            train_val_test_sets_list[run], 
            loss_function, 
            record_interval, 
            pysr_params
        ) for run in tqdm(
            range(n_runs), 
            desc=f"{method_name} "
        )
    )

    # Unpack the outputs into the respective lists
    for run, (training_losses, validation_losses, test_losses, best_eq) in enumerate(outputs):
        # Store results
        results["training_loss"][run] = training_losses
        results["validation_loss"][run] = validation_losses
        results["test_loss"][run] = test_losses
        best_eqs.append(best_eq)

    return results, features, best_eqs

def gpshap(
    train_val_test_sets_list: tuple,
    gp_best_equations: Optional[list] = None,
    loss_function: Callable = nrmse_loss,
    record_interval: int = 1,
    pysr_params: Optional[dict] = None
) -> tuple[dict, list, list]:

    if pysr_params is None: pysr_params = {}
    method_name = "GPSHAP"

    # Determine the total number of iterations and how many times to record losses
    niterations = pysr_params.get("niterations", signature(PySRRegressor).parameters['niterations'].default)
    n_records = niterations // record_interval
    
    # Initialize variables
    n_runs = len(train_val_test_sets_list)
    n_features = train_val_test_sets_list[0][0].shape[1]
    results = {
        'training_loss': np.zeros((n_runs, n_records)),
        'validation_loss': np.zeros((n_runs, n_records)),
        'test_loss': np.zeros((n_runs, n_records))
    }
    best_eqs = []

    X_train_list = [train_val_test_sets[0] for train_val_test_sets in train_val_test_sets_list]
    
    selected_features, _ = shap_feature_selection(
        n_top_features=max(1, round(np.log2(n_features))),
        X_train_list=X_train_list,
        gp_best_equations=gp_best_equations
    )

    train_val_test_sets_list_filtered = [
        (
            X_train[selected_features],
            X_val[selected_features],
            X_test[selected_features],
            y_train,
            y_val,
            y_test
        ) for X_train, X_val, X_test, y_train, y_val, y_test in train_val_test_sets_list
    ]

    print(f"{method_name+' ':-<20}")
    
    # Use Parallel processing to run the function in parallel
    outputs = Parallel(n_jobs=-1)(
        delayed(fit_and_evaluate_best_equation)(
            train_val_test_sets_list_filtered[run], 
            loss_function, 
            record_interval, 
            pysr_params
        ) for run in tqdm(
            range(n_runs), 
            desc=f"{method_name} "
        )
    )

    # Unpack the outputs into the respective lists
    for run, (training_losses, validation_losses, test_losses, best_eq) in enumerate(outputs):
        results["training_loss"][run] = training_losses
        results["validation_loss"][run] = validation_losses
        results["test_loss"][run] = test_losses
        best_eqs.append(best_eq)

    return results, selected_features, best_eqs

def gpcmi(
    train_val_test_sets_list: tuple,
    loss_function: Callable = nrmse_loss,
    record_interval: int = 1,
    pysr_params: Optional[dict] = None,
    fs_params: Optional[dict] = None,
) -> tuple[dict, list, list]:

    if pysr_params is None: pysr_params = {}
    if fs_params is None: fs_params = {}
    method_name = "gpCMI"

    # Determine the total number of iterations and how many times to record losses
    niterations = pysr_params.get("niterations", signature(PySRRegressor).parameters['niterations'].default)
    n_records = niterations // record_interval

    # Initialize variables
    n_runs = len(train_val_test_sets_list)
    results = {
        "training_loss": np.zeros((n_runs, n_records)),
        "validation_loss": np.zeros((n_runs, n_records)),
        "test_loss": np.zeros((n_runs, n_records))
    }
    selected_features_list = []
    best_eqs = []

    print(f"{method_name+" ":-<20}")

    selected_features_list = Parallel(n_jobs=-1)(
        delayed(lambda X_train, y_train, fs_params: cmi_feature_selection(X_train, y_train, **fs_params)[0])(
            X_train, y_train, fs_params
        ) for X_train, _, _, y_train, _, _ in tqdm(
            train_val_test_sets_list,
            desc=f"{method_name} feature selection"
        )
    )

    train_val_test_sets_list_filtered = []
    for run in range(n_runs):
        X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_sets_list[run]
        selected_features = selected_features_list[run]

        # Filter the train/val/test sets based on selected features
        X_train_filtered = X_train[selected_features]
        X_val_filtered = X_val[selected_features]
        X_test_filtered = X_test[selected_features]
        
        train_val_test_sets_list_filtered.append(
            (X_train_filtered, X_val_filtered, X_test_filtered, y_train, y_val, y_test)
        )

    # Use Parallel processing to run the function in parallel
    outputs = Parallel(n_jobs=-1)(
        delayed(fit_and_evaluate_best_equation)(
            train_val_test_sets_list_filtered[run], 
            loss_function, 
            record_interval, 
            pysr_params
        ) for run in tqdm(
            range(n_runs), 
            desc=f"{method_name} "
        )
    )

    # Unpack the outputs into the respective lists
    for run, (training_losses, validation_losses, test_losses, best_eq) in enumerate(outputs):
        results["training_loss"][run] = training_losses
        results["validation_loss"][run] = validation_losses
        results["test_loss"][run] = test_losses
        best_eqs.append(best_eq)

    return results, selected_features_list, best_eqs

def new_method(
    method_params: dict,
    n_submodels: int = 2,
    method_function: Callable = gp,
) -> tuple[dict, list, list]:

    if 'pysr_params' not in method_params:
        niterations = signature(PySRRegressor).parameters['niterations'].default
        method_params['pysr_params'] = {}
    elif method_params['pysr_params'] is None:
        niterations = signature(PySRRegressor).parameters['niterations'].default
        method_params['pysr_params'] = {}
    else:
        niterations = method_params['pysr_params'].get("niterations", signature(PySRRegressor).parameters['niterations'].default)
    
    record_interval = method_params.get("record_interval", signature(method_function).parameters['record_interval'].default)
    n_records = niterations // record_interval
    n_runs = len(method_params['train_val_test_sets_list'])
    

    results = {
        "training_loss": np.zeros((n_runs, n_records)),
        "validation_loss": np.zeros((n_runs, n_records)),
        "test_loss": np.zeros((n_runs, n_records))
    }
    selected_features_list = [[] for _ in range(n_runs)]
    best_eqs = [[] for _ in range(n_runs)]
    lambda_models = [[] for _ in range(n_runs)]

    train_val_test_sets_list_original = method_params['train_val_test_sets_list'].copy()

    for sub_i in range(n_submodels):
        submethod_name = f"{method_function.__name__}_{sub_i+1}/{n_submodels}"
        print(f"{submethod_name+' ':-<20}")

        low = sub_i * n_records // n_submodels
        high = (sub_i + 1) * n_records // n_submodels

        method_params['pysr_params']['niterations'] = (high - low) * record_interval

        # Run the method function
        temp_results, temp_selected_features_list, temp_best_eqs = method_function(**method_params)

        # Store the results in the main results dictionary
        results["training_loss"][:, low:high] = temp_results["training_loss"]
        results["validation_loss"][:, low:high] = temp_results["validation_loss"]
        results["test_loss"][:, low:high] = temp_results["test_loss"]

        for run in range(n_runs):
            selected_features_list[run].append(temp_selected_features_list[run])
            best_eqs[run].append(temp_best_eqs[run])
        
            lambda_expr = temp_best_eqs[run].lambda_format
            selected_features = temp_selected_features_list[run]
            lambda_models[run].append((lambda_expr, selected_features))

            X_train, X_val, X_test = train_val_test_sets_list_original[run][:3]
            X_train_curr = X_train[selected_features]
            X_val_curr = X_val[selected_features]
            X_test_curr = X_test[selected_features]

            y_train_residual, y_val_residual, y_test_residual = method_params['train_val_test_sets_list'][run][3:]

            y_train_residual -= lambda_expr(X_train_curr)
            y_val_residual -= lambda_expr(X_val_curr)
            y_test_residual -= lambda_expr(X_test_curr)

            method_params['train_val_test_sets_list'][run] = (
                X_train, 
                X_val, 
                X_test, 
                y_train_residual, 
                y_val_residual, 
                y_test_residual
            )

            # Compute cumulative predictions from current submodels
            # train_cumulative_pred = cumulative_lambda(X_train, lambda_models[run])
            # val_cumulative_pred = cumulative_lambda(X_val, lambda_models[run])
            # test_cumulative_pred = cumulative_lambda(X_test, lambda_models[run])
    
    return results, selected_features_list, best_eqs