# Libraries and modules
import numpy as np
# import pandas as pd

from joblib import Parallel, delayed
from typing import Callable, Optional
from inspect import signature
from tqdm import tqdm

from pysr import PySRRegressor
from pysr_utils import nrmse_loss, cumulative_lambda
from pysr_utils import fit_and_evaluate_best_equation
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
    best_eqs_list = []

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
    for run, (training_losses, validation_losses, test_losses, best_eqs) in enumerate(outputs):
        # Store results
        results["training_loss"][run] = training_losses
        results["validation_loss"][run] = validation_losses
        results["test_loss"][run] = test_losses
        best_eqs_list.append(best_eqs)

    return results, features, best_eqs_list

def gpshap(
    train_val_test_sets_list: tuple,
    gp_best_equations: Optional[list] = None,
    loss_function: Callable = nrmse_loss,
    record_interval: int = 1,
    pysr_params: Optional[dict] = None
) -> tuple[dict, list, list]:

    if pysr_params is None: pysr_params = {}
    method_name = "GPSHAP"  # NOTE: It's not necessary, maybe you can remove it later

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
    best_eqs_list = []

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
    for run, (training_losses, validation_losses, test_losses, best_eqs) in enumerate(outputs):
        results["training_loss"][run] = training_losses
        results["validation_loss"][run] = validation_losses
        results["test_loss"][run] = test_losses
        best_eqs_list.append(best_eqs)

    return results, selected_features, best_eqs_list

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
    best_eqs_list = []

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
    for run, (training_losses, validation_losses, test_losses, best_eqs) in enumerate(outputs):
        results["training_loss"][run] = training_losses
        results["validation_loss"][run] = validation_losses
        results["test_loss"][run] = test_losses
        best_eqs_list.append(best_eqs)

    return results, selected_features_list, best_eqs_list

def new_method(
    method_params: dict,
    n_submodels: int = 2,
    method_function: Callable = gpcmi,
) -> tuple[dict, list, list]:

    if ('pysr_params' not in method_params) or (method_params['pysr_params'] is None):
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
    best_eqs_lists = [[] for _ in range(n_runs)]
    lambda_models = [[[] for _ in range(n_records)] for _ in range(n_runs)]
    lambda_exprs = [[] for _ in range(n_runs)]

    train_val_test_sets_list_original = method_params['train_val_test_sets_list'].copy()

    for sub_i in range(n_submodels):
        submethod_name = f"{method_function.__name__}_{sub_i+1}/{n_submodels}"
        print(f"{submethod_name+' ':-<20}")

        low = sub_i * n_records // n_submodels
        high = (sub_i + 1) * n_records // n_submodels
        n_points = high - low

        method_params['pysr_params']['niterations'] = n_points * record_interval

        # Run the method function
        _, temp_selected_features_list, temp_best_eqs_list = method_function(**method_params)

        for run in range(n_runs):
            selected_features = temp_selected_features_list[run]
            interval_best_eqs = temp_best_eqs_list[run]

            selected_features_list[run].append(selected_features)
            best_eqs_lists[run].extend(interval_best_eqs)
            lambda_exprs[run].extend([best_eq.lambda_format for best_eq in interval_best_eqs])

            X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_sets_list_original[run]

            for interval_idx in range(low, high):
                lambda_models[run][interval_idx].append((lambda_exprs[run][interval_idx], selected_features))

                # Compute cumulative predictions from current submodels
                train_cumulative_pred = cumulative_lambda(X_train, lambda_models[run][interval_idx])
                val_cumulative_pred = cumulative_lambda(X_val, lambda_models[run][interval_idx])
                test_cumulative_pred = cumulative_lambda(X_test, lambda_models[run][interval_idx])

                # Compute losses for cumulative predictions
                results["training_loss"][run, interval_idx] = method_params['loss_function'](
                    y_train, train_cumulative_pred
                )
                results["validation_loss"][run, interval_idx] = method_params['loss_function'](
                    y_val, val_cumulative_pred
                )
                results["test_loss"][run, interval_idx] = method_params['loss_function'](
                    y_test, test_cumulative_pred
                )
            
            for interval_idx in range(high+1, n_records):
                lambda_models[run][interval_idx].append((lambda_exprs[run][high-1], selected_features))

            X_train_curr = X_train[selected_features]
            X_val_curr = X_val[selected_features]
            X_test_curr = X_test[selected_features]

            y_train_residual, y_val_residual, y_test_residual = method_params['train_val_test_sets_list'][run][3:]

            y_train_residual -= lambda_exprs[run][-1](X_train_curr)
            y_val_residual -= lambda_exprs[run][-1](X_val_curr)
            y_test_residual -= lambda_exprs[run][-1](X_test_curr)

            method_params['train_val_test_sets_list'][run] = (
                X_train, 
                X_val, 
                X_test, 
                y_train_residual, 
                y_val_residual, 
                y_test_residual
            )

    return results, selected_features_list, best_eqs_lists