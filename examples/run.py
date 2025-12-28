from dotenv import load_dotenv, find_dotenv

# Find and load .env file
load_dotenv(find_dotenv())

import os
import datetime
import sympy as sp

from copy import deepcopy  
from functools import partial
from numpy import arange, array
from numpy.random import default_rng
from joblib import delayed
from tqdm_joblib import ParallelPbar

from symbolic_regression.methods.gp import GP
from symbolic_regression.methods.gppi import GPPI
from symbolic_regression.methods.gpshap import GPSHAP
from symbolic_regression.methods.gpcmi import GPCMI
from symbolic_regression.methods.rfgp import RFGP

from symbolic_regression.utils.datasets import load_datasets
from symbolic_regression.utils.data_utils import organize_results, results_to_dataframe, train_val_test_split
from symbolic_regression.utils.io_utils import load_pickle_files, persist, send_email
from symbolic_regression.utils.system_utils import timeit, warnings_manager
from symbolic_regression.utils.model_utils import process_task
from symbolic_regression.utils.losses import nrmse_loss

# Main function
def main() -> None: 
    # Find the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # create a data directory if it doesn't exist
    data_dir = os.path.join(script_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)

    # create a unique output directory for this run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = os.path.join(data_dir, timestamp)
    run_params_dir = "_".join([run_output_dir, "params"])

    # ---------------- Parameters ----------------

    # Suppress specific warnings
    warning_filters = [
        {"action": "ignore", "message": r"invalid value encountered in .*"},
        {"action": "ignore", "message": r"divide by zero encountered in .*"},
        {"action": "ignore", "message": r"overflow encountered in .*"},
    ]

    # For GPSHAP
    n_top_features = None

    # For GPCMI
    n_permutations = 100
    alpha = 0.01
    k_nearest_neighbors = 5

    # For RFGP
    n_submodels = 2

    # General
    n_runs = 2
    test_size = 0.2
    val_size = 0.2
    record_interval = 10
    resplit_interval = 10
    n_jobs = -1
    random_state = 27

    # Set random seed for reproducibility
    rng = default_rng(random_state)

    # Choose datasets to run
    dataset_names = [
        "F1",
        # "F2",
        #"Friedman1",
        #"Friedman2",
        #"Friedman3",
        # "542_pollution",
        # ("4544_GeographicalOriginalofMusic", "4544_GOM"),
        # "505_tecator",
    	# ("Communities and Crime", "CCN"),
        # ("Communities and Crime Unnormalized", "CCUN"),
        # ("Superconductivty Data", "Superconductivity"), 
    ]
    datasets = load_datasets(dataset_names) # Load datasets

    # PySR parameters
    pysr_params = {
        "populations": 2,
        "population_size": 20,
        "niterations": 20,
        "binary_operators": ["+", "-", "*"],
        "unary_operators": ["sqrt", "inv(x) = 1/x", "sin", "atan"],
        "extra_sympy_mappings": {
            "inv": lambda x: 1/x,
            "sqrt": lambda x, evaluate=False: sp.sqrt(x, evaluate=evaluate),
            "sin": lambda x: sp.sin(x),
            "atan": lambda x: sp.atan(x),
        },

        "parallelism": "serial",
        "deterministic": False if random_state is None else True,
        "batching": True,
        "batch_size": 100,

        "verbosity": 0,
        "input_stream": 'devnull',
    }

    # GP method parameters
    gp_params = {
        "loss_function": nrmse_loss,
        "record_interval": record_interval,
        "resplit_interval": resplit_interval,
        "pysr_params": pysr_params,
    }

    # GPPI method parameters
    gppi_params = {
        "test_size": test_size,
        "val_size": val_size,
        "n_runs": n_runs,
        **gp_params,
    }

    # GPSHAP method parameters
    gpshap_params = {
        "n_top_features": n_top_features,
        **gppi_params
    }

    # GPCMI method parameters
    gpcmi_params = {
        "n_permutations": n_permutations,
        "alpha": alpha,
        "k_nearest_neighbors": k_nearest_neighbors,
        **gp_params
    }

    # RFGPCMI method parameters
    rfgpcmi_params = {
        "n_submodels": n_submodels,
        "method_class": GPCMI,
        "method_params": gpcmi_params
    }

    independent_methods = {
        "GP": GP(**gp_params),
        # "GPCMI": GPCMI(**gpcmi_params),
        # "RFGPCMI": RFGP(**rfgpcmi_params),
    }
    dependent_methods = {
        # "GPPI": GPPI(**gppi_params),
        "GPSHAP": GPSHAP(**gpshap_params),
    }
    all_methods = {**independent_methods, **dependent_methods}

    # Get number of iterations from one of the methods
    niterations = all_methods[next(iter(all_methods))].pysr_params["niterations"]

    # Define epochs based on record interval
    epochs = arange(record_interval, niterations+1, record_interval)

    # ---------------- End Parameters ----------------

    delayed_tasks = [] # To hold all delayed tasks
    splits = {} # To hold data splits for each dataset
        
    # Iterate over datasets
    for dataset_name, dataset in datasets.items():
        X = dataset["X"] # Features
        y = dataset["y"] # Target

        # Create splits for each run
        splits[dataset_name] = ParallelPbar(f"Creating splits for {dataset_name}")(n_jobs=n_jobs)(
            delayed(train_val_test_split)(X, y, test_size, val_size, rng.integers(0, 2**32))
            for _ in range(n_runs)
        )

        # Iterate over independent methods
        for method_name, method in independent_methods.items():
            # Determine if results should be returned for this method
            return_results = True if (method_name == "GP") and (
                ("GPPI" in dependent_methods) or ("GPSHAP" in dependent_methods)
            ) else False

            # Create a delayed task for each run
            delayed_tasks.extend(
                delayed(persist)(
                    func=partial(warnings_manager, func=process_task),
                    pickle_dir=os.path.join(run_params_dir, dataset_name, method_name),
                    filename=f"run_{run}.pkl",
                    execute=True,
                    save_result=False,
                    filters=warning_filters,
                    dataset_name=dataset_name, 
                    method_name=method_name, 
                    run=run, 
                    train_val_test_set=split, 
                    method=method, 
                    output_dir=run_output_dir, 
                    return_results=return_results,
                    random_state=rng.integers(0, 2**32)
                ) for run, split in enumerate(splits[dataset_name])
            )

    # Execute all independent method tasks in parallel
    method_results = ParallelPbar("Processing independent methods")(n_jobs=n_jobs)(delayed_tasks)
    method_results = [result for result in method_results if result] # Filter out None results

    # Special handling for GPPI and GPSHAP, which require precomputed features
    if dependent_methods:
        preparing_tasks = [] # To hold GPPI and GPSHAP preparing tasks
        delayed_tasks = [] # To hold all delayed tasks

        # Iterate over datasets
        for dataset_name, dataset in datasets.items():
            # Check if GP results are available for this dataset
            if 'GP' in independent_methods: 
                # Extract GP results for this dataset
                gp_results = [
                    result for result in method_results if (
                        (result['method_name'] == 'GP') and 
                        (result['dataset_name'] == dataset_name)
                    )
                ]
                gp_equations = [gp_result['equations'][-1] for gp_result in gp_results]

                if "GPPI" in dependent_methods:
                    test_sets = [(split[2], split[5]) for split in splits[dataset_name]] # Test sets from splits
                    err_org = array([result['losses']["test_losses"][-1] for result in gp_results]) # Original test errors from GP runs

                    # Create a task for precomputing features from pretrained GP models
                    preparing_tasks.append(
                        delayed(warnings_manager)(
                            dependent_methods["GPPI"].precompute_features_from_pretrained_models,
                            warning_filters,
                            test_sets, err_org, gp_equations, rng.integers(0, 2**32)
                        )
                    )

                if "GPSHAP" in dependent_methods:
                    X_trains = [split[0] for split in splits[dataset_name]] # X_train splits

                    # Create a task for precomputing features from pretrained GP models
                    preparing_tasks.append(
                        delayed(warnings_manager)(
                            dependent_methods["GPSHAP"].precompute_features_from_pretrained_models,
                            warning_filters,
                            X_trains, gp_equations, n_top_features, rng.integers(0, 2**32)
                        )
                    )

            else: 
                X = dataset["X"] # Features
                y = dataset["y"] # Target

                # Create precomputing tasks for dependent methods
                for method in dependent_methods.values():
                    preparing_tasks.append(
                        delayed(warnings_manager)(
                            method.precompute_features,
                            warning_filters,
                            X, y, rng.integers(0, 2**32)
                        )
                    )

        # Execute precomputing tasks in parallel
        ParallelPbar("Precomputing features for dependent methods")(n_jobs=n_jobs)(preparing_tasks)

        # Create picklable copies of dependent methods
        picklable_methods = {}
        for method_name, method in dependent_methods.items():
            picklable_methods[method_name] = deepcopy(method)
            picklable_methods[method_name]._feature_cache = dict(picklable_methods[method_name]._feature_cache)

        # Iterate over datasets to create dependent method tasks
        for dataset_name, dataset in datasets.items():
            for method_name, method in dependent_methods.items():
                delayed_tasks.extend([
                    delayed(persist)(
                        func=partial(warnings_manager, func=process_task),
                        pickle_dir=os.path.join(run_params_dir, dataset_name, method_name),
                        filename=f"run_{run}.pkl",
                        execute=True,
                        save_result=False,
                        exclude_keys=['method'],
                        extra_data={'method': picklable_methods[method_name]},
                        filters=warning_filters,
                        dataset_name=dataset_name, 
                        method_name=method_name, 
                        run=run, 
                        train_val_test_set=split, 
                        method=method,
                        output_dir=run_output_dir,
                        return_results=False,
                        random_state=rng.integers(0, 2**32)
                    ) for run, split in enumerate(splits[dataset_name])
                ])

        # Execute all dependent method tasks in parallel
        ParallelPbar("Processing dependent method tasks")(n_jobs=n_jobs)(delayed_tasks)

    # Load all task results from the run output directory
    task_results = load_pickle_files(run_output_dir)
    print(f"\nLoaded {len(task_results)} task results from {run_output_dir}")

    # Organize and convert results to a DataFrame
    results, equations, features = organize_results(task_results)
    results_df = results_to_dataframe(results, epochs)

    # Save results to a pickle file
    filename = persist(
        filename=f"data_{timestamp}.pkl",
        df=results_df, 
        equations=equations, 
        features=features
    )
    print(f"\nResults saved to {filename}")

if __name__ == '__main__':
    # Calculate elapsed time
    elapsed_time = timeit(main, n_runs=1)['time']

    # Format the elapsed time as H:MM:SS
    td_str = str(datetime.timedelta(seconds=elapsed_time))

    # Split to remove microseconds
    time_parts = td_str.split('.')

    # Create message
    message = f"Script finished running in {time_parts[0]} (H:MM:SS)"
    print(message)

    # Read email credentials from environment variables
    sender_email = os.getenv("EMAIL_SENDER")
    app_password = os.getenv("EMAIL_APP_PASSWORD")
    smtp_server = os.getenv("SMTP_SERVER")
    smtp_port = os.getenv("SMTP_PORT")

    # Send email notification if credentials are available
    if sender_email and app_password and smtp_server and smtp_port:
        success, error = send_email(
            subject="Script Finished Running",
            body_message=message,
            sender_email=sender_email,
            receiver_email=sender_email,
            app_password=app_password,
            smtp_server=smtp_server,
            smtp_port=int(smtp_port),
        )

        if success: print("Email notification sent successfully.")
        else: print(error)
    else: print("Email credentials not found in environment variables.")