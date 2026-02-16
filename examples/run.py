from dotenv import load_dotenv, find_dotenv

# Find and load .env file
load_dotenv(find_dotenv())

import os
import datetime

from inspect import signature
from functools import partial
from numpy import arange
from numpy.random import default_rng
from joblib import delayed
from tqdm_joblib import ParallelPbar
from pysr import PySRRegressor

from symbolic_regression.methods.gp import GP
from symbolic_regression.methods.gppi import GPPI
from symbolic_regression.methods.gpshap import GPSHAP
from symbolic_regression.methods.gpcmi import GPCMI
from symbolic_regression.methods.rfgp import RFGP

from symbolic_regression.utils.datasets import load_datasets
from symbolic_regression.utils.data_utils import (
    organize_results, results_to_dataframe, 
    train_val_test_split, get_numpy_pandas_size
)
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

    # For GPPI
    sub_test_size = 0.3

    # For GPSHAP and GPCMI
    fs_runs = 30

    # For GPCMI
    n_permutations = 100
    alpha = 0.01
    k_nearest_neighbors = 3

    # For RFGP
    n_submodels = 2

    # General
    n_runs = 10
    test_size = 0.3
    val_size = 0.2
    record_interval = 10
    resplit_interval = 10
    n_jobs = -1
    random_state = 2026

    # Set random seed for reproducibility
    rng = default_rng(random_state)

    # Choose datasets to run
    dataset_names = [
        "F1",
        # "F2",
        "Friedman1",
        # "Friedman2",
        # "Friedman3",
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
        "populations": 12,
        "population_size": 20,
        "niterations": 50,
        "binary_operators": ["+", "-", "*"],
        "unary_operators": ["sqrt", "inv", "sin", "atan"],

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
        "sub_test_size": sub_test_size,
        "n_runs": fs_runs,
        **gp_params,
    }

    # GPSHAP method parameters
    gpshap_params = {
        "n_top_features": n_top_features,
        "n_runs": fs_runs,
        **gp_params
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

    # Define the methods to run
    methods = {
        "GP": GP(**gp_params),
        "GPPI": GPPI(**gppi_params),
        "GPSHAP": GPSHAP(**gpshap_params),
        "GPCMI": GPCMI(**gpcmi_params),
        "RFGPCMI": RFGP(**rfgpcmi_params),
    }

    # Get default niterations from PySRRegressor if not provided
    niterations = pysr_params.get("niterations", signature(PySRRegressor).parameters['niterations'].default)

    # Define epochs based on record interval
    epochs = arange(record_interval, niterations+1, record_interval)

    # ---------------- End Parameters ----------------

    delayed_tasks = [] # To hold all delayed tasks
    splits = {} # To hold data splits for each dataset
    exclude_split = {} # To track whether to exclude train_val_test_set from pickling
        
    # Iterate over datasets
    for dataset_name, dataset in datasets.items():
        X = dataset["X"] # Features
        y = dataset["y"] # Target

        # Create splits for each run
        splits[dataset_name] = ParallelPbar(f"Creating splits for {dataset_name}")(n_jobs=n_jobs)(
            delayed(train_val_test_split)(X, y, test_size, val_size, rng.integers(0, 2**32))
            for _ in range(n_runs)
        )

        # Calculate the size of the splits to determine if they should be excluded from pickling
        split_size = sum([get_numpy_pandas_size(obj) for obj in splits[dataset_name][0]])
        exclude_split[dataset_name] = split_size > 1024 ** 2

        # Iterate over methods
        for method_name, method in methods.items():
            # Create a delayed task for each run
            delayed_tasks.extend(
                delayed(persist)(
                    func=partial(warnings_manager, func=process_task),
                    pickle_dir=os.path.join(run_params_dir, dataset_name, method_name),
                    filename=f"run_{run}.pkl",
                    execute=True,
                    save_result=False,
                    exclude_keys=["train_val_test_set" if exclude_split[dataset_name] else None],
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
            )

    # Execute all method tasks in parallel
    ParallelPbar("Processing methods")(n_jobs=n_jobs)(delayed_tasks)

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