import os
from dotenv import load_dotenv, find_dotenv

# Find and load .env file
load_dotenv(find_dotenv())

import datetime
import warnings
from numpy import arange

from dask.delayed import delayed
from dask.base import compute
from dask.diagnostics.progress import ProgressBar

from symbolic_regression.methods.gp import GP
from symbolic_regression.methods.gpshap import GPSHAP
from symbolic_regression.methods.gpcmi import GPCMI
from symbolic_regression.methods.rfgp import RFGP

from symbolic_regression.datasets import load_datasets
from symbolic_regression.utils.pysr_utils import (
    nrmse_loss,
    results_to_dataframe, 
    train_val_test_split, 
    process_task, 
    send_email, 
    gather_splits, 
    extract_equations, 
    inv, 
    sqrt_sympy,
    timeit,
    load_task_results,
    collect_results,
    save_results
)

# Suppress specific warnings
# warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in sqrt",
    category=RuntimeWarning
)

def main(): 
    # Find the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # create a data directory if it doesn't exist
    data_dir = os.path.join(script_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)

    # create a unique output directory for this run
    run_output_dir = os.path.join(data_dir, datetime.datetime.now().strftime(r"%Y%m%d%H%M%S"))
    os.makedirs(run_output_dir)

    # ---------------- Parameters ----------------

    n_runs = 100
    test_size = 0.2
    val_size = 0.2
    n_top_features = None
    ns = 100
    ci = 0.99
    k = 5
    record_interval = 5
    resplit_interval = 5
    n_submodels = 2
    num_workers = (os.cpu_count() or 2) // 2
    threads_per_worker = 4

    pysr_params = {
        "populations": 3,
        "population_size": 33,
        "niterations": 100,
        "binary_operators": ["+", "-", "*"],
        "unary_operators": ["sqrt", "inv(x) = 1/x"],
        "extra_sympy_mappings": {
            "inv": inv,
            "sqrt": sqrt_sympy
        },

        "verbosity": 0,
        "input_stream": 'devnull',
        "parallelism": "serial",
        "deterministic": False,
    }

    # Choose datasets to run
    dataset_names = [
        "F1",
        # "F2",
        # ("4544_GeographicalOriginalofMusic", "4544_GOM"),
        # "505_tecator",
    	# ("Communities and Crime", "CCN"),
        # ("Communities and Crime Unnormalized", "CCUN"),   
    ]
    datasets = load_datasets(dataset_names) # Load datasets

    gp_params = {
        "loss_function": nrmse_loss,
        "record_interval": record_interval,
        "resplit_interval": resplit_interval,
        "pysr_params": pysr_params,
    }

    gpshap_params = {
        "test_size": test_size,
        "val_size": val_size,
        "n_runs": n_runs,
        "n_top_features": n_top_features,
        **gp_params,
    }

    gpcmi_params = {
        "ns": ns,
        "ci": ci,
        "k": k,
        **gp_params
    }

    rfgpcmi_params = {
        "n_submodels": n_submodels,
        "method_class": GPCMI,
        "method_params": gpcmi_params
    }

    methods = {
        "GP": GP(**gp_params),
        "GPSHAP": GPSHAP(**gpshap_params),
        "GPCMI": GPCMI(**gpcmi_params),
    	"RFGPCMI": RFGP(**rfgpcmi_params),
    }

    # Get number of iterations from one of the methods
    niterations = methods[next(iter(methods))].pysr_params["niterations"]

    # Define epochs based on record interval
    epochs = arange(record_interval, niterations+1, record_interval)

    # ---------------- End Parameters ----------------

    delayed_tasks = {} # To hold all delayed tasks

    # Iterate over datasets
    for dataset_name, dataset in datasets.items():
        delayed_tasks[dataset_name] = {} # To hold tasks for each dataset

        X = dataset["X"] # Features
        y = dataset["y"] # Target

        # Create delayed tasks for data splits
        delayed_splits = [delayed(train_val_test_split, pure=False)(X, y, test_size, val_size) for _ in range(n_runs)]
        
        # Iterate over methods
        for method_name, method in methods.items():
            delayed_tasks[dataset_name][method_name] = [] # To hold tasks for each method

            # Determine if results should be returned for this method
            return_results = True if (method_name == "GP") and ("GPSHAP" in methods) else False

            if method_name == "GPSHAP": continue  # Skip GPSHAP for now, as it requires precomputed features

            # Create a delayed task for each run
            for run in range(n_runs):
                delayed_tasks[dataset_name][method_name].append(
                    delayed(process_task, pure=False)(
                        dataset_name, 
                        method_name, 
                        run, 
                        delayed_splits[run], 
                        method, 
                        run_output_dir, 
                        return_results
                    )
                )
        
        # Special handling for GPSHAP, which requires precomputed features
        if ("GPSHAP" in methods):
            if ('GP' in methods):
                # Gather X_train from splits
                delayed_X_trains = delayed(gather_splits)(delayed_splits, 0)  

                # Extract equations from GP runs
                delayed_gp_equations = delayed(extract_equations)(delayed_tasks[dataset_name]["GP"], -1)

                # Use GP's equations for GPSHAP
                delayed_precomputed_features_task = delayed(methods["GPSHAP"].precompute_features_from_pretrained_models)(
                    delayed_X_trains, delayed_gp_equations, n_top_features
                )

            else:
                # Use current dataset for GPSHAP
                delayed_precomputed_features_task = delayed(methods["GPSHAP"].precompute_features)(X, y)

            # Create delayed tasks for each GPSHAP run
            for run in range(n_runs):
                delayed_tasks[dataset_name]["GPSHAP"].append(
                    delayed(process_task, pure=False)(
                        dataset_name, 
                        'GPSHAP', 
                        run, 
                        delayed_splits[run], 
                        methods["GPSHAP"], 
                        run_output_dir, 
                        False, 
                        delayed_precomputed_features_task
                    )
                )

    # Flatten all tasks into a single list for computation
    tasks_to_run = [
        task 
        for methods_dict in delayed_tasks.values() 
        for task_list in methods_dict.values() 
        for task in task_list
    ]
    
    print("Starting computation with Dask...")  
    with ProgressBar():
        compute(
            *tasks_to_run,
            traverse=False, 
            scheduler='processes',
            num_workers=num_workers,
            threads_per_worker=threads_per_worker 
        )
    print("All tasks completed successfully.")

    # Load and aggregate results
    task_results = load_task_results(run_output_dir)
    print(f"\nLoaded {len(task_results)} task results from {run_output_dir}")

    # Collect results into a structured format
    results, equations, features = collect_results(task_results, datasets, methods)

    # Convert results to a DataFrame
    results_df = results_to_dataframe(results, epochs)

    # Save results to a pickle file
    filename = save_results(results_df, equations, features, prefix="data")
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

    # Send email notification if credentials are available
    if sender_email and app_password:
        send_email(
            subject="Script Finished Running",
            body_message=message,
            sender_email=sender_email,
            receiver_email=sender_email,
            app_password=app_password,
            smtp_server="smtp.gmail.com",
            smtp_port=465,
    )
        
    else:
        print("Email credentials not found in environment variables.")
        print("Skipping email notification.")

    