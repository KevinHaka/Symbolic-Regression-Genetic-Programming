import numpy as np
import pickle
import datetime
import warnings
import time

import sympy as sp

# NOTE: This script assumes that the NPEET repository is cloned in the same directory as this script.
# I should have try to fix that by creating a virtual environment and installing NPEET as a package.
import sys
import os

# Add the path to the cloned NPEET repository
script_dir = os.path.dirname(os.path.abspath(__file__))
npeet_path = os.path.join(script_dir, 'NPEET')
if os.path.isdir(npeet_path):
    sys.path.append(npeet_path)

# Add the project's source directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from dask.delayed import delayed
from dask.base import compute
from dask.diagnostics.progress import ProgressBar

from symbolic_regression.methods.gp import GP
from symbolic_regression.methods.gpshap import GPSHAP
from symbolic_regression.methods.gpcmi import GPCMI
from symbolic_regression.methods.rfgp import RFGP

from symbolic_regression.utils.pysr_utils import nrmse_loss, train_val_test_split, process_task, send_email
from symbolic_regression.datasets import load_datasets

warnings.filterwarnings("ignore", category=RuntimeWarning)

def gather_splits(split_results, index):
    """Gathers a specific part of the train-val-test split results (e.g., index 0 for X_train)."""
    return tuple(split[index] for split in split_results)

def extract_equations(results, index):
    """Extracts the equations from the results."""
    return [result['equations'][index] for result in results]

def inv(x):
    return 1 / x

def sqrt_sympy(x, evaluate=True):
    return sp.sqrt(x, evaluate=evaluate)


def main(): 
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)

    run_output_dir = os.path.join(data_dir, datetime.datetime.now().strftime(r"%Y%m%d%H%M%S"))
    os.makedirs(run_output_dir)

    # create a log file
    log_file = os.path.join(script_dir, "logfile.log")
    if os.path.exists(log_file): os.remove(log_file)

    n_runs = 10
    test_size = 0.2
    val_size = 0.25
    n_top_features = None
    ns = 100
    ci = 0.99
    k = 5
    record_interval = 5
    n_submodels = 3

    pysr_params = {
        "populations": 2,
        "population_size": 25,
        "niterations": 100,
        "binary_operators": ["+", "-", "*"],
        "unary_operators": ["sqrt", "inv(x) = 1/x"],
        "extra_sympy_mappings": {
            "inv": inv,
            "sqrt": sqrt_sympy
        },

        "verbosity": 0,
        "parallelism": "serial",
        # "deterministic": True,
        "input_stream": 'devnull',
    }
    dataset_names = [
        "F1",
        #"F2",
        ("4544_GeographicalOriginalofMusic", "4544_GOM"),
        "505_tecator",
    	("Communities and Crime", "CCN"),
        ("Communities and Crime Unnormalized", "CCUN"),
    ]
    datasets = load_datasets(dataset_names)

    gp_params = {
        "loss_function": nrmse_loss,
        "record_interval": record_interval,
        **pysr_params,
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

    n_records = methods[list(methods.keys())[0]].n_records
    delayed_tasks = {}

    if "GPSHAP" in methods:
        methods["GPSHAP"].clear_cache()  # Clear cache before starting new task

    for dataset_name, dataset in datasets.items():
        delayed_tasks[dataset_name] = {}

        X = dataset["X"]
        y = dataset["y"]

        delayed_splits = [delayed(train_val_test_split)(X, y) for _ in range(n_runs)]
        
        for method_name, method in methods.items():
            delayed_tasks[dataset_name][method_name] = []
            return_results = True if method_name == "GP" and ("GPSHAP" in methods) else False

            if method_name == "GPSHAP": continue  # Skip GPSHAP for now, as it requires precomputed features

            # Create a delayed task for each method and dataset
            for run in range(n_runs):
                delayed_tasks[dataset_name][method_name].append(
                    delayed(process_task)(
                        dataset_name, 
                        method_name, 
                        run, 
                        delayed_splits[run], 
                        method, 
                        run_output_dir, 
                        return_results
                    )
                )
        
        if ("GPSHAP" in methods):
            if ('GP' in methods):
                delayed_X_trains = delayed(gather_splits)(delayed_splits, 0)  # Gather X_train from splits
                delayed_gp_equations = delayed(extract_equations)(delayed_tasks[dataset_name]["GP"], -1)

                # Use GP's equations for GPSHAP
                delayed_precomputed_features_task = delayed(methods["GPSHAP"].precompute_features_from_pretrained_models)(
                    delayed_X_trains, delayed_gp_equations, n_top_features
                )

            else:
                delayed_precomputed_features_task = delayed(methods["GPSHAP"].precompute_features)(X, y)

            for run in range(n_runs):
                delayed_tasks[dataset_name]["GPSHAP"].append(
                    delayed(process_task)(
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

    tasks_to_run = [
        task 
        for methods_dict in delayed_tasks.values() 
        for task_list in methods_dict.values() 
        for task in task_list
    ]
    
    # Use threaded scheduler to avoid process spawn overhead on Windows
    print("Starting computation with Dask...")  
    # with mp.Pool(processes=os.cpu_count()) as pool:
    with ProgressBar():
        compute(
            *tasks_to_run,
            traverse=False, 
            scheduler='processes',
            # pool=pool,  
        )
    print("\nAll tasks completed successfully.")

    # Read all saved results from individual task files
    task_results = []
    for filename in os.listdir(run_output_dir):
        if filename.endswith('.pkl'):
            filepath = os.path.join(run_output_dir, filename)

            with open(filepath, 'rb') as f:
                result = pickle.load(f)
                task_results.append(result)

    print(f"Loaded {len(task_results)} task results from {run_output_dir}")

    results = {}
    equations = {}
    features = {}

    for dataset_name in datasets.keys():
        results[dataset_name] = {}
        equations[dataset_name] = {}
        features[dataset_name] = {}

        for method_name in methods.keys():
            results[dataset_name][method_name] = {
                "training_losses": np.empty((n_runs, n_records)),
                "validation_losses": np.empty((n_runs, n_records)),
                "test_losses": np.empty((n_runs, n_records)),
            }
            equations[dataset_name][method_name] = []
            features[dataset_name][method_name] = []

    for result in task_results:
        dataset_name = result['dataset_name']
        method_name = result['method_name']
        run = result['run']

        results[dataset_name][method_name]["training_losses"][run, :] = result['losses'][0]
        results[dataset_name][method_name]["validation_losses"][run, :] = result['losses'][1]   
        results[dataset_name][method_name]["test_losses"][run, :] = result['losses'][2]
        
        equations[dataset_name][method_name].append(result['equations'])
        features[dataset_name][method_name].append(result['features'])

    data = {
        "results": results,
        "equations": equations,
        "features": features
    }
    timestamp = datetime.datetime.now().strftime(r"%Y-%m-%d__%H-%M-%S")
    filename = f"data_{timestamp}.pickle"
    with open(filename, "wb") as f:
        pickle.dump(data, f)
    print(f"\nResults saved to {filename}")

if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()

    # Calculate elapsed time
    elapsed_time = end_time - start_time
    # Format the elapsed time as H:MM:SS
    td_str = str(datetime.timedelta(seconds=elapsed_time))
    # Split to remove microseconds
    time_parts = td_str.split('.')

    message = f"Script finished running in {time_parts[0]} seconds."
    print(message)

    send_email(
        subject="Script Finished Running",
        body_message=message,
        sender_email="kevinhaka98@gmail.com",
        app_password="higx cunc swrs tiyr",
        smtp_server="smtp.gmail.com",
        smtp_port=465,
    )

    