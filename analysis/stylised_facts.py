import numpy as np
import pandas as pd
from pathlib import Path
import scipy.stats as stats
import matplotlib.pyplot as plt
from .utils import load_yaml, load_macro_data, box_plot_scenarios

### Path to database ###

DATABASE_PATH = "F:\\Documents 202507\\University\\University of Sussex\\BSc Dissertation\\Publishing\\MacroABM\\data"

### Plot Parameters ###

# change matplotlib font to serif
plt.rcParams['font.family'] = ['serif']
# figure size
x_figsize = 10
y_figsize = x_figsize/2
# fontsize
fontsize = 25

### Paths ###

# current working directory path
cwd_path = Path.cwd()
# analysis path
analysis_path = cwd_path / "analysis"
# figure path
figure_path = analysis_path / "figures" / "macro_batch"
# create figure path if it doesn't exist
figure_path.mkdir(parents=True, exist_ok=True)
# parameters path 
params_path = cwd_path / "src" / "macroabm" / "config" / "parameters.yaml"

### load model parameters ###

# parameters
params = load_yaml(params_path)
# analysis parameters
steps = params['simulation']['steps']
num_years = params['simulation']['years']
start = params['simulation']['start']*steps
end = (start + params['simulation']['years'])*steps
middle = int((end + start)/2)
years = np.linspace(0, num_years, num_years*steps)

### paths to data ###
# get database_path from parameters
data_path = cwd_path / "data"
# check if dynamically set database path exists
if data_path.exists() and data_path.is_dir():
    # get database names
    databases_paths = [f for f in data_path.iterdir() if f.is_file()]
else:
    # other wise use manual database path
    data_path = Path(DATABASE_PATH)
    try:
        databases_paths = [f for f in data_path.iterdir() if f.is_file()]
    except FileNotFoundError as e:
        # raise an error if neither exist
        print(f"Error: no data folder with databases and no folder found at manual location {DATABASE_PATH}")
        print(e)  # prints the original FileNotFoundError message
        databases_paths = []
        
