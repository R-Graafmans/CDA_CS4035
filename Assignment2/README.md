# Running the code
## Installing Jupyter Notebook
In order to run the code, hou have to Install Jupyter notebook. Using Anaconda most libraries will be included from install. Make sure to use the python 3.7 version.
Anaconda can be downloaded [here](https://www.anaconda.com/distribution/) Installation help can be found [here](https://docs.anaconda.com/anaconda/install/)

## installing dependencies
A few additional packages should be installed. The following instructions are based on the fact that you installed jupyter notebook via anaconda.
```conda install -c conda-forge tslearn```
```conda install -c anaconda statsmodels ```
```conda install -c anaconda keras ```
```conda install -c anaconda seaborn ```
```conda install -c anaconda ipython ```

## Running the code
Now you have installed Jupyter notebook you can start the service from the command line using the command: ```jupyter notebook```. Open the .ipynb file and run cell after cell in a sequential order. Enjoy!

Note: determing the optimal arma parameters takes multiple hours to run. To do this, remove the arma_param.csv.