# knn-stratigraphic-visualization

TODO: abstract

## How to use

### Download the code

The code can be found in the repository, it can be downloaded as ZIP by clicking in the geen Code button. The necessary files are the notebook `knn.ipynb` and the auxiliar module `functions.py`.

### Download the data

The data can be found in the `data` folder. Only two files are necessary: `deltacontourn.csv`, that contains the points of the contour of the Delta; and `hsd new basements.xls`, that contains the data from the wells. It should be noted that the original data from the boreholes only stored the first (most superficial) basement found in the borehole, since it is known that basement is found in all the points below. The file `horizontal sections data.xlsx` contains the original data from the boreholes, the Python script `new_basements.py` reads this data and adds new basement below the existing one to the maximum depth found in the dataset, generating the file `hsd new basements.xls`.

### How to run the notebook

TODO
