# knn-stratigraphic-visualization

## A k-Nearest Neighbors Algorithm in Python for Visualizing the 3D Stratigraphic Architecture of the Llobregat River Delta in NE Spain

The k-nearest neighbors (KNN) algorithm is a non-parametric supervised machine learning classifier, which uses proximity and similarity to make classifications or predictions about the grouping of an individual data point. This ability makes the KNN algorithm ideal for classifying datasets of geological variables and parameters prior to 3D visualization. This paper introduces a machine learning KNN algorithm and Python libraries for visualizing the 3D stratigraphic architecture of porous sedimentary media. A first HTML model shows the consecutive 5-m-equispaced set of horizontal sections of the granulometry classes created with the KNN algorithm from 0 to 120 m b.s.l. in the onshore LRD. A second HTML model shows the 3D mapping of the main Quaternary gravel and coarse sand sedimentary bodies (lithosomes) and the basement (Pliocene and older rocks) top surface created with Python libraries. These results reproduce well the complex sedimentary structure reported in recent scientific publications and prove the suitability of the KNN algorithm and Python libraries for visualizing the 3D stratigraphic structure of sedimentary media, which is a crucial stage to take decisions in different environmental and economic geology disciplines.

## How to use

### Download the code

The code can be found in the repository, it can be downloaded as ZIP by clicking in the geen Code button. The necessary files are the notebook `knn.ipynb` and the auxiliar module `functions.py`.

### Download the data

The data can be found in the `data` folder. Only two files are necessary: `deltacontourn.csv`, that contains the points of the contour of the Delta; and `hsd new basements.xls`, that contains the data from the wells. It should be noted that the original data from the boreholes only stored the first (most superficial) basement found in the borehole, since it is known that basement is found in all the points below. The file `horizontal sections data.xlsx` contains the original data from the boreholes, the Python script `new_basements.py` reads this data and adds new basement below the existing one to the maximum depth found in the dataset, generating the file `hsd new basements.xls`.

### How to run the notebook

TODO 
