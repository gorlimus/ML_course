# Complete ML by ZTM (2023)

## **Data Science Environment Setup**

### Introducing Our Tools

- Steps to learn machine learning [Recall]
  - Create a framework [Done] Refer to Section 3
  - Match to data science and machine learning tools
  - Learn by doing
- Your computer -> Setup Miniconda + Conda for Data Science
  - [Anaconda](https://www.anaconda.com/): Hardware Store = 3GB
  - [Miniconda](https://docs.conda.io/en/latest/miniconda.html): Workbench = 200 MB
  - [Anaconda vs. miniconda](https://stackoverflow.com/questions/45421163/anaconda-vs-miniconda)
  - [Conda](https://docs.conda.io/en/latest/): Personal Assistant
- Conda -> setup the rest of tools
  - Data Analysis:[pandas](https://pandas.pydata.org/), [Matplotlib](https://matplotlib.org/), [NumPy](https://numpy.org/)
  - Machine Learning: TensorFlow, PyTorch, scikit-learn, XGBoost, CatBoost

### What is Conda?

- [Anaconda](https://www.anaconda.com/): Software Distributions
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html): Software Distributions
- [Anaconda vs. miniconda](https://stackoverflow.com/questions/45421163/anaconda-vs-miniconda)
- [Conda](https://docs.conda.io/en/latest/): Package Manager
- Your computer -> Miniconda + Conda -> install other tools
  - Data Analysis:[pandas](https://pandas.pydata.org/), [Matplotlib](https://matplotlib.org/), [NumPy](https://numpy.org/)
  - Machine Learning: [TensorFlow](https://www.tensorflow.org/), [PyTorch](https://pytorch.org/), [scikit-learn](https://scikit-learn.org/stable/), [XGBoost](https://xgboost.ai/), [CatBoost](https://catboost.ai/)
- Conda -> Project 1: sample-project
- Resources
  - [Conda Cheatsheet](conda-cheatsheet.pdf)
  - [Getting started with conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html)
  - [Getting your computer ready for machine learning](https://www.mrdbourke.com/get-your-computer-ready-for-machine-learning-using-anaconda-miniconda-and-conda/)

### Conda Environments

- New Project: Heart disease?
- Your computer -> Project folder = Data + Conda Environment
- Your computer -> share Project folder -> Someone else's computer
- Someone else's computer -> Project folder = Data + Conda Environment

### Mac Environment Setup

- Resources
  - [Getting Started Anaconda, Miniconda and Conda](https://whimsical.com/BD751gt65nKjAD5i1CNEXU)
  - [Miniconda installers](https://docs.conda.io/en/latest/miniconda.html) - Choose latest pkg version
- Create conda environment: goto [sample-project](https://github.com/chesterheng/machine-learning-data-science/tree/sample-project) folder
  - `conda create --prefix ./env pandas numpy matplotlib scikit-learn`
- Activate conda environment: `conda activate /Users/xxx/Desktop/sample-project/env`
- List Conda environments: `conda env list`
  - `cd ~/.conda` -> `environments.txt`
- Deactivate conda environment: `conda deactivate`

### Mac Environment Setup 2

- Install Jupyter: `conda install jupyter`
- Run Jupyter Notebook: `jupyter notebook`
- Remove packages: `conda remove openpyxl xlrd`
- List all packages: `conda list`

### Sharing your Conda Environment

- Share a .yml file of your Conda environment: `conda env export --prefix ./env > environment.yml`
  - [Sharing an environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#sharing-an-environment)
- Create an environment called env_from_file from a .yml file: `conda env create --file environment.yml --name env_from_file`
  - [Creating an environment from an environment.yml file](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)

### Jupyter Notebook Walkthrough

- Project Folder
- Data -> Environment
- Data -> Jupyter Notebook (Workspace) -> matplotlib, numpy, pandas -> scikit-learn

## **Pandas: Data Analysis**

### Pandas Introduction

- Why pandas?
  - Simple to use
  - Integrated with many other data science & ML Python Tools
  - Helps you get your data ready for machine learning
- What are we going to cover?
  - Most useful functions
  - pandas Datatypes
  - Importing & exporting data
  - Describing data
  - Viewing & Selecting data
  - Manipulating data
- Where can you get help?
  - Follow along with the code
  - Try it for yourself
  - Search for it - stackoverflow, [pandas documentation](https://pandas.pydata.org/pandas-docs/stable/)
  - Try again
  - Ask
- Let's code

### Series, Data Frames and CSVs

- 2 main datatypes

```python
  # 1-dimenional data (Column)
  series = pd.Series(["BMW", "Toyota", "Honda"])
  colours = pd.Series(["Red", "Blue", "White"])
  # DataFrame: 2-dimenional data (Table)
  car_data = pd.DataFrame({ "Car make": series, "Colour": colours })
```

- Import data and export to csv

```python
  car_sales = pd.read_csv("car-sales.csv")
  car_sales.to_csv("exported-car-sales.csv", index=False)
  export_car_sales = pd.read_csv("exported-car-sales.csv")
```

- Import data and export to excel

```python
  car_sales = pd.read_csv("car-sales.csv")
  car_sales.to_excel("exported-car-sales.xlsx", index=False)
  export_car_sales = pd.read_excel("exported-car-sales.xlsx")
```

- `conda install openpyxl xlrd` cannot work -> ModuleNotFoundError
- `pip3 install openpyxl xlrd` work

### Describing Data with Pandas

```python
# Attribute - information
car_sales.dtypes

# Function - contain code to execute
# car_sales.to_csv()

car_sales_columns = car_sales.columns # get all columns
car_sales_index = car_sales.index # get index column
car_sales.describe() # get count, mean, std, min, max, percentile
car_sales.info() # get details of car_sales
car_sales.mean()
car_prices = pd.Series([3000, 1500, 111250])
car_prices.mean()
car_sales.sum()
car_sales["Doors"].sum()
len(car_sales)
```

### Selecting and Viewing Data with Pandas

```python
car_sales.head() # get top 5 rows of car_sales
car_sales.head(7) # get top 7 rows of car_sales
car_sales.tail() # get bottom 5 rows of car_sales

# index [0, 3, 9, 8, 3] => ["cat", "dog", "bird", "panda", "snake"]
animals = pd.Series(["cat", "dog", "bird", "panda", "snake"], index=[0, 3, 9, 8, 3])
animals.loc[3]  # loc refers to index
animals.iloc[3] # iloc refers to position
car_sales.loc[3]  # car_sales item has same position and index
car_sales.iloc[3]

animals.iloc[:3]  # 1st to 3rd positions, 4th is excluded
car_sales.loc[:3] # index 0 to 3 (included)

car_sales["Make"] # get column Make method 1 - column name can be more than 2 words with space
car_sales.Make  # get column Make method 2 - column name must be 1 word without space

car_sales[car_sales["Make"] == "Toyota"] # select rows with criteria - ["Make"] == "Toyota"
car_sales[car_sales["Odometer (KM)"] > 100000] # select rows with criteria - ["Odometer (KM)"] > 100000
pd.crosstab(car_sales["Make"], car_sales["Doors"]) # show the relationshop of "Make" and "Doors"
car_sales.groupby(["Make", "Colour"]).mean() # group row by "Make", then "Colour"

car_sales["Odometer (KM)"].plot() # plot a line graph
car_sales["Odometer (KM)"].hist() # plot a histogram
car_sales["Price"].dtype # check data type of "Price" column
# convert "Price" column value to integer type
car_sales["Price"] = car_sales["Price"].str.replace('[\$\,\.]','').astype(int)
```

### Manipulating Data

- Data Manipulation with Pandas

```python
car_sales["Make"].str.lower()
car_sales["Make"] = car_sales["Make"].str.lower()

car_sales_missing = pd.read_csv("car-sales-missing-data.csv")
odometer-mean = car_sales_missing["Odometer"].mean() # get the mean value of Odometer column

car_sales_missing["Odometer"].fillna(odometer-mean) #   replace NaN with mean value
# update car_sales_missing method 1 - inplace=True
car_sales_missing["Odometer"].fillna(odometer-mean, inplace=True)
# update car_sales_missing method 2 - assign new values to car_sales_missing["Odometer"]
car_sales_missing["Odometer"] = car_sales_missing["Odometer"].fillna(car_sales_missing["Odometer"].mean())

car_sales_missing.dropna(inplace=True)
car_sales_missing_dropped = car_sales_missing.dropna()
car_sales_missing_dropped.to_csv("car-sales-missing-dropped.csv")

# Create a column from series
seats_column = pd.Series([5, 5, 5, 5, 5])
car_sales["Seats"] = seats_column
car_sales["Seats"].fillna(5, inplace=True)

# Create a column from Python list
# list must have same length as exsiting data frame
fuel_economy = [7.5, 9.2, 5.0, 9.6, 8.7, 4.7, 7.6, 8.7, 3.0, 4.5]
car_sales["Fuel per 100KM"] = fuel_economy

# Derived a column
car_sales["Total fuel used (L)"] = car_sales["Odometer (KM)"] / 100 * car_sales["Fuel per 100KM"]
car_sales["Total fuel used"] = car_sales["Odometer (KM)"] / 100 * car_sales["Fuel per 100KM"]

# Create a column from a single value
car_sales["Number of wheels"] = 4
car_sales["Passed road safety"] = True

# Delete a column
# axis=1 - refer to column
car_sales.drop("Total fuel used", axis=1, inplace=True)

# get a sample data set - 20% of data
car_sales_shuffled = car_sales.sample(frac=0.2)

# reset index column to original value
car_sales_shuffled.reset_index(drop=True, inplace=True)

# apply lambda function to Odometer (KM) column
car_sales["Odometer (KM)"] = car_sales["Odometer (KM)"].apply(lambda x: x / 1.6)
```

## **NumPy**

### NumPy Introduction

- Machine learning start with data.
  - Example: data frame
  - Numpy turn data into a series of numbers
  - A machine learning algorithm work out the patterns in those numbers
- Why NumPy?
  - It's fast
  - Behind the scenes optimizations written in C
    - vector is a 1D array
    - matrix is a 2D array
    - vectorization: perform math operations on 2 vectors
    - broadcasting: extend an array to a shape that will allow it to successfully take part in a vectorized calculation
  - Backbone of other Python scientific packages
- What are we going to to cover?
  - Most useful functaions
  - NumPy datatypes & attributes (ndarray)
  - Creating arrays
  - Viewing arrays & matrices
  - Manipulating & comparing arrays
  - Sorting arrays
  - Use cases
- Where can you get help?
  - Follow along with the code
  - Try it for yourself
  - Search for it - stackoverflow, [NumPy Documentation](https://numpy.org/doc/)
  - Try again
  - Ask
- Let's code

### NumPy DataTypes and Attributes

```python
import numpy as np

a1 = np.array([1, 2, 3])
a2 = np.array([[1, 2, 3.3],
               [4, 5, 6.5]])
a3 = np.array([[[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]],
                [[10, 11, 12],
                 [13, 14, 15],
                 [16, 17, 18]]])
a1.shape, a2.shape, a3.shape
a1.ndim, a2.ndim, a3.ndim
a1.dtype, a2.dtype, a3.dtype
a1.size, a2.size, a3.size
type(a1), type(a2), type(a3)

import pandas as pd
df = pd.DataFrame(a2)
```

### Creating NumPy Arrays

```python
import numpy as np

sample_array = np.array([1, 2, 3])
ones = np.ones((2, 3))
zeros = np.zeros((2, 3))
range_array = np.arange(0, 10, 2) # array([0, 2, 4, 6, 8])
random_array = np.random.randint(0, 10, size=(3, 5))
random_array_2 = np.random.random((5, 3))
random_array_3 = np.random.rand(5, 3)
```

### NumPy Random Seed

```python
import numpy as np

np.random.seed(seed=0) # define a seed for random number
random_array_4 = np.random.randint(10, size=(5, 3))

np.random.seed(7)
random_array_5 = np.random.random((5, 3))
```

### Viewing Arrays and Matrices

```python
import numpy as np

np.unique(random_array_4)

a3[:2, :2, :2]

a4 = np.random.randint(10, size=(2, 3, 4, 5))
a4[:, :, :, :4] # Get the first 4 numbers of the inner most arrays
```

### Manipulating Arrays

```python
import numpy as np

# Arithmetic
ones = np.ones(3)
a1 + ones
a1 - ones
a1 * ones
a1 / ones
a2 // a1  # Floor division removes the decimals (rounds down)
a2 ** 2
np.square(a2)
np.add(a1, ones)
a1 % 2
np.exp(a1)
np.log(a1)

# Aggregation
massive_array = np.random.random(100000)
%timeit sum(massive_array) # Measure Python's sum () execution time
%timeit np.sum(massive_array) # Measure NumPy's sum () execution time

np.mean(a2)
np.max(a2)
np.min(a2)
```

### Standard Deviation and Variance

- Standard Deviation and Variance
- Outlier Detection Methods
  - If a value is a certain number of standard deviations away from the mean, that data point is identified as an outlier.
  - The specified number of standard deviations is called the threshold. The default value is 3.

```python
import numpy as np

# Standard deviation
# a measure of how spread out a group of numbers is from the mean
np.std(a2)

# measure of the average degree to which each number is different
# Higher variance = wider range of numbers
# Lower variance = lower range of numbers
np.var(a2)
np.sqrt(np.var(a2)) # Standard deviation = squareroot of variance

high_var_array = np.array([1, 100, 200, 300, 4000, 5000])
low_var_array = np.array([2, 4, 6, 8, 10])
np.var(high_var_array), np.var(low_var_array)
np.std(high_var_array), np.std(low_var_array)
np.mean(high_var_array), np.mean(low_var_array)

%matplotlib inline
import matplotlib.pyplot as plt
plt.hist(high_var_array)
plt.show()

plt.hist(low_var_array)
plt.show()
```

### Reshape and Transpose

```python
import numpy as np

a2_reshape = a2.reshape((2, 3, 1))
a2_reshape * a3

a2.T  # Transpose - switches the axis
a3.T.shape

```

### Dot Product vs Element Wise

- Matrix Multiplication

```python
import numpy as np

np.random.seed(0)
mat1 = np.random.randint(10, size=(5, 3))
mat2 = np.random.randint(10, size=(5, 3))
mat1.shape, mat2.shape

# Element-wise multiplication, also known as Hadamard product
mat1 * mat2

mat1.shape, mat2.T.shape
mat3 = np.dot(mat1, mat2.T)

```

### Exercise: Nut Butter Store Sales

```python
np.random.seed(0)
# Number of jars sold
sales_amounts = np.random.randint(20, size=(5,3))
# Create weekly_sales DataFrame
weekly_sales = pd.DataFrame(sales_amounts,
                            index=["Mon", "Tues", "Wed", "Thurs", "Fri"],
                            columns=["Almond butter", "Peanut butter", "Cashew butter"])

# Create prices array
prices = np.array([10, 8, 12])
# Create butter_prices DataFrame
butter_prices = pd.DataFrame(prices.reshape(1, 3),
                             index=["Price"],
                             columns=["Almond butter", "Peanut butter", "Cashew butter"])

total_sales = prices.dot(sales_amounts.T)
daily_sales = butter_prices.dot(weekly_sales.T)
weekly_sales["Total ($)"] = daily_sales.T
```

### Comparison Operators

```python
a1 > a2
bool_array = a1 >= a2
type(bool_array), bool_array.dtype

a1 > 5
a1 < 5
a1 == a1
a1 == a2
```

### Sorting Arrays

```python
random_array = np.random.randint(10, size=(3, 5))
np.sort(random_array)
np.argsort(random_array) # sort and shiw show index

np.argmin(a1)
np.argmax(a1)

np.argmax(random_array, axis=0) # compare elements in a column
np.argmax(random_array, axis=1) # compare elements in a row
```

### Turn Images Into NumPy Arrays

```python
from matplotlib.image import imread
panda = imread("numpy-panda.png")
panda.size, panda.shape, panda.ndim
panda[:5]
```

## **Matplotlib: Plotting and Data Visualization**

### Matplotlib

- What is Matplotlib
  - Python ploting library
  - Turn date into visualisation
- Why Matplotlib?
  - BUilt on NumPy arrays (and Python)
  - Integrates directly with pandas
  - Can create basic or advanced plots
  - Simple to use interface (once you get the foundations)
- What are we going to cover?
  - A Matplotlib workflow
    - Create data
    - Create plot (figure)
    - Plot data (axes on figure)
    - Customise plot
    - Save/share plot
  - Importing Matplotlib and the 2 ways of plotting Plotting data - from NumPy arrays
  - Plotting data from pandas DataFrames Customizing plots
  - Saving and sharing plots

```python
# Potential function
def plotting_workflow(data):

  # 1. Manipulate data

  # 2. Create plot

  # 3. Plot data

  # 4. Customize plot

  # 5. Save plot

  # 6. Return plot
  return plot
```

### Importing And Using Matplotlib

- Which one should you use? (pyplpt vs matplotlib OO method?)
  - When plotting something quickly, okay to use pyplot method
  - When plotting something more customized and advanced, use the OO method

```python
%matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Pyplot interface
# based on MATLAB and uses a state-based interface
plt.plot()
plt.plot(); #add ; to remove []
plt.plot()
plt.show()
plt.plot([1, 2, 3, 4]) # assume x = [0, 1, 2, 3]
x = [1, 2, 3, 4]
y = [11, 22, 33, 44]
plt.plot(x, y)

# Object-Oriented (OO) interface
# utilize an instance of axes.Axes in order to
# render visualizations on an instance of figure.Figure

# 1st method
fig = plt.figure() # creates a figure
ax = fig.add_subplot() # adds some axes
plt.show()

# 2nd method
fig = plt.figure() # creates a figure
ax = fig.add_axes([1, 1, 1, 1])
ax.plot(x, y) # add some data
plt.show()

# 3rd method (recommended)
fig, ax = plt.subplots()
ax.plot(x, y); # add some data
```

### Anatomy Of A Matplotlib Figure

- [Anatomy of a figure](https://matplotlib.org/examples/showcase/anatomy.html)

```python
# 0. import matplotlib and get it ready for plotting in Jupyter
%matplotlib inline
import matplotlib.pyplot as plt

# 1. Prepare data
x = [1, 2, 3, 4]
y = [11, 22, 33, 44]

# 2. Setup plot
fig, ax = plt.subplots(figsize=(10,10)) # figsize dimension is inches

# 3. Plot data
ax.plot(x, y)

# 4. Customize plot
ax.set(title="Sample Simple Plot", xlabel="x-axis", ylabel="y-axis")

# 5. Save & show
fig.savefig("images/simple-plot.png")
```

### Scatter Plot And Bar Plot

```python
import numpy as np
x = np.linspace(0, 10, 100)

# Plot the data and create a line plot
fig, ax = plt.subplots()
ax.plot(x, x**2);

# Use same data to make a scatter
fig, ax = plt.subplots()
ax.scatter(x, np.exp(x));

# Make a Bar plot from dictionary
nut_butter_prices = {"Almond butter": 10,
                     "Peanut butter": 8,
                     "Cashew butter": 12}
fig, ax = plt.subplots()
ax.bar(nut_butter_prices.keys(), nut_butter_prices.values())
ax.set(title="Dan's Nut Butter Store", ylabel="Price ($)");

# Make a horizontal bar plot
fig, ax = plt.subplots()
ax.barh(list(nut_butter_prices.keys()), list(nut_butter_prices.values()));
```

### Histograms

```python
# Make a Histogram plot
x = np.random.randn(1000) # Make some data from a normal distribution
fig, ax = plt.subplots()
ax.hist(x);

x = np.random.random(1000) # random data from random distribution
fig, ax = plt.subplots()
ax.hist(x);
```

### Subplots

```python
# Subplots Option 1: Create multiple subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2,
                                             ncols=2,
                                             figsize=(10, 5))
# Plot data to each different axis
ax1.plot(x, x/2);
ax2.scatter(np.random.random(10), np.random.random(10));
ax3.bar(nut_butter_prices.keys(), nut_butter_prices.values());
ax4.hist(np.random.randn(1000));

# Subplots Option 2: Create multiple subplots
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 5))
# Plot to each different index
ax[0, 0].plot(x, x/2);
ax[0, 1].scatter(np.random.random(10), np.random.random(10));
ax[1, 0].bar(nut_butter_prices.keys(), nut_butter_prices.values());
ax[1, 1].hist(np.random.randn(1000));
```

### Plotting From Pandas DataFrames

- Which one should you use? (pyplpt vs matplotlib OO method?)
  - When plotting something quickly, okay to use pyplot method
  - When plotting something more advanced, use the OO method

```python
import pandas as pd

ts = pd.Series(np.random.randn(1000),
               index=pd.date_range('1/1/2020', periods=1000))
ts = ts.cumsum() # Return cumulative sum over a DataFrame or Series
ts.plot();

# Make a dataframe
car_sales = pd.read_csv("data/car-sales.csv")

# Remove price column symbols
car_sales["Price"] = car_sales["Price"].str.replace('[\$\,\.]', '')
type(car_sales["Price"][0])
# Remove last two zeros from price
#  4    0   0   0   0   0
# [-6][-5][-4][-3][-2][-1]
car_sales["Price"] = car_sales["Price"].str[:-2]

car_sales["Sale Date"] = pd.date_range("1/1/2020", periods=len(car_sales))
type(car_sales["Price"][0])

car_sales["Total Sales"] = car_sales["Price"].astype(int).cumsum()

car_sales.plot(x="Sale Date", y="Total Sales");
car_sales["Price"] = car_sales["Price"].astype(int) # Reassign price column to int

# Plot scatter plot with price column as numeric
car_sales.plot(x="Odometer (KM)", y="Price", kind="scatter");

# How aboute a bar graph?
x = np.random.rand(10, 4)
df = pd.DataFrame(x, columns=['a', 'b', 'c', 'd'])
df.plot.bar();
df.plot(kind='bar');  # Can do the same thing with 'kind' keyword

car_sales.plot(x='Make', y='Odometer (KM)', kind='bar');

# How about Histograms?
car_sales["Odometer (KM)"].plot.hist();
# bins=10 default , bin width = 25,car_sales["Price"].plot.hist(bins=10);000
car_sales["Odometer (KM)"].plot(kind="hist");
# Default number of bins is 10, bin width = 12,500
car_sales["Odometer (KM)"].plot.hist(bins=20);

# Let's try with another dataset
heart_disease = pd.read_csv("data/heart-disease.csv")
# Create a histogram of age
heart_disease["age"].plot.hist(bins=50);
heart_disease.plot.hist(figsize=(10, 30), subplots=True);

over_50 = heart_disease[heart_disease["age"] > 50]
# Pyplot method
# c: change colur of plot base on target value [0, 1]
over_50.plot(kind='scatter',
             x='age',
             y='chol',
             c='target',
             figsize=(10, 6));

# OO method
fig, ax = plt.subplots(figsize=(10, 6))
over_50.plot(kind='scatter',
             x="age",
             y="chol",
             c='target',
             ax=ax);
ax.set_xlim([45, 100]);
over_50.target.values
over_50.target.unique()

# Make a bit more of a complicated plot

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the data
scatter = ax.scatter(over_50["age"],
                     over_50["chol"],
                     c=over_50["target"])

# Customize the plot
ax.set(title="Heart Disease and Cholesterol Levels",
       xlabel="Age",
       ylabel="Cholesterol");

# Add a legend
ax.legend(*scatter.legend_elements(), title="Target");

# Add a horizontal line
ax.axhline(over_50["chol"].mean(), linestyle="--");

# Setup plot (2 rows, 1 column)
fig, (ax0, ax1) = plt.subplots(nrows=2, # 2 rows
                               ncols=1,
                               sharex=True,
                               figsize=(10, 8))

# Add data for ax0
scatter = ax0.scatter(x=over_50["age"],
                      y=over_50["chol"],
                      c=over_50["target"])
# Customize ax0
ax0.set(title="Heart Disease and Cholesterol Levels",
#         xlabel="Age",
        ylabel="Cholesterol")
ax0.legend(*scatter.legend_elements(), title="Target")

# Setup a mean line
ax0.axhline(y=over_50["chol"].mean(),
            color='b',
            linestyle='--',
            label="Average")

# Add data for ax1
scatter = ax1.scatter(over_50["age"],
                      over_50["thalach"],
                      c=over_50["target"])

# Customize ax1
ax1.set(title="Heart Disease and Max Heart Rate Levels",
        xlabel="Age",
        ylabel="Max Heart Rate")
ax1.legend(*scatter.legend_elements(), title="Target")

# Setup a mean line
ax1.axhline(y=over_50["thalach"].mean(),
            color='b',
            linestyle='--',
            label="Average")

# Title the figure
fig.suptitle('Heart Disease Analysis', fontsize=16, fontweight='bold');
```

### Customizing Your Plots

[Choosing Colormaps in Matplotlib](https://matplotlib.org/3.1.1/tutorials/colors/colormaps.html#sphx-glr-tutorials-colors-colormaps-py)

```python
plt.style.available
plt.style.use('seaborn-whitegrid')

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the data
scatter = ax.scatter(over_50["age"],
                     over_50["chol"],
                     c=over_50["target"],
                     cmap="winter") # this changes the color scheme

# Customize the plot
ax.set(title="Heart Disease and Cholesterol Levels",
       xlabel="Age",
       ylabel="Cholesterol");

# Add a legend
ax.legend(*scatter.legend_elements(), title="Target");

# Add a horizontal line
ax.axhline(over_50["chol"].mean(), linestyle="--");
```

```python
# Customizing the y and x axis limitations

# Setup plot (2 rows, 1 column)
fig, (ax0, ax1) = plt.subplots(nrows=2, # 2 rows
                               ncols=1,
                               sharex=True,
                               figsize=(10, 8))

# Add data for ax0
scatter = ax0.scatter(x=over_50["age"],
                      y=over_50["chol"],
                      c=over_50["target"],
                      cmap="winter") # this changes the color scheme
# Customize ax0
ax0.set(title="Heart Disease and Cholesterol Levels",
#         xlabel="Age",
        ylabel="Cholesterol")
ax0.set_xlim([50, 80])  # change the x axis limit
ax0.legend(*scatter.legend_elements(), title="Target")

# Setup a mean line
ax0.axhline(y=over_50["chol"].mean(),
            color='r',
            linestyle='--',
            label="Average")

# Add data for ax1
scatter = ax1.scatter(over_50["age"],
                      over_50["thalach"],
                      c=over_50["target"],
                      cmap="winter") # this changes the color scheme

# Customize ax1
ax1.set(title="Heart Disease and Max Heart Rate Levels",
        xlabel="Age",
        ylabel="Max Heart Rate")
ax1.set_xlim([50, 80])  # change the x axis limit
ax1.set_ylim([60, 200]) # change the y axis limit
ax1.legend(*scatter.legend_elements(), title="Target")

# Setup a mean line
ax1.axhline(y=over_50["thalach"].mean(),
            color='r',
            linestyle='--',
            label="Average")

# Title the figure
fig.suptitle('Heart Disease Analysis', fontsize=16, fontweight='bold');
```

## **Scikit-learn: Creating Machine Learning Models**

### Scikit-learn

- What is Scikit-Learn (sklearn)?
  - Scikit-Learn is a python machine learning library
  - Data -> Scikit-Learn -> machine learning model
  - machine learning model learn patterns in the data
  - machine learning model make prediction
- Why Scikit-Learn?
  - Built on NumPy and Matplotlib (and Python)
  - Has many in-built machine learning models
  - Methods to evaluate your machine learning models
  - Very well-designed API
    Scikit-Learn workflow
  - Get data ready (to be used with machine learning models)
  - Pick a machine learning model
  - Fit a model to the data (learning patterns)
  - Make predictions with a model (using patterns)
  - Evaluate the model
  - Improving model predictions through experimentation
  - Saving and loading models
- Where can you get help?
  - Follow along with the code
  - Try it for yourself
  - Press SHIFT + TAB to read the docstring
  - Search for it
  - Try again
  - Ask

### Refresher: What Is Machine Learning?

- Programming: input -> function -> output
- Machine Learning: input (data) and desired output
  - machine figure out the function
  - a computer writing his own function
  - also know as model, alogrithm, bot
  - machine is the brain

### Typical scikit-learn Workflow

- An end-to-end Scikit-Learn workflow
  - Getting the data ready -> `heart-disease.csv`
  - Choose the right estimator/algorithm for our problems -> Random Forest Classifier
  - Fit the model/algorithm and use it to make predictions on our data
  - Evaluating a model
  - Improve a model
  - Save and load a trained model
  - Putting it all together!

```python
import numpy as np

# 1. Get the data ready
import pandas as pd
heart_disease = pd.read_csv("data/heart-disease.csv")

# Create X (features matrix) choose from age to thal
X = heart_disease.drop("target", axis=1)

# Create y (labels)
y = heart_disease["target"] # 0: no heart disease, 1: got heart disease

# 2. Choose the right model and hyperparameters
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)

# We'll keep the default hyperparameters
model.get_params()

# 3. Fit the model to the training data
from sklearn.model_selection import train_test_split

# test_size=0.2, 80% of data for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Build a forest of trees from the training set (X, y)
model.fit(X_train, y_train);

# make a prediction
y_preds = model.predict(np.array(X_test))

# 4. Evaluate the model on the training data and test data

# Returns the mean accuracy on the given test data and labels
model.score(X_train, y_train)
model.score(X_test, y_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(classification_report(y_test, y_preds))

# Compute confusion matrix to evaluate the accuracy of a classification
confusion_matrix(y_test, y_preds)

# Accuracy classification score
accuracy_score(y_test, y_preds)

# 5. Improve a model
# Try different amount of n_estimators
np.random.seed(42)
for i in range(10, 100, 10):
    print(f"Trying model with {i} estimators...")
    clf = RandomForestClassifier(n_estimators=i).fit(X_train, y_train)
    print(f"Model accuracy on test set: {clf.score(X_test, y_test) * 100:.2f}%")
    print("")

# 6. Save a model and load it
import pickle # Python object serialization

pickle.dump(clf, open("random_forst_model_1.pkl", "wb")) # write binary

loaded_model = pickle.load(open("random_forst_model_1.pkl", "rb")) # read binary
loaded_model.score(X_test, y_test)
```

### Optional: Debugging Warnings In Jupyter

- Updating packages

```python
import warnings
warnings.filterwarnings("default")
warnings.filterwarnings("ignore")

import sklearn
sklearn.show_versions()
```

- `conda list scikit-learn`
- `conda list python`
- `conda remove package`
- `conda install scikit-learn=0.22`

### Getting Your Data Ready: Splitting Your Data

Three main things we have to do:

- Split the data into features and labels (usually X & y)
  - Different names for X = features, features variables, data
  - Different names for y = labels, targets, target variables
- Converting non-numerical values to numerical values (also called feature encoding)
  - or one hot encoding
- Filling (also called imputing) or disregarding missing values

```python
# Split the data into features and labels (usually X & y)
X = heart_disease.drop("target", axis=1)
y = heart_disease["target"]

# Split the data into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

### Quick Tip: Clean, Transform, Reduce

Cannot assume all data you have is automatically going to be perfect

- Clean Data -> Transform data -> Reduce Data
- Clean Data: Remove a row or a column that's empty or has missing fields
- Clean Data: Calculate average to fill an empty cell
- Clean Data: Remove outliers in your data
- Transform data: Convert some of our information into numbers
- Transform data: Convert color into numbers
- Transform data is between zeros and ones
  - 0: No heart disease
  - 1: Heart disease
- Transform data: Data across the board uses the same units
- Reduce Data: More data more CPU
- Reduce Data: More energy it takes for us to run our computation
- Reduce Data: Same result on less data
- Reduce Data: Dimensionality reduction or column reduction
- Reduce Data: Remove irrelevant columns

### Getting Your Data Ready: Convert Data To Numbers

```python
car_sales = pd.read_csv("data/car-sales-extended.csv")
car_sales.head()
# treat Doors as categorical
car_sales["Doors"].value_counts()
len(car_sales)
car_sales.dtypes

# Split into X/y
X = car_sales.drop("Price", axis=1)
y = car_sales["Price"]

# show one hot encoding
dummies = pd.get_dummies(car_sales[["Make", "Colour", "Doors"]])

# Turn the categories into numbers with one hot encoding
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Encode categorical integer features as a one-hot numeric array
categorical_features = ["Make", "Colour", "Doors"]
one_hot = OneHotEncoder()

# Applies transformers to columns of an array or pandas DataFrame
transformer = ColumnTransformer([("one_hot", one_hot, categorical_features)], remainder="passthrough")
transformed_X = transformer.fit_transform(X)
pd.DataFrame(transformed_X)
```

### Getting Your Data Ready: Handling Missing Values With Pandas

```python
car_sales_missing = pd.read_csv("data/car-sales-extended-missing-data.csv")
car_sales_missing.head()

# show number of column with missing value
car_sales_missing.isna().sum()

car_sales_missing["Doors"].value_counts()

# Fill the "Make" column
car_sales_missing["Make"].fillna("missing", inplace=True)

# Fill the "Colour" column
car_sales_missing["Colour"].fillna("missing", inplace=True)

# Fill the "Odometer (KM)" column. Filled with mean values
car_sales_missing["Odometer (KM)"].fillna(car_sales_missing["Odometer (KM)"].mean(), inplace=True)

# Fill the "Doors" column. Most cars have 4 doors
car_sales_missing["Doors"].fillna(4, inplace=True)

# Remove rows with missing Price value
car_sales_missing.dropna(inplace=True)

# show number of column with missing value
car_sales_missing.isna().sum()
len(car_sales_missing)
```

### Extension: Feature Scaling

- Make sure all of your numerical data is on the same scale
- Normalization: rescales all the numerical values to between 0 and 1
- Standardization: z = (x - u) / s
  - z: standard score of a sample x
  - x: sample x
  - u: mean of the training samples
  - s: standard deviation of the training samples
  - Feature scaling usually isn't required for your target variable
  - Feature scaling is usually not required with tree-based models (e.g. Random Forest) since they can handle varying features

### Getting Your Data Ready: Handling Missing Values With Scikit-learn

The main takeaways:

- Split your data first (into train/test)
- Fill/transform the training set and test sets separately

```python
car_sales_missing = pd.read_csv("data/car-sales-extended-missing-data.csv")
car_sales_missing.head()
car_sales_missing.isna().sum()

# Drop the rows with no labels
car_sales_missing.dropna(subset=["Price"], inplace=True)

# Split into X & y
X = car_sales_missing.drop("Price", axis=1)
y = car_sales_missing["Price"]

# Split data into train and test
np.random.seed(42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Fill missing values with Scikit-Learn
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# Fill categorical values with 'missing' & numerical values with mean
cat_imputer = SimpleImputer(strategy="constant", fill_value="missing")
door_imputer = SimpleImputer(strategy="constant", fill_value=4)
num_imputer = SimpleImputer(strategy="mean")

# Define columns
cat_features = ["Make", "Colour"]
door_feature = ["Doors"]
num_features = ["Odometer (KM)"]

# Create an imputer (something that fills missing data)
imputer = ColumnTransformer([
    ("cat_imputer", cat_imputer, cat_features),
    ("door_imputer", door_imputer, door_feature),
    ("num_imputer", num_imputer, num_features)
])

# Fill train and test values separately
filled_X_train = imputer.fit_transform(X_train)
filled_X_test = imputer.transform(X_test)

# Get our transformed data array's back into DataFrame's
car_sales_filled_train = pd.DataFrame(filled_X_train,
                                      columns=["Make", "Colour", "Doors", "Odometer (KM)"])
car_sales_filled_test = pd.DataFrame(filled_X_test,
                                     columns=["Make", "Colour", "Doors", "Odometer (KM)"])
```

### Choosing The Right Model For Your Data

Scikit-Learn uses estimator as another term for machine learning model or algorithm

- Choosing the right estimator
- Regression - predicting a number
- Classification - predicting whether a sample is one thing or another

```python
# Import Boston housing dataset
from sklearn.datasets import load_boston
boston = load_boston()

# convert dataset into pandas dataframe
boston_df = pd.DataFrame(boston["data"], columns=boston["feature_names"])
boston_df["target"] = pd.Series(boston["target"])

# How many samples?
len(boston_df)

# Let's try the Ridge Regression model
from sklearn.linear_model import Ridge

# Setup random seed
np.random.seed(42)

# Create the data
X = boston_df.drop("target", axis=1)
y = boston_df["target"]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Instantiate Ridge model
model = Ridge()
model.fit(X_train, y_train)

# Check the score of the Ridge model on test data
model.score(X_test, y_test)
```

### Choosing The Right Model For Your Data 2 (Regression)

```python
# Let's try the Random Forst Regressor
from sklearn.ensemble import RandomForestRegressor

# Setup random seed
np.random.seed(42)

# Create the data
X = boston_df.drop("target", axis=1)
y = boston_df["target"]

# Split the data
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Instatiate Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100)
rf.fit(X_train, y_train)

# Evaluate the Random Forest Regressor
rf.score(X_test, y_test)

# Check the Ridge model again
model.score(X_test, y_test)
```

### Choosing The Right Model For Your Data 3 (Classification)

Tidbit:

- If you have structured data (heart_disease), used ensemble methods
- If you have unstructured data (image, audio), use deep learning or transfer learning

```python
heart_disease = pd.read_csv("data/heart-disease.csv")
len(heart_disease)

# Import the LinearSVC estimator class
from sklearn.svm import LinearSVC

# Setup random seed
np.random.seed(42)

# Make the data
X = heart_disease.drop("target", axis=1)
y = heart_disease["target"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Instantiate LinearSVC
clf = LinearSVC(max_iter=10000)
clf.fit(X_train, y_train)

# Evaluate the LinearSVC
clf.score(X_test, y_test)

heart_disease["target"].value_counts()

# Import the RandomForestClassifier estimator class
from sklearn.ensemble import RandomForestClassifier

# Setup random seed
np.random.seed(42)

# Make the data
X = heart_disease.drop("target", axis=1)
y = heart_disease["target"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Instantiate Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

# Evaluate the Random Forest Classifier
clf.score(X_test, y_test)
```

### Fitting A Model To The Data

```python
# Import the RandomForestClassifier estimator class
from sklearn.ensemble import RandomForestClassifier

# Setup random seed
np.random.seed(42)

# Make the data
X = heart_disease.drop("target", axis=1)
y = heart_disease["target"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Instantiate Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100)

# Fit the model to the data (training the machine learning model)
clf.fit(X_train, y_train)

# Evaluate the Random Forest Classifier (use the patterns the model has learned)
clf.score(X_test, y_test)
```

### Making Predictions With Our Model

```python
# Compare predictions to truth labels to evaluate the model
y_preds = clf.predict(X_test)
np.mean(y_preds == y_test)

clf.score(X_test, y_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_preds)
```

### predict() vs predict_proba()

```python
# predict_proba() returns probabilities of a classification label
clf.predict_proba(X_test[:5]) # [% for 0, % for 1]
model.score(X_test, y_test) # 0 or 1

heart_disease["target"].value_counts()
```

### Making Predictions With Our Model (Regression)

- predict() can also be used for regression models

```python
np.random.seed(42)

# Create the data
X = boston_df.drop("target", axis=1)
y = boston_df["target"]

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Instantiate and fit model
model = RandomForestRegressor(n_estimators=100).fit(X_train, y_train)

# Make predictions
y_preds = model.predict(X_test)

y_preds[:10]
np.array(y_test[:10])
# Compare the predictions to the truth
from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test, y_preds)

# y_preds = y_test +/- mean_absolute_error
# y_preds = 24 +/- 2.12
# y_preds = 22 to 26
```

### Evaluating A Machine Learning Model (Score)

Three ways to evaluate Scikit-Learn models/esitmators

- Estimator score method
- The scoring parameter
- Problem-specific metric functions.

```python
from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)
X = heart_disease.drop("target", axis=1)
y = heart_disease["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
clf.score(X_train, y_train)
clf.score(X_test, y_test)
```

```python
from sklearn.ensemble import RandomForestRegressor

np.random.seed(42)

# Create the data
X = boston_df.drop("target", axis=1)
y = boston_df["target"]

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Instantiate and fit model
model = RandomForestRegressor(n_estimators=100).fit(X_train, y_train)
model.score(X_train, y_train)
model.score(X_test, y_test)
```

### Evaluating A Machine Learning Model 2 (Cross Validation)

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)
X = heart_disease.drop("target", axis=1)
y = heart_disease["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train);

# Single training and test split score
clf_single_score = clf.score(X_test, y_test)

# Take the mean of 5-fold cross-validation score
clf_cross_val_score = np.mean(cross_val_score(clf, X, y, cv=5))

# Scoring parameter set to None by default
cross_val_score(clf, X, y, cv=5, scoring=None)
```

### Evaluating A Classification Model (Accuracy)

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)
X = heart_disease.drop("target", axis=1)
y = heart_disease["target"]

clf = RandomForestClassifier(n_estimators=100)
cross_val_score = cross_val_score(clf, X, y, cv=5)
np.mean(cross_val_score)
```

### Evaluating A Classification Model (ROC Curve)

- Area under curve (AUC)
- ROC curve

ROC curves are a comparison of a model's true postive rate (tpr) versus a models false positive rate (fpr).

- True positive = model predicts 1 when truth is 1
- False positive = model predicts 1 when truth is 0
- True negative = model predicts 0 when truth is 0
- False negative = model predicts 0 when truth is 1

```python
# Create X_test... etc
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.metrics import roc_curve

# Fit the classifier
clf.fit(X_train, y_train)

# Make predictions with probabilities
y_probs = clf.predict_proba(X_test)
y_probs_positive = y_probs[:, 1]

# Calculate fpr, tpr and thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_probs_positive)
# Plot ROC curve
plot_roc_curve(fpr, tpr)

from sklearn.metrics import roc_auc_score
# area under the curve, max area = 1
roc_auc_score(y_test, y_probs_positive)

# Plot perfect ROC curve and AUC score
fpr, tpr, thresholds = roc_curve(y_test, y_test)
plot_roc_curve(fpr, tpr)

# Perfect AUC score
roc_auc_score(y_test, y_test)
```

```python
# Create a function for plotting ROC curves
import matplotlib.pyplot as plt

def plot_roc_curve(fpr, tpr):
    """
    Plots a ROC curve given the false positive rate (fpr)
    and true positive rate (tpr) of a model.
    """
    # Plot roc curve
    plt.plot(fpr, tpr, color="orange", label="ROC")
    # Plot line with no predictive power (baseline)
    plt.plot([0, 1], [0, 1], color="darkblue", linestyle="--", label="Guessing")

    # Customize the plot
    plt.xlabel("False positive rate (fpr)")
    plt.ylabel("True positive rate (tpr)")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend()
    plt.show()
```

### Evaluating A Classification Model (Confusion Matrix)

- A confusion matrix is a quick way to compare the labels a model predicts and the actual labels it was supposed to predict.
- In essence, giving you an idea of where the model is getting confused.

```python
from sklearn.metrics import confusion_matrix

y_preds = clf.predict(X_test)
confusion_matrix(y_test, y_preds)

# Visualize confusion matrix with pd.crosstab()
pd.crosstab(y_test, y_preds, rownames=["Actual Labels"], colnames=["Predicted Labels"])

# Make our confusion matrix more visual with Seaborn's heatmap()
import seaborn as sns

# Set the font scale
sns.set(font_scale=1.5)

# Create a confusion matrix
conf_mat = confusion_matrix(y_test, y_preds)

# Plot it using Seaborn
sns.heatmap(conf_mat);

plot_conf_mat(conf_mat)

from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(clf, X, y);
```

```python
def plot_conf_mat(conf_mat):
  """
  Plots a confusion matrix using Seaborn's heatmap().
  """
  fig, ax = plt.subplots(figsize=(3,3))
  ax = sns.heatmap(conf_mat,
                    annot=True, # Annotate the boxes with conf_mat info
                    cbar=False)
  plt.xlabel("True label")
  plt.ylabel("Predicted label")
```

### Evaluating A Classification Model 6 (Classification Report)

Precision, Recall & F-Measure

- Classification: Predict Category
- Determine if a sample shoe is Nike or not
- Confusion Matrix
  - True Positive (TP): Predict Nike shoe as Nike (Correct) Example: 0
  - False Positive (FP): Predict Non-Nike shoe as Nike (Wrong) Example: 0
  - False Negative (FN): Predict Nike shoe as Non-Nike (Wrong) Example: 10
  - True Negative (TN): Predict Non-Nike shoe as Non-Nike (Correct) Example: 9990
- Accuracy: % of correct prediction? (TP + TN) / total sample
  - Accuracy]() is a good measure to start with if all classes are balanced (e.g. same amount of samples which are labelled with 0 or 1).
- Precision and recall focus on TP, do not consider TN
- Precision: Of the shoes **classified** Nike, How many are **acutally** Nike?
  - Number of shoes **acutally** Nike = TP
  - Number of shoes **classified** Nike = TP + FP
  - Precision = TP / (TP + FP) = % of correct positive classification over total positive classification
  - When the model predicts a positive, how often is it correct?
- Recall: Of the shoes that are **actually** Nike, How many are **classified** as Nike?
  - Number of shoes **classified** Nike = TP
  - Number of shoes **acutally** Nike = TP + FN
  - Recall = TP / (TP + FN) = % of correct positive classification over total positive
  - When it is actually positive, how often does it predict a positive?
- Precision and recall become more important when classes are imbalanced.
  - If cost of false positive predictions are worse than false negatives, aim for higher precision.
    - For example, in spam detection, a false positive risks the receiver missing an important email due to it being incorrectly labelled as spam.
  - If cost of false negative predictions are worse than false positives, aim for higher recall.
    - For example, in cancer detection and terrorist detection the cost of a false negative prediction is likely to be deadly. Tell a cancer patient you have no cancer.
- F1-score is a combination of precision and recall.
  - Use F1 score if data is imbalanced

```python
from sklearn.metrics import classification_report
print(classification_report(y_test, y_preds))

# Where precision and recall become valuable
disease_true = np.zeros(10000)
disease_true[0] = 1 # only one positive case
disease_preds = np.zeros(10000) # model predicts every case as 0

pd.DataFrame(classification_report(disease_true,
                                   disease_preds,
                                   output_dict=True))

```

### Evaluating A Regression Model 1 (R2 Score)

Regression model evaluation metrics

- R^2 (pronounced r-squared) or coefficient of determination.
- Mean absolute error (MAE)
- Mean squared error (MSE)

Which regression metric should you use?

- R2 is similar to accuracy. It gives you a quick indication of how well your model might be doing. Generally, the closer your R2 value is to 1.0, the better the model. But it doesn't really tell exactly how wrong your model is in terms of how far off each prediction is.
- MAE gives a better indication of how far off each of your model's predictions are on average.
- As for MAE or MSE, because of the way MSE is calculated, squaring the differences between predicted values and actual values, it amplifies larger differences. Let's say we're predicting the value of houses (which we are).
  - Pay more attention to MAE: When being $10,000 off is twice as bad as being $5,000 off.
  - Pay more attention to MSE: When being $10,000 off is more than twice as bad as being $5,000 off.

What R-squared does:

- Compares your models predictions to the mean of the targets. Values can range from negative infinity (a very poor model) to 1.
- For example, if all your model does is predict the mean of the targets, it's R^2 value would be 0.
- And if your model perfectly predicts a range of numbers it's R^2 value would be 1.

```python
from sklearn.ensemble import RandomForestRegressor

np.random.seed(42)

X = boston_df.drop("target", axis=1)
y = boston_df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train);
model.score(X_test, y_test)

from sklearn.metrics import r2_score

# Fill an array with y_test mean
y_test_mean = np.full(len(y_test), y_test.mean())

# Model only predicting the mean gets an R^2 score of 0
r2_score(y_test, y_test_mean)

# Model predicting perfectly the correct values gets an R^2 score of 1
r2_score(y_test, y_test)
```

### Evaluating A Regression Model 2 (MAE)

Mean absolue error (MAE)

- MAE is the average of the aboslute differences between predictions and actual values. It gives you an idea of how wrong your models predictions are.

```python
# Mean absolute error
from sklearn.metrics import mean_absolute_error

y_preds = model.predict(X_test)
mae = mean_absolute_error(y_test, y_preds)

df = pd.DataFrame(data={"actual values": y_test,
                        "predicted values": y_preds})
df["differences"] = df["predicted values"] - df["actual values"]
```

### Evaluating A Regression Model 3 (MSE)

Mean squared error (MSE)

- MSE is the average of the square value of aboslute differences between predictions and actual values.

```python
# Mean squared error
from sklearn.metrics import mean_squared_error

y_preds = model.predict(X_test)
mse = mean_squared_error(y_test, y_preds)

# Calculate MSE by hand
squared = np.square(df["differences"])
squared.mean()
```

### Machine Learning Model Evaluation

- Evaluating the results of a machine learning model is as important as building one.
- But just like how different problems have different machine learning models, different machine learning models have different evaluation metrics.
- Below are some of the most important evaluation metrics you'll want to look into for classification and regression models.

Classification Model Evaluation Metrics/Techniques

- Accuracy - The accuracy of the model in decimal form. Perfect accuracy is equal to 1.0.
- [Precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html#sklearn.metrics.precision_score) - Indicates the proportion of positive identifications (model predicted class 1) which were actually correct. A model which produces no false positives has a precision of 1.0.
- [Recall](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html#sklearn.metrics.recall_score) - Indicates the proportion of actual positives which were correctly classified. A model which produces no false negatives has a recall of 1.0.
- [F1 score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score) - A combination of precision and recall. A perfect model achieves an F1 score of 1.0.
- [Confusion matrix](https://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/) - Compares the predicted values with the true values in a tabular way, if 100% correct, all values in the matrix will be top left to bottom right (diagonal line).
- [Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html) - Splits your dataset into multiple parts and train and tests your model on each part then evaluates performance as an average.
- [Classification report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html) - Sklearn has a built-in function called classification_report() which returns some of the main classification metrics such as precision, recall and f1-score.
- ROC Curve - Also known as receiver operating characteristic is a plot of true positive rate versus false-positive rate.
- [Area Under Curve (AUC) Score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html) - The area underneath the ROC curve. A perfect model achieves an AUC score of 1.0.

Which classification metric should you use?

- Accuracy is a good measure to start with if all classes are balanced (e.g. same amount of samples which are labelled with 0 or 1).
- Precision and recall become more important when classes are imbalanced.
  - If false-positive predictions are worse than false-negatives, aim for higher precision.
  - If false-negative predictions are worse than false-positives, aim for higher recall.
- F1-score is a combination of precision and recall.
- A confusion matrix is always a good way to visualize how a classification model is going.

Regression Model Evaluation Metrics/Techniques

- [R^2 (pronounced r-squared)](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html) or the coefficient of determination - Compares your model's predictions to the mean of the targets. Values can range from negative infinity (a very poor model) to 1. For example, if all your model does is predict the mean of the targets, its R^2 value would be 0. And if your model perfectly predicts a range of numbers it's R^2 value would be 1.
- [Mean absolute error (MAE)](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html) - The average of the absolute differences between predictions and actual values. It gives you an idea of how wrong your predictions were.
- [Mean squared error (MSE)](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html) - The average squared differences between predictions and actual values. Squaring the errors removes negative errors. It also amplifies outliers (samples which have larger errors).

Which regression metric should you use?

- R2 is similar to accuracy. It gives you a quick indication of how well your model might be doing. Generally, the closer your R2 value is to 1.0, the better the model. But it doesn't really tell exactly how wrong your model is in terms of how far off each prediction is.
- MAE gives a better indication of how far off each of your model's predictions are on average.
- As for MAE or MSE, because of the way MSE is calculated, squaring the differences between predicted values and actual values, it amplifies larger differences. Let's say we're predicting the value of houses (which we are).
- Pay more attention to MAE: When being $10,000 off is twice as bad as being $5,000 off.
- Pay more attention to MSE: When being $10,000 off is more than twice as bad as being $5,000 off.

### Evaluating A Model With Cross Validation and Scoring Parameter

- [Metrics and scoring: quantifying the quality of predictions](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Cross-validation: evaluating estimator performance](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation)

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)

X = heart_disease.drop("target", axis=1)
y = heart_disease["target"]

# By default cross_val_score uses the scoring provided in the given estimator,
# which is usually the simplest appropriate scoring method.
# E.g. for most classifiers this is accuracy score and for regressors this is r2 score.
cv_acc = cross_val_score(clf, X, y, cv=5, scoring=None)

# Cross-validated accuracy
print(f'The cross-validated accuracy is: {np.mean(cv_acc)*100:.2f}%')

np.random.seed(42)
cv_acc = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
print(f'The cross-validated accuracy is: {np.mean(cv_acc)*100:.2f}%')

# Precision
cv_precision = cross_val_score(clf, X, y, cv=5, scoring="precision")
np.mean(cv_precision)

# Recall
cv_recall = cross_val_score(clf, X, y, cv=5, scoring="recall")
np.mean(cv_recall)

cv_f1 = cross_val_score(clf, X, y, cv=5, scoring="f1")
np.mean(cv_f1)
```

```python
# How about our regression model?
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

np.random.seed(42)

X = boston_df.drop("target", axis=1)
y = boston_df["target"]

model = RandomForestRegressor(n_estimators=100)

np.random.seed(42)
cv_r2 = cross_val_score(model, X, y, cv=5, scoring=None)
np.mean(cv_r2)

np.random.seed(42)
cv_r2 = cross_val_score(model, X, y, cv=5, scoring="r2")

# Mean absolute error
cv_mae = cross_val_score(model, X, y, cv=5, scoring="neg_mean_absolute_error")

# Mean squared error
cv_mse = cross_val_score(model, X, y, cv=5, scoring="neg_mean_squared_error")
```

### Evaluating A Model With Scikit-learn Functions

```python
# Classification evaluation functions
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

np.random.seed(42)

X = heart_disease.drop("target", axis=1)
y = heart_disease["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

# Make some predictions
y_preds = clf.predict(X_test)

# Evaluate the classifier
print("Classifier metrics on the test set")
print(f"Accuracy: {accuracy_score(y_test, y_preds)*100:.2f}%")
print(f"Precision: {precision_score(y_test, y_preds)}")
print(f"Recall: {recall_score(y_test, y_preds)}")
print(f"F1: {f1_score(y_test, y_preds)}")
```

```python
# Regression evaluation functions
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

np.random.seed(42)

X = boston_df.drop("target", axis=1)
y = boston_df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

# Make predictions using our regression model
y_preds = model.predict(X_test)

# Evaluate the regression model
print("Regression model metrics on the test set")
print(f"R^2: {r2_score(y_test, y_preds)}")
print(f"MAE: {mean_absolute_error(y_test, y_preds)}")
print(f"MSE: {mean_squared_error(y_test, y_preds)}")
```

### Improving A Machine Learning Model

First predictions = baseline predictions. First model = baseline model.

From a data perspective:

- Could we collect more data? (generally, the more data, the better)
- Could we improve our data?

From a model perspective:

- Is there a better model we could use?
- Could we improve the current model?

Hyperparameters vs. Parameters

- Parameters = model find these patterns in data
- Hyperparameters = settings on a model you can adjust to (potentially) improve its ability to find patterns

Three ways to adjust hyperparameters:

- By hand
- Randomly with RandomSearchCV
- Exhaustively with GridSearchCV

```python
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100)
clf.get_params()
```

### Tuning Hyperparameters by hand

- Let's make 3 sets, training, validation and test
- Training set (course materials): 70%
- Validation set (practice exam): 15%
- Test set (final exam): 15%

We're going to try and adjust:

- max_depth
- max_features
- min_samples_leaf
- min_samples_split
- n_estimators

```python
from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)

# Shuffle the data
heart_disease_shuffled = heart_disease.sample(frac=1)

# Split into X & y
X = heart_disease_shuffled.drop("target", axis=1)
y = heart_disease_shuffled["target"]

# Split the data into train, validation & test sets
train_split = round(0.7 * len(heart_disease_shuffled)) # 70% of data
valid_split = round(train_split + 0.15 * len(heart_disease_shuffled)) # 15% of data
X_train, y_train = X[:train_split], y[:train_split] # training set
X_valid, y_valid = X[train_split:valid_split], y[train_split:valid_split] # validation set
X_test, y_test = X[valid_split:], y[valid_split:] # test set

clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Make baseline predictions - Practice exam
y_preds = clf.predict(X_valid)

# Evaluate the classifier on validation set
baseline_metrics = evaluate_preds(y_valid, y_preds)
baseline_metrics

# Make baseline predictions - Final exam
y_preds = clf.predict(X_test)

# Evaluate the classifier on test set
baseline_metrics = evaluate_preds(y_test, y_preds)
baseline_metrics

np.random.seed(42)

# Create a second classifier with different hyperparameters
clf_2 = RandomForestClassifier(n_estimators=100)
clf_2.fit(X_train, y_train)

# Make predictions with different hyperparameters
y_preds_2 = clf_2.predict(X_valid)

# Evalute the 2nd classsifier
clf_2_metrics = evaluate_preds(y_valid, y_preds_2)

def evaluate_preds(y_true, y_preds):
    """
    Performs evaluation comparison on y_true labels vs. y_pred labels
    on a classification.
    """
    accuracy = accuracy_score(y_true, y_preds)
    precision = precision_score(y_true, y_preds)
    recall = recall_score(y_true, y_preds)
    f1 = f1_score(y_true, y_preds)
    metric_dict = {"accuracy": round(accuracy, 2),
                   "precision": round(precision, 2),
                   "recall": round(recall, 2),
                   "f1": round(f1, 2)}
    print(f"Acc: {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 score: {f1:.2f}")

    return metric_dict
```

### Tuning Hyperparameters with RandomizedSearchCV

```python
from sklearn.model_selection import RandomizedSearchCV

grid = {"n_estimators": [10, 100, 200, 500, 1000, 1200],
        "max_depth": [None, 5, 10, 20, 30],
        "max_features": ["auto", "sqrt"],
        "min_samples_split": [2, 4, 6],
        "min_samples_leaf": [1, 2, 4]}

np.random.seed(42)

# Split into X & y
X = heart_disease_shuffled.drop("target", axis=1)
y = heart_disease_shuffled["target"]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Instantiate RandomForestClassifier
clf = RandomForestClassifier(n_jobs=1)

# Setup RandomizedSearchCV
rs_clf = RandomizedSearchCV(estimator=clf,
                            param_distributions=grid,
                            n_iter=10, # number of models to try
                            cv=5,
                            verbose=2)

# Fit the RandomizedSearchCV version of clf
rs_clf.fit(X_train, y_train);
rs_clf.best_params_

# Make predictions with the best hyperparameters - Final exam
rs_y_preds = rs_clf.predict(X_test)

# Evaluate the predictions
rs_metrics = evaluate_preds(y_test, rs_y_preds)
```

### Tuning Hyperparameters with GridSearchCV

- GridSearchCV goes through ALL combinations of hyperparameters in grid2
- Metric Comparison Improvement

```python
# reduce search space of hyperparameter
grid_2 = {'n_estimators': [100, 200, 500],
          'max_depth': [None],
          'max_features': ['auto', 'sqrt'],
          'min_samples_split': [6],
          'min_samples_leaf': [1, 2]}

from sklearn.model_selection import GridSearchCV, train_test_split

np.random.seed(42)

# Split into X & y
X = heart_disease_shuffled.drop("target", axis=1)
y = heart_disease_shuffled["target"]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Instantiate RandomForestClassifier
clf = RandomForestClassifier(n_jobs=1)

# Setup GridSearchCV
gs_clf = GridSearchCV(estimator=clf,
                      param_grid=grid_2,
                      cv=5,
                      verbose=2)

# Fit the GridSearchCV version of clf
gs_clf.fit(X_train, y_train);
gs_clf.best_params_

gs_y_preds = gs_clf.predict(X_test)

# evaluate the predictions
gs_metrics = evaluate_preds(y_test, gs_y_preds)

compare_metrics = pd.DataFrame({"baseline": baseline_metrics,
                                "clf_2": clf_2_metrics,
                                "random search": rs_metrics,
                                "grid search": gs_metrics})
compare_metrics.plot.bar(figsize=(10, 8));
```

### Quick Tip: Correlation Analysis

- Intro to Feature Selection Methods for Data Science
- Correlation Analysis
  - a statistical method used to evaluate the strength of relationship between two quantitative variables
  - A high correlation means that two or more variables have a strong relationship with each other
  - A weak correlation means that the variables are hardly related
- Forward Attribute Selection
  - Forward selection is an iterative method in which we start with having no feature in the model. In each iteration, we keep adding the feature which best improves our model till an addition of a new variable does not improve the performance of the model.
- Backward Attribute Selection
  - In backward elimination, we start with all the features and removes the least significant feature at each iteration which improves the performance of the model. We repeat this until no improvement is observed on removal of features.

### Saving And Loading A Model

Two ways to save and load machine learning models:

- With Python's [pickle](https://docs.python.org/3/library/pickle.html) module
- With the [joblib](https://joblib.readthedocs.io/en/latest/) module

[Model persistence](https://scikit-learn.org/stable/modules/model_persistence.html)

```python
# pickle
# Save an extisting model to file
pickle.dump(gs_clf, open("gs_random_random_forest_model_1.pkl", "wb"))

# Load a saved model
loaded_pickle_model = pickle.load(open("gs_random_random_forest_model_1.pkl", "rb"))

# Make some predictions
pickle_y_preds = loaded_pickle_model.predict(X_test)
evaluate_preds(y_test, pickle_y_preds)

# Joblib
from joblib import dump, load

# Save model to file
dump(gs_clf, filename="gs_random_forest_model_1.joblib")

# Import a saved joblib model
loaded_joblib_model = load(filename="gs_random_forest_model_1.joblib")

# Make and evaluate joblib predictions
joblib_y_preds = loaded_joblib_model.predict(X_test)
evaluate_preds(y_test, joblib_y_preds)
```

### Putting It All Together

Things to remember

- All data should be numerical
- There should be no missing values
- Manipulate the test set the same as the training set
- Never test on data you’ve trained on
- Tune hyperparameters on validation set OR use cross-validation
- One best performance metric doesn’t mean the best model

[Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)

- chain multiple estimators into one
- chain a fixed sequence of steps in preprocessing and modelling

Steps we want to do (all in one cell):

- Fill missing data
- Convert data to numbers
- Build a model on the data

```python
data = pd.read_csv("data/car-sales-extended-missing-data.csv")
data.dtypes
data.isna().sum()

# Getting data ready
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Modelling
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV

# Setup random seed
import numpy as np
np.random.seed(42)

# Import data and drop rows with missing labels
data = pd.read_csv("data/car-sales-extended-missing-data.csv")
data.dropna(subset=["Price"], inplace=True)

# Define different features and transformer pipeline
categorical_features = ["Make", "Colour"]
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))])

door_feature = ["Doors"]
door_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value=4))
])

numeric_features = ["Odometer (KM)"]
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean"))
])

# Setup preprocessing steps (fill missing values, then convert to numbers)
preprocessor = ColumnTransformer(
                    transformers=[
                        ("cat", categorical_transformer, categorical_features),
                        ("door", door_transformer, door_feature),
                        ("num", numeric_transformer, numeric_features)
                    ])

# Creating a preprocessing and modelling pipeline
model = Pipeline(steps=[("preprocessor", preprocessor),
                        ("model", RandomForestRegressor())])

# Split data
X = data.drop("Price", axis=1)
y = data["Price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Fit and score the model
model.fit(X_train, y_train)
model.score(X_test, y_test)

# Use GridSearchCV with our regression Pipeline
from sklearn.model_selection import GridSearchCV

pipe_grid = {
    "preprocessor__num__imputer__strategy": ["mean", "median"],
    "model__n_estimators": [100, 1000],
    "model__max_depth": [None, 5],
    "model__max_features": ["auto"],
    "model__min_samples_split": [2, 4]
}

gs_model = GridSearchCV(model, pipe_grid, cv=5, verbose=2)
gs_model.fit(X_train, y_train)
gs_model.score(X_test, y_test)
```

## [**Milestone Project 1: Supervised Learning (Classification)**](https://github.com/gorlimus/end-to-end-heart-disease-classification)

<!-- ## [**Section 12: Milestone Project 2: Supervised Learning (Time Series Data)**](bulldozer-price-prediction-project/end-to-end-bluebook-bulldozer-price-regression.ipynb)

### [Project Environment Setup]

- Download & install Miniconda
- Start new project
- Create project folder
- Data
  - [Blue Book for Bulldozers](https://www.kaggle.com/c/bluebook-for-bulldozers/overview)
- Create an environment
  - `conda env list`
  - `conda activate /Users/chesterheng/...`
  - `conda env export > environment.yml`
  - `vim environment.yml`
  - `esc + Shift + : + q`
  - `conda deactivate`
  - `conda create --prefix ./env -f environment.yml`
  - `conda create --prefix ./env pandas numpy matplotlib jupyter scikit-learn`
- Jupyter Notebooks
- Data Analysis & Manipulation
- Machine Learning


### Step 1~4 Framework Setup

🚜 Predicting the Sale Price of Bulldozers using Machine Learning

In this notebook, we're going to go through an example machine learning project with the goal of predicting the sale price of bulldozers.

1. Problem defition

> How well can we predict the future sale price of a bulldozer, given its characteristics and previous examples of how much similar bulldozers have been sold for?

2. Data

The data is downloaded from the Kaggle Bluebook for Bulldozers competition: https://www.kaggle.com/c/bluebook-for-bulldozers/data

There are 3 main datasets:

- Train.csv is the training set, which contains data through the end of 2011.
- Valid.csv is the validation set, which contains data from January 1, 2012 - April 30, 2012 You make predictions on this set throughout the majority of the competition. Your score on this set is used to create the public leaderboard.
- Test.csv is the test set, which won't be released until the last week of the competition. It contains data from May 1, 2012 - November 2012. Your score on the test set determines your final rank for the competition.

3. Evaluation

The evaluation metric for this competition is the RMSLE (root mean squared log error) between the actual and predicted auction prices.

For more on the evaluation of this project check: https://www.kaggle.com/c/bluebook-for-bulldozers/overview/evaluation

**Note:** The goal for most regression evaluation metrics is to minimize the error. For example, our goal for this project will be to build a machine learning model which minimises RMSLE.

4. Features

Kaggle provides a data dictionary detailing all of the features of the dataset. You can view this data dictionary on Google Sheets: https://docs.google.com/spreadsheets/d/18ly-bLR8sbDJLITkWG7ozKm8l3RyieQ2Fpgix-beSYI/edit?usp=sharing


### [Exploring Our Data](bulldozer-price-prediction-project/end-to-end-bluebook-bulldozer-price-regression.ipynb)

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

# Import training and validation sets
df = pd.read_csv("data/TrainAndValid.csv",
                 low_memory=False)

df.info()
df.isna().sum()
df.columns

fig, ax = plt.subplots()
ax.scatter(df["saledate"][:1000], df["SalePrice"][:1000])

df.saledate[:1000]
df.saledate.dtype
df.SalePrice.plot.hist()
```

Parsing dates

- When we work with time series data, we want to enrich the time & date component as much as possible.
- We can do that by telling pandas which of our columns has dates in it using the parse_dates parameter.

```python
# Import data again but this time parse dates
df = pd.read_csv("data/TrainAndValid.csv",
                 low_memory=False,
                 parse_dates=["saledate"])

df.saledate.dtype
df.saledate[:1000]

fig, ax = plt.subplots()
ax.scatter(df["saledate"][:1000], df["SalePrice"][:1000])

df.head()
df.head().T
df.saledate.head(20)

# Sort DataFrame in date order
df.sort_values(by=["saledate"], inplace=True, ascending=True)
df.saledate.head(20)

# Make a copy of the original DataFrame to perform edits on
df_tmp = df.copy()
```


### [Feature Engineering](bulldozer-price-prediction-project/end-to-end-bluebook-bulldozer-price-regression.ipynb)

[DatetimeIndex](https://pandas.pydata.org/pandas-docs/version/0.23.4/generated/pandas.DatetimeIndex.html)

```python
# Add datetime parameters for saledate column
df_tmp["saleMonth"] = df_tmp.saledate.dt.month
df_tmp["saleDay"] = df_tmp.saledate.dt.day
df_tmp["saleDayOfWeek"] = df_tmp.saledate.dt.dayofweek
df_tmp["saleDayOfYear"] = df_tmp.saledate.dt.dayofyear

df_tmp.head().T
# Now we've enriched our DataFrame with date time features, we can remove 'saledate'
df_tmp.drop("saledate", axis=1, inplace=True)

# Check the values of different columns
df_tmp.state.value_counts()

df_tmp.head()
len(df_tmp)
```


### [Turning Data Into Numbers](bulldozer-price-prediction-project/end-to-end-bluebook-bulldozer-price-regression.ipynb)

One way we can turn all of our data into numbers is by converting them into pandas catgories.

```python
df_tmp.info()
df_tmp["UsageBand"].dtype
df_tmp.head().T
pd.api.types.is_string_dtype(df_tmp["UsageBand"])

# Find the columns which contain strings
for label, content in df_tmp.items():
    if pd.api.types.is_string_dtype(content):
        print(label)

# This will turn all of the string value into category values
for label, content in df_tmp.items():
    if pd.api.types.is_string_dtype(content):
        df_tmp[label] = content.astype("category").cat.as_ordered()

df_tmp.info()
df_tmp.state.cat.categories
df_tmp.state.cat.codes

# Check missing data
df_tmp.isnull().sum()/len(df_tmp)

# Export current tmp dataframe
df_tmp.to_csv("data/train_tmp.csv",
              index=False)

# Import preprocessed data
df_tmp = pd.read_csv("data/train_tmp.csv",
                     low_memory=False)
df_tmp.head().T
df_tmp.isna().sum()
```

```python
# If you're wondering what df.items() does, here's an example
random_dict = {"key1": "hello",
               "key2": "world!"}

for key, value in random_dict.items():
    print(f"this is a key: {key}",
          f"this is a value: {value}")
```


### [Filling Missing Numerical Values](bulldozer-price-prediction-project/end-to-end-bluebook-bulldozer-price-regression.ipynb)

```python
for label, content in df_tmp.items():
    if pd.api.types.is_numeric_dtype(content):
        print(label)
df_tmp.ModelID

# Check for which numeric columns have null values
for label, content in df_tmp.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            print(label)

# Fill numeric rows with the median
for label, content in df_tmp.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            # Add a binary column which tells us if the data was missing or not
            df_tmp[label+"_is_missing"] = pd.isnull(content)
            # Fill missing numeric values with median
            df_tmp[label] = content.fillna(content.median())

# Check if there's any null numeric values
for label, content in df_tmp.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            print(label)

# Check to see how many examples were missing
df_tmp.auctioneerID_is_missing.value_counts()

df_tmp.isna().sum()
```

```python
# Demonstrate how median is more robust than mean
hundreds = np.full((1000,), 100)
hundreds_billion = np.append(hundreds, 1000000000)
np.mean(hundreds), np.mean(hundreds_billion), np.median(hundreds), np.median(hundreds_billion)
```


### [Filling Missing Categorical Values](bulldozer-price-prediction-project/end-to-end-bluebook-bulldozer-price-regression.ipynb)

```python
# Check for columns which aren't numeric
for label, content in df_tmp.items():
    if not pd.api.types.is_numeric_dtype(content):
        print(label)

# Turn categorical variables into numbers and fill missing
for label, content in df_tmp.items():
    if not pd.api.types.is_numeric_dtype(content):
        # Add binary column to indicate whether sample had missing value
        df_tmp[label+"_is_missing"] = pd.isnull(content)
        # Turn categories into numbers and add +1
        df_tmp[label] = pd.Categorical(content).codes+1

# + 1 to turn -1 to 0, so we know 0 is missing value
pd.Categorical(df_tmp["state"]).codes+1

df_tmp.info()
df_tmp.head().T
df_tmp.isna().sum()
df_tmp.head()
len(df_tmp)
```


### [Fitting A Machine Learning Model](bulldozer-price-prediction-project/end-to-end-bluebook-bulldozer-price-regression.ipynb)

```python
%%time
# Instantiate model
model = RandomForestRegressor(n_jobs=-1,
                              random_state=42)

# Fit the model
model.fit(df_tmp.drop("SalePrice", axis=1), df_tmp["SalePrice"])

# Score the model
model.score(df_tmp.drop("SalePrice", axis=1), df_tmp["SalePrice"])
```


### [Splitting Data](bulldozer-price-prediction-project/end-to-end-bluebook-bulldozer-price-regression.ipynb)

```python
df_tmp.saleYear
df_tmp.saleYear.value_counts()

# Split data into training and validation
df_val = df_tmp[df_tmp.saleYear == 2012]
df_train = df_tmp[df_tmp.saleYear != 2012]

len(df_val), len(df_train)

# Split data into X & y
X_train, y_train = df_train.drop("SalePrice", axis=1), df_train.SalePrice
X_valid, y_valid = df_val.drop("SalePrice", axis=1), df_val.SalePrice

X_train.shape, y_train.shape, X_valid.shape, y_valid.shape
```


### [Challenge: What's wrong with splitting data after filling it?](bulldozer-price-prediction-project/end-to-end-bluebook-bulldozer-price-regression.ipynb)

In the previous few videos we worked on filling the missing data in the training and validation data before splitting it into training and validation sets using the following code:

```python
# Split data into training and validation
df_val = df_tmp[df_tmp.saleYear == 2012]
df_train = df_tmp[df_tmp.saleYear != 2012]
```

The code worked but how might this interfere with our model?

Remember the goal of machine learning: use the past to predict the future.

So if our validation set is supposed to be representative of the future and we’re filling our training data using information from the validation set, what might this mean for our model?

The challenge here comes in two parts.

- What does it mean if we fill our training data with information from the future (validation set)?

- How might you implement a fix to the current way things are being done in the project?

If you need a hint, remember some takeaways from a previous lecture:

- Split your data first (into train/test), always keep your training & test data separate

- Fill/transform the training set and test sets separately (this goes for filling data with pandas as well)

- Don’t use data from the future (test set) to fill data from the past (training set)

Keep these things in mind when we create a data preprocessing function in a few videos time, they'll help you answer the question which gets raised then too.


### [Custom Evaluation Function](bulldozer-price-prediction-project/end-to-end-bluebook-bulldozer-price-regression.ipynb)

```python
# Create evaluation function (the competition uses RMSLE)
from sklearn.metrics import mean_squared_log_error, mean_absolute_error, r2_score

def rmsle(y_test, y_preds):
    """
    Caculates root mean squared log error between predictions and
    true labels.
    """
    return np.sqrt(mean_squared_log_error(y_test, y_preds))

# Create function to evaluate model on a few different levels
def show_scores(model):
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_valid)
    scores = {"Training MAE": mean_absolute_error(y_train, train_preds),
              "Valid MAE": mean_absolute_error(y_valid, val_preds),
              "Training RMSLE": rmsle(y_train, train_preds),
              "Valid RMSLE": rmsle(y_valid, val_preds),
              "Training R^2": r2_score(y_train, train_preds),
              "Valid R^2": r2_score(y_valid, val_preds)}
    return scores
```


### [Reducing Data](bulldozer-price-prediction-project/end-to-end-bluebook-bulldozer-price-regression.ipynb)

```python
# # This takes far too long... for experimenting

# %%time
# model = RandomForestRegressor(n_jobs=-1,
#                               random_state=42)

# model.fit(X_train, y_train)
len(X_train)

# Change max_samples value
model = RandomForestRegressor(n_jobs=-1,
                              random_state=42,
                              max_samples=10000)

%%time
# Cutting down on the max number of samples each estimator can see improves training time
model.fit(X_train, y_train)

# original dataset size = X_train.shape[0] * 100
# new dataset size = 10000 * 100
# 40 times smaller
(X_train.shape[0] * 100) / 1000000
10000 * 100

show_scores(model)
```


### [RandomizedSearchCV](bulldozer-price-prediction-project/end-to-end-bluebook-bulldozer-price-regression.ipynb)

```python
%%time
from sklearn.model_selection import RandomizedSearchCV

# Different RandomForestRegressor hyperparameters
rf_grid = {"n_estimators": np.arange(10, 100, 10),
           "max_depth": [None, 3, 5, 10],
           "min_samples_split": np.arange(2, 20, 2),
           "min_samples_leaf": np.arange(1, 20, 2),
           "max_features": [0.5, 1, "sqrt", "auto"],
           "max_samples": [10000]}

# Instantiate RandomizedSearchCV model
rs_model = RandomizedSearchCV(RandomForestRegressor(n_jobs=-1,
                                                    random_state=42),
                              param_distributions=rf_grid,
                              n_iter=2,
                              cv=5,
                              verbose=True)

# Fit the RandomizedSearchCV model
rs_model.fit(X_train, y_train)

# Find the best model hyperparameters
rs_model.best_params_

# Evaluate the RandomizedSearch model
show_scores(rs_model)
```


### [Improving Hyperparameters](bulldozer-price-prediction-project/end-to-end-bluebook-bulldozer-price-regression.ipynb)

Train a model with the best hyperparamters

Note: These were found after 100 iterations of RandomizedSearchCV.

```python
%%time

# Most ideal hyperparamters
ideal_model = RandomForestRegressor(n_estimators=40,
                                    min_samples_leaf=1,
                                    min_samples_split=14,
                                    max_features=0.5,
                                    n_jobs=-1,
                                    max_samples=None,
                                    random_state=42) # random state so our results are reproducible

# Fit the ideal model
ideal_model.fit(X_train, y_train)

# Scores for ideal_model (trained on all the data)
show_scores(ideal_model)

# Scores on rs_model (only trained on ~10,000 examples)
show_scores(rs_model)
```


### [Preproccessing Our Data](bulldozer-price-prediction-project/end-to-end-bluebook-bulldozer-price-regression.ipynb)

Getting the test dataset in the same format as our training dataset

```python
def preprocess_data(df):
    """
    Performs transformations on df and returns transformed df.
    """
    df["saleYear"] = df.saledate.dt.year
    df["saleMonth"] = df.saledate.dt.month
    df["saleDay"] = df.saledate.dt.day
    df["saleDayOfWeek"] = df.saledate.dt.dayofweek
    df["saleDayOfYear"] = df.saledate.dt.dayofyear

    df.drop("saledate", axis=1, inplace=True)

    # Fill the numeric rows with median
    for label, content in df.items():
        if pd.api.types.is_numeric_dtype(content):
            if pd.isnull(content).sum():
                # Add a binary column which tells us if the data was missing or not
                df[label+"_is_missing"] = pd.isnull(content)
                # Fill missing numeric values with median
                df[label] = content.fillna(content.median())

        # Filled categorical missing data and turn categories into numbers
        if not pd.api.types.is_numeric_dtype(content):
            df[label+"_is_missing"] = pd.isnull(content)
            # We add +1 to the category code because pandas encodes missing categories as -1
            df[label] = pd.Categorical(content).codes+1

    return df

# Process the test data
df_test = preprocess_data(df_test)
df_test.head()
```


### [Making Predictions](bulldozer-price-prediction-project/end-to-end-bluebook-bulldozer-price-regression.ipynb)

```python
# We can find how the columns differ using sets
set(X_train.columns) - set(df_test.columns)

# Manually adjust df_test to have auctioneerID_is_missing column
df_test["auctioneerID_is_missing"] = False
df_test.head()

# Make predictions on the test data
test_preds = ideal_model.predict(df_test)

# Format predictions into the same format Kaggle is after
df_preds = pd.DataFrame()
df_preds["SalesID"] = df_test["SalesID"]
df_preds["SalesPrice"] = test_preds
df_preds

# Export prediction data
df_preds.to_csv("data/test_predictions.csv", index=False)
```


### [Feature Importance](bulldozer-price-prediction-project/end-to-end-bluebook-bulldozer-price-regression.ipynb)

Feature importance seeks to figure out which different attributes of the data were most importance when it comes to predicting the target variable (SalePrice).

```python
# Find feature importance of our best model
ideal_model.feature_importances_

# Helper function for plotting feature importance
def plot_features(columns, importances, n=20):
    df = (pd.DataFrame({"features": columns,
                        "feature_importances": importances})
          .sort_values("feature_importances", ascending=False)
          .reset_index(drop=True))

    # Plot the dataframe
    fig, ax = plt.subplots()
    ax.barh(df["features"][:n], df["feature_importances"][:20])
    ax.set_ylabel("Features")
    ax.set_xlabel("Feature importance")
    ax.invert_yaxis()

plot_features(X_train.columns, ideal_model.feature_importances_)

df["Enclosure"].value_counts()
```

**Question to finish:** Why might knowing the feature importances of a trained machine learning model be helpful?

**Final challenge/extension:** What other machine learning models could you try on our dataset?

**Hint:** https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html check out the regression section of this map, or try to look at something like CatBoost.ai or XGBooost.ai.


## **Section 13: Data Engineering**

### Data Engineering Introduction

Data science is all about using data to make business decisions

Data science is the idea of using data and converting it into something useful for a product or business.

Data analysis is a subset of data science that allows us to analyze the data that we have.

Machine Learning is a technique to allow a computer to learn and figure out the solution to a problem that may be a little too complicated for a human to solve or maybe too tedious and takes too long of a time so we want to automate it.

A company has all these datas are coming from their users from their security cameras from their Web site from IOT devices.

A data engineer takes all this information and then produces it and maintains it in databases or a certain type of computers so that the business has access to this data in an organized fashion.


### What Is Data?

- Part of Product - eg. YouTube recommendation engine
- Are we doing ok? - monitor the company's sales
- Can we do better?

Type of data (organised -> unorganised)

1. Structured Data - from relational Database
2. Semi-Structured Data - eg. XML, CSV, JSON
3. UnStructured Data - eg. pdf, email, document
4. Binary Data - audio, image, video

So one of the tasks of a data engineer is to essentially use the fact that there's all these types of data and somehow combine them or organize them in a way that is useful to the business.


### What Is A Data Engineer?

Data Mining - pre processing and extracting some knowledge from the data

Big Data - data that's so big that you need to have it running on cloud computing or multiple computers

Data Pipeline - build a pipeline that allows us to flow from that unknown large amount of data to a pipeline that extracts data to a more useful form

A data engineer allows us to do this data collection part. They bring in all this information organize it in a way for us to do our data modelling.

And this is what a data engineer built a data engineer starts off with what we call data ingestion that

is acquiring data from various sources and we acquire all these different sources of data and ingested

into what we call a data lake a data lake is a collection.

Well all this data into one location from there we could just leave the lake as it is.

Build the following data pipeline

- Rain -> Data
- Collected into streams and rivers - data ingestion
  - acquire data from various sources and ingested into a data lake
- Lakes / Dam - Data lake (pool of raw data)
- filtration sanitary area - data transformation that is convert data from one format to another
  - data warehouse is a location for structured filtered data that has been processed and has a specific purpose
- plumbing and pipes for us to deliver water

Data Ingestion Tool

- Kafka

Data Lake Tools

- hadoop
- Azure Data lake
- Amazon S3

Data warehouse Tools

- Amazon Athena
- Amazon Redshift
- Google BigQuery

Who use Data Lake?

- Machine Learning
- Data Scientist

Who use Data Warehouse?

- Business intelligent
- business analyst
- data analyst

A software engineer, a software developer, app developer and mobile developer build programs and apps that users and customers use.

The app releases data. A data engineer would build this pipeline for us to ingest data and store it in different services like Hadoop like Google big query so that that data can be accessed by the rest of the business.

Next, data scientists use the data lake to extract information and deliver some sort of business value.

Finally we have data analysts or business intelligence to use something like a data warehouse or structured data to again derive business value.

3 main tasks of data engineer,

- Build ETL pipeline (Extract, Transform and Load into data warehouse)
- Build analysis tools
- Maintain data warehouse and data lakes


### Types Of Databases

Relational Database

- use SQL to make transaction
- [ACID transaction](https://blog.yugabyte.com/a-primer-on-acid-transactions/)

NoSQL - eg. MongoDB,

- distributed database
- Disorganised

NewSQL - eg. VoltDB, CockroachDB

- distributed
- ACID transaction

Usage

- Search - eg. ElasticSearch or solr
- Computation - eg. Apache Spark

[OLTP vs OLAP](https://techdifferences.com/difference-between-oltp-and-olap.html)
OLTP - SQL database, relational database, transactional
OLAP - use for analytical purpose

- view a financial report, or budgeting, marketing management, sales report


### Optional: OLTP Databases

What is a database?

- A database is a collection of data.

Many form of data

- numbers
- dates
- password hashes
- user information

2 types of DBMS

- Relational Database
- NoSQL / Non Relational Database (document oriented)


### Optional: Learn SQL

- [Intro to SQL: Querying and managing data](https://www.khanacademy.org/computing/computer-programming/sql#more-advanced-sql-queries)
- [SQL Tutorial](https://sqlzoo.net/)


### Hadoop, HDFS and MapReduce

Hadoop (store a lots of data across multiple machine)

- HDFS (Hadoop distributed file system)
- MapReduce (batch processing)

Hive - makes your Hadoop cluster feel like it's a relational database


### Apache Spark and Apache Flink

Apache Spark

- run ETL jobs like extract transform load to clean and transform that data

Apache Flink

- real time processing started to happen things like spark streaming


### Kafka and Stream Processing

Batch processing

- Hadoop
- Spark
- AWS S3
- Common Databases

Real time stream processing

- Spark Streaming
- Flink
- Storm
- Kineses

Data -> Ingest data through Kafka -> Real time stream processing
Data -> Ingest data through Kafka -> Batch processing


## **Section 14: Neural Networks: Deep Learning, Transfer Learning and TensorFlow 2**

### Deep Learning and Unstructured Data

[Google Colab](https://colab.research.google.com/notebooks/intro.ipynb#recent=true)

- alternative to jupyter notebook

[TensorFlow](https://www.tensorflow.org/)

- a deep learning or numerical computing library
- use for unstructured data - Photos, Audio waves, natural language text

Why TensorFlow ?

- Write fast deep learning code in Python (able to run on a GPU)
- Able to access many pre-built deep learning models
- Whole stack: preprocess, model, deploy
- Originally designed and used in-house by Google (now open-source)

Choosing a model (throwback)

- Problem 1 (structured data) -> Choose a Model
  - CatBoost, dmlc XGBoost, Random Forest
- Problem 2 (unstructured data) -> Choose a Model
  - Deep Learning use TensorFlow
  - Transfer Learning use TensorFlow Hub

What is deep learning?

- another form of machine learning

What are neural networks?

- type of machine learning algorithm for deep learning

What kind of deep learning problems are there?

- Classification
  - multi-classification of dog breed
  - classification of spam email
- Sequence to sequence (seq2seq)
  - audio to text translation
- Object detection

What is transfer learning? Why use transfer learning?

- Take what you know in one domain and apply it to another.
- Starting from scratch can be expensive and time consuming.
- Why not take advantage of what’s already out there?

A TensorFlow workflow

- Get the data ready (turn into Tensors)
  - An end-to-end multi-class classification workflow with TensorFlow Preprocessing image data (getting it into Tensors)
- Pick a model from TensorFlow Hub
  - Choosing a deep learning model
- Fit the model to the data and make a prediction
  - Fitting a model to the data (learning patterns)
  - Making predictions with a model (using patterns)
- Evaluate the model
  - Evaluating model predictions
- Improve through experimentation
- Save and reload your trained model
  - Saving and loading models
- Using a trained model to make predictions on custom data


### Setting Up Google Colab

- [Using Transfer Learning and TensorFlow 2.0 to Classify Different Dog Breeds](https://github.com/mrdbourke/zero-to-mastery-ml/blob/wip/section-4-unstructured-data-projects/end-to-end-dog-vision.ipynb)
- [Dog Breed Identification](https://www.kaggle.com/c/dog-breed-identification/overview)
- [Kaggle API](https://github.com/Kaggle/kaggle-api)
- [What is Colaboratory?](https://colab.research.google.com/notebooks/intro.ipynb)
- [External data: Local Files, Drive, Sheets, and Cloud Storage](https://colab.research.google.com/notebooks/io.ipynb)


### Google Colab Workspace

- [Welcome To Colaboratory](https://colab.research.google.com/notebooks/intro.ipynb)
- [Colaboratory Frequently Asked Questions](https://research.google.com/colaboratory/faq.html)


### Uploading Project Data

- File > Mount Drive
- Upload dog-breed-identification.zip to "drive/My Drive/Dog Vision"


### Setting Up Our Data

```python
!unzip "drive/My Drive/Dog Vision/dog-breed-identification.zip" -d "drive/My Drive/Dog Vision/"
```


### [Importing TensorFlow 2](https://colab.research.google.com/drive/1OZLn22hAZkNY1hHaK-FePk8sLVnbTBwV#scrollTo=d4DgYi9Zceiy)

```python
# Import necessary tools
import tensorflow as tf
import tensorflow_hub as hub
print("TF version:", tf.__version__)
print("TF Hub version:", hub.__version__)

# Check for GPU availability
print("GPU", "available (YESSSS!!!!!)" if tf.config.list_physical_devices("GPU") else "not available :(")
```


### Using A GPU

[Tensorflow with GPU](https://colab.research.google.com/notebooks/gpu.ipynb)

But we can fix this going to runtime and then changing the runtime type:

- Go to Runtime.
- Click "Change runtime type".
- Where it says "Hardware accelerator", choose "GPU" (don't worry about TPU for now but feel free to research them).
- Click save.
- The runtime will be restarted to activate the new hardware, so you'll have to rerun the above cells.
  - If the steps have worked you should see a print out saying "GPU available".


### [Loading Our Data Labels](https://colab.research.google.com/drive/1OZLn22hAZkNY1hHaK-FePk8sLVnbTBwV#scrollTo=d4DgYi9Zceiy)

[Preparing your training data](https://cloud.google.com/vision/automl/object-detection/docs/prepare)

```python
import pandas as pd
labels_csv = pd.read_csv("drive/My Drive/Dog Vision/labels.csv")
labels_csv.describe()
labels_csv.head()

# How many images are there of each breed?
labels_csv["breed"].value_counts().plot.bar(figsize=(20, 10))

# What's the median number of images per class?
labels_csv["breed"].value_counts().median()
```


### [Preparing The Images](https://colab.research.google.com/drive/1OZLn22hAZkNY1hHaK-FePk8sLVnbTBwV#scrollTo=d4DgYi9Zceiy)

```python
# Let's view an image
from IPython.display import Image
Image("drive/My Drive/Dog Vision/train/001513dfcb2ffafc82cccf4d8bbaba97.jpg")

labels_csv.head()

# Create pathnames from image ID's
filenames = ["drive/My Drive/Dog Vision/train/" + fname + ".jpg" for fname in labels_csv["id"]]

# Check the first 10 filenames
filenames[:10]

# Check whether number of filenames matches number of actual image files
import os
if len(os.listdir("drive/My Drive/Dog Vision/train/")) == len(filenames):
  print("Filenames match actual amount of files!")
else:
  print("Filenames do not match actual amount of files, check the target directory.")

# One more check
Image(filenames[9000])

labels_csv["breed"][9000]
```


### [Turning Data Labels Into Numbers](https://colab.research.google.com/drive/1OZLn22hAZkNY1hHaK-FePk8sLVnbTBwV#scrollTo=d4DgYi9Zceiy)

```python
import numpy as np
labels = labels_csv["breed"].to_numpy()
# labels = np.array(labels) # does same thing as above

len(labels)

# Find the unique label values
unique_breeds = np.unique(labels)
len(unique_breeds)

# Turn a single label into an array of booleans
print(labels[0])
labels[0] == unique_breeds

# Turn every label into a boolean array
boolean_labels = [label == unique_breeds for label in labels]
boolean_labels[:2]
len(boolean_labels)

# Example: Turning boolean array into integers
print(labels[0]) # original label
print(np.where(unique_breeds == labels[0])) # index where label occurs
print(boolean_labels[0].argmax()) # index where label occurs in boolean array
print(boolean_labels[0].astype(int)) # there will be a 1 where the sample label occurs

print(labels[2])
print(boolean_labels[2].astype(int))

filenames[:10]
```


### [Creating Our Own Validation Set](https://colab.research.google.com/drive/1OZLn22hAZkNY1hHaK-FePk8sLVnbTBwV#scrollTo=d4DgYi9Zceiy)

[How (and why) to create a good validation set](https://www.fast.ai/2017/11/13/validation-sets/)

```python
# Setup X & y variables
X = filenames
y = boolean_labels

len(filenames)

# Set number of images to use for experimenting
NUM_IMAGES = 1000 #@param {type:"slider", min:1000, max:10000, step:1000}

# Let's split our data into train and validation sets
from sklearn.model_selection import train_test_split

# Split them into training and validation of total size NUM_IMAGES
X_train, X_val, y_train, y_val = train_test_split(X[:NUM_IMAGES],
                                                  y[:NUM_IMAGES],
                                                  test_size=0.2,
                                                  random_state=42)

len(X_train), len(y_train), len(X_val), len(y_val)

# Check out the training data (image file paths and labels)
X_train[:5], y_train[:2]
```


### [Preprocess Images](https://colab.research.google.com/drive/1OZLn22hAZkNY1hHaK-FePk8sLVnbTBwV#scrollTo=d4DgYi9Zceiy)

- tensors are numerical representation
- like a matrix
- [Load images](https://www.tensorflow.org/tutorials/load_data/images)
- [tf.data: Build TensorFlow input pipelines](https://www.tensorflow.org/guide/data)

```python
# Convert image to NumPy array
from matplotlib.pyplot import imread
image = imread(filenames[42])
image.shape # color channel
# dimensions: 350 x 257
# each pixel has 3 part (rgb)

image.max(), image.min()
image[:2]

# turn image into a tensor
tf.constant(image)[:2]
```

Now we've seen what an image looks like as a Tensor, let's make a function to preprocess them.

We'll create a function to:

1. Take an image filepath as input
2. Use TensorFlow to read the file and save it to a variable, `image`
3. Turn our `image` (a jpg) into Tensors
4. Normalize our image (convert color channel values from from 0-255 to 0-1).
5. Resize the `image` to be a shape of (224, 224)
6. Return the modified `image`

```python
# Define image size
IMG_SIZE = 224

# Create a function for preprocessing images
def process_image(image_path, img_size=IMG_SIZE):
  """
  Takes an image file path and turns the image into a Tensor.
  """
  # Read in an image file
  image = tf.io.read_file(image_path)
  # Turn the jpeg image into numerical Tensor with 3 colour channels (Red, Green, Blue)
  image = tf.image.decode_jpeg(image, channels=3)
  # Convert the colour channel values from 0-255 to 0-1 values
  image = tf.image.convert_image_dtype(image, tf.float32)
  # Resize the image to our desired value (224, 224)
  image = tf.image.resize(image, size=[IMG_SIZE, IMG_SIZE])

  return image
```


### [Turning Data Into Batches](https://colab.research.google.com/drive/1OZLn22hAZkNY1hHaK-FePk8sLVnbTBwV#scrollTo=d4DgYi9Zceiy)

[Yann LeCun Batch Size](https://twitter.com/ylecun/status/989610208497360896?s=20)

```python
# Create a simple function to return a tuple (image, label)
def get_image_label(image_path, label):
  """
  Takes an image file path name and the assosciated label,
  processes the image and reutrns a typle of (image, label).
  """
  image = process_image(image_path)
  return image, label

# Demo of the above
(process_image(X[42]), tf.constant(y[42]))

# Define the batch size, 32 is a good start
BATCH_SIZE = 32

# Create a function to turn data into batches
def create_data_batches(X, y=None, batch_size=BATCH_SIZE, valid_data=False, test_data=False):
  """
  Creates batches of data out of image (X) and label (y) pairs.
  Shuffles the data if it's training data but doesn't shuffle if it's validation data.
  Also accepts test data as input (no labels).
  """
  # If the data is a test dataset, we probably don't have have labels
  if test_data:
    print("Creating test data batches...")
    data = tf.data.Dataset.from_tensor_slices((tf.constant(X))) # only filepaths (no labels)
    data_batch = data.map(process_image).batch(BATCH_SIZE)
    return data_batch

  # If the data is a valid dataset, we don't need to shuffle it
  elif valid_data:
    print("Creating validation data batches...")
    data = tf.data.Dataset.from_tensor_slices((tf.constant(X), # filepaths
                                               tf.constant(y))) # labels
    data_batch = data.map(get_image_label).batch(BATCH_SIZE)
    return data_batch

  else:
    print("Creating training data batches...")
    # Turn filepaths and labels into Tensors
    data = tf.data.Dataset.from_tensor_slices((tf.constant(X),
                                               tf.constant(y)))
    # Shuffling pathnames and labels before mapping image processor function is faster than shuffling images
    data = data.shuffle(buffer_size=len(X))

    # Create (image, label) tuples (this also turns the iamge path into a preprocessed image)
    data = data.map(get_image_label)

    # Turn the training data into batches
    data_batch = data.batch(BATCH_SIZE)
  return data_batch

# Create training and validation data batches
train_data = create_data_batches(X_train, y_train)
val_data = create_data_batches(X_val, y_val, valid_data=True)

# Check out the different attributes of our data batches
train_data.element_spec, val_data.element_spec
```


### [Visualizing Our Data](https://colab.research.google.com/drive/1OZLn22hAZkNY1hHaK-FePk8sLVnbTBwV#scrollTo=d4DgYi9Zceiy)

```python
import matplotlib.pyplot as plt

# Create a function for viewing images in a data batch
def show_25_images(images, labels):
  """
  Displays a plot of 25 images and their labels from a data batch.
  """
  # Setup the figure
  plt.figure(figsize=(10, 10))
  # Loop through 25 (for displaying 25 images)
  for i in range(25):
    # Create subplots (5 rows, 5 columns)
    ax = plt.subplot(5, 5, i+1)
    # Display an image
    plt.imshow(images[i])
    # Add the image label as the title
    plt.title(unique_breeds[labels[i].argmax()])
    # Turn the grid lines off
    plt.axis("off")

train_data

# Now let's visualize the data in a training batch
train_images, train_labels = next(train_data.as_numpy_iterator())
show_25_images(train_images, train_labels)

# Now let's visualize our validation set
val_images, val_labels = next(val_data.as_numpy_iterator())
show_25_images(val_images, val_labels)

# Now let's visualize our validation set
val_images, val_labels = next(val_data.as_numpy_iterator())
show_25_images(val_images, val_labels)
```


### [Preparing Our Inputs and Outputs](https://colab.research.google.com/drive/1OZLn22hAZkNY1hHaK-FePk8sLVnbTBwV#scrollTo=d4DgYi9Zceiy)

Building a model

Before we build a model, there are a few things we need to define:

- The input shape (our images shape, in the form of Tensors) to our model.
- The output shape (image labels, in the form of Tensors) of our model.
- The URL of the model we want to use from TensorFlow Hub https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/4

```python
IMG_SIZE

# Setup input shape to the model
INPUT_SHAPE = [None, IMG_SIZE, IMG_SIZE, 3] # batch, height, width, colour channels

# Setup output shape of our model
OUTPUT_SHAPE = len(unique_breeds)

# Setup model URL from TensorFlow Hub
MODEL_URL = "https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/4"

INPUT_SHAPE
```


### How machines learn and what's going on behind the scenes?

Massive effort getting the data ready for use with a machine learning model! This is one of the most important steps in any machine learning project.

Now you've got the data ready, you're about to dive headfirst into writing deep learning code with TensorFlow 2.x.

Since we're focused on writing code first and foremost, these videos are optional but they're here for those who want to start to get an understanding of what goes on behind the scenes.

How Machines Learn

The first is a video called [How Machines Learn](https://www.youtube.com/watch?v=R9OHn5ZF4Uo) by GCP Grey on YouTube.

It's a non-technical narrative explaining how some of the biggest tech companies in the world use data to improve their businesses. In short, they're leveraging techniques like the ones you've been learning. Instead of trying to think of every possible rule to code, they collect data and then use machines to figure out the patterns for them.

What actually is a neural network?

You're going to be writing code which builds a neural network (a type of machine learning model) so you might start to wonder, what's going on when you run the code?

When you pass inputs (often data and labels) to a neural network and it figures out patterns between them, how is it doing so?

When it tries to make predictions and gets them wrong, how does it improve itself?

[The deep learning series](https://www.youtube.com/watch?v=aircAruvnKk) by 3Blue1Brown on YouTube contains a technical deep-dive into what's going on behind the code you're writing.

Be warned though, it isn't for the faint of heart. The videos explain the topics in a beautiful way but it doesn't mean the topics aren't still difficult to comprehend.

If you're up for it, a good idea would be to watch 1 video in the series one day and then another the day after and so on.

Remember, you don't need to know all of these things to get started writing machine learning code. Focus on solving problems first (like we're doing in this project) and then dive deeper when you need to.

And since these videos are optional, feel free to bookmark them for now, continue with the course and come back later!


### [Building A Deep Learning Model](https://colab.research.google.com/drive/1OZLn22hAZkNY1hHaK-FePk8sLVnbTBwV#scrollTo=LrjvXtrpszqp)

[TensorFlow Hub](https://www.tensorflow.org/hub)
[Papers With Code](https://paperswithcode.com/)
[PyTouch Hub](https://pytorch.org/hub/)
[Model Zoo](https://modelzoo.co/)
[TensorFlow Keras](https://www.tensorflow.org/guide/keras)

```python
# Setup input shape to the model
INPUT_SHAPE = [None, IMG_SIZE, IMG_SIZE, 3] # batch, height, width, colour channels

# Setup output shape of our model
OUTPUT_SHAPE = len(unique_breeds)

# Setup model URL from TensorFlow Hub
MODEL_URL = "https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/4"
```

[A Comprehensive Guide to Convolutional Neural Networks — the ELI5 way](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53)
[Review: MobileNetV2 — Light Weight Model (Image Classification)](https://towardsdatascience.com/review-mobilenetv2-light-weight-model-image-classification-8febb490e61c)
[How to Choose Loss Functions When Training Deep Learning Neural Networks](https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/)
[TensorFlow Metrics](https://www.tensorflow.org/api_docs/python/tf/keras/metrics)

Which activation? Which loss?
| |Binary classification| Multi-class classification|
|-|-|-|
| Activation| Sigmoid | Softmax|
| Loss | Binary Crossentropy| Categorical Crossentropy|

```python
# Create a function which builds a Keras model
def create_model(input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE, model_url=MODEL_URL):
  print("Building model with:", MODEL_URL)

  # Setup the model layers
  model = tf.keras.Sequential([
    hub.KerasLayer(MODEL_URL), # Layer 1 (input layer)
    tf.keras.layers.Dense(units=OUTPUT_SHAPE,
                          activation="softmax") # Layer 2 (output layer)
  ])

  # Compile the model
  model.compile(
      loss=tf.keras.losses.CategoricalCrossentropy(),
      optimizer=tf.keras.optimizers.Adam(),
      metrics=["accuracy"]
  )

  # Build the model
  model.build(INPUT_SHAPE)

  return model

model = create_model()
model.summary()
```


### [Summarizing Our Model](https://colab.research.google.com/drive/1OZLn22hAZkNY1hHaK-FePk8sLVnbTBwV#scrollTo=LrjvXtrpszqp)

[ImageNet](http://www.image-net.org/)


### [Evaluating Our Model](https://colab.research.google.com/drive/1OZLn22hAZkNY1hHaK-FePk8sLVnbTBwV#scrollTo=LrjvXtrpszqp)

Callbacks are helper functions a model can use during training to do such things as save its progress, check its progress or stop training early if a model stops improving.

To setup a TensorBoard callback, we need to do 3 things:
- Load the TensorBoard notebook extension ✅
- Create a TensorBoard callback which is able to save logs to a directory and pass it to our model's `fit()` function. ✅
- Visualize our models training logs with the `%tensorboard` magic function (we'll do this after model training).

```python
# Load TensorBoard notebook extension
%load_ext tensorboard

import datetime

# Create a function to build a TensorBoard callback
# to help track our models progress

def create_tensorboard_callback():
  # Create a log directory for storing TensorBoard logs
  logdir = os.path.join("drive/My Drive/Dog Vision/logs",
                        # Make it so the logs get tracked whenever we run an experiment
                        datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
  return tf.keras.callbacks.TensorBoard(logdir)
```


### [Preventing Overfitting](https://colab.research.google.com/drive/1OZLn22hAZkNY1hHaK-FePk8sLVnbTBwV#scrollTo=LrjvXtrpszqp)

[EarlyStopping Callback](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping)

```
# Create early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy",
                                                  patience=3)
```


### [Training Your Deep Neural Network](https://colab.research.google.com/drive/1OZLn22hAZkNY1hHaK-FePk8sLVnbTBwV#scrollTo=LrjvXtrpszqp)

Our first model is only going to train on 1000 images, to make sure everything is working.

```python
NUM_EPOCHS = 100 #@param {type:"slider", min:10, max:100, step:10}

# Build a function to train and return a trained model
def train_model():
  """
  Trains a given model and returns the trained version.
  """
  # Create a model
  model = create_model()

  # Create new TensorBoard session everytime we train a model
  tensorboard = create_tensorboard_callback()

  # Fit the model to the data passing it the callbacks we created
  model.fit(x=train_data,
            epochs=NUM_EPOCHS,
            validation_data=val_data,
            validation_freq=1,
            callbacks=[tensorboard, early_stopping])
  # Return the fitted model
  return model

# Fit the model to the data
# Train: 800 / 32 = 25
# Validate: 200 / 32 = 7
model = train_model()
```


### [Evaluating Performance With TensorBoard](https://colab.research.google.com/drive/1OZLn22hAZkNY1hHaK-FePk8sLVnbTBwV#scrollTo=LrjvXtrpszqp)

The TensorBoard magic function (%tensorboard) will access the logs directory we created earlier and visualize its contents.

```python
%tensorboard --logdir drive/My\ Drive/Dog\ Vision/logs
```


### [Make And Transform Predictions](https://colab.research.google.com/drive/1OZLn22hAZkNY1hHaK-FePk8sLVnbTBwV#scrollTo=LrjvXtrpszqp)

```python
# Make predictions on the validation data (not used to train on)
predictions = model.predict(val_data, verbose=1)

predictions.shape
len(y_val)
len(unique_breeds)

# First prediction
index = 42
print(predictions[index])
print(f"Max value (probability of prediction): {np.max(predictions[index])}")
print(f"Sum: {np.sum(predictions[index])}")
print(f"Max index: {np.argmax(predictions[index])}")
print(f"Predicted label: {unique_breeds[np.argmax(predictions[index])]}")

unique_breeds[113]
```

**[⬆ back to top](#table-of-contents)**

### [Transform Predictions To Text](https://colab.research.google.com/drive/1OZLn22hAZkNY1hHaK-FePk8sLVnbTBwV#scrollTo=LrjvXtrpszqp)

```python
# Turn prediction probabilities into their respective label (easier to understand)
def get_pred_label(prediction_probabilities):
  """
  Turns an array of prediction probabilities into a label.
  """
  return unique_breeds[np.argmax(prediction_probabilities)]

# Get a predicted label based on an array of prediction probabilities
pred_label = get_pred_label(predictions[81])

# Create a function to unbatch a batch dataset
def unbatchify(data):
  """
  Takes a batched dataset of (image, label) Tensors and reutrns separate arrays
  of images and labels.
  """
  images = []
  labels = []
  # Loop through unbatched data
  for image, label in data.unbatch().as_numpy_iterator():
    images.append(image)
    labels.append(unique_breeds[np.argmax(label)])
  return images, labels

# Unbatchify the validation data
val_images, val_labels = unbatchify(val_data)
val_images[0], val_labels[0]
```

**[⬆ back to top](#table-of-contents)**

### [Visualizing Model Predictions](https://colab.research.google.com/drive/1OZLn22hAZkNY1hHaK-FePk8sLVnbTBwV#scrollTo=LrjvXtrpszqp)

We'll create a function which:

- Takes an array of prediction probabilities, an array of truth labels and an array of images and an integer. ✅
- Convert the prediction probabilities to a predicted label. ✅
- Plot the predicted label, its predicted probability, the truth label and the target image on a single plot. ✅

```python
def plot_pred(prediction_probabilities, labels, images, n=1):
  """
  View the prediction, ground truth and image for sample n
  """
  pred_prob, true_label, image = prediction_probabilities[n], labels[n], images[n]

  # Get the pred label
  pred_label = get_pred_label(pred_prob)

  # Plot image & remove ticks
  plt.imshow(image)
  plt.xticks([])
  plt.yticks([])

  # Change the colour of the title depending on if the prediction is right or wrong
  if pred_label == true_label:
    color = "green"
  else:
    color = "red"

  # Change plot title to be predicted, probability of prediction and truth label
  plt.title("{} {:2.0f}% {}".format(pred_label,
                                    np.max(pred_prob)*100,
                                    true_label),
                                    color=color)

plot_pred(prediction_probabilities=predictions,
          labels=val_labels,
          images=val_images,
          n=0)
```

Now we've got one function to visualize our models top prediction, let's make another to view our models top 10 predictions.

```python
def plot_pred_conf(prediction_probabilities, labels, n=1):
  """
  Plus the top 10 highest prediction confidences along with the truth label for sample n.
  """
  pred_prob, true_label = prediction_probabilities[n], labels[n]

  # Get the predicted label
  pred_label = get_pred_label(pred_prob)

  # Find the top 10 prediction confidence indexes
  top_10_pred_indexes = pred_prob.argsort()[-10:][::-1]
  # Find the top 10 prediction confidence values
  top_10_pred_values = pred_prob[top_10_pred_indexes]
  # Find the top 10 prediction labels
  top_10_pred_labels = unique_breeds[top_10_pred_indexes]

  # Setup plot
  top_plot = plt.bar(np.arange(len(top_10_pred_labels)),
                     top_10_pred_values,
                     color="grey")
  plt.xticks(np.arange(len(top_10_pred_labels)),
             labels=top_10_pred_labels,
             rotation="vertical")

  # Change color of true label
  if np.isin(true_label, top_10_pred_labels):
    top_plot[np.argmax(top_10_pred_labels == true_label)].set_color("green")
  else:
    pass

plot_pred_conf(prediction_probabilities=predictions,
               labels=val_labels,
               n=9)
```

Now we've got some function to help us visualize our predictions and evaluate our modle, let's check out a few.

```python
# Let's check out a few predictions and their different values
i_multiplier = 20
num_rows = 3
num_cols = 2
num_images = num_rows*num_cols
plt.figure(figsize=(10*num_cols, 5*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_pred(prediction_probabilities=predictions,
            labels=val_labels,
            images=val_images,
            n=i+i_multiplier)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_pred_conf(prediction_probabilities=predictions,
                 labels=val_labels,
                 n=i+i_multiplier)
plt.tight_layout(h_pad=1.0)
plt.show()
```

**[⬆ back to top](#table-of-contents)**

### [Saving And Loading A Trained Model](https://colab.research.google.com/drive/1OZLn22hAZkNY1hHaK-FePk8sLVnbTBwV#scrollTo=LrjvXtrpszqp)

[Save and load models](https://www.tensorflow.org/tutorials/keras/save_and_load)

```python
# Create a function to save a model
def save_model(model, suffix=None):
  """
  Saves a given model in a models directory and appends a suffix (string).
  """
  # Create a model directory pathname with current time
  modeldir = os.path.join("drive/My Drive/Dog Vision/models",
                          datetime.datetime.now().strftime("%Y%m%d-%H%M%s"))
  model_path = modeldir + "-" + suffix + ".h5" # save format of model
  print(f"Saving model to: {model_path}...")
  model.save(model_path)
  return model_path

# Create a function to load a trained model
def load_model(model_path):
  """
  Loads a saved model from a specified path.
  """
  print(f"Loading saved model from: {model_path}")
  model = tf.keras.models.load_model(model_path,
                                     custom_objects={"KerasLayer":hub.KerasLayer})
  return model

# Save our model trained on 1000 images
save_model(model, suffix="1000-images-mobilenetv2-Adam")

# Load a trained model
loaded_1000_image_model = load_model('drive/My Drive/Dog Vision/models/20200604-22561591311380-1000-images-mobilenetv2-Adam.h5')

# Evaluate the pre-saved model
model.evaluate(val_data)

# Evaluate the loaded model
loaded_1000_image_model.evaluate(val_data)
```

**[⬆ back to top](#table-of-contents)**

### [Training Model On Full Dataset](https://colab.research.google.com/drive/1OZLn22hAZkNY1hHaK-FePk8sLVnbTBwV#scrollTo=LrjvXtrpszqp)

```python
len(X), len(y)

# Create a data batch with the full data set
full_data = create_data_batches(X, y)

# Create a model for full model
full_model = create_model()

# Create full model callbacks
full_model_tensorboard = create_tensorboard_callback()
# No validation set when training on all the data, so we can't monitor validation accuracy
full_model_early_stopping = tf.keras.callbacks.EarlyStopping(monitor="accuracy",
                                                             patience=3)

# Fit the full model to the full data
full_model.fit(x=full_data,
               epochs=NUM_EPOCHS,
               callbacks=[full_model_tensorboard, full_model_early_stopping])

save_model(full_model, suffix="full-image-set-mobilenetv2-Adam")

# Load in the full model
loaded_full_model = load_model('drive/My Drive/Dog Vision/models/20200205-07041580886291-full-image-set-mobilenetv2-Adam.h5')
```

**[⬆ back to top](#table-of-contents)**

### [Making Predictions On Test Images](https://colab.research.google.com/drive/1OZLn22hAZkNY1hHaK-FePk8sLVnbTBwV#scrollTo=LrjvXtrpszqp)

```python
# Load test image filenames
test_path = "drive/My Drive/Dog Vision/test/"
test_filenames = [test_path + fname for fname in os.listdir(test_path)]

# Create test data batch
test_data = create_data_batches(test_filenames, test_data=True)

# Make predictions on test data batch using the loaded full model
test_predictions = loaded_full_model.predict(test_data,
                                             verbose=1)

# Save predictions (NumPy array) to csv file (for access later)
np.savetxt("drive/My Drive/Dog Vision/preds_array.csv", test_predictions, delimiter=",")

# Load predictions (NumPy array) from csv file
test_predictions = np.loadtxt("drive/My Drive/Dog Vision/preds_array.csv", delimiter=",")

test_predictions.shape
```

**[⬆ back to top](#table-of-contents)**

### [Submitting Model to Kaggle](https://colab.research.google.com/drive/1OZLn22hAZkNY1hHaK-FePk8sLVnbTBwV#scrollTo=LrjvXtrpszqp)

```python
# Create a pandas DataFrame with empty columns
preds_df = pd.DataFrame(columns=["id"] + list(unique_breeds))
preds_df.head()

# Append test image ID's to predictions DataFrame
test_ids = [os.path.splitext(path)[0] for path in os.listdir(test_path)]
preds_df["id"] = test_ids
preds_df.head()

# Add the prediction probabilities to each dog breed column
preds_df[list(unique_breeds)] = test_predictions
preds_df.head()

# Save our predictions dataframe to CSV for submission to Kaggle
preds_df.to_csv("drive/My Drive/Dog Vision/full_model_predictions_submission_1_mobilenetV2.csv",
                index=False)
```

**[⬆ back to top](#table-of-contents)**

### [Making Predictions On Our Images](https://colab.research.google.com/drive/1OZLn22hAZkNY1hHaK-FePk8sLVnbTBwV#scrollTo=LrjvXtrpszqp)

```python
# Get custom image filepaths
custom_path = "drive/My Drive/Dog Vision/my-dog-photos/"
custom_image_paths = [custom_path + fname for fname in os.listdir(custom_path)]

# Turn custom images into batch datasets
custom_data = create_data_batches(custom_image_paths, test_data=True)
custom_data

# Make predictions on the custom data
custom_preds = loaded_full_model.predict(custom_data)

# Get custom image prediction labels
custom_pred_labels = [get_pred_label(custom_preds[i]) for i in range(len(custom_preds))]
custom_pred_labels

# Get custom images (our unbatchify() function won't work since there aren't labels... maybe we could fix this later)
custom_images = []
# Loop through unbatched data
for image in custom_data.unbatch().as_numpy_iterator():
  custom_images.append(image)

# Check custom image predictions
plt.figure(figsize=(10, 10))
for i, image in enumerate(custom_images):
  plt.subplot(1, 3, i+1)
  plt.xticks([])
  plt.yticks([])
  plt.title(custom_pred_labels[i])
  plt.imshow(image)
```

**[⬆ back to top](#table-of-contents)**

## **Section 15: Storytelling + Communication: How To Present Your Work**

### Communicating Your Work

- [How to Think About Communicating and Sharing Your Work](https://www.mrdbourke.com/how-to-think-about-communicating-and-sharing-your-work/)

Now you’ve got skills, what do you do next?
The most important question you can ask

- “Who’s it for?”
- What questions will they have?
- What concerns can you address before they arise?
- Heard but not understood
- Heard and (potentially) understood

“Who’s it for?”

- People on your team: Boss, Project manager and Teammates
- People outside your team: Clients, Customers and Fans

**[⬆ back to top](#table-of-contents)**

### Communicating With Managers

“Who’s it for?”
“What do they need to know?”

- How the project is going
- What’s in the way
- What you’ve done
- What you’re doing next
- Why you’re doing something next
- Who else could help
- What’s not needed Where you’re stuck
- What’s not clear What questions do have
- Are you still working towards the right thing
- Is there any feedback or advice

Example: The Project Manager, Boss, Senior, Lead

- What’s holding you back?

**[⬆ back to top](#table-of-contents)**

### Communicating With Co-Workers

The People You’re Working With, Sitting Next to, in the Group Chat

Break it down

- 6-month project
- 4-week month
- 5-day week

What did you work on today?

- What I worked on today (1-3 points on what you did):
  - What’s working?
  - What’s not working?
  - What could be improved?
- What I’m working on next:

  - What’s your next course of action? (based on the above)
  - Why?
  - What’s holding you back?

- Relate back to overall project goal
- Take note of overlaps

**[⬆ back to top](#table-of-contents)**

### Weekend Project Principle

Start the job before you have it
The weekend project principle

- What you work on in your own time
- Work on your own projects to build specific knowledge
- Compound knowledge that you learn courses into skill which can't be taught in courses

How?

- Documented on your blog
- 6-week project

**[⬆ back to top](#table-of-contents)**

### Communicating With Outside World

People outside your team: Clients, Customers and Fans

- Obvious to you, amazing to others

**[⬆ back to top](#table-of-contents)**

### Storytelling

What story are you trying to tell?
Always ask, Who’s it for?

- “Who’s it for?”
- “What do they need to know?” = Specific = Courage
  Write it down
- What did you work on today?
- What are you working this week?
  Progress, not perfection
- “Here’s what I’ve done.”

**[⬆ back to top](#table-of-contents)**

Communicating and sharing your work: Further reading

- [How to Think About Communicating and Sharing Your Work](https://www.mrdbourke.com/how-to-think-about-communicating-and-sharing-your-work/)
- [The Basecamp Guide to Internal Communication](https://basecamp.com/guides/how-we-communicate)
- [Your own hosted blog](https://www.fast.ai/2020/01/16/fast_template/#you-should-blog)
- [How to Start Your Own Machine Learning Projects](https://www.mrdbourke.com/how-to-start-your-own-machine-learning-projects/)
- [Why you (yes, you) should blog](https://medium.com/@racheltho/why-you-yes-you-should-blog-7d2544ac1045)
- [devblog](https://hashnode.com/devblog)
- [Devblog: How to Launch Your Own Developer Blog on Your Own Domain in Minutes](https://www.freecodecamp.org/news/devblog-launch-your-developer-blog-own-domain/)

**[⬆ back to top](#table-of-contents)**
 -->
