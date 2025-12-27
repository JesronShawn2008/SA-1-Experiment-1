# SA-1-Experiment-1

## AIM
Create a scatter plot between cylinder vs Co2Emission (green color)

## Algorithim
### Step - 1
Import required libraries and load the dataset using read_csv(), then display initial records.

### Step - 2
Select relevant columns (CYLINDERS, ENGINESIZE, FUELCONSUMPTION_COMB, CO2EMISSIONS) for analysis.

### Step - 3
Preview the selected data to confirm columns.

### Step - 4
Create a scatter plot by taking CYLINDERS as the x-axis and CO2EMISSIONS as the y-axis.

### Step - 5
Display the scatter plot with appropriate labels and title to visualize the relationship.

## PROGRAM
```
# ==========================
# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
# --------------------------
# Step 1: Load dataset
# --------------------------
# Replace 'FuelConsumption.csv' with your dataset file name
df = pd.read_csv('FuelConsumption.csv')
# Display first few rows
print("Dataset Preview:")
print(df.head())
# --------------------------
# Step 2: Select useful columns
# --------------------------
data = df[['CYLINDERS', 'ENGINESIZE', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
print("\nSelected Columns:")
print(data.head())
# --------------------------
# Q1: Scatter plot (CYLINDERS vs CO2EMISSIONS)
# --------------------------
plt.figure(figsize=(7,5))
plt.scatter(data['CYLINDERS'], data['CO2EMISSIONS'], color='green')
plt.title('Cylinders vs CO2 Emission')
plt.xlabel('Number of Cylinders')
plt.ylabel('CO2 Emission')
plt.show()
```
## Output
 Dataset Preview:
   MODELYEAR   MAKE       MODEL VEHICLECLASS  ENGINESIZE  CYLINDERS  \
0       2014  ACURA         ILX      COMPACT         2.0          4   
1       2014  ACURA         ILX      COMPACT         2.4          4   
2       2014  ACURA  ILX HYBRID      COMPACT         1.5          4   
3       2014  ACURA     MDX 4WD  SUV - SMALL         3.5          6   
4       2014  ACURA     RDX AWD  SUV - SMALL         3.5          6   

  TRANSMISSION FUELTYPE  FUELCONSUMPTION_CITY  FUELCONSUMPTION_HWY  \
0          AS5        Z                   9.9                  6.7   
1           M6        Z                  11.2                  7.7   
2          AV7        Z                   6.0                  5.8   
3          AS6        Z                  12.7                  9.1   
4          AS6        Z                  12.1                  8.7   

   FUELCONSUMPTION_COMB  FUELCONSUMPTION_COMB_MPG  CO2EMISSIONS  
0                   8.5                        33           196  
1                   9.6                        29           221  
2                   5.9                        48           136  
3                  11.1                        25           255  
4                  10.6                        27           244  

Selected Columns:
   CYLINDERS  ENGINESIZE  FUELCONSUMPTION_COMB  CO2EMISSIONS
0          4         2.0                   8.5           196
1          4         2.4                   9.6           221
2          4         1.5                   5.9           136
3          6         3.5                  11.1           255
4          6         3.5                  10.6           244

![alt text](image.png)

## Result
Thus the python program was able to create a scatter plot between cylinder vs Co2Emission (green color)