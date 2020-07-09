# Datasets for Daisy 

All the public datasets we used in [Relational Data Synthesis using Generative Adversarial Networks: A Design Space Exploration] can be downloaded from this page.

We give a table to introduce the summarizes for each dataset, here is a description of columns:
- \# Rec.: Number of records we used.
- \# Cat.: Number of categorical attributes.
- \# Num.: Number of numerical attributes.
- \# Lab.: Number of unique labels.

You can download the dataset by "Download" links. The "Detail" links point to the detail description of dataset.

<table>
  <thead>
    <tr>
      <th>Type</th>
      <th>Dataset</th>
      <th>Domain</th>
      <th># Rec.</th>
      <th># Num.</th>
      <th># Cat.</th>
      <th># Lab.</th>
      <th>Skewness</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan=4>Low-Dimensional</td>
      <td>HTRU2</td>
      <td>Physical</td>
      <td>17,898</td>
      <td>8</td>
      <td>0</td>
      <td>2</td>
      <td>skew</td>
      <td><a href="http://archive.ics.uci.edu/ml/datasets/HTRU2">Download</a> | 
      <a href="#htru2">Detail</a></td>
    </tr>
    <tr>
      <td>Digits</td>
      <td>Computer</td>
      <td>10,992</td>
      <td>16</td>
      <td>0</td>
      <td>10</td>
      <td>balanced</td>
      <td><a href="https://archive.ics.uci.edu/ml/datasets/Pen-Based+Recognition+of+Handwritten+Digits">Download</a> | 
      <a href="#digits">Detail</a></td>
    </tr>
    <tr>
      <td>Adult</td>
      <td>Social</td>
      <td>41,292</td>
      <td>6</td>
      <td>8</td>
      <td>2</td>
      <td>skew</td>
      <td><a href="https://archive.ics.uci.edu/ml/datasets/Adult">Download</a> | 
      <a href="#adult">Detail</a></td>
    </tr>
    <tr>
      <td>CovType</td>
      <td>Life</td>
      <td>116,204</td>
      <td>10</td>
      <td>2</td>
      <td>7</td>
      <td>skew</td>
      <td><a href="http://archive.ics.uci.edu/ml/datasets/covertype">Download</a> | 
      <a href="#covtype">Detail</a></td>
    </tr>
    <tr>
      <td rowspan=3>High-Dimensional</td>
      <td>SAT</td>
      <td>Physical</td>
      <td>6,435</td>
      <td>36</td>
      <td>0</td>
      <td>6</td>
      <td>balanced</td>
      <td><a href="https://archive.ics.uci.edu/ml/datasets/Statlog+%28Landsat+Satellite%29">Download</a> | 
      <a href="#sat">Detail</a></td>
    </tr>
    <tr>
      <td>Anuran</td>
      <td>Life
      <td>7,195</td>
      <td>22</td>
      <td>0</td>
      <td>10</td>
      <td>skew</td>
      <td><a href="http://archive.ics.uci.edu/ml/datasets/Anuran+Calls+%28MFCCs%29">Download</a> | 
      <a href="#anuran">Detail</a></td>
    </tr>
    <tr>
      <td>Census</td>
      <td>Social</td>
      <td>142,522</td>
      <td>9</td>
      <td>30</td>
      <td>2</td>
      <td>skew</td>
      <td><a href="http://archive.ics.uci.edu/ml/datasets/Census-Income+(KDD)">Download</a> | 
      <a href="#census">Detail</a></td>
    </tr>
  </tbody>
</table>

**Notes:** 
- We remove the records which contain null values from Adult dataset, and the number of remaining records is 41,292.
- For the efficiency of evaluation, we randomly sample 116,204 records and 142,522 records from Covtype and Census datasets respectively in our experiments  

# Dataset Details

## Low-Dimensional (Number of Attributes <= 20)

### HTRU2
HTRU2 dataset is a physical dataset that contains 17, 898 pulsar candidates collected during the High Time Resolution Universe Survey. This dataset has 8 numerical attributes, which are statistics obtained from the integrated pulse profile and the DM-SNR curve, and a binary label (i.e., pulsar and non-pulsar). The label distribution is balanced.

### Digits
Digits dataset contains 10, 992 pen-based handwritten digits. Each digit has 16 numerical attributes collected by a pressure sensitive tablet and processed by normalization methods, and a label indicating the gold-standard number from 0 0 9. The label distribution is balanced.

### Adult
Adult dataset contains personal information of 41, 292 individuals extracted from the 1994 US census with 8 categorical attributes, such as Workclass and Education and 6 numerical attributes, such as Age and Hours-per-Week. We use attribute Income as label and predict whether a person has income larger than 50K per year (positive) or not
(negative), where the label distribution is skew, i.e., the ratio between positive and negative labels is 0.34.

### CovType
CovType dataset contains the information of 116, 204 forest records obtained from US Geological Survey (USGS) and US Forest Service (USFS) data. It includes 2 categorical attributes, Wild-area and Soil-type, and 10 numerical attributes, such as Elavation and Slope. We use attribute Cover-type with 7 distinct values as label and predict forest cover-type from other cartographic variables. The label distribution is also very skew, e.g., there are 46% records with label 2 while only 6% records with label 3.

## High-Dimensional (Number of Attributes > 20)

### SAT
SAT dataset consists of the multi-spectral values of pixels in 3x3 neighborhoods in a satellite image. It has 36 numerical attributes that represent the values in the four spectral bands of the 9 pixels in a neighborhood, and uses a label with 7 unique values indicating the type of the central pixel. The label distribution is balanced in the dataset.

### Anuran
Anuran dataset a dataset from the life domain for anuran species recognition through their calls. It has 22 numerical attributes, which are derived from the audio records belonging to specimens (individual frogs), and associates a label with 10 unique values that indicates the corresponding species. The label distribution is very skew: there are 3, 478 records with label 2 and 68 with label 9.

### Census
Census dataset contains weighted census data extracted from the 1994 and 1995 Current Population Surveys. We use demographic and employment variables, i.e., 9 numerical and 30 categorical attributes, as features, and totalperson-income as label. We remove the records containing null values and then obtain 142, 522 records with very skew label distribution, i.e., 5% records with income larger than 50K vs. 95% with income smaller than 50K.
