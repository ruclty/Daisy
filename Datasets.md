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
      <td><a href="http://archive.ics.uci.edu/ml/datasets/HTRU2.">Download</a> | 
      <a href="#htru2">Detail</a></td>
    </tr>
    <tr>
      <td>Digits</td>
      <td>Computer</td>
      <td>10,992/td>
      <td>16</td>
      <td>0</td>
      <td>10</td>
      <td>balanced</td>
      <td><a href="http://archive.ics.uci.edu/ml/datasets/HTRU2.">Download</a> | 
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
      <td><a href="http://archive.ics.uci.edu/ml/datasets/HTRU2.">Download</a> | 
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
      <td><a href="http://archive.ics.uci.edu/ml/datasets/HTRU2.">Download</a> | 
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
      <td><a href="http://archive.ics.uci.edu/ml/datasets/HTRU2.">Download</a> | 
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
      <td><a href="http://archive.ics.uci.edu/ml/datasets/HTRU2.">Download</a> | 
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
      <td><a href="http://archive.ics.uci.edu/ml/datasets/HTRU2.">Download</a> | 
      <a href="#census">Detail</a></td>
    </tr>
  </tbody>
</table>

**Notes:** 
- For covertype datasets, we randomly sample 116,204 records in our experiments to  

# Dataset Details

## Low-Dimensional (Number of Attributes <= 20)

### HTRU2

### Digits

### Adult

### CovType

## High-Dimensional (Number of Attributes > 20)

### SAT

### Anuran

### Census

