import shap
X, y = shap.datasets.adult()
X_display, y_display = shap.datasets.adult(display=True)
feature_names = list(X.columns)
feature_names

```python
display(X.describe())
hist = X.hist(bins=30, sharey=True, figsize=(20, 10))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Workclass</th>
      <th>Education-Num</th>
      <th>Marital Status</th>
      <th>Occupation</th>
      <th>Relationship</th>
      <th>Race</th>
      <th>Sex</th>
      <th>Capital Gain</th>
      <th>Capital Loss</th>
      <th>Hours per week</th>
      <th>Country</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>32561.000000</td>
      <td>32561.000000</td>
      <td>32561.000000</td>
      <td>32561.000000</td>
      <td>32561.000000</td>
      <td>32561.000000</td>
      <td>32561.000000</td>
      <td>32561.000000</td>
      <td>32561.000000</td>
      <td>32561.000000</td>
      <td>32561.000000</td>
      <td>32561.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>38.581646</td>
      <td>3.868892</td>
      <td>10.080679</td>
      <td>2.611836</td>
      <td>6.572740</td>
      <td>2.494518</td>
      <td>3.665858</td>
      <td>0.669205</td>
      <td>1077.649170</td>
      <td>87.303833</td>
      <td>40.437454</td>
      <td>36.718866</td>
    </tr>
    <tr>
      <th>std</th>
      <td>13.640442</td>
      <td>1.455960</td>
      <td>2.572562</td>
      <td>1.506222</td>
      <td>4.228857</td>
      <td>1.758232</td>
      <td>0.848806</td>
      <td>0.470506</td>
      <td>7385.911621</td>
      <td>403.014771</td>
      <td>12.347933</td>
      <td>7.823782</td>
    </tr>
    <tr>
      <th>min</th>
      <td>17.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>28.000000</td>
      <td>4.000000</td>
      <td>9.000000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>4.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>40.000000</td>
      <td>39.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>37.000000</td>
      <td>4.000000</td>
      <td>10.000000</td>
      <td>2.000000</td>
      <td>7.000000</td>
      <td>3.000000</td>
      <td>4.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>40.000000</td>
      <td>39.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>48.000000</td>
      <td>4.000000</td>
      <td>12.000000</td>
      <td>4.000000</td>
      <td>10.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>45.000000</td>
      <td>39.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>90.000000</td>
      <td>8.000000</td>
      <td>16.000000</td>
      <td>6.000000</td>
      <td>14.000000</td>
      <td>5.000000</td>
      <td>4.000000</td>
      <td>1.000000</td>
      <td>99999.000000</td>
      <td>4356.000000</td>
      <td>99.000000</td>
      <td>41.000000</td>
    </tr>
  </tbody>
</table>
</div>



![png](output_1_1.png)



```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train_display = X_display.loc[X_train.index]
```


```python
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)
X_train_display = X_display.loc[X_train.index]
X_val_display = X_display.loc[X_val.index]
```


```python
import pandas as pd
train = pd.concat([pd.Series(y_train, index=X_train.index,
                             name='Income>50K', dtype=int), X_train], axis=1)
validation = pd.concat([pd.Series(y_val, index=X_val.index,
                            name='Income>50K', dtype=int), X_val], axis=1)
test = pd.concat([pd.Series(y_test, index=X_test.index,
                            name='Income>50K', dtype=int), X_test], axis=1)
```


```python
train
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Income&gt;50K</th>
      <th>Age</th>
      <th>Workclass</th>
      <th>Education-Num</th>
      <th>Marital Status</th>
      <th>Occupation</th>
      <th>Relationship</th>
      <th>Race</th>
      <th>Sex</th>
      <th>Capital Gain</th>
      <th>Capital Loss</th>
      <th>Hours per week</th>
      <th>Country</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10911</th>
      <td>1</td>
      <td>47.0</td>
      <td>4</td>
      <td>9.0</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>4</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>40.0</td>
      <td>39</td>
    </tr>
    <tr>
      <th>17852</th>
      <td>0</td>
      <td>31.0</td>
      <td>4</td>
      <td>13.0</td>
      <td>2</td>
      <td>7</td>
      <td>4</td>
      <td>3</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>36.0</td>
      <td>26</td>
    </tr>
    <tr>
      <th>29165</th>
      <td>1</td>
      <td>32.0</td>
      <td>4</td>
      <td>10.0</td>
      <td>2</td>
      <td>13</td>
      <td>5</td>
      <td>4</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>32.0</td>
      <td>39</td>
    </tr>
    <tr>
      <th>30287</th>
      <td>0</td>
      <td>58.0</td>
      <td>4</td>
      <td>9.0</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>2</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>40.0</td>
      <td>39</td>
    </tr>
    <tr>
      <th>24019</th>
      <td>0</td>
      <td>17.0</td>
      <td>4</td>
      <td>6.0</td>
      <td>4</td>
      <td>6</td>
      <td>3</td>
      <td>4</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>20.0</td>
      <td>39</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>21168</th>
      <td>0</td>
      <td>43.0</td>
      <td>4</td>
      <td>8.0</td>
      <td>2</td>
      <td>14</td>
      <td>4</td>
      <td>4</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>40.0</td>
      <td>39</td>
    </tr>
    <tr>
      <th>6452</th>
      <td>0</td>
      <td>26.0</td>
      <td>4</td>
      <td>9.0</td>
      <td>4</td>
      <td>7</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>52.0</td>
      <td>39</td>
    </tr>
    <tr>
      <th>31352</th>
      <td>0</td>
      <td>32.0</td>
      <td>7</td>
      <td>14.0</td>
      <td>2</td>
      <td>10</td>
      <td>4</td>
      <td>4</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>50.0</td>
      <td>39</td>
    </tr>
    <tr>
      <th>6575</th>
      <td>0</td>
      <td>45.0</td>
      <td>4</td>
      <td>9.0</td>
      <td>4</td>
      <td>6</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>40.0</td>
      <td>39</td>
    </tr>
    <tr>
      <th>23608</th>
      <td>0</td>
      <td>23.0</td>
      <td>4</td>
      <td>9.0</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>40.0</td>
      <td>39</td>
    </tr>
  </tbody>
</table>
<p>19536 rows × 13 columns</p>
</div>




```python
validation
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Income&gt;50K</th>
      <th>Age</th>
      <th>Workclass</th>
      <th>Education-Num</th>
      <th>Marital Status</th>
      <th>Occupation</th>
      <th>Relationship</th>
      <th>Race</th>
      <th>Sex</th>
      <th>Capital Gain</th>
      <th>Capital Loss</th>
      <th>Hours per week</th>
      <th>Country</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>16530</th>
      <td>0</td>
      <td>25.0</td>
      <td>4</td>
      <td>4.0</td>
      <td>2</td>
      <td>6</td>
      <td>4</td>
      <td>4</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>40.0</td>
      <td>26</td>
    </tr>
    <tr>
      <th>26723</th>
      <td>0</td>
      <td>41.0</td>
      <td>6</td>
      <td>9.0</td>
      <td>2</td>
      <td>5</td>
      <td>5</td>
      <td>4</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>40.0</td>
      <td>39</td>
    </tr>
    <tr>
      <th>3338</th>
      <td>0</td>
      <td>79.0</td>
      <td>0</td>
      <td>9.0</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>30.0</td>
      <td>39</td>
    </tr>
    <tr>
      <th>19367</th>
      <td>1</td>
      <td>43.0</td>
      <td>2</td>
      <td>15.0</td>
      <td>2</td>
      <td>10</td>
      <td>4</td>
      <td>4</td>
      <td>1</td>
      <td>15024.0</td>
      <td>0.0</td>
      <td>45.0</td>
      <td>39</td>
    </tr>
    <tr>
      <th>30274</th>
      <td>0</td>
      <td>51.0</td>
      <td>5</td>
      <td>9.0</td>
      <td>4</td>
      <td>12</td>
      <td>2</td>
      <td>4</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>40.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1604</th>
      <td>0</td>
      <td>46.0</td>
      <td>7</td>
      <td>9.0</td>
      <td>2</td>
      <td>13</td>
      <td>4</td>
      <td>4</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>40.0</td>
      <td>39</td>
    </tr>
    <tr>
      <th>5937</th>
      <td>1</td>
      <td>71.0</td>
      <td>4</td>
      <td>10.0</td>
      <td>6</td>
      <td>12</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>35.0</td>
      <td>39</td>
    </tr>
    <tr>
      <th>11034</th>
      <td>0</td>
      <td>36.0</td>
      <td>4</td>
      <td>9.0</td>
      <td>5</td>
      <td>14</td>
      <td>2</td>
      <td>4</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>60.0</td>
      <td>26</td>
    </tr>
    <tr>
      <th>2819</th>
      <td>0</td>
      <td>31.0</td>
      <td>4</td>
      <td>9.0</td>
      <td>4</td>
      <td>8</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>40.0</td>
      <td>39</td>
    </tr>
    <tr>
      <th>14152</th>
      <td>1</td>
      <td>37.0</td>
      <td>4</td>
      <td>10.0</td>
      <td>2</td>
      <td>12</td>
      <td>4</td>
      <td>4</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>50.0</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
<p>6512 rows × 13 columns</p>
</div>




```python
test
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Income&gt;50K</th>
      <th>Age</th>
      <th>Workclass</th>
      <th>Education-Num</th>
      <th>Marital Status</th>
      <th>Occupation</th>
      <th>Relationship</th>
      <th>Race</th>
      <th>Sex</th>
      <th>Capital Gain</th>
      <th>Capital Loss</th>
      <th>Hours per week</th>
      <th>Country</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9646</th>
      <td>0</td>
      <td>62.0</td>
      <td>6</td>
      <td>4.0</td>
      <td>6</td>
      <td>8</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>66.0</td>
      <td>39</td>
    </tr>
    <tr>
      <th>709</th>
      <td>0</td>
      <td>18.0</td>
      <td>4</td>
      <td>7.0</td>
      <td>4</td>
      <td>8</td>
      <td>2</td>
      <td>4</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>25.0</td>
      <td>39</td>
    </tr>
    <tr>
      <th>7385</th>
      <td>1</td>
      <td>25.0</td>
      <td>4</td>
      <td>13.0</td>
      <td>4</td>
      <td>5</td>
      <td>3</td>
      <td>4</td>
      <td>1</td>
      <td>27828.0</td>
      <td>0.0</td>
      <td>50.0</td>
      <td>39</td>
    </tr>
    <tr>
      <th>16671</th>
      <td>0</td>
      <td>33.0</td>
      <td>4</td>
      <td>9.0</td>
      <td>2</td>
      <td>10</td>
      <td>4</td>
      <td>4</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>40.0</td>
      <td>39</td>
    </tr>
    <tr>
      <th>21932</th>
      <td>0</td>
      <td>36.0</td>
      <td>4</td>
      <td>7.0</td>
      <td>4</td>
      <td>7</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>40.0</td>
      <td>39</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>5889</th>
      <td>1</td>
      <td>39.0</td>
      <td>4</td>
      <td>13.0</td>
      <td>2</td>
      <td>10</td>
      <td>5</td>
      <td>4</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>20.0</td>
      <td>39</td>
    </tr>
    <tr>
      <th>25723</th>
      <td>0</td>
      <td>17.0</td>
      <td>4</td>
      <td>6.0</td>
      <td>4</td>
      <td>12</td>
      <td>3</td>
      <td>4</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>20.0</td>
      <td>39</td>
    </tr>
    <tr>
      <th>29514</th>
      <td>0</td>
      <td>35.0</td>
      <td>4</td>
      <td>9.0</td>
      <td>4</td>
      <td>14</td>
      <td>3</td>
      <td>4</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>40.0</td>
      <td>39</td>
    </tr>
    <tr>
      <th>1600</th>
      <td>0</td>
      <td>30.0</td>
      <td>4</td>
      <td>7.0</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>4</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>45.0</td>
      <td>39</td>
    </tr>
    <tr>
      <th>639</th>
      <td>1</td>
      <td>52.0</td>
      <td>6</td>
      <td>16.0</td>
      <td>2</td>
      <td>10</td>
      <td>4</td>
      <td>4</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>60.0</td>
      <td>39</td>
    </tr>
  </tbody>
</table>
<p>6513 rows × 13 columns</p>
</div>




```python
# Use 'csv' format to store the data
# The first column is expected to be the output column
train.to_csv('train.csv', index=False, header=False)
validation.to_csv('validation.csv', index=False, header=False)
```


```python
import sagemaker, boto3, os
bucket = sagemaker.Session().default_bucket()
prefix = "demo-sagemaker-xgboost-adult-income-prediction"

boto3.Session().resource('s3').Bucket(bucket).Object(
    os.path.join(prefix, 'data/train.csv')).upload_file('train.csv')
boto3.Session().resource('s3').Bucket(bucket).Object(
    os.path.join(prefix, 'data/validation.csv')).upload_file('validation.csv')
```


```python
! aws s3 ls {bucket}/{prefix}/data --recursive
```

    2021-03-14 02:33:12     786285 demo-sagemaker-xgboost-adult-income-prediction/data/train.csv
    2021-03-14 02:33:12     262122 demo-sagemaker-xgboost-adult-income-prediction/data/validation.csv



```python
import sagemaker

region = sagemaker.Session().boto_region_name
print("AWS Region: {}".format(region))

role = sagemaker.get_execution_role()
print("RoleArn: {}".format(role))
```

    AWS Region: us-east-2
    RoleArn: arn:aws:iam::411548220055:role/service-role/AmazonSageMaker-ExecutionRole-20210313T171387



```python
from sagemaker.debugger import Rule, rule_configs
from sagemaker.session import TrainingInput

s3_output_location='s3://{}/{}/{}'.format(bucket, prefix, 'xgboost_model')

container=sagemaker.image_uris.retrieve("xgboost", region, "1.2-1")
print(container)

xgb_model=sagemaker.estimator.Estimator(
    image_uri=container,
    role=role,
    instance_count=1,
    instance_type='ml.m4.xlarge',
    volume_size=5,
    output_path=s3_output_location,
    sagemaker_session=sagemaker.Session(),
    rules=[Rule.sagemaker(rule_configs.create_xgboost_report())]
)
```

    257758044811.dkr.ecr.us-east-2.amazonaws.com/sagemaker-xgboost:1.2-1



```python
xgb_model.set_hyperparameters(
    max_depth = 5,
    eta = 0.2,
    gamma = 4,
    min_child_weight = 6,
    subsample = 0.7,
    objective = "binary:logistic",
    num_round = 1000
)
```


```python
from sagemaker.session import TrainingInput

train_input = TrainingInput(
    "s3://{}/{}/{}".format(bucket, prefix, "data/train.csv"), content_type="csv"
)
validation_input = TrainingInput(
    "s3://{}/{}/{}".format(bucket, prefix, "data/validation.csv"), content_type="csv"
)
```


```python
xgb_model.fit({"train": train_input, "validation": validation_input}, wait=True)
```

    2021-03-14 02:37:52 Starting - Starting the training job...
    2021-03-14 02:37:54 Starting - Launching requested ML instancesCreateXgboostReport: InProgress
    ProfilerReport-1615689472: InProgress
    ......
    2021-03-14 02:39:07 Starting - Preparing the instances for training......
    2021-03-14 02:40:11 Downloading - Downloading input data...
    2021-03-14 02:40:51 Training - Downloading the training image..[34m[2021-03-14 02:40:59.792 ip-10-0-208-62.us-east-2.compute.internal:1 INFO utils.py:27] RULE_JOB_STOP_SIGNAL_FILENAME: None[0m
    [34mINFO:sagemaker-containers:Imported framework sagemaker_xgboost_container.training[0m
    [34mINFO:sagemaker-containers:Failed to parse hyperparameter objective value binary:logistic to Json.[0m
    [34mReturning the value itself[0m
    [34mINFO:sagemaker-containers:No GPUs detected (normal if no gpus installed)[0m
    [34mINFO:sagemaker_xgboost_container.training:Running XGBoost Sagemaker in algorithm mode[0m
    [34mINFO:root:Determined delimiter of CSV input is ','[0m
    [34mINFO:root:Determined delimiter of CSV input is ','[0m
    [34mINFO:root:Determined delimiter of CSV input is ','[0m
    [34mINFO:root:Determined delimiter of CSV input is ','[0m
    [34mINFO:root:Single node training.[0m
    [34m[2021-03-14 02:40:59.929 ip-10-0-208-62.us-east-2.compute.internal:1 INFO json_config.py:91] Creating hook from json_config at /opt/ml/input/config/debughookconfig.json.[0m
    [34m[2021-03-14 02:40:59.930 ip-10-0-208-62.us-east-2.compute.internal:1 INFO hook.py:199] tensorboard_dir has not been set for the hook. SMDebug will not be exporting tensorboard summaries.[0m
    [34m[2021-03-14 02:40:59.930 ip-10-0-208-62.us-east-2.compute.internal:1 INFO profiler_config_parser.py:102] User has disabled profiler.[0m
    [34m[2021-03-14 02:40:59.932 ip-10-0-208-62.us-east-2.compute.internal:1 INFO hook.py:253] Saving to /opt/ml/output/tensors[0m
    [34m[2021-03-14 02:40:59.932 ip-10-0-208-62.us-east-2.compute.internal:1 INFO state_store.py:67] The checkpoint config file /opt/ml/input/config/checkpointconfig.json does not exist.[0m
    [34mINFO:root:Debug hook created from config[0m
    [34mINFO:root:Train matrix has 19536 rows[0m
    [34mINFO:root:Validation matrix has 6512 rows[0m
    [34m[0]#011train-error:0.14588#011validation-error:0.15525[0m
    [34m[2021-03-14 02:40:59.966 ip-10-0-208-62.us-east-2.compute.internal:1 INFO hook.py:413] Monitoring the collections: predictions, metrics, labels, feature_importance, hyperparameters[0m
    [34m[2021-03-14 02:40:59.969 ip-10-0-208-62.us-east-2.compute.internal:1 INFO hook.py:476] Hook is writing from the hook with pid: 1
    [0m
    [34m[1]#011train-error:0.14757#011validation-error:0.15587[0m
    [34m[2]#011train-error:0.14430#011validation-error:0.15326[0m
    [34m[3]#011train-error:0.14317#011validation-error:0.15141[0m
    [34m[4]#011train-error:0.14343#011validation-error:0.15203[0m
    [34m[5]#011train-error:0.14210#011validation-error:0.15157[0m
    [34m[6]#011train-error:0.14276#011validation-error:0.15233[0m
    [34m[7]#011train-error:0.14210#011validation-error:0.15249[0m
    [34m[8]#011train-error:0.14169#011validation-error:0.15157[0m
    [34m[9]#011train-error:0.14153#011validation-error:0.15141[0m
    [34m[10]#011train-error:0.14102#011validation-error:0.15157[0m
    [34m[11]#011train-error:0.14056#011validation-error:0.15126[0m
    [34m[12]#011train-error:0.13790#011validation-error:0.15003[0m
    [34m[13]#011train-error:0.13734#011validation-error:0.14926[0m
    [34m[14]#011train-error:0.13739#011validation-error:0.14865[0m
    [34m[15]#011train-error:0.13631#011validation-error:0.14788[0m
    [34m[16]#011train-error:0.13426#011validation-error:0.14558[0m
    [34m[17]#011train-error:0.13524#011validation-error:0.14773[0m
    [34m[18]#011train-error:0.13411#011validation-error:0.14573[0m
    [34m[19]#011train-error:0.13165#011validation-error:0.14327[0m
    [34m[20]#011train-error:0.13206#011validation-error:0.14297[0m
    [34m[21]#011train-error:0.13114#011validation-error:0.14281[0m
    [34m[22]#011train-error:0.13073#011validation-error:0.14297[0m
    [34m[23]#011train-error:0.13089#011validation-error:0.14205[0m
    [34m[24]#011train-error:0.13043#011validation-error:0.14251[0m
    [34m[25]#011train-error:0.13002#011validation-error:0.14251[0m
    [34m[26]#011train-error:0.12945#011validation-error:0.14158[0m
    [34m[27]#011train-error:0.12981#011validation-error:0.14143[0m
    [34m[28]#011train-error:0.12863#011validation-error:0.14097[0m
    [34m[29]#011train-error:0.12858#011validation-error:0.14036[0m
    [34m[30]#011train-error:0.12782#011validation-error:0.13928[0m
    [34m[31]#011train-error:0.12735#011validation-error:0.13821[0m
    [34m[32]#011train-error:0.12674#011validation-error:0.13682[0m
    [34m[33]#011train-error:0.12643#011validation-error:0.13744[0m
    [34m[34]#011train-error:0.12643#011validation-error:0.13728[0m
    [34m[35]#011train-error:0.12561#011validation-error:0.13728[0m
    [34m[36]#011train-error:0.12556#011validation-error:0.13698[0m
    [34m[37]#011train-error:0.12526#011validation-error:0.13667[0m
    [34m[38]#011train-error:0.12490#011validation-error:0.13652[0m
    [34m[39]#011train-error:0.12449#011validation-error:0.13452[0m
    [34m[40]#011train-error:0.12413#011validation-error:0.13467[0m
    [34m[41]#011train-error:0.12336#011validation-error:0.13406[0m
    [34m[42]#011train-error:0.12352#011validation-error:0.13406[0m
    [34m[43]#011train-error:0.12341#011validation-error:0.13452[0m
    [34m[44]#011train-error:0.12305#011validation-error:0.13437[0m
    [34m[45]#011train-error:0.12300#011validation-error:0.13391[0m
    [34m[46]#011train-error:0.12270#011validation-error:0.13360[0m
    [34m[47]#011train-error:0.12265#011validation-error:0.13360[0m
    [34m[48]#011train-error:0.12249#011validation-error:0.13298[0m
    [34m[49]#011train-error:0.12229#011validation-error:0.13360[0m
    [34m[50]#011train-error:0.12224#011validation-error:0.13375[0m
    [34m[51]#011train-error:0.12270#011validation-error:0.13452[0m
    [34m[52]#011train-error:0.12203#011validation-error:0.13375[0m
    [34m[53]#011train-error:0.12152#011validation-error:0.13421[0m
    [34m[54]#011train-error:0.12131#011validation-error:0.13329[0m
    [34m[55]#011train-error:0.12137#011validation-error:0.13360[0m
    [34m[56]#011train-error:0.12121#011validation-error:0.13360[0m
    [34m[57]#011train-error:0.12096#011validation-error:0.13268[0m
    [34m[58]#011train-error:0.12085#011validation-error:0.13222[0m
    [34m[59]#011train-error:0.12024#011validation-error:0.13145[0m
    [34m[60]#011train-error:0.12044#011validation-error:0.13130[0m
    [34m[61]#011train-error:0.12044#011validation-error:0.13160[0m
    [34m[62]#011train-error:0.12055#011validation-error:0.13191[0m
    [34m[63]#011train-error:0.12055#011validation-error:0.13206[0m
    [34m[64]#011train-error:0.12075#011validation-error:0.13084[0m
    [34m[65]#011train-error:0.12090#011validation-error:0.13068[0m
    [34m[66]#011train-error:0.12096#011validation-error:0.13084[0m
    [34m[67]#011train-error:0.12044#011validation-error:0.13068[0m
    [34m[68]#011train-error:0.12024#011validation-error:0.13084[0m
    [34m[69]#011train-error:0.12024#011validation-error:0.13084[0m
    [34m[70]#011train-error:0.12050#011validation-error:0.13022[0m
    [34m[71]#011train-error:0.12050#011validation-error:0.13037[0m
    [34m[72]#011train-error:0.12065#011validation-error:0.13068[0m
    [34m[73]#011train-error:0.12050#011validation-error:0.13068[0m
    [34m[74]#011train-error:0.12034#011validation-error:0.13084[0m
    [34m[75]#011train-error:0.12034#011validation-error:0.13084[0m
    [34m[76]#011train-error:0.12019#011validation-error:0.13114[0m
    [34m[77]#011train-error:0.11998#011validation-error:0.13053[0m
    [34m[78]#011train-error:0.12004#011validation-error:0.13022[0m
    [34m[79]#011train-error:0.11952#011validation-error:0.13068[0m
    [34m[80]#011train-error:0.11952#011validation-error:0.13053[0m
    [34m[81]#011train-error:0.11911#011validation-error:0.13022[0m
    [34m[82]#011train-error:0.11937#011validation-error:0.13022[0m
    [34m[83]#011train-error:0.11932#011validation-error:0.13022[0m
    [34m[84]#011train-error:0.11937#011validation-error:0.13022[0m
    [34m[85]#011train-error:0.11927#011validation-error:0.13022[0m
    [34m[86]#011train-error:0.11906#011validation-error:0.13037[0m
    [34m[87]#011train-error:0.11901#011validation-error:0.13037[0m
    [34m[88]#011train-error:0.11901#011validation-error:0.13037[0m
    [34m[89]#011train-error:0.11901#011validation-error:0.13022[0m
    [34m[90]#011train-error:0.11922#011validation-error:0.13099[0m
    [34m[91]#011train-error:0.11932#011validation-error:0.13084[0m
    [34m[92]#011train-error:0.11881#011validation-error:0.13145[0m
    [34m[93]#011train-error:0.11870#011validation-error:0.13130[0m
    [34m[94]#011train-error:0.11865#011validation-error:0.13160[0m
    [34m[95]#011train-error:0.11860#011validation-error:0.13130[0m
    [34m[96]#011train-error:0.11850#011validation-error:0.13099[0m
    [34m[97]#011train-error:0.11845#011validation-error:0.13099[0m
    [34m[98]#011train-error:0.11845#011validation-error:0.13130[0m
    [34m[99]#011train-error:0.11819#011validation-error:0.13084[0m
    [34m[100]#011train-error:0.11819#011validation-error:0.13068[0m
    [34m[101]#011train-error:0.11819#011validation-error:0.13130[0m
    [34m[102]#011train-error:0.11840#011validation-error:0.13191[0m
    [34m[103]#011train-error:0.11840#011validation-error:0.13145[0m
    [34m[104]#011train-error:0.11840#011validation-error:0.13160[0m
    [34m[105]#011train-error:0.11835#011validation-error:0.13206[0m
    [34m[106]#011train-error:0.11819#011validation-error:0.13206[0m
    [34m[107]#011train-error:0.11819#011validation-error:0.13206[0m
    [34m[108]#011train-error:0.11804#011validation-error:0.13206[0m
    [34m[109]#011train-error:0.11799#011validation-error:0.13176[0m
    [34m[110]#011train-error:0.11799#011validation-error:0.13176[0m
    [34m[111]#011train-error:0.11794#011validation-error:0.13160[0m
    [34m[112]#011train-error:0.11768#011validation-error:0.13130[0m
    [34m[113]#011train-error:0.11804#011validation-error:0.13206[0m
    [34m[114]#011train-error:0.11819#011validation-error:0.13206[0m
    [34m[115]#011train-error:0.11809#011validation-error:0.13145[0m
    [34m[116]#011train-error:0.11804#011validation-error:0.13130[0m
    [34m[117]#011train-error:0.11804#011validation-error:0.13130[0m
    [34m[118]#011train-error:0.11804#011validation-error:0.13114[0m
    [34m[119]#011train-error:0.11804#011validation-error:0.13130[0m
    [34m[120]#011train-error:0.11809#011validation-error:0.13068[0m
    [34m[121]#011train-error:0.11783#011validation-error:0.13145[0m
    [34m[122]#011train-error:0.11794#011validation-error:0.13145[0m
    [34m[123]#011train-error:0.11789#011validation-error:0.13206[0m
    [34m[124]#011train-error:0.11742#011validation-error:0.13099[0m
    [34m[125]#011train-error:0.11742#011validation-error:0.13114[0m
    [34m[126]#011train-error:0.11737#011validation-error:0.13037[0m
    [34m[127]#011train-error:0.11742#011validation-error:0.13068[0m
    [34m[128]#011train-error:0.11727#011validation-error:0.13084[0m
    [34m[129]#011train-error:0.11732#011validation-error:0.13176[0m
    [34m[130]#011train-error:0.11727#011validation-error:0.13130[0m
    [34m[131]#011train-error:0.11747#011validation-error:0.13114[0m
    [34m[132]#011train-error:0.11737#011validation-error:0.13084[0m
    [34m[133]#011train-error:0.11696#011validation-error:0.13068[0m
    [34m[134]#011train-error:0.11701#011validation-error:0.13084[0m
    [34m[135]#011train-error:0.11701#011validation-error:0.13084[0m
    [34m[136]#011train-error:0.11707#011validation-error:0.13068[0m
    [34m[137]#011train-error:0.11696#011validation-error:0.13037[0m
    [34m[138]#011train-error:0.11727#011validation-error:0.13145[0m
    [34m[139]#011train-error:0.11727#011validation-error:0.13130[0m
    [34m[140]#011train-error:0.11717#011validation-error:0.13099[0m
    [34m[141]#011train-error:0.11732#011validation-error:0.13130[0m
    [34m[142]#011train-error:0.11696#011validation-error:0.13053[0m
    [34m[143]#011train-error:0.11676#011validation-error:0.13037[0m
    [34m[144]#011train-error:0.11686#011validation-error:0.13053[0m
    [34m[145]#011train-error:0.11686#011validation-error:0.13068[0m
    [34m[146]#011train-error:0.11681#011validation-error:0.13053[0m
    [34m[147]#011train-error:0.11747#011validation-error:0.13053[0m
    [34m[148]#011train-error:0.11686#011validation-error:0.13099[0m
    [34m[149]#011train-error:0.11712#011validation-error:0.13068[0m
    [34m[150]#011train-error:0.11701#011validation-error:0.13114[0m
    [34m[151]#011train-error:0.11676#011validation-error:0.13037[0m
    [34m[152]#011train-error:0.11681#011validation-error:0.13022[0m
    [34m[153]#011train-error:0.11696#011validation-error:0.13037[0m
    [34m[154]#011train-error:0.11717#011validation-error:0.13037[0m
    [34m[155]#011train-error:0.11701#011validation-error:0.13037[0m
    [34m[156]#011train-error:0.11696#011validation-error:0.13022[0m
    [34m[157]#011train-error:0.11701#011validation-error:0.13022[0m
    [34m[158]#011train-error:0.11707#011validation-error:0.13037[0m
    [34m[159]#011train-error:0.11707#011validation-error:0.13022[0m
    [34m[160]#011train-error:0.11691#011validation-error:0.13037[0m
    [34m[161]#011train-error:0.11686#011validation-error:0.13007[0m
    [34m[162]#011train-error:0.11671#011validation-error:0.13007[0m
    [34m[163]#011train-error:0.11666#011validation-error:0.13007[0m
    [34m[164]#011train-error:0.11666#011validation-error:0.13022[0m
    [34m[165]#011train-error:0.11671#011validation-error:0.13053[0m
    [34m[166]#011train-error:0.11661#011validation-error:0.13053[0m
    [34m[167]#011train-error:0.11630#011validation-error:0.13022[0m
    [34m[168]#011train-error:0.11620#011validation-error:0.13007[0m
    [34m[169]#011train-error:0.11655#011validation-error:0.13022[0m
    [34m[170]#011train-error:0.11661#011validation-error:0.13022[0m
    [34m[171]#011train-error:0.11722#011validation-error:0.13084[0m
    [34m[172]#011train-error:0.11696#011validation-error:0.13099[0m
    [34m[173]#011train-error:0.11691#011validation-error:0.13068[0m
    [34m[174]#011train-error:0.11701#011validation-error:0.13114[0m
    [34m[175]#011train-error:0.11681#011validation-error:0.13114[0m
    [34m[176]#011train-error:0.11676#011validation-error:0.13068[0m
    [34m[177]#011train-error:0.11681#011validation-error:0.13114[0m
    [34m[178]#011train-error:0.11686#011validation-error:0.13114[0m
    [34m[179]#011train-error:0.11681#011validation-error:0.13099[0m
    [34m[180]#011train-error:0.11701#011validation-error:0.13160[0m
    [34m[181]#011train-error:0.11701#011validation-error:0.13145[0m
    [34m[182]#011train-error:0.11666#011validation-error:0.13160[0m
    [34m[183]#011train-error:0.11661#011validation-error:0.13222[0m
    [34m[184]#011train-error:0.11655#011validation-error:0.13222[0m
    [34m[185]#011train-error:0.11630#011validation-error:0.13222[0m
    [34m[186]#011train-error:0.11655#011validation-error:0.13130[0m
    [34m[187]#011train-error:0.11666#011validation-error:0.13176[0m
    [34m[188]#011train-error:0.11681#011validation-error:0.13237[0m
    [34m[189]#011train-error:0.11640#011validation-error:0.13253[0m
    [34m[190]#011train-error:0.11620#011validation-error:0.13222[0m
    [34m[191]#011train-error:0.11625#011validation-error:0.13253[0m
    [34m[192]#011train-error:0.11625#011validation-error:0.13253[0m
    [34m[193]#011train-error:0.11686#011validation-error:0.13206[0m
    [34m[194]#011train-error:0.11696#011validation-error:0.13114[0m
    [34m[195]#011train-error:0.11686#011validation-error:0.13160[0m
    [34m[196]#011train-error:0.11686#011validation-error:0.13176[0m
    [34m[197]#011train-error:0.11696#011validation-error:0.13191[0m
    [34m[198]#011train-error:0.11691#011validation-error:0.13191[0m
    [34m[199]#011train-error:0.11686#011validation-error:0.13160[0m
    [34m[200]#011train-error:0.11691#011validation-error:0.13160[0m
    [34m[201]#011train-error:0.11671#011validation-error:0.13145[0m
    [34m[202]#011train-error:0.11676#011validation-error:0.13145[0m
    [34m[203]#011train-error:0.11650#011validation-error:0.13145[0m
    [34m[204]#011train-error:0.11635#011validation-error:0.13206[0m
    [34m[205]#011train-error:0.11661#011validation-error:0.13160[0m
    [34m[206]#011train-error:0.11655#011validation-error:0.13145[0m
    [34m[207]#011train-error:0.11666#011validation-error:0.13145[0m
    [34m[208]#011train-error:0.11671#011validation-error:0.13145[0m
    [34m[209]#011train-error:0.11671#011validation-error:0.13130[0m
    [34m[210]#011train-error:0.11640#011validation-error:0.13130[0m
    [34m[211]#011train-error:0.11599#011validation-error:0.13130[0m
    [34m[212]#011train-error:0.11599#011validation-error:0.13130[0m
    [34m[213]#011train-error:0.11630#011validation-error:0.13130[0m
    [34m[214]#011train-error:0.11620#011validation-error:0.13160[0m
    [34m[215]#011train-error:0.11614#011validation-error:0.13145[0m
    [34m[216]#011train-error:0.11609#011validation-error:0.13160[0m
    [34m[217]#011train-error:0.11620#011validation-error:0.13176[0m
    [34m[218]#011train-error:0.11620#011validation-error:0.13222[0m
    [34m[219]#011train-error:0.11625#011validation-error:0.13268[0m
    [34m[220]#011train-error:0.11609#011validation-error:0.13191[0m
    [34m[221]#011train-error:0.11614#011validation-error:0.13206[0m
    [34m[222]#011train-error:0.11635#011validation-error:0.13176[0m
    [34m[223]#011train-error:0.11635#011validation-error:0.13237[0m
    [34m[224]#011train-error:0.11640#011validation-error:0.13206[0m
    [34m[225]#011train-error:0.11635#011validation-error:0.13237[0m
    [34m[226]#011train-error:0.11614#011validation-error:0.13237[0m
    [34m[227]#011train-error:0.11614#011validation-error:0.13253[0m
    [34m[228]#011train-error:0.11620#011validation-error:0.13253[0m
    [34m[229]#011train-error:0.11614#011validation-error:0.13191[0m
    [34m[230]#011train-error:0.11614#011validation-error:0.13191[0m
    [34m[231]#011train-error:0.11620#011validation-error:0.13222[0m
    [34m[232]#011train-error:0.11620#011validation-error:0.13206[0m
    [34m[233]#011train-error:0.11604#011validation-error:0.13237[0m
    [34m[234]#011train-error:0.11604#011validation-error:0.13222[0m
    [34m[235]#011train-error:0.11609#011validation-error:0.13253[0m
    [34m[236]#011train-error:0.11609#011validation-error:0.13268[0m
    [34m[237]#011train-error:0.11609#011validation-error:0.13283[0m
    [34m[238]#011train-error:0.11609#011validation-error:0.13298[0m
    [34m[239]#011train-error:0.11594#011validation-error:0.13283[0m
    [34m[240]#011train-error:0.11594#011validation-error:0.13283[0m
    [34m[241]#011train-error:0.11574#011validation-error:0.13268[0m
    [34m[242]#011train-error:0.11584#011validation-error:0.13253[0m
    [34m[243]#011train-error:0.11574#011validation-error:0.13268[0m
    [34m[244]#011train-error:0.11568#011validation-error:0.13222[0m
    [34m[245]#011train-error:0.11548#011validation-error:0.13206[0m
    [34m[246]#011train-error:0.11558#011validation-error:0.13237[0m
    [34m[247]#011train-error:0.11548#011validation-error:0.13237[0m
    [34m[248]#011train-error:0.11594#011validation-error:0.13206[0m
    [34m[249]#011train-error:0.11584#011validation-error:0.13206[0m
    [34m[250]#011train-error:0.11543#011validation-error:0.13191[0m
    [34m[251]#011train-error:0.11538#011validation-error:0.13191[0m
    [34m[252]#011train-error:0.11517#011validation-error:0.13253[0m
    [34m[253]#011train-error:0.11517#011validation-error:0.13253[0m
    [34m[254]#011train-error:0.11527#011validation-error:0.13283[0m
    [34m[255]#011train-error:0.11522#011validation-error:0.13283[0m
    [34m[256]#011train-error:0.11527#011validation-error:0.13253[0m
    [34m[257]#011train-error:0.11517#011validation-error:0.13283[0m
    [34m[258]#011train-error:0.11517#011validation-error:0.13329[0m
    [34m[259]#011train-error:0.11522#011validation-error:0.13298[0m
    [34m[260]#011train-error:0.11533#011validation-error:0.13253[0m
    [34m[261]#011train-error:0.11533#011validation-error:0.13253[0m
    [34m[262]#011train-error:0.11533#011validation-error:0.13222[0m
    [34m[263]#011train-error:0.11538#011validation-error:0.13191[0m
    [34m[264]#011train-error:0.11486#011validation-error:0.13145[0m
    [34m[265]#011train-error:0.11461#011validation-error:0.13130[0m
    [34m[266]#011train-error:0.11471#011validation-error:0.13191[0m
    [34m[267]#011train-error:0.11502#011validation-error:0.13176[0m
    [34m[268]#011train-error:0.11502#011validation-error:0.13176[0m
    [34m[269]#011train-error:0.11512#011validation-error:0.13237[0m
    [34m[270]#011train-error:0.11512#011validation-error:0.13237[0m
    [34m[271]#011train-error:0.11507#011validation-error:0.13206[0m
    
    2021-03-14 02:41:11 Training - Training image download completed. Training in progress.[34m[272]#011train-error:0.11502#011validation-error:0.13191[0m
    [34m[273]#011train-error:0.11502#011validation-error:0.13191[0m
    [34m[274]#011train-error:0.11502#011validation-error:0.13191[0m
    [34m[275]#011train-error:0.11497#011validation-error:0.13191[0m
    [34m[276]#011train-error:0.11466#011validation-error:0.13237[0m
    [34m[277]#011train-error:0.11456#011validation-error:0.13253[0m
    [34m[278]#011train-error:0.11466#011validation-error:0.13237[0m
    [34m[279]#011train-error:0.11492#011validation-error:0.13283[0m
    [34m[280]#011train-error:0.11440#011validation-error:0.13237[0m
    [34m[281]#011train-error:0.11451#011validation-error:0.13298[0m
    [34m[282]#011train-error:0.11440#011validation-error:0.13253[0m
    [34m[283]#011train-error:0.11456#011validation-error:0.13283[0m
    [34m[284]#011train-error:0.11425#011validation-error:0.13268[0m
    [34m[285]#011train-error:0.11451#011validation-error:0.13283[0m
    [34m[286]#011train-error:0.11451#011validation-error:0.13283[0m
    [34m[287]#011train-error:0.11476#011validation-error:0.13237[0m
    [34m[288]#011train-error:0.11476#011validation-error:0.13298[0m
    [34m[289]#011train-error:0.11481#011validation-error:0.13283[0m
    [34m[290]#011train-error:0.11481#011validation-error:0.13268[0m
    [34m[291]#011train-error:0.11481#011validation-error:0.13268[0m
    [34m[292]#011train-error:0.11481#011validation-error:0.13253[0m
    [34m[293]#011train-error:0.11481#011validation-error:0.13253[0m
    [34m[294]#011train-error:0.11481#011validation-error:0.13253[0m
    [34m[295]#011train-error:0.11451#011validation-error:0.13176[0m
    [34m[296]#011train-error:0.11451#011validation-error:0.13176[0m
    [34m[297]#011train-error:0.11461#011validation-error:0.13160[0m
    [34m[298]#011train-error:0.11492#011validation-error:0.13253[0m
    [34m[299]#011train-error:0.11471#011validation-error:0.13222[0m
    [34m[300]#011train-error:0.11446#011validation-error:0.13222[0m
    [34m[301]#011train-error:0.11456#011validation-error:0.13176[0m
    [34m[302]#011train-error:0.11456#011validation-error:0.13176[0m
    [34m[303]#011train-error:0.11435#011validation-error:0.13222[0m
    [34m[304]#011train-error:0.11461#011validation-error:0.13191[0m
    [34m[305]#011train-error:0.11446#011validation-error:0.13222[0m
    [34m[306]#011train-error:0.11451#011validation-error:0.13160[0m
    [34m[307]#011train-error:0.11456#011validation-error:0.13160[0m
    [34m[308]#011train-error:0.11466#011validation-error:0.13176[0m
    [34m[309]#011train-error:0.11446#011validation-error:0.13191[0m
    [34m[310]#011train-error:0.11440#011validation-error:0.13283[0m
    [34m[311]#011train-error:0.11461#011validation-error:0.13222[0m
    [34m[312]#011train-error:0.11451#011validation-error:0.13237[0m
    [34m[313]#011train-error:0.11466#011validation-error:0.13222[0m
    [34m[314]#011train-error:0.11451#011validation-error:0.13237[0m
    [34m[315]#011train-error:0.11461#011validation-error:0.13222[0m
    [34m[316]#011train-error:0.11451#011validation-error:0.13237[0m
    [34m[317]#011train-error:0.11456#011validation-error:0.13206[0m
    [34m[318]#011train-error:0.11451#011validation-error:0.13222[0m
    [34m[319]#011train-error:0.11446#011validation-error:0.13268[0m
    [34m[320]#011train-error:0.11456#011validation-error:0.13222[0m
    [34m[321]#011train-error:0.11451#011validation-error:0.13283[0m
    [34m[322]#011train-error:0.11451#011validation-error:0.13283[0m
    [34m[323]#011train-error:0.11451#011validation-error:0.13283[0m
    [34m[324]#011train-error:0.11451#011validation-error:0.13283[0m
    [34m[325]#011train-error:0.11446#011validation-error:0.13253[0m
    [34m[326]#011train-error:0.11456#011validation-error:0.13283[0m
    [34m[327]#011train-error:0.11451#011validation-error:0.13283[0m
    [34m[328]#011train-error:0.11440#011validation-error:0.13268[0m
    [34m[329]#011train-error:0.11410#011validation-error:0.13268[0m
    [34m[330]#011train-error:0.11425#011validation-error:0.13268[0m
    [34m[331]#011train-error:0.11420#011validation-error:0.13298[0m
    [34m[332]#011train-error:0.11384#011validation-error:0.13314[0m
    [34m[333]#011train-error:0.11384#011validation-error:0.13314[0m
    [34m[334]#011train-error:0.11374#011validation-error:0.13283[0m
    [34m[335]#011train-error:0.11394#011validation-error:0.13298[0m
    [34m[336]#011train-error:0.11369#011validation-error:0.13283[0m
    [34m[337]#011train-error:0.11379#011validation-error:0.13253[0m
    [34m[338]#011train-error:0.11369#011validation-error:0.13283[0m
    [34m[339]#011train-error:0.11384#011validation-error:0.13283[0m
    [34m[340]#011train-error:0.11374#011validation-error:0.13283[0m
    [34m[341]#011train-error:0.11399#011validation-error:0.13314[0m
    [34m[342]#011train-error:0.11364#011validation-error:0.13253[0m
    [34m[343]#011train-error:0.11374#011validation-error:0.13253[0m
    [34m[344]#011train-error:0.11379#011validation-error:0.13298[0m
    [34m[345]#011train-error:0.11389#011validation-error:0.13283[0m
    [34m[346]#011train-error:0.11389#011validation-error:0.13237[0m
    [34m[347]#011train-error:0.11415#011validation-error:0.13237[0m
    [34m[348]#011train-error:0.11405#011validation-error:0.13237[0m
    [34m[349]#011train-error:0.11425#011validation-error:0.13314[0m
    [34m[350]#011train-error:0.11420#011validation-error:0.13268[0m
    [34m[351]#011train-error:0.11410#011validation-error:0.13253[0m
    [34m[352]#011train-error:0.11410#011validation-error:0.13268[0m
    [34m[353]#011train-error:0.11379#011validation-error:0.13298[0m
    [34m[354]#011train-error:0.11384#011validation-error:0.13222[0m
    [34m[355]#011train-error:0.11410#011validation-error:0.13206[0m
    [34m[356]#011train-error:0.11394#011validation-error:0.13222[0m
    [34m[357]#011train-error:0.11364#011validation-error:0.13191[0m
    [34m[358]#011train-error:0.11364#011validation-error:0.13191[0m
    [34m[359]#011train-error:0.11384#011validation-error:0.13268[0m
    [34m[360]#011train-error:0.11415#011validation-error:0.13298[0m
    [34m[361]#011train-error:0.11394#011validation-error:0.13237[0m
    [34m[362]#011train-error:0.11389#011validation-error:0.13237[0m
    [34m[363]#011train-error:0.11374#011validation-error:0.13222[0m
    [34m[364]#011train-error:0.11359#011validation-error:0.13237[0m
    [34m[365]#011train-error:0.11364#011validation-error:0.13222[0m
    [34m[366]#011train-error:0.11348#011validation-error:0.13206[0m
    [34m[367]#011train-error:0.11312#011validation-error:0.13283[0m
    [34m[368]#011train-error:0.11323#011validation-error:0.13268[0m
    [34m[369]#011train-error:0.11338#011validation-error:0.13283[0m
    [34m[370]#011train-error:0.11328#011validation-error:0.13268[0m
    [34m[371]#011train-error:0.11323#011validation-error:0.13314[0m
    [34m[372]#011train-error:0.11307#011validation-error:0.13314[0m
    [34m[373]#011train-error:0.11312#011validation-error:0.13329[0m
    [34m[374]#011train-error:0.11348#011validation-error:0.13345[0m
    [34m[375]#011train-error:0.11343#011validation-error:0.13360[0m
    [34m[376]#011train-error:0.11343#011validation-error:0.13345[0m
    [34m[377]#011train-error:0.11338#011validation-error:0.13345[0m
    [34m[378]#011train-error:0.11353#011validation-error:0.13345[0m
    [34m[379]#011train-error:0.11353#011validation-error:0.13345[0m
    [34m[380]#011train-error:0.11359#011validation-error:0.13329[0m
    [34m[381]#011train-error:0.11359#011validation-error:0.13283[0m
    [34m[382]#011train-error:0.11369#011validation-error:0.13345[0m
    [34m[383]#011train-error:0.11353#011validation-error:0.13360[0m
    [34m[384]#011train-error:0.11369#011validation-error:0.13329[0m
    [34m[385]#011train-error:0.11364#011validation-error:0.13314[0m
    [34m[386]#011train-error:0.11374#011validation-error:0.13314[0m
    [34m[387]#011train-error:0.11348#011validation-error:0.13421[0m
    [34m[388]#011train-error:0.11338#011validation-error:0.13437[0m
    [34m[389]#011train-error:0.11353#011validation-error:0.13452[0m
    [34m[390]#011train-error:0.11328#011validation-error:0.13421[0m
    [34m[391]#011train-error:0.11338#011validation-error:0.13437[0m
    [34m[392]#011train-error:0.11353#011validation-error:0.13452[0m
    [34m[393]#011train-error:0.11343#011validation-error:0.13452[0m
    [34m[394]#011train-error:0.11333#011validation-error:0.13421[0m
    [34m[395]#011train-error:0.11328#011validation-error:0.13421[0m
    [34m[396]#011train-error:0.11353#011validation-error:0.13452[0m
    [34m[397]#011train-error:0.11353#011validation-error:0.13452[0m
    [34m[398]#011train-error:0.11343#011validation-error:0.13452[0m
    [34m[399]#011train-error:0.11364#011validation-error:0.13437[0m
    [34m[400]#011train-error:0.11328#011validation-error:0.13391[0m
    [34m[401]#011train-error:0.11312#011validation-error:0.13421[0m
    [34m[402]#011train-error:0.11307#011validation-error:0.13406[0m
    [34m[403]#011train-error:0.11338#011validation-error:0.13421[0m
    [34m[404]#011train-error:0.11343#011validation-error:0.13406[0m
    [34m[405]#011train-error:0.11343#011validation-error:0.13406[0m
    [34m[406]#011train-error:0.11328#011validation-error:0.13421[0m
    [34m[407]#011train-error:0.11353#011validation-error:0.13421[0m
    [34m[408]#011train-error:0.11353#011validation-error:0.13421[0m
    [34m[409]#011train-error:0.11353#011validation-error:0.13421[0m
    [34m[410]#011train-error:0.11353#011validation-error:0.13421[0m
    [34m[411]#011train-error:0.11353#011validation-error:0.13421[0m
    [34m[412]#011train-error:0.11359#011validation-error:0.13406[0m
    [34m[413]#011train-error:0.11364#011validation-error:0.13452[0m
    [34m[414]#011train-error:0.11369#011validation-error:0.13452[0m
    [34m[415]#011train-error:0.11353#011validation-error:0.13483[0m
    [34m[416]#011train-error:0.11348#011validation-error:0.13483[0m
    [34m[417]#011train-error:0.11374#011validation-error:0.13529[0m
    [34m[418]#011train-error:0.11374#011validation-error:0.13514[0m
    [34m[419]#011train-error:0.11389#011validation-error:0.13483[0m
    [34m[420]#011train-error:0.11389#011validation-error:0.13483[0m
    [34m[421]#011train-error:0.11394#011validation-error:0.13498[0m
    [34m[422]#011train-error:0.11394#011validation-error:0.13498[0m
    [34m[423]#011train-error:0.11405#011validation-error:0.13514[0m
    [34m[424]#011train-error:0.11389#011validation-error:0.13514[0m
    [34m[425]#011train-error:0.11389#011validation-error:0.13514[0m
    [34m[426]#011train-error:0.11384#011validation-error:0.13514[0m
    [34m[427]#011train-error:0.11389#011validation-error:0.13514[0m
    [34m[428]#011train-error:0.11384#011validation-error:0.13514[0m
    [34m[429]#011train-error:0.11394#011validation-error:0.13529[0m
    [34m[430]#011train-error:0.11389#011validation-error:0.13544[0m
    [34m[431]#011train-error:0.11374#011validation-error:0.13560[0m
    [34m[432]#011train-error:0.11394#011validation-error:0.13498[0m
    [34m[433]#011train-error:0.11394#011validation-error:0.13498[0m
    [34m[434]#011train-error:0.11399#011validation-error:0.13529[0m
    [34m[435]#011train-error:0.11389#011validation-error:0.13514[0m
    [34m[436]#011train-error:0.11394#011validation-error:0.13529[0m
    [34m[437]#011train-error:0.11394#011validation-error:0.13529[0m
    [34m[438]#011train-error:0.11394#011validation-error:0.13498[0m
    [34m[439]#011train-error:0.11384#011validation-error:0.13483[0m
    [34m[440]#011train-error:0.11394#011validation-error:0.13514[0m
    [34m[441]#011train-error:0.11389#011validation-error:0.13483[0m
    [34m[442]#011train-error:0.11384#011validation-error:0.13483[0m
    [34m[443]#011train-error:0.11359#011validation-error:0.13514[0m
    [34m[444]#011train-error:0.11364#011validation-error:0.13514[0m
    [34m[445]#011train-error:0.11343#011validation-error:0.13498[0m
    [34m[446]#011train-error:0.11328#011validation-error:0.13514[0m
    [34m[447]#011train-error:0.11338#011validation-error:0.13529[0m
    [34m[448]#011train-error:0.11333#011validation-error:0.13529[0m
    [34m[449]#011train-error:0.11359#011validation-error:0.13560[0m
    [34m[450]#011train-error:0.11359#011validation-error:0.13529[0m
    [34m[451]#011train-error:0.11364#011validation-error:0.13514[0m
    [34m[452]#011train-error:0.11353#011validation-error:0.13529[0m
    [34m[453]#011train-error:0.11353#011validation-error:0.13529[0m
    [34m[454]#011train-error:0.11359#011validation-error:0.13514[0m
    [34m[455]#011train-error:0.11359#011validation-error:0.13544[0m
    [34m[456]#011train-error:0.11348#011validation-error:0.13544[0m
    [34m[457]#011train-error:0.11353#011validation-error:0.13514[0m
    [34m[458]#011train-error:0.11353#011validation-error:0.13560[0m
    [34m[459]#011train-error:0.11343#011validation-error:0.13544[0m
    [34m[460]#011train-error:0.11348#011validation-error:0.13544[0m
    [34m[461]#011train-error:0.11312#011validation-error:0.13452[0m
    [34m[462]#011train-error:0.11312#011validation-error:0.13452[0m
    [34m[463]#011train-error:0.11312#011validation-error:0.13452[0m
    [34m[464]#011train-error:0.11312#011validation-error:0.13452[0m
    [34m[465]#011train-error:0.11405#011validation-error:0.13483[0m
    [34m[466]#011train-error:0.11405#011validation-error:0.13483[0m
    [34m[467]#011train-error:0.11415#011validation-error:0.13498[0m
    [34m[468]#011train-error:0.11379#011validation-error:0.13498[0m
    [34m[469]#011train-error:0.11379#011validation-error:0.13483[0m
    [34m[470]#011train-error:0.11394#011validation-error:0.13483[0m
    [34m[471]#011train-error:0.11399#011validation-error:0.13467[0m
    [34m[472]#011train-error:0.11384#011validation-error:0.13467[0m
    [34m[473]#011train-error:0.11374#011validation-error:0.13467[0m
    [34m[474]#011train-error:0.11364#011validation-error:0.13421[0m
    [34m[475]#011train-error:0.11389#011validation-error:0.13483[0m
    [34m[476]#011train-error:0.11374#011validation-error:0.13467[0m
    [34m[477]#011train-error:0.11379#011validation-error:0.13467[0m
    [34m[478]#011train-error:0.11384#011validation-error:0.13483[0m
    [34m[479]#011train-error:0.11389#011validation-error:0.13483[0m
    [34m[480]#011train-error:0.11384#011validation-error:0.13483[0m
    [34m[481]#011train-error:0.11379#011validation-error:0.13467[0m
    [34m[482]#011train-error:0.11374#011validation-error:0.13483[0m
    [34m[483]#011train-error:0.11379#011validation-error:0.13575[0m
    [34m[484]#011train-error:0.11399#011validation-error:0.13575[0m
    [34m[485]#011train-error:0.11399#011validation-error:0.13529[0m
    [34m[486]#011train-error:0.11389#011validation-error:0.13529[0m
    [34m[487]#011train-error:0.11374#011validation-error:0.13544[0m
    [34m[488]#011train-error:0.11379#011validation-error:0.13544[0m
    [34m[489]#011train-error:0.11374#011validation-error:0.13544[0m
    [34m[490]#011train-error:0.11405#011validation-error:0.13560[0m
    [34m[491]#011train-error:0.11410#011validation-error:0.13544[0m
    [34m[492]#011train-error:0.11410#011validation-error:0.13544[0m
    [34m[493]#011train-error:0.11399#011validation-error:0.13560[0m
    [34m[494]#011train-error:0.11389#011validation-error:0.13560[0m
    [34m[495]#011train-error:0.11359#011validation-error:0.13529[0m
    [34m[496]#011train-error:0.11359#011validation-error:0.13529[0m
    [34m[497]#011train-error:0.11359#011validation-error:0.13529[0m
    [34m[498]#011train-error:0.11359#011validation-error:0.13529[0m
    [34m[499]#011train-error:0.11333#011validation-error:0.13590[0m
    [34m[500]#011train-error:0.11359#011validation-error:0.13636[0m
    [34m[501]#011train-error:0.11359#011validation-error:0.13621[0m
    [34m[502]#011train-error:0.11343#011validation-error:0.13621[0m
    [34m[503]#011train-error:0.11348#011validation-error:0.13652[0m
    [34m[504]#011train-error:0.11374#011validation-error:0.13667[0m
    [34m[505]#011train-error:0.11353#011validation-error:0.13636[0m
    [34m[506]#011train-error:0.11338#011validation-error:0.13621[0m
    [34m[507]#011train-error:0.11251#011validation-error:0.13744[0m
    [34m[508]#011train-error:0.11251#011validation-error:0.13682[0m
    [34m[509]#011train-error:0.11251#011validation-error:0.13667[0m
    [34m[510]#011train-error:0.11277#011validation-error:0.13713[0m
    [34m[511]#011train-error:0.11302#011validation-error:0.13713[0m
    [34m[512]#011train-error:0.11302#011validation-error:0.13713[0m
    [34m[513]#011train-error:0.11297#011validation-error:0.13713[0m
    [34m[514]#011train-error:0.11292#011validation-error:0.13713[0m
    [34m[515]#011train-error:0.11241#011validation-error:0.13728[0m
    [34m[516]#011train-error:0.11241#011validation-error:0.13728[0m
    [34m[517]#011train-error:0.11271#011validation-error:0.13713[0m
    [34m[518]#011train-error:0.11261#011validation-error:0.13728[0m
    [34m[519]#011train-error:0.11312#011validation-error:0.13698[0m
    [34m[520]#011train-error:0.11307#011validation-error:0.13698[0m
    [34m[521]#011train-error:0.11312#011validation-error:0.13698[0m
    [34m[522]#011train-error:0.11312#011validation-error:0.13698[0m
    [34m[523]#011train-error:0.11359#011validation-error:0.13713[0m
    [34m[524]#011train-error:0.11333#011validation-error:0.13667[0m
    [34m[525]#011train-error:0.11338#011validation-error:0.13652[0m
    [34m[526]#011train-error:0.11338#011validation-error:0.13652[0m
    [34m[527]#011train-error:0.11338#011validation-error:0.13667[0m
    [34m[528]#011train-error:0.11338#011validation-error:0.13667[0m
    [34m[529]#011train-error:0.11338#011validation-error:0.13667[0m
    [34m[530]#011train-error:0.11338#011validation-error:0.13652[0m
    [34m[531]#011train-error:0.11333#011validation-error:0.13667[0m
    [34m[532]#011train-error:0.11338#011validation-error:0.13667[0m
    [34m[533]#011train-error:0.11338#011validation-error:0.13652[0m
    [34m[534]#011train-error:0.11338#011validation-error:0.13667[0m
    [34m[535]#011train-error:0.11338#011validation-error:0.13652[0m
    [34m[536]#011train-error:0.11333#011validation-error:0.13652[0m
    [34m[537]#011train-error:0.11338#011validation-error:0.13667[0m
    [34m[538]#011train-error:0.11338#011validation-error:0.13667[0m
    [34m[539]#011train-error:0.11323#011validation-error:0.13682[0m
    [34m[540]#011train-error:0.11271#011validation-error:0.13759[0m
    [34m[541]#011train-error:0.11271#011validation-error:0.13728[0m
    [34m[542]#011train-error:0.11271#011validation-error:0.13759[0m
    [34m[543]#011train-error:0.11271#011validation-error:0.13728[0m
    [34m[544]#011train-error:0.11266#011validation-error:0.13759[0m
    [34m[545]#011train-error:0.11282#011validation-error:0.13744[0m
    [34m[546]#011train-error:0.11266#011validation-error:0.13759[0m
    [34m[547]#011train-error:0.11236#011validation-error:0.13728[0m
    [34m[548]#011train-error:0.11210#011validation-error:0.13698[0m
    [34m[549]#011train-error:0.11205#011validation-error:0.13636[0m
    [34m[550]#011train-error:0.11220#011validation-error:0.13606[0m
    [34m[551]#011train-error:0.11215#011validation-error:0.13636[0m
    [34m[552]#011train-error:0.11246#011validation-error:0.13606[0m
    [34m[553]#011train-error:0.11215#011validation-error:0.13636[0m
    [34m[554]#011train-error:0.11215#011validation-error:0.13636[0m
    [34m[555]#011train-error:0.11231#011validation-error:0.13606[0m
    [34m[556]#011train-error:0.11220#011validation-error:0.13606[0m
    [34m[557]#011train-error:0.11220#011validation-error:0.13636[0m
    [34m[558]#011train-error:0.11174#011validation-error:0.13606[0m
    [34m[559]#011train-error:0.11174#011validation-error:0.13606[0m
    [34m[560]#011train-error:0.11174#011validation-error:0.13606[0m
    [34m[561]#011train-error:0.11174#011validation-error:0.13698[0m
    [34m[562]#011train-error:0.11174#011validation-error:0.13682[0m
    [34m[563]#011train-error:0.11184#011validation-error:0.13621[0m
    [34m[564]#011train-error:0.11179#011validation-error:0.13698[0m
    [34m[565]#011train-error:0.11169#011validation-error:0.13698[0m
    [34m[566]#011train-error:0.11169#011validation-error:0.13728[0m
    [34m[567]#011train-error:0.11169#011validation-error:0.13728[0m
    [34m[568]#011train-error:0.11190#011validation-error:0.13713[0m
    [34m[569]#011train-error:0.11205#011validation-error:0.13713[0m
    [34m[570]#011train-error:0.11200#011validation-error:0.13728[0m
    [34m[571]#011train-error:0.11205#011validation-error:0.13713[0m
    [34m[572]#011train-error:0.11159#011validation-error:0.13698[0m
    [34m[573]#011train-error:0.11149#011validation-error:0.13667[0m
    [34m[574]#011train-error:0.11149#011validation-error:0.13636[0m
    [34m[575]#011train-error:0.11174#011validation-error:0.13621[0m
    [34m[576]#011train-error:0.11174#011validation-error:0.13621[0m
    [34m[577]#011train-error:0.11138#011validation-error:0.13652[0m
    [34m[578]#011train-error:0.11144#011validation-error:0.13636[0m
    [34m[579]#011train-error:0.11149#011validation-error:0.13636[0m
    [34m[580]#011train-error:0.11174#011validation-error:0.13606[0m
    [34m[581]#011train-error:0.11195#011validation-error:0.13590[0m
    [34m[582]#011train-error:0.11190#011validation-error:0.13590[0m
    [34m[583]#011train-error:0.11164#011validation-error:0.13575[0m
    [34m[584]#011train-error:0.11174#011validation-error:0.13590[0m
    [34m[585]#011train-error:0.11164#011validation-error:0.13590[0m
    [34m[586]#011train-error:0.11164#011validation-error:0.13590[0m
    [34m[587]#011train-error:0.11149#011validation-error:0.13606[0m
    [34m[588]#011train-error:0.11128#011validation-error:0.13606[0m
    [34m[589]#011train-error:0.11133#011validation-error:0.13606[0m
    [34m[590]#011train-error:0.11179#011validation-error:0.13698[0m
    [34m[591]#011train-error:0.11190#011validation-error:0.13713[0m
    [34m[592]#011train-error:0.11179#011validation-error:0.13636[0m
    [34m[593]#011train-error:0.11179#011validation-error:0.13667[0m
    [34m[594]#011train-error:0.11154#011validation-error:0.13621[0m
    [34m[595]#011train-error:0.11154#011validation-error:0.13590[0m
    [34m[596]#011train-error:0.11164#011validation-error:0.13575[0m
    [34m[597]#011train-error:0.11174#011validation-error:0.13636[0m
    [34m[598]#011train-error:0.11164#011validation-error:0.13606[0m
    [34m[599]#011train-error:0.11190#011validation-error:0.13606[0m
    [34m[600]#011train-error:0.11190#011validation-error:0.13621[0m
    [34m[601]#011train-error:0.11190#011validation-error:0.13636[0m
    [34m[602]#011train-error:0.11169#011validation-error:0.13652[0m
    [34m[603]#011train-error:0.11169#011validation-error:0.13682[0m
    [34m[604]#011train-error:0.11164#011validation-error:0.13682[0m
    [34m[605]#011train-error:0.11159#011validation-error:0.13713[0m
    [34m[606]#011train-error:0.11144#011validation-error:0.13652[0m
    [34m[607]#011train-error:0.11144#011validation-error:0.13667[0m
    [34m[608]#011train-error:0.11123#011validation-error:0.13652[0m
    [34m[609]#011train-error:0.11128#011validation-error:0.13652[0m
    [34m[610]#011train-error:0.11138#011validation-error:0.13544[0m
    [34m[611]#011train-error:0.11144#011validation-error:0.13560[0m
    [34m[612]#011train-error:0.11144#011validation-error:0.13560[0m
    [34m[613]#011train-error:0.11144#011validation-error:0.13560[0m
    [34m[614]#011train-error:0.11128#011validation-error:0.13560[0m
    [34m[615]#011train-error:0.11103#011validation-error:0.13621[0m
    [34m[616]#011train-error:0.11113#011validation-error:0.13621[0m
    [34m[617]#011train-error:0.11108#011validation-error:0.13621[0m
    [34m[618]#011train-error:0.11128#011validation-error:0.13636[0m
    [34m[619]#011train-error:0.11128#011validation-error:0.13621[0m
    [34m[620]#011train-error:0.11123#011validation-error:0.13636[0m
    [34m[621]#011train-error:0.11103#011validation-error:0.13682[0m
    [34m[622]#011train-error:0.11077#011validation-error:0.13698[0m
    [34m[623]#011train-error:0.11113#011validation-error:0.13759[0m
    [34m[624]#011train-error:0.11118#011validation-error:0.13713[0m
    [34m[625]#011train-error:0.11144#011validation-error:0.13728[0m
    [34m[626]#011train-error:0.11123#011validation-error:0.13713[0m
    [34m[627]#011train-error:0.11133#011validation-error:0.13728[0m
    [34m[628]#011train-error:0.11133#011validation-error:0.13728[0m
    [34m[629]#011train-error:0.11128#011validation-error:0.13698[0m
    [34m[630]#011train-error:0.11118#011validation-error:0.13790[0m
    [34m[631]#011train-error:0.11108#011validation-error:0.13744[0m
    [34m[632]#011train-error:0.11103#011validation-error:0.13759[0m
    [34m[633]#011train-error:0.11098#011validation-error:0.13805[0m
    [34m[634]#011train-error:0.11098#011validation-error:0.13805[0m
    [34m[635]#011train-error:0.11087#011validation-error:0.13790[0m
    [34m[636]#011train-error:0.11087#011validation-error:0.13805[0m
    [34m[637]#011train-error:0.11108#011validation-error:0.13836[0m
    [34m[638]#011train-error:0.11113#011validation-error:0.13821[0m
    [34m[639]#011train-error:0.11062#011validation-error:0.13867[0m
    [34m[640]#011train-error:0.11082#011validation-error:0.13805[0m
    [34m[641]#011train-error:0.11067#011validation-error:0.13851[0m
    [34m[642]#011train-error:0.11067#011validation-error:0.13851[0m
    [34m[643]#011train-error:0.11056#011validation-error:0.13728[0m
    [34m[644]#011train-error:0.11010#011validation-error:0.13759[0m
    [34m[645]#011train-error:0.11010#011validation-error:0.13759[0m
    [34m[646]#011train-error:0.11016#011validation-error:0.13759[0m
    [34m[647]#011train-error:0.11041#011validation-error:0.13713[0m
    [34m[648]#011train-error:0.11056#011validation-error:0.13682[0m
    [34m[649]#011train-error:0.11051#011validation-error:0.13667[0m
    [34m[650]#011train-error:0.11056#011validation-error:0.13682[0m
    [34m[651]#011train-error:0.11051#011validation-error:0.13698[0m
    [34m[652]#011train-error:0.11041#011validation-error:0.13698[0m
    [34m[653]#011train-error:0.11051#011validation-error:0.13682[0m
    [34m[654]#011train-error:0.11056#011validation-error:0.13682[0m
    [34m[655]#011train-error:0.11010#011validation-error:0.13821[0m
    [34m[656]#011train-error:0.11036#011validation-error:0.13759[0m
    [34m[657]#011train-error:0.11036#011validation-error:0.13790[0m
    [34m[658]#011train-error:0.11026#011validation-error:0.13775[0m
    [34m[659]#011train-error:0.11031#011validation-error:0.13775[0m
    [34m[660]#011train-error:0.11031#011validation-error:0.13775[0m
    [34m[661]#011train-error:0.11031#011validation-error:0.13728[0m
    [34m[662]#011train-error:0.11005#011validation-error:0.13652[0m
    [34m[663]#011train-error:0.11000#011validation-error:0.13621[0m
    [34m[664]#011train-error:0.11005#011validation-error:0.13667[0m
    [34m[665]#011train-error:0.11010#011validation-error:0.13667[0m
    [34m[666]#011train-error:0.11000#011validation-error:0.13667[0m
    [34m[667]#011train-error:0.11000#011validation-error:0.13667[0m
    [34m[668]#011train-error:0.11010#011validation-error:0.13667[0m
    [34m[669]#011train-error:0.11010#011validation-error:0.13667[0m
    [34m[670]#011train-error:0.11010#011validation-error:0.13652[0m
    [34m[671]#011train-error:0.11026#011validation-error:0.13713[0m
    [34m[672]#011train-error:0.11010#011validation-error:0.13728[0m
    [34m[673]#011train-error:0.10990#011validation-error:0.13728[0m
    [34m[674]#011train-error:0.10954#011validation-error:0.13698[0m
    [34m[675]#011train-error:0.10959#011validation-error:0.13713[0m
    [34m[676]#011train-error:0.10959#011validation-error:0.13713[0m
    [34m[677]#011train-error:0.10964#011validation-error:0.13744[0m
    [34m[678]#011train-error:0.10964#011validation-error:0.13744[0m
    [34m[679]#011train-error:0.10964#011validation-error:0.13744[0m
    [34m[680]#011train-error:0.11016#011validation-error:0.13652[0m
    [34m[681]#011train-error:0.11005#011validation-error:0.13682[0m
    [34m[682]#011train-error:0.11005#011validation-error:0.13713[0m
    [34m[683]#011train-error:0.11046#011validation-error:0.13652[0m
    [34m[684]#011train-error:0.11010#011validation-error:0.13621[0m
    [34m[685]#011train-error:0.11010#011validation-error:0.13621[0m
    [34m[686]#011train-error:0.11026#011validation-error:0.13606[0m
    [34m[687]#011train-error:0.11041#011validation-error:0.13636[0m
    [34m[688]#011train-error:0.11016#011validation-error:0.13621[0m
    [34m[689]#011train-error:0.11000#011validation-error:0.13606[0m
    [34m[690]#011train-error:0.11005#011validation-error:0.13575[0m
    [34m[691]#011train-error:0.11000#011validation-error:0.13636[0m
    [34m[692]#011train-error:0.11016#011validation-error:0.13636[0m
    [34m[693]#011train-error:0.11010#011validation-error:0.13606[0m
    [34m[694]#011train-error:0.11005#011validation-error:0.13636[0m
    [34m[695]#011train-error:0.11010#011validation-error:0.13606[0m
    [34m[696]#011train-error:0.11026#011validation-error:0.13636[0m
    [34m[697]#011train-error:0.11016#011validation-error:0.13636[0m
    [34m[698]#011train-error:0.11010#011validation-error:0.13636[0m
    [34m[699]#011train-error:0.11010#011validation-error:0.13636[0m
    [34m[700]#011train-error:0.11031#011validation-error:0.13652[0m
    [34m[701]#011train-error:0.11031#011validation-error:0.13682[0m
    [34m[702]#011train-error:0.11026#011validation-error:0.13667[0m
    [34m[703]#011train-error:0.11010#011validation-error:0.13744[0m
    [34m[704]#011train-error:0.11031#011validation-error:0.13713[0m
    [34m[705]#011train-error:0.11005#011validation-error:0.13713[0m
    [34m[706]#011train-error:0.11026#011validation-error:0.13713[0m
    [34m[707]#011train-error:0.11026#011validation-error:0.13713[0m
    [34m[708]#011train-error:0.11021#011validation-error:0.13713[0m
    [34m[709]#011train-error:0.11026#011validation-error:0.13744[0m
    [34m[710]#011train-error:0.11021#011validation-error:0.13744[0m
    [34m[711]#011train-error:0.11026#011validation-error:0.13698[0m
    [34m[712]#011train-error:0.11031#011validation-error:0.13744[0m
    [34m[713]#011train-error:0.11016#011validation-error:0.13682[0m
    [34m[714]#011train-error:0.11021#011validation-error:0.13682[0m
    [34m[715]#011train-error:0.11026#011validation-error:0.13698[0m
    [34m[716]#011train-error:0.11021#011validation-error:0.13667[0m
    [34m[717]#011train-error:0.11046#011validation-error:0.13682[0m
    [34m[718]#011train-error:0.11031#011validation-error:0.13713[0m
    [34m[719]#011train-error:0.11031#011validation-error:0.13713[0m
    [34m[720]#011train-error:0.11031#011validation-error:0.13698[0m
    [34m[721]#011train-error:0.11026#011validation-error:0.13698[0m
    [34m[722]#011train-error:0.11031#011validation-error:0.13713[0m
    [34m[723]#011train-error:0.11026#011validation-error:0.13728[0m
    [34m[724]#011train-error:0.11026#011validation-error:0.13698[0m
    [34m[725]#011train-error:0.11031#011validation-error:0.13682[0m
    [34m[726]#011train-error:0.11021#011validation-error:0.13698[0m
    [34m[727]#011train-error:0.11026#011validation-error:0.13682[0m
    [34m[728]#011train-error:0.11036#011validation-error:0.13682[0m
    [34m[729]#011train-error:0.11021#011validation-error:0.13698[0m
    [34m[730]#011train-error:0.11021#011validation-error:0.13667[0m
    [34m[731]#011train-error:0.11005#011validation-error:0.13636[0m
    [34m[732]#011train-error:0.11000#011validation-error:0.13667[0m
    [34m[733]#011train-error:0.11000#011validation-error:0.13667[0m
    [34m[734]#011train-error:0.11005#011validation-error:0.13652[0m
    [34m[735]#011train-error:0.10995#011validation-error:0.13713[0m
    [34m[736]#011train-error:0.11000#011validation-error:0.13713[0m
    [34m[737]#011train-error:0.10995#011validation-error:0.13713[0m
    [34m[738]#011train-error:0.10995#011validation-error:0.13713[0m
    [34m[739]#011train-error:0.10995#011validation-error:0.13713[0m
    [34m[740]#011train-error:0.10990#011validation-error:0.13713[0m
    [34m[741]#011train-error:0.10995#011validation-error:0.13744[0m
    [34m[742]#011train-error:0.11000#011validation-error:0.13713[0m
    [34m[743]#011train-error:0.10995#011validation-error:0.13698[0m
    [34m[744]#011train-error:0.11005#011validation-error:0.13698[0m
    [34m[745]#011train-error:0.10995#011validation-error:0.13698[0m
    [34m[746]#011train-error:0.10995#011validation-error:0.13698[0m
    [34m[747]#011train-error:0.10990#011validation-error:0.13698[0m
    [34m[748]#011train-error:0.10985#011validation-error:0.13698[0m
    [34m[749]#011train-error:0.11000#011validation-error:0.13698[0m
    [34m[750]#011train-error:0.10995#011validation-error:0.13698[0m
    [34m[751]#011train-error:0.10995#011validation-error:0.13698[0m
    [34m[752]#011train-error:0.11005#011validation-error:0.13698[0m
    [34m[753]#011train-error:0.11000#011validation-error:0.13667[0m
    [34m[754]#011train-error:0.11000#011validation-error:0.13667[0m
    [34m[755]#011train-error:0.11000#011validation-error:0.13652[0m
    [34m[756]#011train-error:0.11000#011validation-error:0.13667[0m
    [34m[757]#011train-error:0.11005#011validation-error:0.13667[0m
    [34m[758]#011train-error:0.10995#011validation-error:0.13698[0m
    [34m[759]#011train-error:0.10995#011validation-error:0.13698[0m
    [34m[760]#011train-error:0.11010#011validation-error:0.13713[0m
    [34m[761]#011train-error:0.10995#011validation-error:0.13698[0m
    [34m[762]#011train-error:0.11005#011validation-error:0.13698[0m
    [34m[763]#011train-error:0.11016#011validation-error:0.13682[0m
    [34m[764]#011train-error:0.11005#011validation-error:0.13682[0m
    [34m[765]#011train-error:0.11005#011validation-error:0.13682[0m
    [34m[766]#011train-error:0.11016#011validation-error:0.13652[0m
    [34m[767]#011train-error:0.11026#011validation-error:0.13682[0m
    [34m[768]#011train-error:0.11016#011validation-error:0.13652[0m
    [34m[769]#011train-error:0.11031#011validation-error:0.13652[0m
    [34m[770]#011train-error:0.11016#011validation-error:0.13652[0m
    [34m[771]#011train-error:0.11026#011validation-error:0.13682[0m
    [34m[772]#011train-error:0.11000#011validation-error:0.13667[0m
    [34m[773]#011train-error:0.11010#011validation-error:0.13652[0m
    [34m[774]#011train-error:0.11062#011validation-error:0.13652[0m
    [34m[775]#011train-error:0.11056#011validation-error:0.13652[0m
    [34m[776]#011train-error:0.11046#011validation-error:0.13652[0m
    [34m[777]#011train-error:0.11036#011validation-error:0.13698[0m
    [34m[778]#011train-error:0.11041#011validation-error:0.13698[0m
    [34m[779]#011train-error:0.11026#011validation-error:0.13744[0m
    [34m[780]#011train-error:0.11031#011validation-error:0.13744[0m
    [34m[781]#011train-error:0.11031#011validation-error:0.13744[0m
    [34m[782]#011train-error:0.11036#011validation-error:0.13713[0m
    [34m[783]#011train-error:0.11036#011validation-error:0.13713[0m
    [34m[784]#011train-error:0.11031#011validation-error:0.13713[0m
    [34m[785]#011train-error:0.11051#011validation-error:0.13759[0m
    [34m[786]#011train-error:0.11036#011validation-error:0.13744[0m
    [34m[787]#011train-error:0.11036#011validation-error:0.13744[0m
    [34m[788]#011train-error:0.11031#011validation-error:0.13728[0m
    [34m[789]#011train-error:0.11036#011validation-error:0.13728[0m
    [34m[790]#011train-error:0.11051#011validation-error:0.13744[0m
    [34m[791]#011train-error:0.11051#011validation-error:0.13759[0m
    [34m[792]#011train-error:0.11087#011validation-error:0.13744[0m
    [34m[793]#011train-error:0.11077#011validation-error:0.13713[0m
    [34m[794]#011train-error:0.11041#011validation-error:0.13728[0m
    [34m[795]#011train-error:0.11016#011validation-error:0.13775[0m
    [34m[796]#011train-error:0.11016#011validation-error:0.13759[0m
    [34m[797]#011train-error:0.11000#011validation-error:0.13805[0m
    [34m[798]#011train-error:0.11016#011validation-error:0.13805[0m
    [34m[799]#011train-error:0.11026#011validation-error:0.13775[0m
    [34m[800]#011train-error:0.10985#011validation-error:0.13775[0m
    [34m[801]#011train-error:0.10995#011validation-error:0.13744[0m
    [34m[802]#011train-error:0.11021#011validation-error:0.13805[0m
    [34m[803]#011train-error:0.11041#011validation-error:0.13836[0m
    [34m[804]#011train-error:0.11046#011validation-error:0.13836[0m
    [34m[805]#011train-error:0.11051#011validation-error:0.13851[0m
    [34m[806]#011train-error:0.11041#011validation-error:0.13836[0m
    [34m[807]#011train-error:0.11046#011validation-error:0.13836[0m
    [34m[808]#011train-error:0.11041#011validation-error:0.13836[0m
    [34m[809]#011train-error:0.11036#011validation-error:0.13821[0m
    [34m[810]#011train-error:0.11046#011validation-error:0.13836[0m
    [34m[811]#011train-error:0.11041#011validation-error:0.13821[0m
    [34m[812]#011train-error:0.11051#011validation-error:0.13851[0m
    [34m[813]#011train-error:0.11056#011validation-error:0.13821[0m
    [34m[814]#011train-error:0.11021#011validation-error:0.13790[0m
    [34m[815]#011train-error:0.11010#011validation-error:0.13805[0m
    [34m[816]#011train-error:0.11016#011validation-error:0.13821[0m
    [34m[817]#011train-error:0.11021#011validation-error:0.13821[0m
    [34m[818]#011train-error:0.11036#011validation-error:0.13821[0m
    [34m[819]#011train-error:0.11021#011validation-error:0.13851[0m
    [34m[820]#011train-error:0.11021#011validation-error:0.13821[0m
    [34m[821]#011train-error:0.11016#011validation-error:0.13851[0m
    [34m[822]#011train-error:0.11010#011validation-error:0.13851[0m
    [34m[823]#011train-error:0.11021#011validation-error:0.13851[0m
    [34m[824]#011train-error:0.11016#011validation-error:0.13821[0m
    [34m[825]#011train-error:0.11021#011validation-error:0.13851[0m
    [34m[826]#011train-error:0.11016#011validation-error:0.13821[0m
    [34m[827]#011train-error:0.11016#011validation-error:0.13821[0m
    [34m[828]#011train-error:0.11016#011validation-error:0.13821[0m
    [34m[829]#011train-error:0.11016#011validation-error:0.13821[0m
    [34m[830]#011train-error:0.11016#011validation-error:0.13821[0m
    [34m[831]#011train-error:0.11016#011validation-error:0.13821[0m
    [34m[832]#011train-error:0.11021#011validation-error:0.13805[0m
    [34m[833]#011train-error:0.11016#011validation-error:0.13821[0m
    [34m[834]#011train-error:0.11031#011validation-error:0.13836[0m
    [34m[835]#011train-error:0.11010#011validation-error:0.13836[0m
    [34m[836]#011train-error:0.11005#011validation-error:0.13836[0m
    [34m[837]#011train-error:0.11010#011validation-error:0.13836[0m
    [34m[838]#011train-error:0.11010#011validation-error:0.13836[0m
    [34m[839]#011train-error:0.11041#011validation-error:0.13851[0m
    [34m[840]#011train-error:0.11021#011validation-error:0.13851[0m
    [34m[841]#011train-error:0.11016#011validation-error:0.13851[0m
    [34m[842]#011train-error:0.11000#011validation-error:0.13836[0m
    [34m[843]#011train-error:0.11016#011validation-error:0.13836[0m
    [34m[844]#011train-error:0.11010#011validation-error:0.13836[0m
    [34m[845]#011train-error:0.11010#011validation-error:0.13836[0m
    [34m[846]#011train-error:0.11016#011validation-error:0.13851[0m
    [34m[847]#011train-error:0.10990#011validation-error:0.13851[0m
    [34m[848]#011train-error:0.10980#011validation-error:0.13851[0m
    [34m[849]#011train-error:0.11005#011validation-error:0.13821[0m
    [34m[850]#011train-error:0.10990#011validation-error:0.13836[0m
    [34m[851]#011train-error:0.10980#011validation-error:0.13836[0m
    [34m[852]#011train-error:0.11000#011validation-error:0.13836[0m
    [34m[853]#011train-error:0.11016#011validation-error:0.13790[0m
    [34m[854]#011train-error:0.10990#011validation-error:0.13836[0m
    [34m[855]#011train-error:0.10970#011validation-error:0.13836[0m
    [34m[856]#011train-error:0.11000#011validation-error:0.13836[0m
    [34m[857]#011train-error:0.11005#011validation-error:0.13805[0m
    [34m[858]#011train-error:0.11016#011validation-error:0.13851[0m
    [34m[859]#011train-error:0.10990#011validation-error:0.13790[0m
    [34m[860]#011train-error:0.10990#011validation-error:0.13775[0m
    [34m[861]#011train-error:0.11000#011validation-error:0.13821[0m
    [34m[862]#011train-error:0.10980#011validation-error:0.13759[0m
    [34m[863]#011train-error:0.10970#011validation-error:0.13790[0m
    [34m[864]#011train-error:0.10975#011validation-error:0.13851[0m
    [34m[865]#011train-error:0.10975#011validation-error:0.13882[0m
    [34m[866]#011train-error:0.10964#011validation-error:0.13851[0m
    [34m[867]#011train-error:0.10970#011validation-error:0.13821[0m
    [34m[868]#011train-error:0.10964#011validation-error:0.13882[0m
    [34m[869]#011train-error:0.10980#011validation-error:0.13897[0m
    [34m[870]#011train-error:0.10975#011validation-error:0.13836[0m
    [34m[871]#011train-error:0.10975#011validation-error:0.13836[0m
    [34m[872]#011train-error:0.10964#011validation-error:0.13882[0m
    [34m[873]#011train-error:0.10970#011validation-error:0.13882[0m
    [34m[874]#011train-error:0.10964#011validation-error:0.13867[0m
    [34m[875]#011train-error:0.10970#011validation-error:0.13851[0m
    [34m[876]#011train-error:0.10898#011validation-error:0.13821[0m
    [34m[877]#011train-error:0.10908#011validation-error:0.13867[0m
    [34m[878]#011train-error:0.10913#011validation-error:0.13805[0m
    [34m[879]#011train-error:0.10918#011validation-error:0.13759[0m
    [34m[880]#011train-error:0.10918#011validation-error:0.13759[0m
    
    2021-03-14 02:41:51 Uploading - Uploading generated training model[34m[881]#011train-error:0.10964#011validation-error:0.13759[0m
    [34m[882]#011train-error:0.10928#011validation-error:0.13744[0m
    [34m[883]#011train-error:0.10964#011validation-error:0.13728[0m
    [34m[884]#011train-error:0.10959#011validation-error:0.13728[0m
    [34m[885]#011train-error:0.10959#011validation-error:0.13713[0m
    [34m[886]#011train-error:0.10959#011validation-error:0.13713[0m
    [34m[887]#011train-error:0.10964#011validation-error:0.13728[0m
    [34m[888]#011train-error:0.10959#011validation-error:0.13728[0m
    [34m[889]#011train-error:0.10959#011validation-error:0.13682[0m
    [34m[890]#011train-error:0.10949#011validation-error:0.13698[0m
    [34m[891]#011train-error:0.10954#011validation-error:0.13698[0m
    [34m[892]#011train-error:0.10964#011validation-error:0.13698[0m
    [34m[893]#011train-error:0.10964#011validation-error:0.13728[0m
    [34m[894]#011train-error:0.10954#011validation-error:0.13728[0m
    [34m[895]#011train-error:0.10959#011validation-error:0.13698[0m
    [34m[896]#011train-error:0.10964#011validation-error:0.13728[0m
    [34m[897]#011train-error:0.10949#011validation-error:0.13713[0m
    [34m[898]#011train-error:0.10918#011validation-error:0.13698[0m
    [34m[899]#011train-error:0.10954#011validation-error:0.13713[0m
    [34m[900]#011train-error:0.10949#011validation-error:0.13759[0m
    [34m[901]#011train-error:0.10949#011validation-error:0.13744[0m
    [34m[902]#011train-error:0.10949#011validation-error:0.13744[0m
    [34m[903]#011train-error:0.10949#011validation-error:0.13728[0m
    [34m[904]#011train-error:0.10949#011validation-error:0.13744[0m
    [34m[905]#011train-error:0.10949#011validation-error:0.13759[0m
    [34m[906]#011train-error:0.10949#011validation-error:0.13759[0m
    [34m[907]#011train-error:0.10949#011validation-error:0.13744[0m
    [34m[908]#011train-error:0.10954#011validation-error:0.13759[0m
    [34m[909]#011train-error:0.10949#011validation-error:0.13744[0m
    [34m[910]#011train-error:0.10949#011validation-error:0.13759[0m
    [34m[911]#011train-error:0.10923#011validation-error:0.13728[0m
    [34m[912]#011train-error:0.10918#011validation-error:0.13744[0m
    [34m[913]#011train-error:0.10918#011validation-error:0.13775[0m
    [34m[914]#011train-error:0.10918#011validation-error:0.13775[0m
    [34m[915]#011train-error:0.10908#011validation-error:0.13759[0m
    [34m[916]#011train-error:0.10913#011validation-error:0.13744[0m
    [34m[917]#011train-error:0.10913#011validation-error:0.13744[0m
    [34m[918]#011train-error:0.10913#011validation-error:0.13713[0m
    [34m[919]#011train-error:0.10923#011validation-error:0.13728[0m
    [34m[920]#011train-error:0.10939#011validation-error:0.13759[0m
    [34m[921]#011train-error:0.10939#011validation-error:0.13759[0m
    [34m[922]#011train-error:0.10939#011validation-error:0.13790[0m
    [34m[923]#011train-error:0.10944#011validation-error:0.13775[0m
    [34m[924]#011train-error:0.10944#011validation-error:0.13775[0m
    [34m[925]#011train-error:0.10908#011validation-error:0.13775[0m
    [34m[926]#011train-error:0.10893#011validation-error:0.13728[0m
    [34m[927]#011train-error:0.10898#011validation-error:0.13759[0m
    [34m[928]#011train-error:0.10898#011validation-error:0.13744[0m
    [34m[929]#011train-error:0.10898#011validation-error:0.13790[0m
    [34m[930]#011train-error:0.10888#011validation-error:0.13775[0m
    [34m[931]#011train-error:0.10903#011validation-error:0.13744[0m
    [34m[932]#011train-error:0.10903#011validation-error:0.13744[0m
    [34m[933]#011train-error:0.10898#011validation-error:0.13744[0m
    [34m[934]#011train-error:0.10883#011validation-error:0.13775[0m
    [34m[935]#011train-error:0.10862#011validation-error:0.13805[0m
    [34m[936]#011train-error:0.10877#011validation-error:0.13775[0m
    [34m[937]#011train-error:0.10857#011validation-error:0.13759[0m
    [34m[938]#011train-error:0.10857#011validation-error:0.13744[0m
    [34m[939]#011train-error:0.10847#011validation-error:0.13759[0m
    [34m[940]#011train-error:0.10847#011validation-error:0.13744[0m
    [34m[941]#011train-error:0.10857#011validation-error:0.13744[0m
    [34m[942]#011train-error:0.10862#011validation-error:0.13805[0m
    [34m[943]#011train-error:0.10841#011validation-error:0.13775[0m
    [34m[944]#011train-error:0.10841#011validation-error:0.13744[0m
    [34m[945]#011train-error:0.10841#011validation-error:0.13744[0m
    [34m[946]#011train-error:0.10836#011validation-error:0.13759[0m
    [34m[947]#011train-error:0.10831#011validation-error:0.13759[0m
    [34m[948]#011train-error:0.10821#011validation-error:0.13759[0m
    [34m[949]#011train-error:0.10831#011validation-error:0.13775[0m
    [34m[950]#011train-error:0.10836#011validation-error:0.13759[0m
    [34m[951]#011train-error:0.10836#011validation-error:0.13759[0m
    [34m[952]#011train-error:0.10836#011validation-error:0.13744[0m
    [34m[953]#011train-error:0.10831#011validation-error:0.13744[0m
    [34m[954]#011train-error:0.10836#011validation-error:0.13744[0m
    [34m[955]#011train-error:0.10852#011validation-error:0.13759[0m
    [34m[956]#011train-error:0.10836#011validation-error:0.13775[0m
    [34m[957]#011train-error:0.10826#011validation-error:0.13775[0m
    [34m[958]#011train-error:0.10836#011validation-error:0.13790[0m
    [34m[959]#011train-error:0.10836#011validation-error:0.13790[0m
    [34m[960]#011train-error:0.10852#011validation-error:0.13759[0m
    [34m[961]#011train-error:0.10847#011validation-error:0.13821[0m
    [34m[962]#011train-error:0.10877#011validation-error:0.13805[0m
    [34m[963]#011train-error:0.10872#011validation-error:0.13744[0m
    [34m[964]#011train-error:0.10883#011validation-error:0.13759[0m
    [34m[965]#011train-error:0.10883#011validation-error:0.13728[0m
    [34m[966]#011train-error:0.10898#011validation-error:0.13698[0m
    [34m[967]#011train-error:0.10862#011validation-error:0.13790[0m
    [34m[968]#011train-error:0.10893#011validation-error:0.13790[0m
    [34m[969]#011train-error:0.10883#011validation-error:0.13790[0m
    [34m[970]#011train-error:0.10872#011validation-error:0.13790[0m
    [34m[971]#011train-error:0.10862#011validation-error:0.13775[0m
    [34m[972]#011train-error:0.10888#011validation-error:0.13759[0m
    [34m[973]#011train-error:0.10867#011validation-error:0.13744[0m
    [34m[974]#011train-error:0.10867#011validation-error:0.13744[0m
    [34m[975]#011train-error:0.10857#011validation-error:0.13759[0m
    [34m[976]#011train-error:0.10867#011validation-error:0.13759[0m
    [34m[977]#011train-error:0.10872#011validation-error:0.13759[0m
    [34m[978]#011train-error:0.10867#011validation-error:0.13744[0m
    [34m[979]#011train-error:0.10852#011validation-error:0.13744[0m
    [34m[980]#011train-error:0.10852#011validation-error:0.13744[0m
    [34m[981]#011train-error:0.10862#011validation-error:0.13759[0m
    [34m[982]#011train-error:0.10852#011validation-error:0.13759[0m
    [34m[983]#011train-error:0.10862#011validation-error:0.13744[0m
    [34m[984]#011train-error:0.10867#011validation-error:0.13759[0m
    [34m[985]#011train-error:0.10883#011validation-error:0.13759[0m
    [34m[986]#011train-error:0.10867#011validation-error:0.13759[0m
    [34m[987]#011train-error:0.10847#011validation-error:0.13744[0m
    [34m[988]#011train-error:0.10857#011validation-error:0.13744[0m
    [34m[989]#011train-error:0.10857#011validation-error:0.13728[0m
    [34m[990]#011train-error:0.10857#011validation-error:0.13728[0m
    [34m[991]#011train-error:0.10841#011validation-error:0.13728[0m
    [34m[992]#011train-error:0.10872#011validation-error:0.13698[0m
    [34m[993]#011train-error:0.10867#011validation-error:0.13682[0m
    [34m[994]#011train-error:0.10867#011validation-error:0.13652[0m
    [34m[995]#011train-error:0.10867#011validation-error:0.13682[0m
    [34m[996]#011train-error:0.10867#011validation-error:0.13682[0m
    [34m[997]#011train-error:0.10826#011validation-error:0.13698[0m
    [34m[998]#011train-error:0.10821#011validation-error:0.13667[0m
    [34m[999]#011train-error:0.10821#011validation-error:0.13667[0m



```python

```
