# Applying Linear Models on Higher Dimensional Data

Trying different Linear Models on a very high dimensional dataset and trying different preprocessign techniques to achieve a better model with reduced errors and higher accuracy.

## 21st Century Bordeaux Wine Dataset

**Source** : [21st Century Bordeaux Wine Dataset , Kaggle](https://www.kaggle.com/datasets/mexwell/21st-century-bordeaux-wine-dataset)

It contains 14,349 rows and 987 column which is considerd as a so called high dimensional datset.

### Preprocessing Involves

*   Dropping the Duplicate Columns[No Duplicate found].
*   Removing $ from the `Price` column, and filling the NaN values uing ***KNN Imputer***.