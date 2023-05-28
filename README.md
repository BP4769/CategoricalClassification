# CategoricalClassification

Source code for CategoricalClassification library useful for easy categorical dataset creation.

## Dependencies

- numpy

## Usage
_CategoricalClassification_ is a library designed to quickly and easily generate binary categorical datasets. It supports both linearly and non-linearly separable dataset generation, as well as various noise simulating functions.

### Importing
Once copied to your working directory, _CategoricalClassification_ can be imported as any other Python library. 
```python
from CategoricalClassification import CategoricalClassification
cc = CategoricalClassification()
```
### Generating a linearly separable datasets
Generates a linearly separable dataset with 100 relevant features, 400 irrelevant features, 10000 samples, with a seed of 42.
```python
X,y = cc.generate_linear_binary_data(100, 400, samples=10000, seed=42)
```
Generates a linearly separable dataset with 100 relevant features and 400 irrelevant features from a label array.
```python
labels = cc.generate_binary_labels(10000, 0.5, seed=42)
X,y = cc.generate_linear_binary_data(100,400, labels=labels, seed=42)
```
Generates a non-linearly separable dataset with 100 relevant features, 400 irrelevant features, 10000 samples, with a seed of 42.
```python
X,y = cc.generate_nonlinear_data(100, 10000, p=0.5, n_irrelevant=400, seed=42)
```
Generates a non-linearly separable dataset with 100 relevant features and 400 irrelevant features from a label array.
```python
labels = cc.generate_binary_labels(10000, 0.5, seed=42)
X,y = cc.generate_nonlinear_data(100, 10000, n_irrelevant=400, labels=labels, seed=42)
```
### Applying noise to datasets
Applying cardinal noise to any binary or categorical dataset X, cardinality of 10 to class label 1.
```python
X = cc.replace_with_cardinality(X, [10, 1], seed=42)
```
Applying categorical noise to 20% of any binary dataset X.
```python
X = cc.noisy_data_cat(X, p=0.2, seed=42)
```
Applying missing values to 35% of any dataset X.
```python
X = cc.replace_with_none(X, 0.35, seed=42)
```

X, y = cc.generate_linear_binary_data(n_informative, n_redundant, samples=n_samples, seed=seed)
```
The output consists of a dataset with the informative features on lowest indices and redundant concatenated after that and a list of labels.

## License

CategoricalClassification is released under the Apache License 2.0. See the [LICENSE](LICENSE) file for more details.


