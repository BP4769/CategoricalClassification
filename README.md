# CategoricalClassification

Source code for CategoricalClassification library useful for easy categorical dataset creation.

## Dependencies

- numpy

## Usage

1. Import the required modules:

```python
import numpy
from CategoricalClassification import CategoricalClassification
```

2. Creating an instance of the CategoricalClassification and generating a linearly seperable dataset with it:

```python
cc = CategoricalClassification()
n_informative = 3
n_total = 10
n_redundant = n_total - n_informative
n_samples = 10
seed = 42

X, y = cc.generate_linear_binary_data(n_informative, n_redundant, samples=n_samples, seed=seed)
```
The output consists of a dataset with the informative features on lowest indices and redundant concatenated after that and a list of labels.

## License

CategoricalClassification is released under the Apache License 2.0. See the [LICENSE](LICENSE) file for more details.


