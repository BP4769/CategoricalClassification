import numpy as np
import math
from itertools import combinations

class CategoricalClassification:

    ##############################################################################################################
    ###     DATA GENERATION     ##################################################################################
    ##############################################################################################################
    

    def generate_features(self, label, logical_expression, n_features=2, seed=None):
        """
            Generates features for a given label using a logical expression
                Args:
                    label: the label to generate features for
                    logical_expression: the logical expression to use
                    n_features: the number of features to generate
                    seed: random seed
                Returns:
                    a numpy array of features

            features = generate_features(label, xor, n_features=2, seed=42)
        """
        # Parameter validation:
        if n_features < 1:
            raise ValueError("n_features must be greater than 0")
        if not isinstance(label, list):
            raise ValueError("label must be a list")
        if not callable(logical_expression):
            raise ValueError("logical_expression must be a function")
        if not isinstance(n_features, int):
            raise ValueError("n_features must be an integer")
        if seed is not None and not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        
        # set random seed
        if seed is not None:
            np.random.seed(seed)

        # generate all possible combinations of features, number of features is the number parameters in the logical expression
        features = np.array(np.meshgrid(*[np.arange(2)] * n_features)).T.reshape(-1, n_features)

        # evaluate the logical expression for each possible feature value
        res = logical_expression(features)
        # return res, features
        # randomly resample featrues to match the label
        # if the label is 1, then the expression must be true, so we sample from the features where the expression is true
        # if the label is 0, then the expression must be false, so we sample from the features where the expression is false
        # we do this by randomly sampling from the features where the expression is true, and then flipping the bits
        # this is equivalent to sampling from the features where the expression is false
        feature_set = []
        for l in label:
            ix = np.where(res == l)[0]
            sel = np.random.choice(ix, 1)
            feature_set.append(features[sel])

        return np.array(feature_set)  # .reshape(-1,n_features)

    def corrAL(self, x):
        """
            Logic expression used in the corrAL dataset:
                returns (A or B) and (C or D)
        """
        a, b, c, d = np.hsplit(x, 4)
        return (a | b) & (c | d)

    def simple_or(self, x):
        """
            Logical OR:
                returns A or B
        """
        a, b = np.hsplit(x, 2)
        return a | b

    def simple_and(self, x):
        """
            Logical AND:
                returns A and B
        """
        a, b = np.hsplit(x, 2)
        return a & b

    def simple_xor(self, x):
        """
           Logical XOR:
                returns A xor B
        """
        a, b = np.hsplit(x, 2)
        return a ^ b


    def generate_binary_labels(self, N, p, seed=None):
        """
            Generates a vector of binary labels with probability p.
                Args:
                    N: Number of labels to generate.
                    p: Probability of a label being 1.
                    seed: Random seed.
                Returns:
                    A vector of binary labels.

            labels = generate_binary_labels(50, 0.5, seed=42)
        """
        # Parameter validation:
        if N < 1:
            raise ValueError("N must be greater than 0")
        if p < 0 or p > 1:
            raise ValueError("p must be between 0 and 1")
        if not isinstance(N, int):
            raise ValueError("N must be an integer")
        if not isinstance(p, float):
            raise ValueError("p must be a float")
        if seed is not None and not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        
        # set random seed
        if seed is not None:
            np.random.seed(seed)

        return np.random.binomial(1, p, N)
    
    def _generate_hypercube(self, samples, dimensions, replace=False, seed=None):
        """
            Returns distinct binary samples of length dimensions. If dimensions is greater than 30, the samples are
            generated in a recursive manner.
                Args:
                    samples: Number of samples to generate.
                    dimensions: Number of dimensions of each sample.
                    replace: Whether to sample with replacement.
                    seed: Random seed.
                Returns:
                    A numpy array of binary samples.

            hcube = _generate_hypercube(10, 10, replace=True, seed=42)
        """
        # Parameter validation:
        if samples < 1:
            raise ValueError("Number of samples must be greater than 0")
        if dimensions < 1:
            raise ValueError("Number of dimensions must be greater than 0")
        if not isinstance(samples, int):
            raise ValueError("Number of samples must be an integer")
        if not isinstance(dimensions, int):
            raise ValueError("Number of dimensions must be an integer")
        if not isinstance(replace, bool):
            raise ValueError("replace must be a boolean")
        if seed is not None and not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        
        # set random seed
        if seed is not None:
            np.random.seed(seed)

        if dimensions > 30:
            return np.hstack(
                [
                    np.random.randint(2, size=(samples, 30)),
                    self._generate_hypercube(samples, dimensions - 30, replace=True),
                ]
            )
        out = np.random.choice(2 ** dimensions, size=samples, replace=replace).astype(">u4")
        out = np.unpackbits(out.view(">u1")).reshape((-1, 32))[:, -dimensions:]
        return out
    
    
    def generate_linear_binary_data(self, dimensions, useless_dimensions = None, samples = None, labels = None, seed = None):
        """
            Generates linearly separable binary data with categorical features and controlled class imbalance.
                Args:
                    dimensions: Number of dimensions of each sample.
                    samples: Number of samples to generate.
                    labels: Labels to generate.
                Returns:
                    A tuple of (features, labels).

            X,y = generate_linear_binary_data(10, useless_dimensions=40, samples=500, seed=42)
        """
        # Parameter validation:
        if dimensions < 1:
            raise ValueError("Number of dimensions must be greater than 0")
        if samples is None and labels is None:
            raise ValueError("Either samples or labels must be specified")
        if samples is not None and labels is not None:
            raise ValueError("Only one of samples or labels must be specified")
        if samples is not None and samples < 1:
            raise ValueError("Number of samples must be greater than 0")
        if labels is not None and len(labels) < 1:
            raise ValueError("Number of labels must be greater than 0")
        if not isinstance(dimensions, int) or (useless_dimensions is not None and not isinstance(useless_dimensions, int)):
            raise ValueError("Number of dimensions must be an integer")
        if samples is not None and not isinstance(samples, int):
            raise ValueError("Number of samples must be an integer")
        if labels is not None and not isinstance(labels, np.ndarray):
            raise ValueError("labels must be a numpy array")
        if labels is not None and not np.all(np.isin(labels, [0, 1])) and not np.issubdtype(labels.dtype, np.integer):
            raise ValueError("labels must be a numpy array of binary integers")
        if seed is not None and not isinstance(seed, int):
            raise ValueError("seed must be an integer")


        n_classes = 2  # Number of classes

        # Set seed
        if seed is not None:
            np.random.seed(seed)


        if useless_dimensions is None:
            useless_dimensions = 0

        if labels is not None:
            # tak a bigger sample size to account for the fact that we will be 
            # dropping some samples if the ratio is not balanced
            samples = len(labels)
            samples_3 = 3 * samples
            # Generate hypercube samples with categorical features
            X = self._generate_hypercube(samples_3, dimensions, replace=True)
        else:
            # Generate hypercube samples with categorical features
            X = self._generate_hypercube(samples, dimensions, replace=True)

        # Generate clusters for each class
        total_clusters = n_classes
        coefficients = np.zeros((total_clusters, dimensions + 1))  # Initialize coefficients

        # Compute the true center of the hypercube (for each dimension, the center is 0.5)
        center = np.ones(dimensions) * 0.5

        # Generate hyperplane coefficients passing through the center
        for i in range(total_clusters):
            random_point = np.random.randn(dimensions)  # Generate a random point
            hyperplane_normal = random_point - center  # Compute the vector from the center to the random point
            hyperplane_normal /= np.linalg.norm(hyperplane_normal)  # Normalize the vector
        
            coefficients[i, :-1] = hyperplane_normal  # Set the coefficients for the hyperplane equation
            coefficients[i, -1] = -np.dot(hyperplane_normal, center)  # Set the bias term

        # Add a column of ones to X for the bias term in the hyperplane equations
        if labels is not None:
            X_with_bias = np.hstack((X, np.ones((samples_3, 1))))
        else:
            X_with_bias = np.hstack((X, np.ones((samples, 1))))
        # Compute the dot product between X and the coefficients for each cluster
        dot_product = np.dot(X_with_bias, coefficients.T)
        # Assign labels based on the cluster with the maximum dot product
        y = np.argmax(dot_product, axis=1)
        # for each label sample randomly select a candidate
        if labels is not None:
            # get candidates for label 1 or 0
            candidates_1 = np.where(y == 1)[0]
            candidates_0 = np.where(y == 0)[0]
            # randomly select samples for each label
            selected_1 = np.random.choice(candidates_1, sum(labels), replace=False)
            selected_0 = np.random.choice(candidates_0, len(labels) - sum(labels), replace=False)
            # create new X with selected_1 where labels == 1 and selected_0 where labels == 0
            X = np.vstack((X[selected_1], X[selected_0]))
            y = np.hstack((y[selected_1], y[selected_0]))
            # reshuflle X and y according to order
            order = np.hstack((np.where(labels == 1)[0], np.where(labels == 0)[0]))
            sorted_indices = np.argsort(order)
            X = X[sorted_indices]
            y = y[sorted_indices]

        # add number of useless features
        if useless_dimensions > 0:
            X_useless = self._generate_hypercube(samples, useless_dimensions, replace=True)
            X = np.hstack((X, X_useless))

        return X, y

    def _change_variations(self, center, n, M):

        """
            Finds M or all points in distance n, within hypersphere defined by center
                Args:
                    center: center of hypersphere
                    n: square of distance
                    M: maximum needed points
                Returns:
                    List of points within distance n from center

            points = _change_variations(c, 2, 10)
        """

        variations = [center]  # Initialize the list with the original vector
        queue = [(center, 0, 0)]  # Initialize the queue with the original vector, zeros changed, and ones changed

        visited = set([tuple(center)])

        m = 0
        while queue:
            current, zeros_changed, ones_changed = queue.pop(0)

            # Check if the maximum number of changes has been reached
            if zeros_changed + ones_changed >= n:
                break

            for i in range(len(current)):
                if current[i] == 0 and zeros_changed < n:
                    new_vector = current.copy()
                    new_vector[i] = 1

                    if tuple(new_vector) not in visited:
                        variations.append(new_vector)
                        m += 1
                        visited.add(tuple(new_vector))
                        queue.append((new_vector, zeros_changed + 1, ones_changed))
                elif current[i] == 1 and ones_changed < n:
                    new_vector = current.copy()
                    new_vector[i] = 0

                    if tuple(new_vector) not in visited:
                        visited.add(tuple(new_vector))
                        variations.append(new_vector)
                        m += 1
                        queue.append((new_vector, zeros_changed, ones_changed + 1))
                if m >= M:
                    return variations
        return variations

    def _check_overlap(self, h_list, c, norm=2):
        """
            Checks if hypersphere c overlaps with any hypersphere in h_list
                Args:
                    h_list: list of centres of hyperspheres
                    c: point in space
                    norm: optional norm, default 2, used to check overlap
                Returns:
                    bool

            overlap = _check_overlap(centers, c, norm=1)
        """
        for h in h_list:
            if np.linalg.norm(h - c) <= norm:
                return True
        return False

    def generate_nonlinear_data_hyperspheres(self, n_features, n_samples, p=0.5, labels=None, n_irrelevant=0, reporting=True, seed=42):
        """
            Generates relevant, nonlinear, binary dataset using hyperspheres
                Args:
                    n_features: int, number of relevant features
                    n_samples: int, number of samples
                    p: float [0.0, 1.0], class distribution, default 0.5
                    labels: list, class labels
                    n_irrelevant: int, number of irrelevant features, default 0
                    reporting: prints info - number of generated hyperspheres, shape of data
                    seed: seed for numpy random
                Returns:
                    A tuple of (samples, labels)

            X,y = generate_nonlinear_data_hyperspheres(10, 500, p=0.5, n_irrelevant=50)
        """

        if not isinstance(n_features, int):
            raise ValueError("Number of features must be an integer")
        if not isinstance(n_samples, int):
            raise ValueError("Number of samples must be an integer")
        if not isinstance(seed, int):
            raise ValueError("Seed must be an integer")
        if not isinstance(n_irrelevant, int):
            raise ValueError("Number of irrelevant features must be an integer")
        if n_features < 1:
            raise ValueError("Number of features must be greater than 0")
        if n_samples < 1:
            raise ValueError("Number of samples must be greater than 0")
        if p < 0 or p > 1:
            raise ValueError("Class distribution p must be in range [0.0, 1.0]")
        if labels is not None and n_samples != len(labels):
            n_samples = len(labels)

        if 2 ** n_features < 2 * n_samples:
            raise ValueError("2**n_features must be at least 2*n_samples")

        np.random.seed(seed)

        n_class_0 = int(n_samples / 2)
        n_class_1 = n_samples - n_class_0

        if p is not None:
            n_class_1 = int(n_samples * p)
            n_class_0 = n_samples - n_class_1
        if labels is not None:
            n_class_0 = (labels == 0).sum()
            n_class_1 = n_samples - n_class_0

        # print(n_class_1, n_class_0)
        p = n_class_1 / n_samples
        
        generated_labels = np.append(np.ones(n_class_1), np.zeros(n_class_0))

        class_1_features = []
        hyperspheres = []

        t = n_class_1

        while (t > 0):
            if len(hyperspheres) > 0:
                c = np.random.randint(0, 2, size=n_features)
                o = 0
                overlap = self._check_overlap(hyperspheres, c)
                while overlap:
                    c = np.random.randint(0, 2, size=n_features)
                    overlap = self._check_overlap(hyperspheres, c)
                    o += 1
                    if o > 100:
                        print("Warning! Some hyperspheres WILL overlap.")
                        overlap = False

                hyperspheres.append(c)
            else:
                c = np.random.randint(0, 2, size=n_features)
                hyperspheres.append(c)

            vecs = self._change_variations(c, 2, n_class_1)
            # print("vecs:", len(vecs))

            if t == n_class_1:
                class_1_features = vecs
            else:
                class_1_features = np.concatenate((class_1_features, vecs), axis=0)

            # print("t:", t)
            t -= len(class_1_features)

        #print("c1:", len(class_1_features))

        class_0_features = set()
        found = set()
        t = n_class_0
        n = 0
        while (t > 0):
            f = np.random.randint(0, 2, size=n_features)
            ft = tuple(f)
            if ft not in found:
                found.add(ft)
            if len(found) == 2 ** n_features:
                raise ValueError(
                    "No space for desired class distribution. Increase dimensionality or decrease number of samples.")
            if self._check_overlap(hyperspheres, f) == False:
                if ft not in class_0_features:
                    class_0_features.add(ft)
                    t -= 1

        # print("c0:", len(class_0_features))

        class_1_features = np.array(class_1_features)
        class_0_features = np.array(list(class_0_features))
        class_0_features = np.array([np.array(t) for t in class_0_features])

        class_1_features = class_1_features[:n_class_1]
        class_0_features = class_0_features[:n_class_0]

        # print(class_1_features.shape)
        # print(class_0_features.shape)

        X = np.concatenate((class_1_features, class_0_features), axis=0)
        y = generated_labels

        if labels is not None:
            t = []
            for l in labels:
                ix = np.where(l == generated_labels)[0][0]
                t.append(ix)
                generated_labels[ix] = -1

            X = X[t]
            y = labels
        else:
            combined = np.column_stack((X, y))
            np.random.shuffle(combined)
            X = combined[:, :-1]
            y = combined[:, -1]

        if n_irrelevant > 0:
            X_irrelevant = self._generate_hypercube(n_samples, n_irrelevant, replace=True)
            X = np.hstack((X, X_irrelevant))

        if reporting:
            if len(hyperspheres) > 1:
                print("Generated", len(hyperspheres), "hyperspheres.")
                print("X:", X.shape, "y:", y.shape)
            else:
                print("Generated", len(hyperspheres), "hypersphere.")
                print("X:", X.shape, "y:", y.shape)

        return X, y

    def generate_nonlinear_data(self, n_features, n_samples, p=0.5, n_irrelevant=0, labels=None, seed=42):
        """
                   Generates nonlinear, binary dataset by squaring the sum of randomly weighted and biased features and applying an adaptive threshold
                       Args:
                           n_features: int, number of relevant features
                           n_samples: int, number of samples
                           p: float [0.0, 1.0], class distribution, default 0.5
                           labels: list, class labels
                           n_irrelevant: int, number of irrelevant features, default 0
                           reporting: prints info - number of generated hyperspheres, shape of data
                           seed: seed for numpy random
                       Returns:
                           A tuple of (samples, labels)

                   X,y = generate_nonlinear_data(10, 500, p=0.5, n_irrelevant=40)
               """

        if not isinstance(n_features, int):
            raise ValueError("Number of features must be an integer")
        if not isinstance(n_samples, int):
            raise ValueError("Number of samples must be an integer")
        if not isinstance(seed, int):
            raise ValueError("Seed must be an integer")
        if not isinstance(n_irrelevant, int):
            raise ValueError("Number of irrelevant features must be an integer")
        if n_features < 1:
            raise ValueError("Number of features must be greater than 0")
        if n_samples < 1:
            raise ValueError("Number of samples must be greater than 0")
        if p < 0 or p > 1:
            raise ValueError("Class distribution p must be in range [0.0, 1.0]")
        if labels is not None and n_samples != len(labels):
            n_samples = len(labels)

        np.random.seed(seed)

        X = np.random.randint(0, 2, size=(n_samples, n_features))

        n_class_0 = int(n_samples / 2)
        n_class_1 = n_samples - n_class_0

        if p is not None:
            n_class_1 = int(n_samples * p)
            n_class_0 = n_samples - n_class_1
        if labels is not None:
            n_class_0 = (labels == 0).sum()
            n_class_1 = n_samples - n_class_0

        p = n_class_1 / n_samples
        
        #print(n_samples, p, n_class_1, n_class_0)
        generated_labels = np.append(np.ones(n_class_1), np.zeros(n_class_0))

        for i in range(n_features):
            weight = np.random.uniform(-1, 1)
            bias = np.random.uniform(-1, 1)

            generated_labels += weight * (X[:, i] - bias) ** 2

        y_min = np.min(generated_labels)
        y_max = np.max(generated_labels)
        generated_labels = (generated_labels - y_min) / (y_max - y_min)
        #print("dolzina:", len(generated_labels))
        
        if p < 1.0:
            threshold = np.percentile(generated_labels, 100 - p * 100)
            generated_labels = (generated_labels > threshold).astype(int)
        else:
            generated_labels = np.ones(n_samples)

        y = generated_labels

        if labels is not None:
            t = []
            for l in labels:
                ix = np.where(l == generated_labels)[0][0]
                t.append(ix)
                generated_labels[ix] = -1

            X = X[t]
            y = labels
        else:
            combined = np.column_stack((X, y))
            np.random.shuffle(combined)
            X = combined[:, :-1]
            y = combined[:, -1]

        if n_irrelevant > 0:
            X_irrelevant = self._generate_hypercube(n_samples, n_irrelevant, replace=True)
            X = np.hstack((X, X_irrelevant))

        return X, y



    ##############################################################################################################
    ###     SIMULATING NOISE     #################################################################################
    ##############################################################################################################

    def cardinality_noise(self, labels, cardinality, number_of_relevant_features, number_of_irrelevant_features,
                          logical_combination_function, replace_redundant, seed = None):
        """
        Adds noise to the cardinality of a set of labels.
            Args:
                labels: A list of labels.
                cardinality: Vector of cardinalities of features corresponding to each label.
                number_of_relevant_features: Number of features that are relevant to the label.
                number_of_irrelevant_features: Number of features that are irrelevant to the label.
                logical_combination_function: A function that takes in a set of features and returns a logical combination of them.
                replace_redundant: Whether to perform the replacement of 0/1 after or before adding redundant features.
                seed: Random seed.
            Returns:
                The noisy dataset.

        """
        # Parameter validation:
        if not isinstance(labels, list):
            raise ValueError("labels must be a list")
        if not isinstance(cardinality, list) and len(cardinality) == 2 and type(cardinality[0]) == int and type(cardinality[1]) == int:
            raise ValueError("cardinality must be a list of two integers")
        if number_of_relevant_features < 1:
            raise ValueError("number_of_relevant_features must be greater than 0")
        if not callable(logical_combination_function):
            raise ValueError("logical_combination_function must be a function")
        if not isinstance(replace_redundant, bool):
            raise ValueError("replace_redundant must be a boolean")
        if seed is not None and not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        

        if seed is not None:
            np.random.seed(seed)

        new_features = self.generate_features(labels, logical_combination_function, number_of_relevant_features)[:, 0]

        if not replace_redundant:
            # replace each 1 and 0 with a different random number according to the cardinality
            one_indices = np.where(new_features == 1)
            zero_indices = np.where(new_features == 0)
            if cardinality[0] > 1:
                for i, j in zip(*zero_indices):
                    new_features[i, j] = np.random.randint(0, cardinality[0])

            if cardinality[1] > 1:
                for i, j in zip(*one_indices):
                    new_features[i, j] = np.random.randint(cardinality[0], cardinality[0] + cardinality[1])

        complete_cardinality = 2
        if not replace_redundant:
            complete_cardinality = cardinality[0] + cardinality[1]

        new_redundant_features = np.random.randint(complete_cardinality,
                                                   size=(len(labels), number_of_irrelevant_features))
        new_features = np.concatenate([new_features, new_redundant_features], axis=1)

        if replace_redundant:
            # replace each 1 and 0 with a different random number according to the cardinality
            one_indices = np.where(new_features == 1)
            zero_indices = np.where(new_features == 0)
            if cardinality[0] > 1:
                for i, j in zip(*zero_indices):
                    new_features[i, j] = np.random.randint(0, cardinality[0])

            if cardinality[1] > 1:
                for i, j in zip(*one_indices):
                    new_features[i, j] = np.random.randint(cardinality[0], cardinality[0] + cardinality[1])

        return new_features

    def replace_with_cardinality(self, arr, cardinality, seed=None):
        """
            Replaces each 1 and 0 with a different random number according to the cardinality.
                Args:
                    arr: A numpy array.
                    cardinality: Vector of cardinalities of features corresponding to each label.
                    seed: Random seed.
                Returns:
                    The noisy dataset.
        """
        # Parameter validation:
        if not isinstance(arr, np.ndarray):
            raise ValueError("arr must be a numpy array")
        if not isinstance(cardinality, list) and len(cardinality) == 2 and type(cardinality[0]) == int and type(cardinality[1]) == int:
            raise ValueError("cardinality must be a list of two integers")
        if seed is not None and not isinstance(seed, int):
            raise ValueError("seed must be an integer")

        if seed is not None:
            np.random.seed(seed)

        one_indices = np.where(arr == 1)
        zero_indices = np.where(arr == 0)
        if cardinality[0] > 1:
            for i, j in zip(*zero_indices):
                arr[i, j] = np.random.randint(0, cardinality[0])

        if cardinality[1] > 1:
            for i, j in zip(*one_indices):
                arr[i, j] = np.random.randint(cardinality[0], cardinality[0] + cardinality[1])

        return arr

    def replace_with_none(self, arr, f, seed=None):
        """
            Simulates missing data by replacing random values with None
                Args:
                    arr: input array
                    f: percentage of missing data
                    seed: int, seed
                Returns:
                    input array with missing data
        """
        # Parameter validation:
        if not isinstance(arr, np.ndarray):
            raise ValueError("arr must be a numpy array")
        if not isinstance(f, float) and f >= 0 and f <= 1:
            raise ValueError("f must be a float between 0 and 1")
        if not isinstance(seed, int) and seed is not None:
            raise ValueError("seed must be an integer")
        
        # Set seed
        if seed is not None:
            np.random.seed(seed)

        n = int(math.ceil(arr.size * f))
        indices = np.random.choice(arr.size, size=n, replace=False)
        output = arr.ravel().astype(object)
        # output[indices] = None
        output[indices] = -1  # as kNN doesn't handle None values well
        return np.array(output.reshape(arr.shape))

    def rand_bin_features(self, N, n, seed):
        """
            Generates random, irrelevant features
                Args:
                    n: int, number of features
                    N: int, number of samples
                    seed: int, seed
                Returns:
                    2D array of shape (N, n)

            X_rand = rand_bin_features(8, 46, 42)

        """
        # Parameter validation:
        if not isinstance(N, int) and N > 0:
            raise ValueError("N must be a positive integer")
        if not isinstance(n, int) and n > 0:
            raise ValueError("n must be a positive integer")
        if not isinstance(seed, int) and seed is not None:
            raise ValueError("seed must be an integer")

        # Set seed
        if seed is not None:
            np.random.seed(seed)

        return np.random.choice([0, 1], size=(N, n))

    def missing_values_noise(self, labels, **args):
        """
            Generates binary dataset on given labels, given parameters:
                Args:
                    logic: func, dictates the logic function for labels, default xor
                    n_relevant: int, number of relevant features, default 2
                    n_total: int, number of total features, default 100
                    noise: float, amount of noise, default 0.0, applied to relevant features
                    missing: float, amount of missing data, default 0.0, applied to all features
                    seed: int, seed applied to np.random

                Returns:
                    2D array of shape (len(labels), n_total)

            X = missing_values_noise(labels, logic=corrAL, n_relevant=4, n_total=50, noise=0.1, missing=0.2, seed=42)

        """
        # DEFAULT
        logic = self.simple_xor
        n_relevant = 2
        n_total = 100
        noise = 0.0
        missing = 0.0
        seed = None

        if len(args) > 0:
            if 'logic' in args:
                logic = args['logic']
            if 'n_relevant' in args:
                n_relevant = args['n_relevant']
            if 'n_total' in args:
                n_total = args['n_total']
            if 'noise' in args:
                noise = args['noise']
            if 'missing' in args:
                missing = args['missing']
            if 'seed' in args:
                seed = args['seed']

        # Parameter validation:
        if not isinstance(labels, list):
            raise ValueError("labels must be a list")
        if not callable(logic):
            raise ValueError("logic must be a function")
        if not isinstance(n_relevant, int) and n_relevant > 0:
            raise ValueError("n_relevant must be a positive integer")
        if not isinstance(n_total, int) and n_total > 0:
            raise ValueError("n_total must be a positive integer")
        if not isinstance(noise, float) and noise >= 0 and noise <= 1:
            raise ValueError("noise must be a float between 0 and 1")
        if not isinstance(missing, float) and missing >= 0 and missing <= 1:
            raise ValueError("missing must be a float between 0 and 1")
        if not isinstance(seed, int) and seed is not None:
            raise ValueError("seed must be an integer")
        
        #set seed
        if seed is not None:
            np.random.seed(seed)


        # generate irrelevant features
        n_irr = n_total - n_relevant
        # np.random.seed(seed)
        irr_features = self.rand_bin_features(len(labels), n_irr, seed)

        # print("irrelevant shape:", irr_features.shape)

        # generate relevant features
        rel_features = self.generate_features(labels, logic, n_relevant)
        # removes unnecessary dimension
        rel_features = np.squeeze(rel_features)

        """
        for i in range(len(labels)):
            print(labels[i], rel_features[i])
        """

        # print("relevant shape:", rel_features.shape)



        # generate 2D numpy featureset
        feature_set = np.empty((len(labels), n_total), dtype='int16')
        for i in range(len(feature_set)):
            feature_set[i] = np.concatenate((rel_features[i], irr_features[i]), axis=None)

        # remove random data from our feature set
        if missing > 0:
            feature_set = self.replace_with_none(feature_set, missing)

        """
        for i in range(len(labels)):
            print(labels[i], feature_set[i])
        """

        print("feature set shape:", feature_set.shape)
        return feature_set

    def noisy_column_bin(self, x, p=0.1, seed=None):
        """
            Adds noise to binary column
                Args:
                    x: 1D array of binary data
                    p: float, probability of a value being flipped
                    seed: int, seed applied to np.random
                Returns:
                    1D numpy array of noisy data

            x = noisy_column_bin(x, p=0.1)
        """
        # Parameter validation:
        if not isinstance(x, np.ndarray):
            raise ValueError("x must be a numpy array")
        if not isinstance(p, float) and p >= 0 and p <= 1:
            raise ValueError("p must be a float between 0 and 1")
        if not isinstance(seed, int) and seed is not None:
            raise ValueError("seed must be an integer")
        
        if seed is not None:
            np.random.seed(seed)

        # convert to numpy array, type int, reshape to column vector
        x = np.array(x)
        x = x.astype(int)
        x = x.reshape((len(x), 1))

        # get number of samples
        n = x.shape[0]
        # get number of samples to flip
        n_flip = int(n * p)
        # get indexes to flip
        ix = np.random.choice(n, n_flip, replace=False)
        # flip values
        x[ix] = np.abs(x[ix] - 1)
        return x

    def noisy_column_cat(self, x, p=0.1, seed=None):
        """
            Adds noise to categorical column
                Args:
                    x: 1D array of categorical data
                    p: float, probability of a value being flipped
                    seed: int, seed applied to np.random
                Returns:
                    1D numpy array of noisy data

            x = noisy_column_cat(x, p=0.1)
        """
        # Parameter validation:
        if not isinstance(x, np.ndarray):
            raise ValueError("x must be a numpy array")
        if not isinstance(p, float) and p >= 0 and p <= 1:
            raise ValueError("p must be a float between 0 and 1")
        if not isinstance(seed, int) and seed is not None:
            raise ValueError("seed must be an integer")
        
        if seed is not None:
            np.random.seed(seed)
        

        # convert to numpy array, type int, reshape to column vector
        x = np.array(x)
        x = x.reshape((len(x), 1))

        # get number of samples
        n = x.shape[0]
        # get number of samples to flip
        n_flip = int(n * p)
        # get indexes to flip
        ix = np.random.choice(n, n_flip, replace=False)
        # get unique values
        u = np.unique(x)
        # get number of unique values
        n_u = len(u)

        # if only one unique value, return original data
        if n_u <= 1:
            return x

        for i in ix:
            # turn unique into set
            set_u = set(u)
            # remove value xi from set
            set_u.remove(x[i][0])
            # get random value
            r = np.random.randint(0, n_u - 1)
            # set value to random value
            x[i] = list(set_u)[r]

        return x

    def noisy_data_cat(self, X, p=0.1, seed=None):
        """"
            Adds noise to categorical data (each column is treated as a categorical variable)
                Args:
                    X: 2D array of categorical data
                    p: float, probability of a value being flipped
                        or list of floats, probability of a value being flipped for each column
                    seed: int, seed applied to np.random
                Returns:
                    noisy data X

            X = noisy_data_cat(X, p=0.1)
        """
        # Parameter validation:
        if not isinstance(X, np.ndarray):
            raise ValueError("X must be a numpy array")
        if not isinstance(p, float) and not isinstance(p, list):
            raise ValueError("p must be a float or a list of floats")
        if not isinstance(seed, int) and seed is not None:
            raise ValueError("seed must be an integer")

        # convert to numpy array
        x = np.array(X)

        if isinstance(p, float):
            p = [p] * x.shape[1]
        elif isinstance(p, list):
            if len(p) != x.shape[1]:
                raise ValueError(
                    'p must be a float or a list of floats with length equal to the number of columns in X')

        # loop over each column
        for i in range(x.shape[1]):
            x[:, i] = self.noisy_column_cat(x[:, i], p[i], seed).ravel()
        return x

    def gen_categorical_noise(self, labels, logic, n_relevant=2, n_total=100, noise=0.1, seed=None):
        """
            Generates binary dataset on given labels, given parameters:
                Args:
                    labels: 1D array of labels
                    logic: func, dictates the logic function for labels, default xor
                    n_relevant: int, number of relevant features, default 2
                    n_total: int, number of total features, default 100
                    noise: float, amount of noise, default 0.0, applied to relevant features
                    seed: int, seed applied to np.random
                Returns:
                    2D array of shape (len(labels), n_total)

            X = gen_categorical_noise(labels, logic=corrAL, n_relevant=4, n_total=50, noise=0.1)

        """
        # Parameter validation:
        if not isinstance(labels, list):
            raise ValueError("labels must be a list")
        if not isinstance(n_relevant, int) and n_relevant > 0:
            raise ValueError("n_relevant must be a positive integer")
        if not isinstance(n_total, int) and n_total > 0:
            raise ValueError("n_total must be a positive integer")
        if not isinstance(noise, float) and noise >= 0 and noise <= 1:
            raise ValueError("noise must be a float between 0 and 1")
        if not isinstance(seed, int) and seed is not None:   
            raise ValueError("seed must be an integer")
        
        # set seed
        if seed is not None:
            np.random.seed(seed)
        

        if logic is None:
            logic = self.simple_xor

        # generate irrelevant features
        n_irr = n_total - n_relevant
        irr_features = self.rand_bin_features(len(labels), n_irr)

        # print("irrelevant shape:", irr_features.shape)

        # generate relevant features
        rel_features = self.generate_features(labels, logic, n_relevant)
        # removes unnecessary dimension
        rel_features = np.squeeze(rel_features)

        """
        for i in range(len(labels)):
            print(labels[i], rel_features[i])
        """

        # print("relevant shape:", rel_features.shape)

        # add noise to relevant features by randomly changing their value
        rel_f_noise = self.noisy_data_cat(rel_features, noise)

        # generate 2D numpy featureset
        feature_set = np.empty((len(labels), n_total), dtype='int16')
        for i in range(len(feature_set)):
            feature_set[i] = np.concatenate((rel_f_noise[i], irr_features[i]), axis=None)

        """
        for i in range(len(labels)):
            print(labels[i], feature_set[i])
        """

        print("feature set shape:", feature_set.shape)
        return feature_set