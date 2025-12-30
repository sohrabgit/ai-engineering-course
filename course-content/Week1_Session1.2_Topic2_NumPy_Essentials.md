# Week 1, Session 1.2, Topic 2: NumPy Essentials
## Duration: 30 minutes

---

## Learning Objectives
By the end of this topic, students will be able to:
1. Understand why NumPy is essential for AI/ML
2. Create and manipulate NumPy arrays
3. Perform vectorized operations (much faster than loops!)
4. Use array indexing, slicing, and reshaping
5. Apply linear algebra operations for ML
6. Understand broadcasting for efficient computations

---

## Introduction & Why NumPy Matters (2 minutes)

**Instructor Script:**

"Alright! You know Python basics. Now let's talk about **NumPy** - the absolute foundation of AI/ML in Python.

**Quick question:** Why do we need NumPy at all? Python has lists, right?

Let me show you why with a simple example..."

**Code Cell 1: The NumPy Performance Difference**

```python
import numpy as np
import time

# Create a large list of numbers
size = 1_000_000
python_list = list(range(size))
numpy_array = np.arange(size)

# Operation: Square every number

# Python way (using list comprehension)
start = time.time()
squared_list = [x ** 2 for x in python_list]
python_time = time.time() - start

# NumPy way
start = time.time()
squared_array = numpy_array ** 2
numpy_time = time.time() - start

print(f"Python list time: {python_time:.4f} seconds")
print(f"NumPy array time: {numpy_time:.4f} seconds")
print(f"NumPy is {python_time / numpy_time:.1f}x faster! 🚀")
```

**Expected Output:**
```
Python list time: 0.0876 seconds
NumPy array time: 0.0012 seconds
NumPy is 73.0x faster! 🚀
```

**Instructor Commentary:**

"NumPy is 50-100x faster than pure Python! Why?

1. **Written in C**: Low-level, optimized code
2. **Vectorized operations**: No slow Python loops
3. **Contiguous memory**: Data stored efficiently
4. **SIMD instructions**: CPU parallelism

**Bottom line:** For AI/ML with millions of numbers, NumPy isn't optional—it's essential!

Every major Python ML library uses NumPy:
- Pandas (built on NumPy)
- Scikit-learn (uses NumPy arrays)
- TensorFlow/PyTorch (similar array concepts)
- SciPy (extends NumPy)

Master NumPy = Master the foundation of Python AI! ✓"

---

## Part 1: NumPy Arrays Basics (8 minutes)

### 1.1 Creating Arrays (3 minutes)

**Instructor Script:**

"Let's start with the fundamental building block: the NumPy array (ndarray)."

**Code Cell 2: Creating Arrays**

```python
import numpy as np

# From Python list
list_1d = [1, 2, 3, 4, 5]
array_1d = np.array(list_1d)
print("1D array:")
print(array_1d)
print(f"Type: {type(array_1d)}")
print(f"Shape: {array_1d.shape}")
print(f"Data type: {array_1d.dtype}")
print()

# 2D array (matrix)
list_2d = [[1, 2, 3], 
           [4, 5, 6], 
           [7, 8, 9]]
array_2d = np.array(list_2d)
print("2D array:")
print(array_2d)
print(f"Shape: {array_2d.shape}")  # (rows, columns)
print(f"Dimensions: {array_2d.ndim}")
print(f"Total elements: {array_2d.size}")
print()

# 3D array (like a stack of matrices)
array_3d = np.array([
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]]
])
print("3D array:")
print(array_3d)
print(f"Shape: {array_3d.shape}")  # (depth, rows, columns)
```

**Code Cell 3: Special Array Creation Functions**

```python
# Arrays of zeros (common for initialization)
zeros = np.zeros((3, 4))  # 3 rows, 4 columns
print("Zeros array:")
print(zeros)
print()

# Arrays of ones
ones = np.ones((2, 3))
print("Ones array:")
print(ones)
print()

# Identity matrix (1s on diagonal)
identity = np.eye(4)
print("Identity matrix:")
print(identity)
print()

# Array with range of values
range_array = np.arange(0, 10, 2)  # start, stop, step
print(f"Range array: {range_array}")

# Evenly spaced values
linspace = np.linspace(0, 1, 5)  # start, stop, count
print(f"Linspace: {linspace}")
print()

# Random arrays (very common in ML!)
np.random.seed(42)  # For reproducibility
random_array = np.random.rand(3, 3)  # Uniform [0, 1)
print("Random array:")
print(random_array)
print()

# Random integers
random_ints = np.random.randint(0, 10, size=(2, 4))
print("Random integers:")
print(random_ints)
print()

# Random normal distribution (Gaussian)
random_normal = np.random.randn(3, 3)  # Mean=0, Std=1
print("Random normal:")
print(random_normal)
```

**Why This Matters:**

"In ML:
- **zeros/ones**: Initialize weights, create masks
- **random**: Initialize neural network weights
- **arange/linspace**: Create training iterations, ranges
- **eye**: Identity matrix for linear algebra

Example:
```python
# Initialize neural network weights randomly
weights_layer1 = np.random.randn(784, 128) * 0.01  # Small random values
biases_layer1 = np.zeros(128)
print(f"Weights shape: {weights_layer1.shape}")
print(f"Biases shape: {biases_layer1.shape}")
```
"

---

### 1.2 Array Attributes and Data Types (2 minutes)

**Code Cell 4: Understanding Array Attributes**

```python
# Create sample array
arr = np.array([[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]])

print("Array:")
print(arr)
print()

# Key attributes
print(f"Shape (dimensions): {arr.shape}")      # (3, 4)
print(f"Size (total elements): {arr.size}")    # 12
print(f"Number of dimensions: {arr.ndim}")     # 2
print(f"Data type: {arr.dtype}")               # int64
print(f"Item size (bytes): {arr.itemsize}")    # 8
print(f"Total bytes: {arr.nbytes}")            # 96
print()

# Data types matter for memory and speed!
int_array = np.array([1, 2, 3], dtype=np.int32)
float_array = np.array([1, 2, 3], dtype=np.float32)

print(f"Int32 array: {int_array.dtype}, {int_array.nbytes} bytes")
print(f"Float32 array: {float_array.dtype}, {float_array.nbytes} bytes")
```

**Instructor Commentary:**

"Data types are crucial in ML:
- **float32**: Most common for deep learning (balance of precision/speed)
- **float64**: Higher precision, more memory
- **int32/int64**: For labels, indices
- **bool**: For masks, filtering

Choose the right type to save memory and speed up computation!"

---

### 1.3 Array Indexing and Slicing (3 minutes)

**Code Cell 5: Indexing Arrays**

```python
# 1D array indexing
arr_1d = np.array([10, 20, 30, 40, 50])

print("1D array:", arr_1d)
print(f"First element: {arr_1d[0]}")
print(f"Last element: {arr_1d[-1]}")
print(f"Slice [1:4]: {arr_1d[1:4]}")  # Elements 1, 2, 3
print(f"Every other: {arr_1d[::2]}")  # Step of 2
print()

# 2D array indexing
arr_2d = np.array([[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12]])

print("2D array:")
print(arr_2d)
print()

# Access single element
print(f"Element at [0, 0]: {arr_2d[0, 0]}")    # 1
print(f"Element at [1, 2]: {arr_2d[1, 2]}")    # 7
print()

# Access rows
print(f"First row: {arr_2d[0]}")       # or arr_2d[0, :]
print(f"Last row: {arr_2d[-1]}")
print()

# Access columns
print(f"First column: {arr_2d[:, 0]}")
print(f"Second column: {arr_2d[:, 1]}")
print()

# Slicing rows and columns
print("Subarray [0:2, 1:3]:")
print(arr_2d[0:2, 1:3])  # First 2 rows, columns 1-2
```

**Code Cell 6: Boolean Indexing (Powerful for ML!)**

```python
# Boolean indexing - filter arrays based on conditions
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Create boolean mask
mask = arr > 5
print(f"Array: {arr}")
print(f"Mask (arr > 5): {mask}")
print(f"Filtered values: {arr[mask]}")
print()

# One line
print(f"Values > 5: {arr[arr > 5]}")
print(f"Even values: {arr[arr % 2 == 0]}")
print()

# 2D boolean indexing
arr_2d = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

print("2D array:")
print(arr_2d)
print(f"Values > 5:\n{arr_2d[arr_2d > 5]}")
print()

# ML Example: Filter predictions
predictions = np.array([0.2, 0.8, 0.4, 0.9, 0.3, 0.7])
confident = predictions[predictions > 0.5]
print(f"Confident predictions (>0.5): {confident}")
```

**Why This Matters:**

"Boolean indexing is HUGE in ML:
- Filter training data
- Remove outliers
- Apply thresholds to predictions
- Mask invalid values

You'll use this constantly!"

---

## Part 2: Array Operations (10 minutes)

### 2.1 Vectorized Operations (3 minutes)

**Instructor Script:**

"This is where NumPy shines! Operations on entire arrays without loops."

**Code Cell 7: Basic Vectorized Operations**

```python
# Element-wise arithmetic
arr = np.array([1, 2, 3, 4, 5])

print(f"Original: {arr}")
print(f"Add 10: {arr + 10}")
print(f"Multiply by 2: {arr * 2}")
print(f"Square: {arr ** 2}")
print(f"Square root: {np.sqrt(arr)}")
print()

# Operations between arrays
arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([10, 20, 30, 40])

print(f"arr1: {arr1}")
print(f"arr2: {arr2}")
print(f"Addition: {arr1 + arr2}")
print(f"Multiplication: {arr1 * arr2}")  # Element-wise, not matrix!
print(f"Division: {arr2 / arr1}")
print()

# 2D operations
matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])

print("Original matrix:")
print(matrix)
print("\nMultiply by 10:")
print(matrix * 10)
print("\nSquare each element:")
print(matrix ** 2)
```

**Code Cell 8: Mathematical Functions**

```python
# Universal functions (ufuncs)
arr = np.array([0, np.pi/2, np.pi])

print(f"Array: {arr}")
print(f"sin: {np.sin(arr)}")
print(f"cos: {np.cos(arr)}")
print(f"exp: {np.exp([1, 2, 3])}")
print(f"log: {np.log([1, 2, 3, 4, 5])}")
print()

# Aggregation functions
data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

print(f"Data: {data}")
print(f"Sum: {np.sum(data)}")
print(f"Mean: {np.mean(data)}")
print(f"Median: {np.median(data)}")
print(f"Std deviation: {np.std(data)}")
print(f"Min: {np.min(data)}")
print(f"Max: {np.max(data)}")
print(f"Arg max (index of max): {np.argmax(data)}")
print()

# 2D aggregations
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

print("Matrix:")
print(matrix)
print(f"Sum of all elements: {np.sum(matrix)}")
print(f"Sum of each column (axis=0): {np.sum(matrix, axis=0)}")
print(f"Sum of each row (axis=1): {np.sum(matrix, axis=1)}")
print(f"Mean of each column: {np.mean(matrix, axis=0)}")
```

**ML Example:**

```python
# Common ML scenario: normalize data
data = np.array([10, 20, 30, 40, 50])

# Z-score normalization
mean = np.mean(data)
std = np.std(data)
normalized = (data - mean) / std

print(f"Original: {data}")
print(f"Mean: {mean}, Std: {std}")
print(f"Normalized: {normalized}")
print(f"New mean: {np.mean(normalized):.10f}")  # ~0
print(f"New std: {np.std(normalized):.10f}")    # ~1
```

---

### 2.2 Broadcasting (4 minutes)

**Instructor Script:**

"Broadcasting is one of NumPy's most powerful features. It lets you operate on arrays of different shapes without copying data!"

**Code Cell 9: Broadcasting Basics**

```python
# Broadcasting: operating on arrays of different shapes

# Example 1: Array + Scalar
arr = np.array([1, 2, 3, 4])
print(f"Array: {arr}")
print(f"Array + 10: {arr + 10}")  # 10 is "broadcast" to match array shape
print()

# Example 2: 2D array + 1D array
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
row = np.array([10, 20, 30])

print("Matrix:")
print(matrix)
print(f"\nRow to add: {row}")
print("\nMatrix + row (broadcasts row to each matrix row):")
print(matrix + row)
print()

# Example 3: Column broadcasting
col = np.array([[10],
                [20],
                [30]])

print("Column to add:")
print(col)
print("\nMatrix + column (broadcasts column to each matrix column):")
print(matrix + col)
```

**Code Cell 10: Broadcasting Rules and Examples**

```python
# Broadcasting rules:
# 1. If arrays have different number of dimensions, prepend 1s to smaller shape
# 2. Arrays are compatible if dimensions are equal or one is 1
# 3. After broadcasting, each array behaves as if it had the larger shape

# Example: Different shapes
a = np.array([[1, 2, 3],
              [4, 5, 6]])  # Shape: (2, 3)
b = np.array([10, 20, 30])  # Shape: (3,)

print(f"a shape: {a.shape}")
print(f"b shape: {b.shape}")
print("\na:")
print(a)
print(f"\nb: {b}")
print("\na + b (b broadcasts to each row):")
print(a + b)
print()

# ML Example: Centering data (subtract mean)
data = np.array([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]])

column_means = np.mean(data, axis=0)  # Mean of each column
print("Data:")
print(data)
print(f"\nColumn means: {column_means}")
print("\nCentered data (subtract column means):")
centered = data - column_means  # Broadcasting!
print(centered)
print(f"\nNew column means: {np.mean(centered, axis=0)}")  # Should be ~0
```

**Code Cell 11: Broadcasting in Practice**

```python
# Real ML scenario: Normalizing a dataset

# Dataset: 5 samples, 3 features each
X = np.array([[1.0, 2.0, 3.0],
              [4.0, 5.0, 6.0],
              [7.0, 8.0, 9.0],
              [2.0, 4.0, 6.0],
              [3.0, 6.0, 9.0]])

print("Original data:")
print(X)
print(f"Shape: {X.shape}")
print()

# Min-Max normalization per feature
# Formula: (x - min) / (max - min)
min_vals = X.min(axis=0)  # Min of each column
max_vals = X.max(axis=0)  # Max of each column

print(f"Min values per feature: {min_vals}")
print(f"Max values per feature: {max_vals}")
print()

# Normalize (all operations broadcast!)
X_normalized = (X - min_vals) / (max_vals - min_vals)

print("Normalized data (0-1 range):")
print(X_normalized)
print(f"\nNew min per feature: {X_normalized.min(axis=0)}")
print(f"New max per feature: {X_normalized.max(axis=0)}")
```

**Why Broadcasting is Amazing:**

"Without broadcasting:
```python
# Would need loops (slow!)
for i in range(len(matrix)):
    for j in range(len(matrix[i])):
        result[i][j] = matrix[i][j] + row[j]
```

With broadcasting:
```python
result = matrix + row  # One line, super fast!
```

This is why NumPy is so powerful for ML!"

---

### 2.3 Reshaping Arrays (3 minutes)

**Code Cell 12: Reshaping Operations**

```python
# Reshaping - change array dimensions without changing data

# Create 1D array
arr = np.arange(12)
print(f"Original 1D array: {arr}")
print(f"Shape: {arr.shape}")
print()

# Reshape to 2D
arr_2d = arr.reshape(3, 4)  # 3 rows, 4 columns
print("Reshaped to (3, 4):")
print(arr_2d)
print()

# Reshape to different 2D
arr_2d_alt = arr.reshape(4, 3)  # 4 rows, 3 columns
print("Reshaped to (4, 3):")
print(arr_2d_alt)
print()

# Reshape to 3D
arr_3d = arr.reshape(2, 2, 3)  # 2 matrices of 2x3
print("Reshaped to (2, 2, 3):")
print(arr_3d)
print()

# Use -1 to let NumPy figure out dimension
arr_auto = arr.reshape(3, -1)  # 3 rows, NumPy figures out columns
print(f"Reshaped with -1 to (3, -1): shape {arr_auto.shape}")
print(arr_auto)
```

**Code Cell 13: Flatten and Transpose**

```python
# Flatten - convert to 1D
matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])

print("Original matrix:")
print(matrix)
print(f"Shape: {matrix.shape}")
print()

flattened = matrix.flatten()
print(f"Flattened: {flattened}")
print(f"Shape: {flattened.shape}")
print()

# Alternative: ravel (similar to flatten)
raveled = matrix.ravel()
print(f"Raveled: {raveled}")
print()

# Transpose - swap rows and columns
print("Original:")
print(matrix)
print("\nTransposed:")
print(matrix.T)
print(f"Original shape: {matrix.shape}")
print(f"Transposed shape: {matrix.T.shape}")
print()

# Adding dimensions
arr_1d = np.array([1, 2, 3])
print(f"1D array: {arr_1d}, shape: {arr_1d.shape}")

# Add axis to make row vector
row_vector = arr_1d[np.newaxis, :]  # or arr_1d.reshape(1, -1)
print(f"Row vector: {row_vector}, shape: {row_vector.shape}")

# Add axis to make column vector
col_vector = arr_1d[:, np.newaxis]  # or arr_1d.reshape(-1, 1)
print("Column vector:")
print(col_vector)
print(f"Shape: {col_vector.shape}")
```

**ML Example:**

```python
# Common in ML: reshape image data

# Image represented as 28x28 pixels
image = np.random.rand(28, 28)
print(f"Image shape: {image.shape}")

# Flatten for input to neural network
flattened_image = image.reshape(-1)  # or image.flatten()
print(f"Flattened shape: {flattened_image.shape}")  # (784,)

# Batch of images: 100 images of 28x28
batch = np.random.rand(100, 28, 28)
print(f"\nBatch of images shape: {batch.shape}")

# Flatten each image
batch_flattened = batch.reshape(100, -1)
print(f"Flattened batch shape: {batch_flattened.shape}")  # (100, 784)
```

---

## Part 3: Linear Algebra Operations (5 minutes)

**Instructor Script:**

"Linear algebra is the language of machine learning. Neural networks, deep learning - it's all linear algebra with NumPy!"

**Code Cell 14: Matrix Operations**

```python
# Matrix multiplication (dot product)
A = np.array([[1, 2],
              [3, 4]])
B = np.array([[5, 6],
              [7, 8]])

print("Matrix A:")
print(A)
print("\nMatrix B:")
print(B)
print()

# Element-wise multiplication (NOT matrix multiplication!)
print("Element-wise multiplication (A * B):")
print(A * B)
print()

# Matrix multiplication (proper)
print("Matrix multiplication (A @ B) or np.dot(A, B):")
print(A @ B)  # Python 3.5+
# Or: print(np.dot(A, B))
print()

# Vector dot product
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

dot_product = np.dot(v1, v2)
print(f"v1: {v1}")
print(f"v2: {v2}")
print(f"Dot product: {dot_product}")  # 1*4 + 2*5 + 3*6 = 32
```

**Code Cell 15: Neural Network Forward Pass Example**

```python
# Simulated neural network layer
# Formula: output = input @ weights + bias

# Input: 1 sample with 4 features
X = np.array([[1.0, 2.0, 3.0, 4.0]])  # Shape: (1, 4)

# Weights: 4 inputs to 3 neurons
W = np.array([[0.1, 0.2, 0.3],   # From input 1 to each neuron
              [0.4, 0.5, 0.6],   # From input 2
              [0.7, 0.8, 0.9],   # From input 3
              [1.0, 1.1, 1.2]])  # From input 4
# Shape: (4, 3)

# Biases: one per neuron
b = np.array([[0.1, 0.2, 0.3]])  # Shape: (1, 3)

print("Input (X):")
print(X)
print(f"Shape: {X.shape}")
print()

print("Weights (W):")
print(W)
print(f"Shape: {W.shape}")
print()

print("Biases (b):")
print(b)
print(f"Shape: {b.shape}")
print()

# Forward pass: linear transformation
Z = X @ W + b  # Matrix multiplication + broadcasting!
print("Output (Z = X @ W + b):")
print(Z)
print(f"Shape: {Z.shape}")
print()

# Apply activation function (ReLU: max(0, x))
A = np.maximum(0, Z)
print("After ReLU activation:")
print(A)
```

**Code Cell 16: Batch Processing**

```python
# Process multiple samples at once (batching)

# 3 samples (batch), 4 features each
X_batch = np.array([[1.0, 2.0, 3.0, 4.0],
                    [2.0, 3.0, 4.0, 5.0],
                    [3.0, 4.0, 5.0, 6.0]])  # Shape: (3, 4)

# Same weights and biases
W = np.array([[0.1, 0.2, 0.3],
              [0.4, 0.5, 0.6],
              [0.7, 0.8, 0.9],
              [1.0, 1.1, 1.2]])  # Shape: (4, 3)

b = np.array([[0.1, 0.2, 0.3]])  # Shape: (1, 3)

print("Batch input:")
print(X_batch)
print(f"Shape: {X_batch.shape}")
print()

# Process entire batch at once!
Z_batch = X_batch @ W + b  # Broadcasting handles the bias
print("Batch output:")
print(Z_batch)
print(f"Shape: {Z_batch.shape}")
print()

A_batch = np.maximum(0, Z_batch)
print("After ReLU:")
print(A_batch)
```

**Why This Matters:**

"This is EXACTLY how neural networks work:
1. Input data (X)
2. Multiply by weights (W)
3. Add bias (b)
4. Apply activation function
5. Repeat for each layer

NumPy makes this fast for millions of parameters!"

**Code Cell 17: Other Linear Algebra Operations**

```python
# Common linear algebra operations

# Matrix inverse
A = np.array([[1, 2],
              [3, 4]])
A_inv = np.linalg.inv(A)
print("Matrix A:")
print(A)
print("\nInverse of A:")
print(A_inv)
print("\nA @ A_inv (should be identity):")
print(A @ A_inv)
print()

# Eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"Eigenvalues: {eigenvalues}")
print("Eigenvectors:")
print(eigenvectors)
print()

# Determinant
det = np.linalg.det(A)
print(f"Determinant: {det}")
print()

# Norm (magnitude)
v = np.array([3, 4])
norm = np.linalg.norm(v)
print(f"Vector: {v}")
print(f"Norm (magnitude): {norm}")  # sqrt(3^2 + 4^2) = 5
```

---

## Part 4: Practical ML Examples (5 minutes)

**Code Cell 18: Complete ML Workflow Example**

```python
# Realistic ML data processing with NumPy

# Generate synthetic dataset
np.random.seed(42)
n_samples = 100
n_features = 5

# Features: random values
X = np.random.randn(n_samples, n_features)

# Labels: binary classification (0 or 1)
y = np.random.randint(0, 2, size=n_samples)

print("Dataset created:")
print(f"X shape: {X.shape} - {n_samples} samples, {n_features} features")
print(f"y shape: {y.shape} - {n_samples} labels")
print()

# 1. Data exploration
print("Data Statistics:")
print(f"Mean per feature: {X.mean(axis=0)}")
print(f"Std per feature: {X.std(axis=0)}")
print(f"Class distribution: {np.bincount(y)}")
print()

# 2. Standardize features (mean=0, std=1)
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_standardized = (X - X_mean) / X_std

print("After standardization:")
print(f"New mean per feature: {X_standardized.mean(axis=0)}")
print(f"New std per feature: {X_standardized.std(axis=0)}")
print()

# 3. Train-test split (80-20)
split_idx = int(0.8 * n_samples)
X_train = X_standardized[:split_idx]
X_test = X_standardized[split_idx:]
y_train = y[:split_idx]
y_test = y[split_idx:]

print("Data split:")
print(f"Training: {X_train.shape}, {y_train.shape}")
print(f"Testing: {X_test.shape}, {y_test.shape}")
print()

# 4. Simple linear model weights (random initialization)
n_outputs = 1
weights = np.random.randn(n_features, n_outputs) * 0.01
bias = np.zeros((1, n_outputs))

print("Model initialized:")
print(f"Weights shape: {weights.shape}")
print(f"Bias shape: {bias.shape}")
print()

# 5. Make predictions
predictions = X_test @ weights + bias
print("Sample predictions (raw scores):")
print(predictions[:5].flatten())
print()

# 6. Apply threshold for classification
threshold = 0.0
binary_predictions = (predictions > threshold).astype(int).flatten()
print("Sample binary predictions:")
print(binary_predictions[:10])
print(f"Actual labels:")
print(y_test[:10])
print()

# 7. Calculate accuracy
accuracy = np.mean(binary_predictions == y_test)
print(f"Accuracy: {accuracy:.2%}")
```

**Code Cell 19: Image Processing Example**

```python
# Working with image data (common in CV)

# Simulate a grayscale image (28x28 pixels)
image = np.random.randint(0, 256, size=(28, 28))
print("Grayscale image shape:", image.shape)
print(f"Min pixel value: {image.min()}")
print(f"Max pixel value: {image.max()}")
print()

# Normalize pixel values to [0, 1]
image_normalized = image / 255.0
print(f"Normalized min: {image_normalized.min()}")
print(f"Normalized max: {image_normalized.max()}")
print()

# Flatten for neural network input
image_flat = image_normalized.flatten()
print(f"Flattened shape: {image_flat.shape}")
print()

# Batch of images
batch_size = 32
batch_images = np.random.randint(0, 256, size=(batch_size, 28, 28))
print(f"Batch shape: {batch_images.shape}")

# Normalize and flatten entire batch
batch_normalized = batch_images / 255.0
batch_flat = batch_normalized.reshape(batch_size, -1)
print(f"Processed batch shape: {batch_flat.shape}")
print()

# Add channel dimension for RGB (color images)
rgb_image = np.random.randint(0, 256, size=(28, 28, 3))  # Height, Width, Channels
print(f"RGB image shape: {rgb_image.shape}")
```

**Code Cell 20: Statistical Operations**

```python
# Common statistical operations in ML

# Create sample data
data = np.random.randn(1000, 10)  # 1000 samples, 10 features

# Correlation matrix (relationship between features)
correlation = np.corrcoef(data.T)  # Transpose so features are rows
print("Correlation matrix shape:", correlation.shape)
print("Sample correlations:")
print(correlation[:3, :3])
print()

# Covariance matrix
covariance = np.cov(data.T)
print("Covariance matrix shape:", covariance.shape)
print()

# Percentiles
print("Data statistics:")
print(f"25th percentile: {np.percentile(data, 25, axis=0)[:3]}")
print(f"50th percentile (median): {np.percentile(data, 50, axis=0)[:3]}")
print(f"75th percentile: {np.percentile(data, 75, axis=0)[:3]}")
print()

# Outlier detection using Z-scores
z_scores = np.abs((data - data.mean(axis=0)) / data.std(axis=0))
outliers = (z_scores > 3).any(axis=1)  # Any feature > 3 std devs
print(f"Number of outliers: {outliers.sum()} out of {len(data)}")
```

---

## Closing & NumPy Cheat Sheet (2 minutes)

**Code Cell 21: NumPy Quick Reference**

```python
# NumPy Essentials - Quick Reference

print("="*60)
print("NUMPY QUICK REFERENCE FOR AI/ML")
print("="*60)
print()

print("1. ARRAY CREATION")
print("  np.array([1,2,3])           # From list")
print("  np.zeros((3,4))             # Array of zeros")
print("  np.ones((2,3))              # Array of ones")
print("  np.arange(0,10,2)           # Range with step")
print("  np.random.rand(3,3)         # Random [0,1)")
print("  np.random.randn(3,3)        # Random normal")
print()

print("2. ARRAY OPERATIONS")
print("  arr + 10                    # Add scalar (broadcast)")
print("  arr1 + arr2                 # Element-wise addition")
print("  arr ** 2                    # Element-wise power")
print("  arr @ matrix                # Matrix multiplication")
print()

print("3. AGGREGATIONS")
print("  np.sum(arr)                 # Sum all elements")
print("  np.mean(arr, axis=0)        # Mean per column")
print("  np.std(arr)                 # Standard deviation")
print("  np.max(arr), np.min(arr)    # Max, min")
print()

print("4. INDEXING & SLICING")
print("  arr[0]                      # First element")
print("  arr[1:4]                    # Slice")
print("  arr[arr > 5]                # Boolean indexing")
print("  matrix[0, :]                # First row")
print("  matrix[:, 1]                # Second column")
print()

print("5. RESHAPING")
print("  arr.reshape(3, 4)           # Change shape")
print("  arr.flatten()               # To 1D")
print("  arr.T                       # Transpose")
print()

print("6. ML COMMON PATTERNS")
print("  (X - X.mean()) / X.std()    # Standardization")
print("  (X - X.min()) / (X.max()-X.min())  # Min-max norm")
print("  X @ W + b                   # Neural network layer")
print("  np.maximum(0, Z)            # ReLU activation")
print()

print("Remember: NumPy is FAST because it's vectorized!")
print("Avoid Python loops - use NumPy operations instead!")
print("="*60)
```

**Instructor Script:**

"Fantastic! You've learned NumPy essentials! Let's recap:

**What We Covered:**
- ✅ Why NumPy is 50-100x faster than Python lists
- ✅ Creating and manipulating arrays
- ✅ Vectorized operations (no loops!)
- ✅ Broadcasting (different shapes)
- ✅ Reshaping and indexing
- ✅ Linear algebra for neural networks
- ✅ Real ML workflows

**Key Takeaways:**
1. **NumPy is the foundation** - everything in Python ML uses it
2. **Think in arrays** - not loops!
3. **Broadcasting saves memory** - operates on different shapes
4. **Vectorization is fast** - let NumPy do the work
5. **Linear algebra = ML** - matrix operations everywhere

**The NumPy Mindset:**

❌ Don't think: 'Loop through each element'
✓ Think: 'What operation on the whole array?'

Example:
```python
# Slow Python way
result = []
for x in data:
    result.append(x ** 2)

# Fast NumPy way
result = data ** 2  # Done!
```

**Next Up:**

NumPy gives us fast arrays, but working with labeled data (like spreadsheets) is still tedious. That's where **Pandas** comes in!

Pandas = NumPy + Labels + SQL-like operations

Ready to make data manipulation even easier? Let's go! 🐼"

---

## Practice Exercises (For Students)

**Exercise 1: Array Manipulation**
```python
# Create a 5x5 array of random integers from 1-100
# Calculate: mean, standard deviation, max, min
# Find all values > 50
# Normalize to range [0, 1]

# Your code here
```

**Solution:**
```python
arr = np.random.randint(1, 101, size=(5, 5))
print("Array:")
print(arr)
print(f"\nMean: {np.mean(arr):.2f}")
print(f"Std: {np.std(arr):.2f}")
print(f"Max: {np.max(arr)}")
print(f"Min: {np.min(arr)}")
print(f"\nValues > 50:\n{arr[arr > 50]}")

normalized = (arr - arr.min()) / (arr.max() - arr.min())
print(f"\nNormalized:\n{normalized}")
```

**Exercise 2: Neural Network Layer**
```python
# Implement a neural network layer
# Input: 10 samples, 4 features
# Output: 3 neurons
# Use random weights and zero biases
# Apply ReLU activation

# Your code here
```

**Solution:**
```python
X = np.random.randn(10, 4)
W = np.random.randn(4, 3) * 0.01
b = np.zeros((1, 3))

Z = X @ W + b
A = np.maximum(0, Z)

print(f"Input shape: {X.shape}")
print(f"Output shape: {A.shape}")
print(f"Output:\n{A}")
```

---

## Common Student Questions & Answers

**Q: "When should I use NumPy vs. Python lists?"**
A: "Use NumPy for numerical data (especially large arrays). Use Python lists for small collections of mixed types. In ML, you'll use NumPy 95% of the time for data."

**Q: "What's the difference between * and @ for arrays?"**
A: "`*` is element-wise multiplication. `@` is matrix multiplication (dot product). For ML, you usually want `@` for layer operations."

**Q: "Why does my array shape sometimes have a trailing comma like (5,)?"**
A: "That indicates a 1D array with 5 elements. Compare: (5,) is 1D, (5,1) is 2D column vector, (1,5) is 2D row vector."

**Q: "How do I remember when to use axis=0 vs axis=1?"**
A: "axis=0 operates down columns (along rows), axis=1 operates across rows (along columns). I remember: axis=0 makes columns disappear, axis=1 makes rows disappear."

**Q: "Broadcasting seems confusing. How do I know if it will work?"**
A: "Arrays are compatible if dimensions are equal or one is 1. NumPy will tell you if it fails! Start simple and experiment. Most ML operations you need broadcast naturally."

**Q: "Do I need to understand all the linear algebra?"**
A: "Not deeply! Understand that ML is matrix operations, and NumPy does them fast. You'll build intuition as you use it. Focus on patterns, not proofs."

---

## Instructor Notes

**Pacing:**
- This is content-heavy (30 min for all NumPy essentials)
- Focus on most-used features, not exhaustive coverage
- Live code everything - don't just show slides
- Encourage experimentation

**Key Messages:**
1. NumPy is fast because it's vectorized
2. Think in arrays, not loops
3. Broadcasting is powerful once you get it
4. This is the foundation of ALL Python ML

**Teaching Tips:**
- Start with the speed comparison - hooks students
- Use lots of visual analogies for reshaping
- Show side-by-side: loop vs. vectorized
- Relate everything back to ML use cases
- Make mistakes and debug together

**Common Pitfalls:**
- Students reverting to loop thinking
- Confusion about array shapes and dimensions
- Broadcasting rules seeming arbitrary
- Forgetting axis parameter on aggregations
- Matrix multiplication (@) vs element-wise (*)

**Watch For:**
- Glazed looks during broadcasting (slow down, use visuals)
- Students not typing along (encourage participation)
- Shape errors (teachable moments!)
- Questions about "why NumPy" (reinforce the speed)

---

## Transition to Next Topic

**Instructor Script:**

"Incredible work! You now understand NumPy - the engine under the hood of Python ML.

But let's be honest: working with raw NumPy arrays is still low-level. Imagine you have a dataset with:
- Names (strings)
- Ages (integers)  
- Salaries (floats)
- Departments (categories)

NumPy treats everything as numbers. You'd need to manage column names, handle different types, filter rows... it gets messy!

Enter **Pandas**! 🐼

Pandas gives you:
- DataFrames (like Excel, but programmatic)
- Column names (no more 'column 3')
- Easy filtering and grouping (SQL-like)
- Built on NumPy (so still fast!)

Think: NumPy is assembly language, Pandas is Python!

Ready to make data manipulation elegant? Let's learn Pandas!"

---

**Next Topic:** Pandas for Data Manipulation (20 minutes)
