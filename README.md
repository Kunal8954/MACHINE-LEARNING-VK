# 🧠 Neural Network Loss Functions & Optimizers

A practical guide to choosing the correct **activation function, loss function, optimizer, and evaluation metric** for Machine Learning and Deep Learning problems.

---

## 1. The Basic Pipeline

A neural network usually follows:

```text
Input
  ↓
Hidden Layers
  ↓
Activation
  ↓
Output Layer
  ↓
Loss Function
  ↓
Optimizer
  ↓
Backpropagation
  ↓
Updated Weights
```

### Remember:

* **Activation** → model output ko transform karta hai.
* **Loss** → prediction kitni wrong hai, measure karta hai.
* **Optimizer** → loss ko reduce karne ke liye weights update karta hai.
* **Metric** → human-readable performance measure karta hai.

---

# 2. Classification — Which Loss?

## 🟢 Binary Classification

### Problem

Exactly **2 classes**:

```text
Spam / Not Spam
Pass / Fail
Disease / No Disease
Cat / Not Cat
0 / 1
```

### Output

```python
Dense(1, activation="sigmoid")
```

### Loss

```python
loss="binary_crossentropy"
```

### Example

```python
model = keras.Sequential([
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
```

### Labels

```python
y = [0, 1, 0, 1, 1]
```

### Rule

```text
Binary Classification
        ↓
Sigmoid
        ↓
Binary Cross-Entropy
```

---

# 3. Multiclass Classification

Suppose:

```text
Cat
Dog
Horse
```

There are 3 mutually exclusive classes.

---

## 🟢 Multiclass + Integer Labels

Use **Sparse Categorical Cross-Entropy**.

### Labels

```python
Cat   → 0
Dog   → 1
Horse → 2
```

Example:

```python
y = [0, 2, 1, 0, 2]
```

### Output

```python
Dense(3, activation="softmax")
```

### Loss

```python
loss="sparse_categorical_crossentropy"
```

### Example

```python
model = keras.Sequential([
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(3, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
```

### Rule

```text
Multiclass
   +
Integer labels
        ↓
Softmax
        ↓
Sparse Categorical Cross-Entropy
```

---

# 4. Multiclass + One-Hot Labels

Same problem:

```text
Cat
Dog
Horse
```

But labels are one-hot encoded.

```text
Cat   → [1,0,0]
Dog   → [0,1,0]
Horse → [0,0,1]
```

Use:

```python
Dense(3, activation="softmax")
```

and:

```python
loss="categorical_crossentropy"
```

### Example

```python
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
```

### Rule

```text
Multiclass
   +
One-Hot labels
        ↓
Softmax
        ↓
Categorical Cross-Entropy
```

---

# 5. Multilabel Classification

This is different from multiclass.

### Multiclass

Only **one class** can be correct:

```text
Image → Cat
```

### Multilabel

Multiple labels can be true simultaneously:

```text
Image → [Dog, Animal, Outdoor]
```

Suppose labels are:

```text
Dog
Cat
Car
Person
```

An image could contain:

```text
[1, 0, 1, 1]
```

### Output

```python
Dense(4, activation="sigmoid")
```

### Loss

```python
loss="binary_crossentropy"
```

Because each label is an independent binary decision.

### Rule

```text
Multilabel
    ↓
Multiple Sigmoids
    ↓
Binary Cross-Entropy
```

---

# 6. Regression

Regression predicts a **continuous number**.

Examples:

```text
House price
Temperature
Salary
Stock value
Age
Sales
```

### Output

Usually:

```python
Dense(1)
```

No sigmoid/softmax is generally needed for an unconstrained continuous target.

---

## MSE — Mean Squared Error

```python
loss="mse"
```

Formula:

$$
MSE = \frac{1}{n}\sum(y-\hat y)^2
$$

Large errors receive stronger punishment because the error is squared.

Good default for many regression problems.

---

## MAE — Mean Absolute Error

```python
loss="mae"
```

Formula:

$$
MAE = \frac{1}{n}\sum |y-\hat y|
$$

MAE is generally less sensitive to large outliers than MSE.

### Rule

```text
Regression
   ↓
Linear output
   ↓
MSE / MAE
```

---

# 7. Huber Loss

Huber loss combines ideas from MSE and MAE.

It behaves roughly like:

```text
Small errors → MSE-like
Large errors → MAE-like
```

Useful when your regression data contains outliers but you still want smooth optimization behavior.

```python
loss = keras.losses.Huber()
```

---

# 8. Quick Loss Decision Tree

```text
                    What are you predicting?
                            │
              ┌─────────────┴─────────────┐
              ↓                           ↓
         Classification               Regression
              │                           │
       ┌──────┴──────┐              ┌─────┴─────┐
       ↓             ↓              ↓           ↓
    Binary       Multiple        MSE/MAE      Huber
       │          classes
       ↓             │
     Sigmoid    ┌────┴────┐
       ↓        ↓         ↓
     BCE     Integer    One-hot
                ↓         ↓
             Sparse      CCE
              CCE
```

---

# 9. What Is an Optimizer?

Loss tells us:

> "Tumhari prediction kitni wrong hai."

Optimizer decides:

> "Weights ko kis direction mein change karna hai?"

Training:

```text
Prediction
    ↓
Loss
    ↓
Gradient
    ↓
Optimizer
    ↓
Update Weights
    ↓
Better Prediction
```

---

# 10. Adam — Best Starting Point

For most beginner-to-intermediate neural network projects:

```python
optimizer="adam"
```

is an excellent starting choice.

Example:

```python
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
```

Adam combines adaptive learning-rate ideas with momentum-like behavior.

### Use Adam when:

* Starting a new neural network
* Classification
* Regression
* CNN
* Basic NLP
* General deep learning experiments

---

# 11. SGD

SGD = Stochastic Gradient Descent.

```python
optimizer=keras.optimizers.SGD(
    learning_rate=0.01
)
```

SGD can work extremely well, but usually requires more careful tuning.

Common parameters:

```python
learning_rate
momentum
```

Example:

```python
optimizer=keras.optimizers.SGD(
    learning_rate=0.01,
    momentum=0.9
)
```

---

# 12. RMSprop

RMSprop adapts the learning rate based on recent gradient information.

```python
optimizer="rmsprop"
```

It has historically been useful in recurrent neural networks and some noisy/non-stationary optimization settings.

---

# 13. AdamW

AdamW is a variant of Adam that handles weight decay separately from the gradient update.

```python
optimizer = keras.optimizers.AdamW(
    learning_rate=0.001,
    weight_decay=1e-4
)
```

It is a strong choice when regularization/generalization matters, especially in modern deep-learning workflows.

---

# 14. Optimizer Cheat Sheet

| Optimizer | Starting Point | Main Idea                                      |
| --------- | -------------- | ---------------------------------------------- |
| Adam      | ⭐⭐⭐⭐⭐          | Adaptive + momentum-like                       |
| AdamW     | ⭐⭐⭐⭐⭐          | Adam + decoupled weight decay                  |
| SGD       | ⭐⭐⭐⭐           | Simple, strong with tuning                     |
| RMSprop   | ⭐⭐⭐            | Adaptive learning rate                         |
| Adagrad   | ⭐⭐             | Adaptive learning rate, can decay aggressively |

### Beginner rule:

```text
Start → Adam
        ↓
Need stronger regularization?
        ↓
Try AdamW
        ↓
Want careful/tuned training?
        ↓
Try SGD + Momentum
```

---

# 15. Learning Rate

Optimizer choose karna enough nahi hai.

Learning rate is extremely important.

Example:

```python
Adam(learning_rate=0.001)
```

### Too high

```text
0.1
```

Possible behavior:

```text
Loss ↓
Loss ↑
Loss ↓
Loss ↑
```

Training unstable ho sakti hai.

### Too low

```text
0.000001
```

Training extremely slow ho sakti hai.

### Common starting point

```python
learning_rate=0.001
```

But it is **not a universal rule**. The appropriate value depends on architecture, optimizer, batch size, data, and training setup.

---

# 16. Activation + Loss + Optimizer Cheat Sheet

## Binary Classification

```python
Dense(1, activation="sigmoid")
```

```python
loss="binary_crossentropy"
optimizer="adam"
```

---

## Multiclass Classification

```python
Dense(num_classes, activation="softmax")
```

Integer labels:

```python
loss="sparse_categorical_crossentropy"
```

One-hot labels:

```python
loss="categorical_crossentropy"
```

Optimizer:

```python
optimizer="adam"
```

---

## Multilabel Classification

```python
Dense(num_labels, activation="sigmoid")
```

```python
loss="binary_crossentropy"
optimizer="adam"
```

---

## Regression

```python
Dense(1)
```

Usually:

```python
loss="mse"
```

or:

```python
loss="mae"
```

or:

```python
loss=keras.losses.Huber()
```

Optimizer:

```python
optimizer="adam"
```

---

# 17. Complete Examples

## Binary Classification

```python
model = keras.Sequential([
    keras.layers.Input(shape=(10,)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
```

---

## Multiclass Classification

```python
model = keras.Sequential([
    keras.layers.Input(shape=(10,)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(5, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
```

Here:

```text
5 → number of classes
```

---

## Regression

```python
model = keras.Sequential([
    keras.layers.Input(shape=(10,)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)
```

---

# 18. Most Important Table

| Problem                     | Output Layer | Activation   | Loss              | Starting Optimizer |
| --------------------------- | ------------ | ------------ | ----------------- | ------------------ |
| Binary classification       | `Dense(1)`   | Sigmoid      | Binary CE         | Adam               |
| Multiclass + integer labels | `Dense(N)`   | Softmax      | Sparse CCE        | Adam               |
| Multiclass + one-hot labels | `Dense(N)`   | Softmax      | Categorical CE    | Adam               |
| Multilabel classification   | `Dense(N)`   | Sigmoid      | Binary CE         | Adam               |
| Regression                  | `Dense(1)`   | Usually none | MSE / MAE / Huber | Adam               |

---

# 19. 🧠 Golden Rules

### Rule 1

```text
2 mutually exclusive classes
→ Sigmoid + Binary Cross-Entropy
```

### Rule 2

```text
N mutually exclusive classes
→ Softmax + Cross-Entropy
```

### Rule 3

```text
N classes + integer labels
→ Sparse Categorical CE
```

### Rule 4

```text
N classes + one-hot labels
→ Categorical CE
```

### Rule 5

```text
Multiple independent labels
→ Sigmoid + Binary CE
```

### Rule 6

```text
Continuous number
→ Regression loss
```

### Rule 7

```text
New neural network
→ Start with Adam
```

Then tune based on validation performance.

---

# 20. Final Mental Model

```text
                WHAT IS THE TASK?
                       │
          ┌────────────┴────────────┐
          ↓                         ↓
    Classification              Regression
          │                         │
     ┌────┼────┐                    ↓
     ↓    ↓    ↓                  Dense(1)
   Binary Multi Multi-label          ↓
     ↓    ↓      ↓              MSE/MAE/Huber
  Sigmoid Softmax Sigmoid
     ↓      ↓       ↓
    BCE   CCE      BCE
```

And optimizer:

```text
Loss
  ↓
Gradient
  ↓
Optimizer
  ↓
Weight Update
  ↓
Better Model
```

## 🚀 Recommended learning order

For your TensorFlow journey:

```text
1. Activation Functions
        ↓
2. Loss Functions
        ↓
3. Optimizers
        ↓
4. Forward Propagation
        ↓
5. Backpropagation
        ↓
6. Gradient Descent
        ↓
7. Learning Rate
        ↓
8. Epochs & Batch Size
        ↓
9. Overfitting / Underfitting
        ↓
10. Regularization
        ↓
11. Dropout
        ↓
12. Batch Normalization
        ↓
13. Callbacks
        ↓
14. CNN
        ↓
15. Transfer Learning
        ↓
16. NLP / Transformers
```

**The 4 things you should be able to decide automatically are:**

```text
Problem
   ↓
Output Layer
   ↓
Activation
   ↓
Loss
   ↓
Optimizer
```

Once this becomes automatic, building neural networks becomes much easier.
