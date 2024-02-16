## 1. Boolean Functions

1. Easy. Need to 3 produce boolean function for inputs.
2. With new data how many error each of your boolean func produce.
3. Is linearly separably.

## 2. Feature Transformation

1. Given a func fr defined by radius r. This func is not linearly separable.
2. Need to map this Phi func 2D space and make it linearly separable on new space. Showing the proof.

Given: This Phi func should not have radius.

## 3. Mistake Bound Model of Learning

1. Concept class.
2. Concept class and CON, Halving algorithms.

## 4. Perceptron

## Answers

### 1. Boolean Functions

1. Three boolean functions

- x2 ^ x4
- x3 ^ x4
- x2 ^ x3 ^ x4

2. Adding more data points

- 1
- 1
- 2

3. Is linearly separable?

- Yes linearly separable. y = x4

### 2. Feature transformations

### 3. Mistake Bound Model of Learning

1a. n. Every instance of z has a unique function. Thus, the number of instance (n) of z is possible member of concept class.

1b.

f0 f1 f2 f3
1 0 0 0
random chose fo and predict 1. no mistake.
0 1 0 0
random chose f1 and predict 1. no mistake.

0 1 0 0
random chose f0 and predicts 0. mistake. Eliminates f0 f2 f3.

Solution is with one mistake all the functions will removed.

Among all C1 there is one function if that feature (x) is present then it is the function fz(x).

An algo is mistake bound if makes at most polynomial mistake.

2a. CON: Picks a random func and eliminates all the func that does not satisfy.

|c2| - 1. If it never picks the right feature.

2b. Halving:

log n
