---
title: K-Means tricks for fun and profit
published: true
description: K-Means is an interesting, simple and pretty intuitive algorithm. It turns out it can do more than just clustering.
tags: machine learning, clustering, artificial inteligence, k-means, svm, kernel trick
layout: post
date:   2021-06-18 00:10:00 +0200
categories: posts
permalink: /posts/2021-06-18-kmeans-trick
comments: true
---

# Prologue

This will be a small post, but an interesting one nevertheless.

We all know, or at least heard about the K-Means clustering algorithm. I remember when I first found out about it it seemed pretty nice, an unsupervised learning algorithm that is fairly intuitive and mostly works well. But then in time the interest faded away, while understanding it's limitations. And then, I found out about a few neet tricks of how to use K-Means. So here it goes.

# K-Means as a source of new features

We'll start with a semi-obvious one, that is, using K-Means predictions as new features for your dataset.
```python
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

import numpy as np

X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=17)

svm = LinearSVC(random_state=17)
svm.fit(X_train, y_train)
svm.score(X_test, y_test) # should be ~0.93
```

Now let's try using K-Means cluster assignments as features

```python
# imports from the example above

svm = LinearSVC(random_state=17)
kmeans = KMeans(n_clusters=3, random_state=17)
X_clusters = kmeans.fit_predict(X_train).reshape(-1, 1)

svm.fit(np.hstack([X_train, X_clusters]), y_train)
svm.score(np.hstack([X_test, kmeans.predict(X_test).reshape(-1, 1)]), y_test) # should be ~0.937
```

<!-- insert it ain't much but it's honest work TK -->

These features are categorical, but we can ask the model to output distances to all the centroids, thus obtaining (hopefully) more informative features.

```python
# imports from the example above

svm = LinearSVC(random_state=17)
kmeans = KMeans(n_clusters=3, random_state=17)
X_clusters = kmeans.fit_transform(X_train)
#                       ^^^^^^^^^
#                       Notice the `transform` instead of `predict`
# Scikit-learn supports this method as early as version 0.15

svm.fit(np.hstack([X_train, X_clusters]), y_train)
svm.score(np.hstack([X_test, kmeans.transform(X_test)]), y_test) # should be ~0.727
```

Well, what if we use only the distances to the cluster means as features, will it work then?

```python
# imports from the example above

svm = LinearSVC(random_state=17)
kmeans = KMeans(n_clusters=3, random_state=17)
X_clusters = kmeans.fit_transform(X_train)

svm.fit(X_clusters, y_train)
svm.score(kmeans.transform(X_test), y_test) # should be ~0.951
```

Much better, because previously the features were corellated with the distances. You probably know that we want our features independent of each other for a reason.
With this example, you can see that we can actually use KMeans as a way to do dimensionality reduction. Neat.

So far so good. But the piece de resistance is yet to be showed.

# K-Means as a substitute for the kernel trick

You heard me right. You can, for example define _more_ centroids for the K-Means algorithm to fit than there are features, much more.

```python
# imports from the example above

svm = LinearSVC(random_state=17)
kmeans = KMeans(n_clusters=250, random_state=17)
X_clusters = kmeans.fit_transform(X_train)

svm.fit(X_clusters, y_train)
svm.score(kmeans.transform(X_test), y_test) # should be ~0.944
```

Well, not as good, but pretty decent. In practice, the greates benefit of this approach is when you have a lot of data. Also predictive performance-wise your mileage may vary, I for one had run this method with `n_clusters=1000` and it worked better than only with a few clusters.

SVMs are known to be slow to train on big datasets. Really slow, been there, done that. That's why for example there are numerous techniques to approximate the kernel trick with much less computational resources.

By the way, let's compare how this K-Means trick will do against classic SVM and some alternative kerner approximation methods.

The code below is inspired by [these](https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_kernel_approximation.html) [two](https://scikit-learn.org/stable/auto_examples/kernel_approximation/plot_scalable_poly_kernels.html) scikit-learn examples.


```python
# adjust this example TK

import matplotlib.pyplot as plt
import numpy as np
from time import time

from sklearn.datasets import load_digits, load_breast_cancer
from sklearn.svm import LinearSVC, SVC
from sklearn import pipeline
from sklearn.kernel_approximation import RBFSampler, Nystroem, PolynomialCountSketch
from sklearn.preprocessing import MinMaxScaler, Normalizer
from sklearn.model_selection import train_test_split
from sklearn.cluster import MiniBatchKMeans



mm = pipeline.make_pipeline(MinMaxScaler(), Normalizer())

X, y = load_breast_cancer(return_X_y=True)
X = mm.fit_transform(X)

data_train, data_test, targets_train, targets_test = train_test_split(X, y, random_state=17)


# Create a classifier: a support vector classifier
kernel_svm = SVC(gamma=.2, random_state=17)
linear_svm = LinearSVC(random_state=17)

# create pipeline from kernel approximation
# and linear svm
feature_map_fourier = RBFSampler(gamma=.2, random_state=17)
feature_map_nystroem = Nystroem(gamma=.2, random_state=17)
feature_map_poly_cm = PolynomialCountSketch(degree=4, random_state=17)
feature_map_kmeans = MiniBatchKMeans(random_state=17)
fourier_approx_svm = pipeline.Pipeline([("feature_map", feature_map_fourier),
                                        ("svm", LinearSVC(random_state=17))])

nystroem_approx_svm = pipeline.Pipeline([("feature_map", feature_map_nystroem),
                                        ("svm", LinearSVC(random_state=17))])

poly_cm_approx_svm = pipeline.Pipeline([("feature_map", feature_map_poly_cm),
                                        ("svm", LinearSVC(random_state=17))])

kmeans_approx_svm = pipeline.Pipeline([("feature_map", feature_map_kmeans),
                                        ("svm", LinearSVC(random_state=17))])


# fit and predict using linear and kernel svm:
kernel_svm_time = time()
kernel_svm.fit(data_train, targets_train)
kernel_svm_score = kernel_svm.score(data_test, targets_test)
kernel_svm_time = time() - kernel_svm_time

linear_svm_time = time()
linear_svm.fit(data_train, targets_train)
linear_svm_score = linear_svm.score(data_test, targets_test)
linear_svm_time = time() - linear_svm_time

sample_sizes = 30 * np.arange(1, 10)
fourier_scores = []
nystroem_scores = []
poly_cm_scores = []
kmeans_scores = []

fourier_times = []
nystroem_times = []
poly_cm_times = []
kmeans_times = []

for D in sample_sizes:
    fourier_approx_svm.set_params(feature_map__n_components=D)
    nystroem_approx_svm.set_params(feature_map__n_components=D)
    poly_cm_approx_svm.set_params(feature_map__n_components=D)
    kmeans_approx_svm.set_params(feature_map__n_clusters=D)
    start = time()
    nystroem_approx_svm.fit(data_train, targets_train)
    nystroem_times.append(time() - start)

    start = time()
    fourier_approx_svm.fit(data_train, targets_train)
    fourier_times.append(time() - start)

    start = time()
    poly_cm_approx_svm.fit(data_train, targets_train)
    poly_cm_times.append(time() - start)

    start = time()
    kmeans_approx_svm.fit(data_train, targets_train)
    kmeans_times.append(time() - start)

    fourier_score = fourier_approx_svm.score(data_test, targets_test)
    fourier_scores.append(fourier_score)
    nystroem_score = nystroem_approx_svm.score(data_test, targets_test)
    nystroem_scores.append(nystroem_score)
    poly_cm_score = poly_cm_approx_svm.score(data_test, targets_test)
    poly_cm_scores.append(poly_cm_score)
    kmeans_score = kmeans_approx_svm.score(data_test, targets_test)
    kmeans_scores.append(kmeans_score)

# plot the results:
plt.figure(figsize=(16, 4))
accuracy = plt.subplot(121)
# second y axis for timings
timescale = plt.subplot(122)

accuracy.plot(sample_sizes, nystroem_scores, label="Nystroem approx. kernel")
timescale.plot(sample_sizes, nystroem_times, '--',
               label='Nystroem approx. kernel')

accuracy.plot(sample_sizes, fourier_scores, label="Fourier approx. kernel")
timescale.plot(sample_sizes, fourier_times, '--',
               label='Fourier approx. kernel')

accuracy.plot(sample_sizes, poly_cm_scores, label="Polynomial Count-Min approx. kernel")
timescale.plot(sample_sizes, poly_cm_times, '--',
               label='Polynomial Count-Min approx. kernel')

accuracy.plot(sample_sizes, kmeans_scores, label="K-Means approx. kernel")
timescale.plot(sample_sizes, kmeans_times, '--',
               label='K-Means approx. kernel')

# horizontal lines for exact rbf and linear kernels:
accuracy.plot([sample_sizes[0], sample_sizes[-1]],
              [linear_svm_score, linear_svm_score], label="linear svm")
timescale.plot([sample_sizes[0], sample_sizes[-1]],
               [linear_svm_time, linear_svm_time], '--', label='linear svm')

accuracy.plot([sample_sizes[0], sample_sizes[-1]],
              [kernel_svm_score, kernel_svm_score], label="rbf svm")
timescale.plot([sample_sizes[0], sample_sizes[-1]],
               [kernel_svm_time, kernel_svm_time], '--', label='rbf svm')


# legends and labels
accuracy.set_title("Classification accuracy")
timescale.set_title("Training times")
accuracy.set_xlim(sample_sizes[0], sample_sizes[-1])
accuracy.set_xticks(())
accuracy.set_ylim(np.min(fourier_scores), 1)
timescale.set_xlabel("Sampling steps = transformed feature dimension")
accuracy.set_ylabel("Classification accuracy")
timescale.set_ylabel("Training time in seconds")
accuracy.legend(loc='best')
timescale.legend(loc='best')
plt.tight_layout()
plt.show()
```

<!-- insert plot TK -->

Even if it's the slowest, K-Means as an approximation of the RBF Kernel is still a nice option, because you can use `.partial_fit` and combining this with a model that has `.partial_fit` too, like a `PassiveAggressiveClassifier` one can create a pretty interesting solution.


<!-- https://towardsdatascience.com/k-means-clustering-from-scratch-6a9d19cafc25 -->
<!-- TK Spherical k-means -->
<!-- http://www.jcomputers.us/vol8/jcp0810-25.pdf
     https://jakevdp.github.io/PythonDataScienceHandbook/05.11-k-means.html
     https://sites.google.com/site/dataclusteringalgorithms/kernel-k-means-clustering-algorithm
     https://datascience.stackexchange.com/questions/24324/how-to-use-k-means-outputs-extracted-features-as-svm-inputs
  -->


# Epilogue

TK