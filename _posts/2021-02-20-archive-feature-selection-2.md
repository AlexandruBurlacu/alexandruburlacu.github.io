---
title: Feature Selection for Machine Learning (2/2)
published: true
# description: An overview of methods to speed up training of convolutional neural networks without significant impact on the accuracy.
tags: scikit-learn, ai, machinelearning
layout: post
date:   2019-09-15 23:20:00 +0200
categories: posts
---

{% gtm body %}

_Originally published on Medium, quite some time ago, [here](https://medium.com/@alexburlacu1996/feature-selection-for-machine-learning-2-2-1a5a5b822581)_

In this part, we’re gonna see how to use information-theoretic concepts like mutual information, to identify the most relevant features for a machine learning model. Also, the Boruta algorithm will de described and some heuristics on how to choose FS algorithms based on dataset size and sparseness will be given.

> The first part of this series can be accessed [here]().

## Boruta: when forests help you find the way

Boruta is an algorithm, originally written for R, that uses random forests and so-called shadow features, which are permuted original features, to iteratively select and rank important features in a dataset. It’s quite a powerful algorithm capable of clearly deciding if a variable is significant or not and the level of statistical significance of these variables.

Another interesting thing about Boruta, from its original FAQ

> It is an all relevant feature selection method, while most other are minimal optimal; this means it tries to find all features carrying information usable for prediction, rather than finding a possibly compact subset of features on which some classifier has a minimal error.

A great (I can’t do better + too much copy-pasting in the blog post is just not fair) explanation of how Boruta works, how to properly use it, and even why the name, can be found on its [wiki/FAQ page here](https://notabug.org/mbq/Boruta/wiki/FAQ). If you intend to use it, check the link.

The best thing is, it is also available in Python, [here](https://github.com/scikit-learn-contrib/boruta_py). Being part of scikit-learn-contrib group of packages, it’s interface is the all-to-familiar `fit_transform/fit+transform` methods. It is more flexible than the original implementation, with the possibility to use your estimator (as long as it has `feature_importances_` attribute) and tune the threshold for comparison between shadow and real features. Here's a code example from the project:

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy# load X and y
# NOTE BorutaPy accepts numpy arrays only, hence the .values attribute
X = pd.read_csv('examples/test_X.csv', index_col=0).values
y = pd.read_csv('examples/test_y.csv', header=None, index_col=0).values
y = y.ravel()# define random forest classifier, with utilising all cores and
# sampling in proportion to y labels
rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)# define Boruta feature selection method
feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1)# find all relevant features - 5 features should be selected
feat_selector.fit(X, y)# check selected features - first 5 features are selected
feat_selector.support_# check ranking of features
feat_selector.ranking_# call transform() on X to filter it down to selected features
X_filtered = feat_selector.transform(X)
```

## Mutual Information: what is it?

Mutual Information is a measure of how much the presence/absence of some variable will affect the correct classification of the target variable.

Given two discrete random variables X and Y with probability mass functions p( x), p( y) and joint probability mass function p( x, y) we have the following formula for correlation

> Cov(X, Y) = E(XY) − E(X)E(Y) = ∑p(x, y)xy − ∑p(x)x ⋅ ∑p(y)y ⇒ Cov(X, Y) = ∑[p(x, y)−p(x)p(y)]xy

and for mutual information

> I(X, Y) = H(XY) — H(X)H(Y) = E(lnp(x, y)p(x)p(y))
> ⇒ I(X, Y) = ∑p(x, y)[lnp(x,y)−lnp(x)p(y)]

So, these two are not opposites in any way, just two different perspectives on some distributions.

Compared with correlation, MI is not concerned whenever the variables are linearly dependent or not, but on the other hand, it requires knowledge of the underlining distribution of the variables, which is not always possible, or very hard to approximate.

So, it’s a trade-off, in which people usually prefer correlation-based methods.

### MI algorithms for feature selection

As per the previous section, computing mutual information between 2 variables might be hard, because knowledge about the underlying distributions is necessary. Usually, one uses approximation methods.
But now, for feature selection, one has to consider many combinations of features with the target variable, in fact, exponentially many combinations.

To make the problem more tractable, some assumptions are used. One of them, the fact that already selected features are independent of the features not selected but under consideration. This allows for a greedy selection process and a simplified formula to solve:

> argmax I(xi; y) — [αI(xi; xs)-βI(xi; xs|y)] where i not in S

The left side of the formula searches for maximum relevancy and the right one for minimal redundancy. Different MI-based feature selection algorithms vary α and β parameters, thus changing some of the other assumptions of the algorithm. For example, the JMI, aka joint mutual information algorithm assigns the 1/#selected features-1 to both α and β. And then we arrive at mRMR.

mRMR (minimal-redundancy-maximum-relevance) is a class of algorithms that try to identify features with most mutual information with the target variable, yet minimum overlap between them. It assigns 1/#selected features-1 to α and 0 to β from the formula above. The [original paper](http://home.penglab.com/papersall/docpdf/2005_TPAMI_FeaSel.pdf) described 2 methods, MID and MIQ, that is, minimum information distance and minimum information quotient, respectively.

A nice package with scikit-learn compatible interface can be found [here](https://github.com/danielhomola/mifs). Here’s a code sample from them:

```python
import pandas as pd
import mifs # load X and y
X = pd.read_csv('my_X_table.csv', index_col=0).values
y = pd.read_csv('my_y_vector.csv', index_col=0).values # define MI_FS feature selection method
feat_selector = mifs.MutualInformationFeatureSelector(method="MRMR") # find all relevant features
feat_selector.fit(X, y) # check selected features
feat_selector.support_ # check ranking of features
feat_selector.ranking_ # call transform() on X to filter it down to selected features
X_filtered = feat_selector.transform(X)
```

For a more in depth view, with more theory, see this amazing blog post. Also, just so you know, you could use sklearn’s `mutual_info_classif` or `mutual_info_regression` functions in combination with `SelectKBest` or `SelectPercentile` classes, but using MIFS package is faster and more flexible.
Selecting feature selection methods

Now that you know all these different feature selection methods, how do you pick one? The answer as usual in ML is — experiment, see for yourself.

Not very inspiring, isn’t it? That’s why here are some consideration for you, to at least minimize the search space.

- When having a small dataset (less than a couple of thousands) prefer model-based selection, for example using trees or forests. These are powerful methods that are compute expensive, so applying them for larger datasets will be prohibitive. Also, you could afford cross-validation in such a setting.
- When having a very big and sparse dataset, opt for `SGDClassifier`-based model with L1 penalty.
- For low sample size and big dimensionality datasets, the best option will be to use something like Lasso algorithm.
- For quick EDA-like workflow on low dimensionality datasets (<100 features), don’t forget to find the correlation coeficients between all features in your dataset.
- MI-based feature selection is best when having big (that’s mandatory, to estimate the distribution) and dense (optional, works pretty good with sparse data too) datasets.
- Statistical feature selection (chi2, correlation, et al) are best when running cheap and fast is important and the dataset is not too big (<100k or so).

# Epilogue

Note that all feature selection, both in this blogpost and in the previous one are applicable on feature vectors. That means, no method from here could be applied for vision problems, for example, unless you want to have fewer “features” from the last layer of a CNN, which might be a good idea, idk.

This series is by no means a definitive guide, but rather a list of handy tools and how to use them. If interested in FS, you should google how to perform wrapper methods efficiently, using optimization techniques like genetic algorithms, simulated annealing, or other methods alike.

Also, remember this, there’s no free lunch — think which trade-offs are you ready to make, pick a couple of methods, experiment, choose the best one for your problem.

If you’re reading this, I’d like to thank you and hope all of the above written will be of great help for you, as it was for me. Let me know what are your thoughs about it in the comments section. Your feedback is valuable for me.

Given my history of publishing, the next post will be sometime in January 😃 I’ll try to do better.