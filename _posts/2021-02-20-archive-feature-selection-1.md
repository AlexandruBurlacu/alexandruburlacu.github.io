---
title: Feature Selection for Machine Learning (1/2)
published: true
# description: An overview of methods to speed up training of convolutional neural networks without significant impact on the accuracy.
tags: scikit-learn, ai, machinelearning
layout: post
date:   2019-09-08 23:20:00 +0200
categories: posts
---



_Originally published on Medium, quite some time ago, [here](https://towardsdatascience.com/feature-selection-for-machine-learning-1-2-1597d9ccb54a)_

Feature selection, also known as variable selection, is a powerful idea, with major implications for your machine learning workflow.

Why would you ever need it?

Well, how do you like to reduce your number of features 10x? Or if doing NLP, even 1000x. What about besides smaller feature space, resulting in faster training and inference, also to have an observable improvement in accuracy, or whatever metric you use for your models? If that doesn’t grab your attention, I don’t know what does.

Don’t believe me? This literally happened to me a couple of days ago at work.

So, this is a 2 part blog post where I’m going to explain, and show, how to do automated feature selection in Python, so that you can level up your ML game.

Only filter methods will be presented because they are more generic and less compute hungry than wrapper methods, while embedded feature selection methods being, well, embedded in the model, aren’t as flexible as filter methods.

Sit tight 😉

> Second part can be accessed [here]({{ site.url }}/posts/posts/archive-feature-selection-2/)

# First things first: the basics

So, you need to find the most powerful, that is, important features for your model. We will assert that between an important feature and the target variable there’s a meaningful relationship (not necessarily linear, but more on that later), that is something like `target ~ f(feature)`. The simplest relationship is the linear one, and a powerful tool to identify such a relationship is **correlation**. Correlation means association, or dependence between 2 variables, just what we need. There are a number of ways to compute it, but given that the aim of this blog post is to be densely packed with practical advice, here’re the “simplest” 2: Spearman and Pearson methods, in pandas

```python
>>> # shamelessly taken from here: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.corr.html
>>> df = pd.DataFrame([(.2, .3), (.0, .6), (.6, .0), (.2, .1)],
...                   columns=['dogs', 'cats'])
>>> # df.corr(method="spearman")
>>> df.corr(method="pearson")
      dogs  cats
dogs   1.0   0.3
cats   0.3   1.0
```

So, once computed, we can get the indices and use them to select only highly correlated features, that will further be used for model training. In practice, if there’s indeed a significant linear correlation between variables, it should be above 0.7. The nice thing about these correlation tests is that in practice these are quite robust, and might identify sometimes even non-linear dependencies, if possible to approximate locally with a line, for example, 2nd order polynomial dependencies, or logs and square root dependencies, or even exponentials. The correlation coefficient will be smaller, maybe between 0.5 and 0.7. When getting such values, turn EDA mode end plot the values, maybe you can spot some sort of dependency. I sometimes do.

```python
pearson = pd.concat([features_df, target_df], axis=1).corr(method="pearson")
indices = pearson[abs(pearson["prediction"]) > 0.55].index
```

Another method is using chi2 test. Something like this:

```python
chi_sq = feature_selection.chi2(X, y)
corr_table = pd.DataFrame(zip(*chi_sq), columns = ("Correlation", "P-value"))

top_features = corr_table.sort_values("Correlation", ascending=False).head()["Correlation"]
```

One last thing, that you need to keep in mind: Usually, features in datasets not only correlate with the target variable, but between themselves too. **You don’t want this!** When selecting features you (or the algo) should pick the minimal number of most important features that are as orthogonal/uncorrelated between themselves, as possible. In the second blog post some methods that can achieve this will be presented, so stay tuned.

# The bigger guns: feature imporance and model based selection

Alright, a lot of these methods are grounded in statistics, and this is neat, but sometimes, you just need a bit less formalism and a bit more Scikit-Learn.

Some models in scikit-learn have `coef_` or `feature_importances_` attributes. Once these models are trained, the attributes are populated with information that is highly valuable for feature selection. Here are 2 examples, using decision trees and L1 regularization.


### Decision-trees-based method


```python
feature_importance_tree = tree.DecisionTreeClassifier()
feature_importance_tree.fit(X, y)

feature_importance_list = feature_importance_tree.feature_importances_.tolist()
indices = zip(*sorted(enumerate(feature_importance_list), key=lambda x: x[1], reverse=True)[:5])[0]

X_tree = X[:, indices]

scores = [model.fit(X_tree[train], y[train]).score(X_tree[test], y[test]) for train, test in kfcv]
```


Now that I grabbed your attention, let me explain. Decision trees are pretty nice tools, highly interpretable, and it turns out, useful beyond just classification/regression problems. In this example, a `DecisionTreeClassifier` is fitted quickly on a dataset and then the `feature_importances_` are used to pick the most relevant features and train a bigger, more complex, and slower model. In practice, if you have a lot of data, you might opt for a variation of the next method, but for smaller data, this one is beyond decent, and capable of capturing features with non-linear dependencies. Also, an `ExtraTreesClassifier` could work well too for bigger data, if regularized (shallower trees, more samples per leaf) even better. Always experiment.

### L1-based method

For those who don’t know yet, L1 regularization, due to it’s nature, introduces sparseness into models… Just what we need, indeed!

```python
clf = linear_model.LassoCV()

sfm = feature_selection.SelectFromModel(clf, threshold=0.002)
sfm.fit(X, y)

X_l1 = sfm.transform(X)

scores = [model.fit(X_l1[train], y[train]).score(X_l1[test], y[test]) for train, test in kfcv]
```

Like the tree example above, a small model is trained, but unlike the example, `coef_` is what drives the sklearn implemented feature selection. Because L1 is used, most of the coefficients in the model will be 0 or close to it, so anything bigger could count as a significant feature. This works with linear dependencies well. For non linear ones, maybe try and SVM, or use `RBFSampler` from sklearn before the linear model.

### An important note on performance and big datasets.

Say you have an NLP problem, and use TF-IDF. On a reasonable dataset, you’re gonna have a huge output matrix, something like many thousands of rows (documents) with a couple of millions of columns (n-grams). Running any model on such a matrix is time and memory consuming, so you better use something fast. In such a case I would definitely recommend the L1 approach, but instead of `LassoCV` to use `SGDClassifier(penalty="l1")`. Both methods are almost equivalent, but on big datasets, the later runs almost an order of magnitude faster. So keep it in mind.

Also, keep in mind that you don’t need your feature selection models trained until convergence, you’re not going to use these for predictions, and the most relevant features will be selected among first in the model anyway.

# Epilogue

Most of the code here is from an old project of mine on GitHub, [here](https://github.com/AlexandruBurlacu/MLExperiments/blob/master/machine-learning-and-a-bit-of-data-science/Breast_Cancer_feature_selection.ipynb). It should work, but if it doesn’t, don’t be shy — HMU.

Note that all feature selection, both in this blogpost and in the next one are applicable on feature vectors. That means, no method from here could be applied for vision problems, for example, unless you want to have fewer “features” from the last layer of a CNN, which might be a good idea, idk.

Also, remember this, there’s no free lunch — think which trade-offs are you ready to make, pick a couple of methods, experiment, choose the best one for your problem.

If you’re reading this, I’d like to thank you and hope all of the above written will be of great help for you, as it was for me. Let me know what are your thoughs about it in the comments section. Your feedback is valuable for me.