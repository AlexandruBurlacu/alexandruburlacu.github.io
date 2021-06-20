---
title: Going beyond simple error analysis of ML systems
published: true
description: When deploying machine learning algorithms in production the stakes are much higher than in any toy problem or competition. For this reason we need a much more thorough evaluation of our models, to make sure it is indeed good.
tags: machine learning, machine learning debugging, error analysis, deep learning, machine learning evaluation, machine learning testing, artificial inteligence, fairness
layout: post
date:   2021-06-20 02:10:00 +0200
categories: posts
permalink: /posts/2021-06-15-ml-error-analysis
comments: true
---

# First there was a story...

Imagine yourself working as an ML engineer... you rock!

So yeah, congratulations, pat yourself on the back, your family must be proud. Second, depending on the company size, culture and the maturity of the machine learning team, you're most likely in for a wild ride through many computer science and software engineering domains. Again, pat yourself on the back. Now, let's get to the chase.

As an MLE part of your work is to pick, tune and deploy ML models. I believe I don't need to explain you that this is actually not so trivial. But most likely, you believe that the hardest part of this process is to tune the model, don't you? Or maybe that it is the deployment of the algorithm? Although these are indeed non-trivial, especially the later one, here's _The Question ©_ for you: __*How do you make sure you have a high quality model in production?*__

If you're gonna tell me that you tested your model on a held-out dataset, and that your metric of choice was something like accuracy or the mean squared error, just run. Fast. Far away. Also be prepared to recieve questions on whenever or not you (1) had a baseline, (2) was the dataset balanced, (3) did you use the held-out dataset for tuning/hyperparameter search... and so on.

<!-- TK insert padme anakin right? meme -->

I guess you figured out by now that a simple train/test split and a few error metrics, like accuracy or maybe even F1 score are not nearly enough to answer _The Question ©_. But what _would_ be enough? Well, it depends, as all things in software engineering, but long story short, you need to at least understand that reducing your model characteristic to only one or a few scalars will forfeit way to much information about the model.


# ... and then words of wisdom* followed

Keep in mind that depending on the domain you apply machine learning to, a subpar model could be anything between a little annoyance for your users from which they can easily opt-out, in the best case scenarios, to a complete dumpster fire that amplifies biases and actually makes your customers run away from your business. We don't want that, your employer certainly doesn't.

<!-- TK maybe insert something here -->

You know what, let me first define a few ML evaluation maturity levels. For now, don't bother about the meaning of some more advanced terms here, I will explain them right after this section.

<!-- TK check ML pyramid of needs -->

- L0: Having a train+test split and one or too few generic metrics, like MSE or Accuracy. At this level, deploying the ML model is not advised (read: irresponssible at best).
- L1: Previous level, but using cross-validation if possible, or worst case scenario, having a huge, diverse test set. Per-class metrics for classification problems, multiple metrics for regression problems (MAPE+RMSE+R^2 are a good combination) <!-- TK Check this -->. In case of regression try to have at least one metric robust to outliers.
- L1.1: Check most wrong predictions, that is, entries with high prediction confidence, but that are actually wrongly predicted. It can help you uncover error patterns, maybe even biases.
- L2.1: Rerturbation analysis using counterfactuals and random alterations of input values. Usually such approach also permits understanding of feature importance for each individual entry, but that is more like a bonus you have to work towards.
- L2.2: Maybe using ICE/PDP/ALE plots to better understand feature importance.
- L2.3: Maybe using surogate local explanations (usually LIME) to understand model predictions before approving it for deployment.
- L3: Cohort-based model inspection. Error grouping/Manifold-like model inspection.
      One more important thing: taking into account the changes in data distributions and evaluating on data from different periods (if needed). Believe me when I tell you this, sometimes features/label distribution changes even in domains where you don't expect them to. And not accounting for this will give you some royal headaches.
- (Optional) L4: Adversarial examples checking. Also, stuff like Anchors is at this level. In principle any other advanced model interpretability/explainability or security auditing is at this level.

Normally you would want to be at L1 when launching a model in a beta, L2.1 when it's in production, and from there grow to L3. L4 is more specific and not every use case requires it. Maybe you are using your ML algorithms internally, and there's low risk for some malicious agents trying to screw you, in this case I doubt you need to examine the behaviour of your model when fed adversarial examples, but use your own judgement.

Note that although I mention regression use-cases, I omitted a lot of info about time-series forcasting. That was done intentionally, because the topic is huge, and this post is already a long-read. But if you have a basic understanding of what's going on here, you will be able to map different time-series analysis tools onto this levels.


# Methods

A little disclaimer: I had [an older post]({{ site.url }}/posts/2021-05-09-archive-understanding-a-black-box) tangential to this topic, but the focus in it is on interpretability/explainability methods. In this blog post I focus more on how to assess the errors of machine learning models. If you think these topics are pretty close to each other, somewhat overlapping, you are right, to better evaluate a model we sometimes need to understand the "reasoning" it puts into making a prediction.

So, just keep in mind that the motif of this article is _understanding how, by how much, and (maybe) why a machine learning model fails?_.

Let's roughly cluster evaluation/error analysis methods into 3 broad categories: (1) metrics, (2) groupings, and (3) interpretations. Metrics is kinda obvious, groupings are probably the most abstract ones. We put here train/test splits, cross-validation, input data cohort, and error groupings in this... oh god... group (no pun intended). Finally, under the interpretation umbrella fall such things as surrogate local explanations, feature importance, and even analyzing the most wrong predictions, among other things.

## Metrics

I won't dive deep into metrics-based evaluations, but will mention that depending on your use case you might want to consider metrics that are non-linear in their relation to how wrong the prediction is. Maybe you're fine with a bit of error, but if the model is very wrong, or wrong very often you want to penalize it disproportionally more. Or, on the contrary, as there are more wrong predictions, or the total loss of the model is growing, you want to have a log-like behaviour for your metric, that is the metric will attenuate its growth as the model is more wrong.

Also, on the matter of metrics that are robust to outliers, sometimes these are nice to have, if you do some outlier removal beforehand, or these might be a necessity, where you can't or specifically don't remove the outliers, for whatever reason. Keep that in mind.

Finally, most of the time in production scenarios you will want to asses your model performance on different cohorts, and maybe even based on these cohorts to use different models. A cohort means a group of entities, with a specific grouping criterion, like an age bracket, or location-based, or maybe something else.

## Groupings

So I mentioned about cohorts in the paragraph above, so will make sense to follow-up on it. Cohorts are important because your stakeholders are interested in these, sometimes you might be too, but the business is usually the number one "fan" of cohorts. Why? Well, maybe they are especially interested to provide top-notch services for a special group of customers, or maybe they must comply with some regulations that ask them for some specific level of performance for anyone and everyone.

Also, your dataset is most certainly skeweed, if it's real world data. Meaning you will have underrepresented classes, all sorts of disbalances, and even different distributions for your features for each class/group of classes. It wouldn't be ok for the business to give mediocre recommendations for people outside US and Canada, or to predict that a person of color is some kind of ape <!-- TK link to that story -->.

So, we need to create cohorts, or groups, based on some characteristics, and track the performance of our machine learning systems across all these. Often you will discover that teams conscious about their cohorts will deploy different models for different cohorts, to ensure high-quality service for all of these.

But groupings aren't just cohorts based on input data characteristics. Sometimes for model analysis it makes sense to create groupings based on errors. Some sort of groupings by error profile. Maybe for some inputs your model, or models, are giving low errors, for other inputs some very high errors, and for yet another group if inputs an entirely different error distribution. To uncover and understand these, you could use [K-Means]({{ site.url }}/posts/2021-06-18-kmeans-trick) to cluster your losses and identify the reason your model might fail or just underperform. That's what Manifold from Uber does, and that's just brilliant!

Finally, groupings are also about how you group your data into train/test splits, or maybe more splits like evaluation during the training of your model, to notice when the model starts to overfit or whatever. Also, special care should be taken when doing hyperparameter search. For fast to train models a technique called nested cross validation <!-- TK add info --> is an incredibly good way to ensure the model is really good.

Also, an often overlooked issue is the target distribution of the dataset. It might be heavely imbalanced, and as a reasult, special care should be taken when sampling from it for train/validation/test splits. That's why you should almost always search for a way to have your splits _stratified_ (see scikit-learn's `StratifiedKFold`, also `train_test_split` has a `stratify=` parameter and for multioutput datasets check out `multioutput_crossvalidation` package).

## Interpretations

This category is pretty abstract, and some might argue that these are not really related to model evaluation, but rather ML interpretability/explainability. To which I say that these methods allow to uncover hidden errors, biases, and based on these you can pick one model over another, thus being in fact useful for evaluation. These tools are especially useful in __right answer - wrong method__ scenarios, which will pass without any issue metrics and groupings.

Methods:
- Perturbation analysis
    - Counterfactuals (What-If)
    - Adversarials (CleverHands/Foolbox)
    - Random alterations (Robustness to change/Common Sense-ness)
- Check most wrong predictions (wrong+high softmax)

- Similarities (neighbor instances)
- LIME (local linear/surogate explanations)
- SHAP (global/local explanations)
- Anchors (local rule-based explanations)
- Concept-identification ([Tensorflow's TCAV tool](https://github.com/tensorflow/tcav))


##  Personal recommendations

We're nearing the end of this post, so I would like to give you some recomendations on how to proceed when evaluating ML models, as if those maturity levels weren't enough. These recommendations are more low-level and practical, some gotchas if you will.

- Of course start with a couple of appropriate evaluation metrics. Don't use just one. If you can, cross-validate. If doing HPO, have 2 testing splits. For classification I would recommend at least some loss and some score function + scikit-learn's `classification_report` and if you don't have a ton of classes, the confusion matrix is your friend. Some people use AUC and ROC-DET curves, they are nice, I'm just not used to these, maybe after this blogpost I will start using them. (do as I say, not as I do)
- I usually do perturbation analysis (random and counterfactuals) after this. Looking for the top-k most wrong predictions also helps, but I rarely do it (do as I say, not as I do, #2).
- If I'm not satisfied yet, I will certainly check for error groups a la Manifold and/or surrogate local explanations (LIME-like, I mostly use the `eli5` package). I preffer not to do the later because it takes a looooot of time, especially with bigger sized input. Also regarding local explanations with surrogate models, sometimes I find it necessary to adjust the surrogate, using the default might be just too simplistic. I do NLP and both points are a real issue.

Also, sometimes, especially in the early stages of development I could do kind of "exploratory testing" of model predictions, namely feed out-of-distribution data and look what will happen.

For personal experiments, not work-related, I also sometimes use SHAP but I find it a bit frustrating that it's hard to export the graphics and works best when working from Jupyter. Also, it's slow, but that's a general issue for all surrogate explanations.

I am yet to play arround with Anchors, adversarial examples and doing stuff like "Find the most similar entry with a different class" or "Find the most similar entries to this one". The later two can be done in principle using kNN in an embedding space, it's just not all algorithms have one explicitly defined.


# A few references
- [Interpretable Machine Learning by Christoph Molnar](https://christophm.github.io/interpretable-ml-book/); amazing work, a lot of info, a lot of details
- [Gamut paper]({{ site.url }}/_data/ml_debugging/19_gamut_chi.pdf) to help you ask the right questions about a model
- [Manifold paper]({{ site.url }}/_data/ml_debugging/1808.00196.pdf) and [Manifold GitHub repo](https://github.com/uber/manifold)
- Github repos which also contain links to their respective papers:
    - [LIME GitHub repo](https://github.com/marcotcr/lime)
    - [SHAP GitHub repo](https://github.com/slundberg/shap)
    - [Anchors GitHub repo](https://github.com/marcotcr/anchor)


*. More like personal war stories.


<!-- # Annex A: A few words about increasing the predictive performance of mostly classifiers

Robustification
- adversarial training
- focal loss for tails
- label smoothing
- self-distilation -->
