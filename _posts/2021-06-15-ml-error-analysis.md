---
title: Going beyond simple error analysis of ML systems
published: true
description: When deploying machine learning algorithms in production the stakes are much higher than in any toy problem or competition. For this reason we need a much more thorough evaluation of the model, to make sure it is indeed good.
tags: machine learning, machine learning debugging, error analysis, deep learning, machine learning evaluation, machine learning testing, artificial inteligence, fairness
layout: post
date:   2021-06-15 02:10:00 +0200
categories: posts
permalink: /posts/2021-06-15-ml-error-analysis
comments: true
---

# First there was a story...

Imagine yourself working as an ML engineer... First, congratulations, pat yourself on the shoulder. Second, depending on the company size, culture and the maturity of the machine learning team, you're most likely in for a wilde ride through many computer science and software engineering domains. Again, pat yourself on the shoulder. Now, let's get to the chase.

As an MLE part of your work is to pick, tune and deploy ML models. I believe I don't need to explain you that this is actually not so trivial, but most likely, you believe that the hardest part of this process is to tune the model, don't you? Or maybe that it is the deployment of the algorithm?

Here's a question for you: __*How do you make sure you have a high quality model in production?*__

I guess you figured out by now that you won't get away with a simple train/test split and a few error metrics, like accuracy or maybe F1 score. But what _would_ be enough? Well, it depends, as all things software engineering, but long story short, you need to at least understand that reducing your model characteristic to only one or a few scalars will forfeit way to much information about the model.


# ... and then words of wisdom* followed

Keep in mind that depending on the domain you apply machine learning to, a subpar model could be anything between a little annoyance for your users from which they can opt-out, in the best case scenarios, to a complete garbage fire that amplifies biases and actually makes your customers run away from your business.

You know what, let me first define a few ML evaluation maturity levels. For now, don't bother about the meaning of some more advanced terms here, I will explain them right after this section.

- L0: Having a train+test split and one or two generic metrics, like MSE or Accuracy. At this level, deploying the ML model is not advised (read: irresponssible at best).
- L1: Previous level, but using cross-validation if possible, or worst case scenario, having a huge test set. Per-class metrics for classification problems, multiple metrics for regression problems (MAPE+RMSE+R^2 are a good combination). In case of regression try to have at least one metric robust to outliers.
- L1.1: Check most wrong predictions, that is, entries with high prediction confidence, but that are actually wrongly predicted.
- L2.1: Rerturbation analysis using counterfactuals and random alterations of input values. Usually such approach also permits understanding of feature importance for each individual entry, but that is more like a bonus you have to work towards.
- L2.2: Maybe using ICE/PDP/ALE plots.
- L2.3: Maybe using local surogate explanations (usually LIME) to understand model predictions before approving it for deployment.
- L3: Cohort-based model inspection. Error grouping/Manifold-like model inspection.
      Also, stuff like Anchors is at this level.
      One more important thing: taking into account the changes in data distributions and evaluating on data from different periods (if needed). Believe me when I tell you this, sometimes features/label distribution changes even in domains where you don't expect them to.
- (Optional) L4: Adversarial examples checking.

Normally you would want to be at L1 when launching a model in a beta, L2.1 when it's in production, and from there grow to L3. L4 is more specific and not every use case requires it. Maybe you are using your ML algorithms internally, and there's low risk for some malicious agent trying to screw you, in this case I doubt you need to examine the behaviour of your model when fed adversarial examples, but use your own judgement.

Note that although I mention regression use-cases, I omitted a lot of info about time-series forcasting. That was done intentionally, because the topic is huge, and this post is already a long-read. But if you have a basic understanding of what's going on here, you will be able to map different time-series analysis tools onto this levels.


# Methods

A little disclaimer: I had [an older post]({{ site.url }}/posts/2021-05-09-archive-understanding-a-black-box) tangential to this topic, but the focus in it is on interpretability/explainability methods. In this blog post I focus more on how to assess the errors of machine learning models. If you think these topics are pretty close to each other, somewhat overlapping, you are right, to better evaluate a model we sometimes need to understand the "reasoning" it puts into making a prediction.

<!-- That article tries do answer the question *How does this model makes a decision (and what tools to use)?* versus *How does it fail?* See the difference? -->

<!-- Let's cluster evaluation methods into 3 broad categories: metrics,  -->

I won't dive deep into metrics-based evaluations, but will mention that depending on your use case you might want to consider metrics that are either non-linear in their relation to how wrong the prediction is. Maybe you're fine with a bit of error, but if the model is very wrong, or wrong very often you want to penalize it disproportionally more. Or, on the contrary, as there are more wrong predictions, or the total loss of the model is growing, you want to have a log-like behaviour for your metric, that is the metric will attenuate its growth as the model is more wrong.

Also, on the matter of metrics that are robust to outliers, sometimes these are nice to have, if you do some outlier removal beforehand, or these might be a necessity, where you can't or specifically don't remove the outliers, for whatever reason.

Finally, most of the time in production scenarios you will want to asses your model performance on different cohorts, and maybe even based on these cohorts to use different models. A cohort means a group of entities, with a specific grouping criterion, like an age bracket, or location-based, or maybe something else.


Methods:
- Perturbation analysis
    - Counterfactuals (What-If)
    - Adversarials (CleverHands/Foolbox)
    - Random alterations (Robustness to change/Common Sense-ness)
- Check most wrong predictions (wrong+high softmax)
- Estimate errors per cohort
    - Manifold-like, aka find automatically cohorts based on error profile, or some other way

- Similarities (neighbor instances)
- LIME (local linear/surogate explanations)
- SHAP (global/local explanations)
- Anchors (local rule-based explanations)
- Concept-identification ([Tensorflow's TCAV tool](https://github.com/tensorflow/tcav))


##  Personal recommendations

TK

- Of course start with a couple of appropriate evaluation metrics. Don't use just one. If you can, cross-validate. If doing HPO, have 2 testing splits. For classification I would recommend some loss/score function + scikit-learn's `classification_report` and if you don't have a ton of classes, the confusion matrix is your friend. Some use AUC and ROC-DET curves, they are nice, I'm just not used to these, maybe after this blogpost I will start using them. (do as I say, not as I do)
- I usually do perturbation analysis (random and counterfactuals) after this. Looking for the top-k most wrong predictions also helps, but I rarely do it (do as I say, not as I do, #2).
- If I'm not satisfied yet, I will certainly check for error groups a la Manifold and/or surrogate local explanations (LIME). I preffer not to do the later because it takes a looooot of time, especially with bigger sized input. Also regarding local explanations with surrogate models, sometimes I find it necessary to adjust the surrogate, using the default might be just too simplistic. I do NLP and both points are a real issue.

Also, sometimes, especially in the early stages of development I could do kind of "exploratory testing" of model predictions, namely feed out-of-distribution data and look what will happen.

For personal experiments, not work-related, I also sometimes use SHAP but I find it a bit frustrating that it's hard to export the graphics and works best when working from Jupyter. Also, it's slow, but that's a general issue for all surrogate explanations.

I am yet to play arround with Anchors, Adversarials and doing stuff like "Find the most similar entry with a different class" or "Find the most similar entries to this one". The later two can be done in principle using kNN in an embedding space, it's just not all algorithms have one explicitly defined.

References:
- [Interpretable Machine Learning by Christoph Molnar](https://christophm.github.io/interpretable-ml-book/)
- [Gamut paper]({{ site.url }}/_data/ml_debugging/19_gamut_chi.pdf)
- [Manifold paper]({{ site.url }}/_data/ml_debugging/1808.00196.pdf) and [Manifold GitHub repo](https://github.com/uber/manifold)
- Github repos which also contain links to their respective papers:
    - [LIME GitHub repo](https://github.com/marcotcr/lime)
    - [SHAP GitHub repo](https://github.com/slundberg/shap)
    - [Anchors GitHub repo](https://github.com/marcotcr/anchor)

*. More like personal war stories.


# Annex A: A few words about increasing the predictive performance of fostly classifiers

Robustification
- adversarial training
- focal loss for tails
- label smoothing
- self-distilation
