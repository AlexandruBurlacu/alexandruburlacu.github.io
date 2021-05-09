---
title: Understanding a Black-Box
published: true
description: An overview of model interpretability methods… and why it’s important.
tags: machine learning, deep learning, fairness, ai, xai, explainability
layout: post
date: 2018-04-12 17:12:00 +0100
categories: posts
permalink: /posts/2021-05-09-archive-understanding-a-black-box
---
![](https://thepracticaldev.s3.amazonaws.com/i/bakqd3d3w3l7dhfz2zh8.jpeg)

An overview of model interpretability methods… and why it’s important. Originally published [here](https://towardsdatascience.com/understanding-a-black-box-896df6b82c6e).

Before we dive into some popular and quite powerful methods to crack open black box machine learning models, like deep learning ones, let’s first make clear why it is so important.

You see, there are a lot of domains that would benefit from understandable models, like self-driving cars, or ad targeting, and there are even more that demand this interpretability, like creditworthiness assignment, banking, healthcare, human resources. Being able to audit the model for these critical domains is very important.

Understanding the most important features of a model gives us insights into its inner workings and gives directions for improving its performance and removing bias.

Besides that, sometimes it helps to debug models (happens all the time). The most important reason, however, for providing explanations along with the predictions is that explainable ML models are necessary to gain end-user trust (think of medical applications as an example).

I hope now you also believe that understandable machine learning is of high importance, so let’s dive into concrete examples to solve this problem.

---

# Simple(st) methods

The simplest method one can think of is slight alterations of input data to observe how the underlying black box is reacting. For visual data usage of partially occluded images is the easiest method. For text — the substitution of words, and for numerical/categorical data — alteration of variables. Easy as that!

The greatest benefit of this method — it is model-agnostic, you can even check someone else’s models without direct access to it.

Even if it sounds easy, the benefits are immense. I used this method numerous times to debug both Machine-Learning-as-a-Service solutions and neural networks trained on my own machine to find that the trained models choose irrelevant features to decide the class of images, thus saving hours of work. Truly 80/20 rule in action.

---

# GradCAM

**Gradient-weighted Class Activation Maps** — a more advanced and specialized method. The constraints of this method are that you need to have access to the model’s internals, and it should work with images. To give you a simple intuition of the method, given a sample of data (image), it will output a heat map of the regions of the image where the neural network had the most and greatest activations, therefore the features in the image that model correlates the most with the class.

Essentially, you get a more fine-grained understanding of what are the important features for the model than in the previous model.

Here’s a nice demo of the GradCAM interpretability method.
To learn how the GradCAM works, check the [“Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization”](https://arxiv.org/pdf/1610.02391.pdf) paper.

---

# LIME

Maybe you’ve heard about this one. If not, first take a look at this short intro.

<!-- [![IMAGE ALT TEXT](http://img.youtube.com/vi/hUnRCxnydCc/0.jpg)](http://www.youtube.com/watch?v=hUnRCxnydCc "KDD2016 paper 573") -->
<iframe width="560" height="315" src="https://www.youtube.com/embed/hUnRCxnydCc" title="KDD2016 paper 573" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>


In 2016 was published “Why Should I Trust You?: Explaining the Predictions of Any Classifier” paper that introduced **LIME — Local Interpretable Model-agnostic Explanations**. Let’s derive its capabilities from the name!

**Local Interpretable** — You should know that higher the complexity of the machine learning model, the less interpretable the model is. That is, logistic regression and decision trees are much more interpretable than, say, random forests and neural networks. The assumption of the LIME method is that non-linear, complex models like random forests or neural networks can be linear and simple locally, that is on small patches of the whole decision boundary. And recall, we said that simple models are interpretable.

**Model-agnostic** — This part is easier. LIME doesn’t have any assumptions about the model that is interpreted.

The best thing about LIME is that it is also available as a PyPI package. `pip install lime` and you’re ready to go! For more information, [here](https://github.com/marcotcr/lime)’s their GitHub repo with benchmarks and some tutorial notebooks, and [here](https://arxiv.org/pdf/1602.04938.pdf) is the link to their paper. [FYI, LIME was also used by the Fast Forward Labs (now part of Cloudera) in their demo on the importance of model interpretability](https://blog.fastforwardlabs.com/2017/08/02/interpretability.html).

---

# SHAP

**SHapley Additive exPlanations** — a more recent solution for understanding black-box models. In a way, it is very similar to LIME. Both are powerful, unified solutions, that are model-agnostic and relatively easy to get started.

But what is special about SHAP is that it uses LIME internally. SHAP actually has a plethora of interpretable models behind it, and it selects the most appropriate for the problem at hand, giving you the needed explanations, using the right tool for it.

Moreover, if we break down the capabilities of this solution, we actually find out that SHAP explains the output of any machine learning model using Shapley values. This means that SHAP assigns a value to each feature for each prediction (i.e. feature attribution); the higher the value, the larger the feature’s attribution to the specific prediction. It also means that the sum of these values should be close to the original model prediction.

Check their [GitHub repo](https://github.com/slundberg/shap), just like LIME, they have some tutorials and it is also possible to install SHAP via pip. Also, for more nitty-gritty details, check their [NIPS pre-print paper](http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions.pdf).

---

# Final Notes

Machine Learning model interpretability and explainability is a hot topic in the AI community, with immense implications. In order for AI-enabled products and services to enter new, highly regulated markets, it is mandatory to understand how these novel algorithms are making decisions.

Moreover, knowing the reasons of a machine learning model provides an outstanding advantage in debugging it, and even improving it.

It is highly advisable to design Deep Learning/Machine Learning systems with interpretability in mind, so that is is always easy to inspect the model and in a critical situation, to suppress its decisions.

If you’ve made it so far, thank you! I encourage you to make your own research in this new domain and share your findings with us in the comments section.

Also, don’t forget to clap if you liked the article 😏 or even follow me for more articles on miscellaneous topics from machine learning and deep learning.

P.S. Sorry, no Colab Notebook this time because there are already lots of very good tutorials on this topic to get you started.


