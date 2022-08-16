---
title: Interviewing for a Senior ML Engineer position
published: true
description: My experience interviewing for a few Senior ML and MLOps roles. You will learn what are the common steps, quirks, and tips how to nail an interview for senior ML engineer positions.
tags: machine learning, career, career advice, senior engineer, leadership, programming, interviews
keywords: machine learning, career, career advice, senior engineer, leadership, programming, interviews
layout: post
date:   2022-07-23 03:00:00 +0200
categories: posts
permalink: /posts/2022-07-23-senior-ml-interview
comments: true
---

Interviewing is always a tiring and sometimes awkward process. Thankfully there are lots of resources online to help you prepare. But what if you need more specific advice for a more niche position?

This post is based on my personal experience going through the interviewing process at 5 not-FAANG companies. I also had some experience interviewing for not-senior ML Engineering roles at another 3 companies last year. So, I will also do a comparative analysis.


## Before we begin...

Let me start with a short prologue to explain why I'm writing this piece. In January 2022, I decided, again, it was time to search for another job outside of my home country. But this time, I decided to be sneaky/smart about it, so I changed my LinkedIn address to show that I'm in London. I also groomed a bit more my LinkedIn page to show some highlights of my recent experience. And then magic happened. For weeks I had recruiters invite me to interviews. I didn't even have to apply myself to anything, only to accept or reject opportunities arriving from recruiters. What surprised me was that the majority of options were senior or even lead roles. So, I felt like an imposter, but I still accepted a few of these and started the process. And then I searched for tips on how to nail senior ML engineering interviews... and found almost nothing. Sh*t. And that's how I ~~met your mother~~ decided to write this blog post.

I brushed up my interviewing skills through mock interviews. I was also searching for technical questions for Senior ML roles. Surprisingly, I couldn't find anything. All the info was only for MLE roles. It seemed a bit strange. In retrospect, it all makes sense now.

I know you are eager to find out why, so I'll just give the TL;DR right away - **ML and Senior ML have more or less the same complexity/hardness for technical questions**. Surprise! 


I bet you didn't expect that. I know I didn't. But then, what **is different**? And how does the interviewing process works for Senior ML Engineers?


## Senior vs non-senior ML interviews

Based on my experience, I haven't noticed much difference between senior ML and ML engineering interviews at the technical level.

What I did notice is the focus on soft skills for senior positions, and I don't necessarily mean communication skills. Instead, how a candidate handled failures, team-level conflicts, cross-team communication, how they solved their most challenging problems, or how they handled a poor decision. 

I recall the first technical interview for a Senior ML role I had. I was anxious about what kind of questions will I receive. It wasn't so bad, I had tougher questions than that, but the focus was undoubtedly higher on how I handled some scenarios or how I would do it now.


Aspect| ML engineer interview | Senior ML engineer interview
------|-----------------------|------------------------------
Coding | Your usual leetcode-medium questions | Same, haven't seen dynamic programming at this stage
Take-home assignment | Either do EDA or deploy an ML model, focus on code quality, ease of use and tests | Same, take-home assignments are not harder for senior positions
ML Trivia | How algs. work? What would be the best solution for a type of problem | On average, the same as for ML engineer
System Design | How to implement a system for a given scenario? Data collection issues? | On average, same as for ML engineers, just be more conscious of budget constraints
**Behavioral** | **Focus on collaboration, individual growth, and adaptability** | **Focus on failures, conflict management, and cross-team collaboration**

One position for which I did notice some big differences when it comes to the technical questions is **Research Engineer**. I'm talking questions like [how does JPEG compresses](https://www.image-engineering.de/library/technotes/745-how-does-the-jpeg-compression-work) images, how to compute [nth Fibonacci in O(log n) time](https://baioc.github.io/blog/fibonacci/#fft-the-fast-fibonacci-transform), or [how to compute PCA from scratch](https://drscotthawley.github.io/blog/2019/12/21/PCA-From-Scratch.html). Now, for a research engineering position, these kinds of questions do make sense because of the innovative and research-oriented nature of the projects they have to work on. These frequently can involve a lot of _convert-math-to-code_ or _let's-break-it-down-and-then-improve_ type of tasks.

Anyway, to give you a more detailed view, let's see what is the general interviewing process when it comes to these kinds of roles.

<!-- 
Graphcore      - Interviewer -> Take home project    -> Technical discussion -> Behavioral
ASOS           - Interviewer -> Take home project    -> Technical discussion + Behavioral
Yelp           - Interviewer -> Coding challenge     -> System design + Coding interview + 2 x Behavioral
Toptal         - Interviewer -> Coding challenge     -> Coding interview + Technical discussion -> Take home project + Technical discussion
Sprout.ai      - Interviewer -> Take home project    -> Technical discussion -> Behavioral
THG            - Interviewer -> Behavioral           -> Technical discussion
Hyperscience   - Interviewer -> Technical discussion -> Behavioral
Rasa.ai        - Interviewer -> Coding challenge     -> TBA
Tessian        - Interviewer -> Coding interview     -> Technical discussion -> System design + Behavioral
Audio Analytic - Interviewer -> Technical discussion + (Behavioral + Technical) + Behavioral -> Behavioral?
Zensors        - Behavioral/Interviewer -> Technical discussion + Coding interview -> ML Coding interview -> Behavioral
 -->


## The general interviewing flow

First, let's go over the main steps in the process. Generally, there are at least 4 steps:
1. You have the first call with a recruiter or hiring manager. You get to know each other, go over your CV in general, discuss what makes you search for jobs, or accept invitations to interview, what you know about the company, what you are searching for, and so on. A pretty simple step if you ask me. Then, suppose the hiring manager thinks your goals and interests align with what the company seeks. In that case, you will be invited to the second, **technical** step. The dreaded one.
2. I call this step just technical for a reason. Some companies split it into 2, a take-home assignment and then a discussion based on it. Others have the typical coding interview. And others yet just have a technical discussion. The technical discussion usually covers ML theory and some specifics, like what is transfer learning, or what transformer architectures are. It might also be a pen-and-paper exercise where you can be asked to infer how PCA works. The latter is more common for more research-oriented roles.
3. Most of the time, there are two technical interviews, the second being more focused on system design interview. Or maybe some more technical challenges and discussions, YMMV, because this is very company- and team- specific.
4. Finally, the last round of interviews is usually reserved for everything else that wasn't covered in the previous steps, usually the behavior interview. Some companies have three rounds, combining the 3rd step with the 4th.

Now, let's dive into details.


## 1st interview

Pretty simple. Make sure to learn about the company, even if you were invited to interview with them. At this point, the company searching for candidates has a few objectives:
- to understand how interested you are in the company/position
- are there any legal constraints that need to be acknowledged, like visa status
- or personal constraints, like the necessity to work remotely
Also, at this stage, the recruiter is looking whether you'd be a good fit based on your career aspirations, personal opinions, and past experiences.

But don't be fooled, there's a probability of failure even at this stage. For example, if the recruiter feels you're not interested in the position or if your career plans don't align with the responsibilities of this position.


## 2nd/3rd interview

As mentioned, different companies do this stage differently. I found three types. Given that we have two steps here, most companies do a mix of these three methods.

### The "take-home-assignment tribe"

Take home - either an ML serving solution or EDA + modeling. No one will expect you to deliver a robust, production-ready solution for the ML serving project, nor will anyone complain that your Jupyter notebook doesn't contain a SotA ML model for a given dataset. The focus is on code quality, the presence of tests and features, ease of running the code for the former, and reproducibility and soundness of the solution for the latter.

Focus on quality over quantity. A good way to show professionalism is to follow up with clarifying questions once you receive the task. And please, read it carefully. Too often have I seen people doing it all wrong and not even bothering to check the exact constraints for the homework.

### The "coding challengers"

Too much was said about it. One point I consider worth reiterating is how important it is to actually talk through your problem-solving process and ask clarifying questions. I would argue that this could be even more important than solving the problem. Also, don't forget about:
- Asking about possible edge cases and then covering them.
- Explaining the time and space complexity of your solution.
- If you have the time, extra points for going through your code "debugger-style". That is, step-by-step while telling what the current values of all your variables are.

### The "technical discussionists"

Discussion with a team of engineers. It usually goes like this: `Technical/ML Trivia + NotSoOptional[ML System Design] + Optional[Behavioral]`. ML questions are mostly one of:
- "How would you handle X scenario"
- "What is Y? How does this work?"
- Occasionally, for research-heavy roles - "Could you compute Z from scratch, here's a Google Doc", as a follow-up to the previous questions.

Where $$ Y \in \{BatchNorm, DropOut, SkipConnections, DataAugmentation, SGD, Transformers, Attention, et al.\} $$
$$ Z \in \{PCA, Linear Regression, kNN, kMeans\} $$

Sometimes technical discussions take a more ML-System-Design flavor.

It's (was) COVID, so system design is usually only verbal unless you can also text-draw a solution while sharing your screen. Pseudo-code also helps.
ML System Design seems not to be any different. It's still one of "Design a Search Engine for X", or "How are you going to design an X-which-is-actually-a-recommender-system".

```

---------   r/w  ----------    ----------   HTTP/2
|  DB   | <------| API    |<-- | NGINX  |  <-------  Client
|       |        |        |    ---------- 
---------        ----------

```
<center><i>Example of "text-drawing" #1</i></center>


```
                                 /-------> Users Service --> MySQL
                                /
Client w/ Browser Cache ---> Gateway -----> Posts Service  --> Cassandra x 6
                                                |                 write_to: 2
                                              Redis               read_from: 1

```
<center><i>Example of "text-drawing" #2</i></center>

Extra points for talking through efficiency/budget/business considerations at this step. For example, proposing to split the application in two, with ML logic on a GPU-enabled machine and business logic on a more conventional server. Or thinking out loud about a buy vs. build decision about some sub-component.


## Some personal opinions

I prefer take-home projects + technical discussions. This combination makes for a more meaningful technical discussion. It allows the candidate to express their ideas about how a proper production system should be designed based on the take-home assignment. Plus, a good take-home project can highlight candidates' abilities to write code and how they handle logging, testing, documentation, and deployment. I would argue it's much better than just solving leetcode problems.

I even used take-home assignments to filter candidates when we were hiring for my team. I know the main cons of it, but I believe that a well-defined problem can be solved in one or two evenings, a couple hours each. Not great, but I feel much more relaxed than doing a 45m coding interview. Speaking of the devil...

I don't like coding challenges. IMO, it's usually just lazy bs. These kinds of practices can be understandable for FAANG ([well, more like MANGA nowadays](https://www.reddit.com/r/csMajors/comments/qhtqre/faang_manga/)) companies because of their scale*. But, when coding challenges are done by small companies, I mostly find this as just bad taste.

> Disclaimer *: I don't mean that at Google-scale, they need their devs to know very well how to sort an array or find 2 numbers that add up to something. I mean that they have to go through so many candidates that they need a standardized, time-efficient, and repeatable way to check their capabilities. It doesn't seem realistic for companies this big to give take-home assignments and thoroughly check these without incurring significant time and productivity losses. That's the sad reality.

To add to the mess of coding interviews, companies are actually misusing them. Coding interviews are supposed to check for a candidate's problem-solving **and** communication skills. You need to show the interviewer *what is your thought process* and *how are you tackling a new problem*. Usually, it shouldn't matter much if the solution you implemented is optimal or not. You need to be aware of this, though. Regretfully, interviewers usually just look for the "correct" answers, like it's an exam and not a discussion, making the whole experience miserable.

In theory, coding tests are even worse. Because there's no way to see the candidate's *thought process* and *the way they are tackling problems*. Thus, it becomes just a timed exam that has no actual value in assessing how good a candidate is. In practice, because most interviewers are no better, I would take a coding test over a coding interview almost any day of the week.

So, if I were to rank coding interviews, I would arrange them like this: 
1. "Discussion" coding interview
2. Coding test with no interviewer at all
3. Exam-like coding interview, without much support from the interviewer


Of course, there are exceptions. One time, [at band camp](https://www.youtube.com/watch?v=e-ftdcWqhUs) (jk), I had a fantastic experience with a no-interviewer coding challenge. It was a 3.5h HackerRank challenge, in 3 stages, for a research engineering position. The questions ranged from probability to ML model serving, numerical stability, and basic ML theory. Then, for the second stage, it was a code review exercise! I was given a piece of code and had to identify a bug and suggest an improvement. How cool is that?! The final part was an actual coding challenge to implement a graph algorithm. It was exhausting, but at least it wasn't generic, and because it was so diverse, I felt like it enabled people to show where their true strength lies.

Alright, I'll stop complaining and move on to the next section of this post.


## 4th interview

This one is primarily behavioral. Although I would say the candidate is always asked behavioral questions, it's just at this stage, it is the primary focus.

I really like the questions about past experiences and how they can be improved, or if something didn't work, why?
I feel these questions correlate more with actual skill rather than generic theory questions.

A few questions that I really liked were:
- If I ask your manager what's your greatest weakness, what would they tell me?
- What was a situation in which you made a mistake? How would you prevent it now by having more experience?
- Give me an example where you made a poor technical decision and then had to fix it. How did you do it?

Generally, any question which asks to reflect on past mistakes is especially cool. Why? They help uncover how you grew since then, how humble you are, and how your critical thinking works.

I have no recollection of such questions in a non-senior ML interview, but plenty of those for senior/lead positions. So maybe think about such scenarios before your next interview.


## Some final tips to prepare

To really nail that interview process, I like doing mock interviews. The best way to do it (that I found) is [Pramp.com](https://pramp.com). It's not an advertisement, you can check the link - it has no referral code or anything. I just really find them helpful, especially for coding interviews and somewhat for system design interviews.

For ML system design, the best thing I have found so far is Chip Huyen's booklet - [Machine Learning Systems Design](https://huyenchip.com/machine-learning-systems-design/toc.html). And of course, for generic system design - [The System Design Primer](https://github.com/donnemartin/system-design-primer).

And remember, to really prepare for the behavioral interviews. Be ready to answer questions about how you failed and what you learned from it. Focus more on behavioral questions, specifically ones highlighting your leadership potential and learning-from-mistakes type of situations. For a good list of behavioral questions, see this [PDF from LinkedIn](https://business.linkedin.com/content/dam/me/business/en-us/talent-solutions/resources/pdfs/linkedin-30-questions-to-identify-high-potential-candidates-ebook-8-7-17-uk-en.pdf).

Throughout the process, ask questions and show your interviewers that you are engaged in conversations with them and are interested in the role. Ask them about their technical and business priorities, how specific processes are implemented in the organization, and their current pain points. [Here's a good list](https://github.com/viraptor/reverse-interview) of questions you can ask.

Interested in becoming a senior engineer? You'll need both strong ML and superior soft skills to get that senior position. Also, maybe check my post [_Becoming a Senior Engineer_]({{ site.url }}/posts/2022-05-23-becoming-senior), which should help you define your own roadmap.



#### A little disclaimer (last one in this post)

These posts were almost done since February, but due to the tragic events unfolding in Ukraine, I thought it wouldn't be nice, to say the least, to post it back then. In Moldova, there's a saying "Satu' arde da baba sî chiaptănă" which translates to something like "The (unreasonable) old lady is grooming while the whole village burns". I didn't want to be that lady, so I thought it would be better to wait until things become at least somewhat less chaotic.

\#Слава Україні! \#Героям слава!

