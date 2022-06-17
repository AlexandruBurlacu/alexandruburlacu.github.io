---
title: Choosing programming languages for real-world projects
published: true
description: How to pick a tool, language, or framework when real money and the business is at stake. What to consider when faced with this kind of situation.
tags: software engineering, programming, programming languages, decision making, frameworks, java, kotlin, lisp, python, go, golang, rust, rustlang, erlang, elixir, ocaml, software, engineering, senior, leadership
keywords: software engineering, programming, programming languages, decision making, frameworks, java, kotlin, lisp, python, go, golang, rust, rustlang, erlang, elixir, ocaml, software, engineering, senior, leadership
layout: post
date:   2022-06-18 11:00:00 +0300
categories: posts
permalink: /posts/2022-06-18-choosing-a-tool
comments: true
---

A few years ago, when I was in my senior year at the university, during the distributed systems lecture our professor asked us a very nice question: 
> If we were to choose between a fancy new programming language, or Java/C#, for a greenfield commercial project, what would we choose and why?

If you're wondering what it has to do with distributed systems, I have to say - half of it was about software architecture.

The classroom was split into 2 camps, obviously. The fun and somewhat sad fact was that the Java camp won. I was part of that camp, even though I don't like Java, to say the least. We had much better arguments. So, what were those winning arguments? Rich library and tooling ecosystem, and the relative availability of professionals in our local market, for a fair price too. Our professor deemed us project managers, not real programmers, then said we were right, and for a few seconds the atmosphere in the classroom turned sad and hopeless. Then we moved on with the lecture.

**TL;DR:** We all want to play with the shiniest new toys, but when money is at stake, better stick to something tried and true.

So here are some questions to keep in mind when choosing a programming language, or any software tool for that matter, for a project. The focus will be on commercial projects, but some of the tips work for research projects and simple pet projects too.


## Basic level

Initially, the decision-making process is usually guided by a very narrow understanding of the consequences of choosing a specific tool. In increasing order of maturity, here are some basic reasons to make a choice:
1. *I would like to learn this new tool/language/framework, people say it's hot right now*
2. *People say this is the best tool/language for this kind of problem*
3. *I know this language/tool very well and can be very productive with it*
4. *I and my team know this language/tool quite well and we can all be productive with it*

1 and 2 are only acceptable reasons for a pet project, with a small caveat, which I'll explain later\*. Although I would recommend sometimes taking a look at more niche, possibly peculiar tools to learn. Because, you know, [if a language doesn't change the way you think, it's not worth learning](https://www.goodreads.com/author/quotes/1164347.Alan_J_Perlis).

4 is a decent reason, see Paul Graham's post about [using LISP to build a startup](http://www.paulgraham.com/avg.html), but in the long run, it's not that simple.

## Higher-level decision making

The difference between programming and getting stuff done, and software engineering is that the latter has significantly harder constraints (See [Software Engineering at Google](https://abseil.io/resources/swe-book/html/toc.html)). Not just any code can be developed productively by a changing team of people and maintained over time. And most commercial software isn't one-time scripts, but code that lives on for years, if not decades. That's why, when choosing a tool, language, or an entire stack, try to guide your decision-making with these questions, in no particular order:

- *How well documented this tool/language is?*
- *How actively used/developed is it?*
- *How many dependencies of any sort does it have?*
- *How stable this tool/language is?*
- *What is the size and quality of the ecosystem for this tool/language?*
- *How productive can someone be using this tool/language?*

More constraints, but doable.

## Business-level decision-making

Now we reached the final frontier. Until now, it wasn't particularly hard to make a choice, you just had to do your research. But now, we're gonna have to enter the realm of never-ending trade-offs. Keep in mind that software is written by people, who you have to employ, pay salaries, and ideally have a positive return on investment.

- *How easy is it to teach someone, or how much time does it take to make someone productive with the given tool/language?*
- *How much reachable supply of professionals is out there for this tool/language? Is it sufficient for you?*
- *How much do professionals who are knowledgeable with this tool/language ask for (money, perks, whatever)?*
- *What is the quality of the supply? Are the engineers mostly newbies or seasoned professionals?*
- *How many people would like to work with the chosen tool/language? How excited are they?*

Rarely the raw performance of a tool or language is a big issue. Some domains are indeed interested in that characteristic too, like scientific computing, low-latency systems, and maybe embedded systems. More recently, how energy-efficient, or "green" a language or tool is, is of greater importance. Yes, [I'm not kidding](https://docente.ifsc.edu.br/mello/livros/java/paperSLE.pdf). For example [Amazon cares](https://aws.amazon.com/blogs/opensource/sustainability-with-rust/) about such things, although like all things at this level, it's [not so simple](https://news.ycombinator.com/item?id=30441771).

### An example of picking a language

Let's do a "demo". We will assume that we're a remote-first startup and we want to build ~~a snowman~~ a serverless platform. How do we pick the programming stack? Well, at least the programming language. We will assume that the technical founders are capable of writing any language. No, they are not [spherical](https://en.wikipedia.org/wiki/Spherical_cow).

An important technical constraint for our project is that serverless technology is especially effective when the startup time of a serverless function is quick. If it's not, why bother? Optionally, we might want to dive into serverless edge computing, meaning we need a programming language that can work even on resource-constrained devices. Maybe not microcontrollers, but something like a newer Raspberry Pi shouldn't be considered unrealistic.

We are also budget-constrained because we're a startup. We need to execute fast, or else we might not reach escape velocity, and no one will bother.

With that said, let's prune some candidates. Because of our startup latency constraint, we can't afford to run anything which needs a VM-like runtime. So no Java, C#, and even Erlang or Elixir. Although Erlang and Elixir have less substantial problems with VM cold start, they have another downside of having a smaller talent pool. On yet another hand, this talent pool is usually very enthusiastic and professional. What a shame we're not building a messaging system.

|Language|Verdict|Talent Pool Size|Tooling|Excitement Factor|Startup Latency|
---------|-------|----------------|-------|-----------------|----------------
|Java | No | Very Large | Very Good | Can we go lower? | Half of Java jokes are about this |
|C# | No | Large | Very Good | A bit better than Java | A bit better than Java |
|Elixir/Erlang | No | Small | Good | Almost through the roof | Good, for a VM-based language |

If we are planning for maximum efficiency, maybe we should use C++? Definitely no. C++ is quite dangerous. Besides, we need to keep in mind that we want to develop fast and preferably without much risk of segmentation faults, resource leaks, and other C++ surprises. Btw, a good C++ dev is quite expensive and hard to find nowadays.

<!-- |Java | No | Very Large | Very Good | Can we go lower? | Half of Java jokes are about this |
|C# | No | Large | Very Good | A bit better than Java | A bit better than Java |
|Elixir/Erlang | No | Small | Good | Almost through the roof | Good, for a VM-based language | -->

|Language|Verdict|Talent Pool Size|Tooling|Excitement Factor|Startup Latency|
---------|-------|----------------|-------|-----------------|----------------
|... | ... | ... | ... | ... | ... |
|C++ | No | Moderate | Moderate, hard to use IMO | Depends what kind of person are you | Sonic the hedgehog approves |


We know that development speed is important. But we also want a performant language without VM cold start problems. How about Python, or JS? These are popular, fast to work with, with a considerable talent pool, and JS can be speedy too. To be fair, this wouldn't be the worst idea. Python, specifically CPython, can be slow but with the right tooling, or by substituting it with [PyPy](https://www.pypy.org/), we can solve these problems. As for JS, one issue is that the language is not the most pleasant to debug, with its [unholy trinity of no-values](https://javascriptwtf.com/wtf/javascript-holy-trinity) and subpar traceback messages. Regretfully, there are lots of not-so-good-devs out there professing these tools, so that's and issue. Finally, these are not the best systems programming languages.


<!-- |Java | No | Very Large | Very Good | Can we go lower? | Half of Java jokes are about this |
|C# | No | Large | Very Good | A bit better than Java | A bit better than Java |
|Elixir/Erlang | No | Small | Good | Almost through the roof | Good, for a VM-based language |
|C++ | No | Moderate | Moderate, hard to use IMO | Depends what kind of person are you | Sonic the hedgehog approves | -->

|Language|Verdict|Talent Pool Size|Tooling|Excitement Factor|Startup Latency|
---------|-------|----------------|-------|-----------------|----------------
|... | ... | ... | ... | ... | ... |
|JS | Maybe/No | Very Large | Good | Depends what flavor are you using | Good |
|Python (CPython) | Maybe/No | Very Large | Good | It will be a bummer that it's not used for DS/ML/AI | Good |
|Python (PyPy) | Maybe/Yes | Very Large (but there's a catch) | Good | If you know, you know | Good, and it's very fast overall |

Ok, so I said it, systems programming languages. And we dropped C++. What do we have left? [Go](https://golangdocs.com/system-programming-in-go-1), [Rust](https://msrc-blog.microsoft.com/2019/07/22/why-rust-for-safe-systems-programming/), [Crystal](https://crystal-lang.org/). We drop Crystal right away due to the lack of a sizeable community, talent pool, and libraries. So, it's Go vs Rust? Hold on, there's another contestant - [OCaml](https://ocamlverse.github.io/content/systems_programming.html). So, why did it come to these 3 languages? All of these are very suitable for systems programming, that is, interacting with lower-level OS constructs, are quite efficient at working closer to hardware, and in general, are fast and resource-efficient. Of all 3, Go is the most mainstream, so it's a plus. Also, it's easy to onboard people to use it. On the other hand, Rust and OCaml provide nicer guarantees for the programs you write, and although less popular, the quality of developers using them is usually pretty high. OCaml and Rust are pretty close idiomatically, but Rust syntax will be much more familiar to non-hardcore FP people, aka common folk, so it's probably 10 points to Rust. All in all, let's see the final table.

|Language|Verdict|Talent Pool Size|Tooling|Excitement Factor|Startup Latency|
---------|-------|----------------|-------|-----------------|----------------
|Java | No | Very Large | Very Good | Can we go lower? | Half of Java jokes are about this |
|C# | No | Large | Very Good | A bit better than Java | A bit better than Java |
|Elixir/Erlang | No | Small | Good | Almost through the roof | Good, for a VM-based language |
|C++ | No | Moderate | Moderate, hard to use IMO | Depends what kind of person are you | Sonic the hedgehog approves |
|JS | Maybe/No | Very Large | Good | Depends what flavor are you using | Good |
|Python (CPython) | Maybe/No | Very Large | Good | It will be a bummer that it's not used for DS/ML/AI | Good |
|Python (PyPy) | Maybe/Yes | Very Large (but there's a catch) | Good | If you know, you know | Good, and it's very fast overall |
|Crystal | No | Very Small | So-so | If you know, you know v2 | Very Good, and it's blazing fast overall |
|Rust | Maybe/Strong Yes | Small-Moderate | Moderate | Almost through the roof | Very good, and it's very fast overall |
|Go | Yes | Large | Good | Pretty good | Good, and it's very fast overall |
|OCaml | Maybe/Yes | Small | Moderate | Almost through the roof, but only for FP geeks | Very good, and it's very fast overall |


All things considered, probably the safest choice would be to use Go. And the next best thing would be Rust. A very good option would be PyPy, IMO. It's almost 1 to 1 equivalent to CPython, but considerably faster. If you like it more hardcore FP, you could try OCaml. You could in fact go polyglot, and pick 2 languages, but don't escalate to more than that. There's a reason most full-stack engineers are writing JS-only.


## \*Time to discuss that caveat.

Yes, picking a tool only because it's *hot* or seems interesting but is risky will rarely be a good idea, except when it is. You see, a tool is usually "hot" for a reason. Maybe it's solving a common pain in the industry, and does so elegantly. Or maybe, it boosts productivity, efficiency, or the long-term maintainability of a system. Still, this isn't enough to make such a risky move. 

On the other hand, there's an interesting aspect here. If a tool is hot people will want to work with it. This phenomenon boosts the desire to work for your team/business because you're using this New Hot Thing ©. Combined with the intrinsic qualities of the new tool, it might make sense to actually give it a try. It is just as risky to never take a risk. Failing to grow and innovate will leave your business hard to hire for, your talent pool shrinking, and your operational efficiency slowly dying.

<center><img src="/_data/bell_curve_languages.jpg"/></center>
<center><i>Follow sage's advice 😏 Made with: imgflip.com</i></center>


## A substitute for a conclussion

I hope I haven't fried your brains with this many things to consider. Even I sometimes don't do the whole process, or am being sloppy when assesing some of the aspects. Still, having a checklist of things to consider is always a good thing, so I hope you'll benefit from this.

Maybe a bit anti-climactic, but consider this - if you picked the wrong tool, it will rarely doom your project for failure. What will is not realising you made a bad choice, and trying to fix it. Technical stacks are problems which can be fixed with money, and that's a good thing.

Not the ending you expected? 😏

### P.S.
I should add a clarification about Java. Don't get me wrong - I don't "hate" Java, I just like pointing to its flaws, sometimes vehemently 😀. Java's unnecessary verbosity is the main issue that I have with it. It wasn't the only issue, but with the sped-up release cycle and a lot of ideas borrowed from other languages and communities, it's becoming a better language. Brilliant engineers use Java for many important, actively developed projects with no plans to retire or rewrite these. Ergo, it can't be an objectively "bad" language.

<!-- Also, on a more philosophical note, keep in mind - Java was created for mass producing of software, where developers would become interchangeble. From a business point of view, this is a very good idea. But from a craftsman's point of view, this is sad and uninspiring. Also this thing become so popular because Sun marketed it as hell and people started to believe Java is good. -->

#### A little disclaimer

These posts were almost done since February, but due to the tragic events unfolding in Ukraine, I thought it wouldn't be nice, to say the least, to post it back then. In Moldova, there's a saying "Satu' arde da baba sî chiaptănă" which translates to something like "The (unreasonable) old lady is grooming while the whole village burns". I didn't want to be that lady, so I thought it would be better to wait until things become at least somewhat less chaotic.

\#Слава Україні! \#Героям слава!
