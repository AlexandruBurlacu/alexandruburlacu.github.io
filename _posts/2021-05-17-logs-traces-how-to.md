---
title: Loging, Tracing, Monitoring, et al.
published: true
description: When it comes to production-ready systems we need a way to know what's going on in it, aiding us in debugging it, when the time comes.
tags: logging, logs, tracing, traces, observability, telemetry, monitoring, alerting, distributed-systems
layout: post
date:   2021-05-19 00:10:00 +0200
categories: posts
permalink: /posts/2021-05-17-logs-traces-how-to
---


# So, you want to launch your code/app/system in production?

Wait, before you do, ask yourself this question: _If something goes south, how will I what **exactly** happened?_

A good question, isn't it?

If you're a more seasoned engineer, you might say: _**I will use logs!!!**_ Good, good. But what if I tell you, logs are only the begging?

# Act 1: I'll setup logs, alright...

So, what exactly is a log?

![](https://media.giphy.com/media/xUOxfbAOLZmR356YgM/giphy.gif)

Technically it's a log, but I'm talking about other kinds of logs.

Logs are a record about some event in a system. Pretty abstract, huh? Well, what are you gonna do. Basically a log is like an entry in a journal about something that happened, maybe with some context. Like Apple-reporter's Twitter feed during and WWDC event. You have time, you have a record of something that just happened, and maybe you have context too. Now, jokes aside, logs are important. They help you uncover what was happening moments before application crash. Or mallicous activity. Or other stuff. Bur how do we make **good** logs?

## Tenets of a good log message

So, how should we design our logs? Here are some tenets:

- Thy logs must be **hierarchical**: we need to respect the distinction between `DEBUG/INFO/WARNING/ERROR` and possibly other levels, and not to crowd the system with `WARNING` logs when `INFO` or `DEBUG` are more appropriate. Not-to-crowd also refers to how much information a log contains. That being said, it's a good idea for a `WARNING` or `ERROR` log to contain as much information as possible, to aid in debugging. `DEBUG` logs can be used to register information about what setting the program is using, even how much time or resources some subroutine is using, but don't abuse this. As for `INFO` logs, anything in between, like information about a top-level handler being called, as in HTTP servers.

- Thy logs must be **informative**: A good rule of thumb is to log everything that might help you in debugging your system. If an error happens, you will want to log the traceback, but also the context in which the error happened, that is, some surrounding variables, which might have something to do with the failure. If your system is multiprocess/multithreaded/multi-whatever, do yourself a favor and also log the PID/Thread ID respectively. Also, be very careful with how you represent time, won't explain why, but time in computer systems is a pain, [see for yourself](https://www.youtube.com/watch?v=-5wpm-gesOY).

```
ERROR: Error name, message, traceback, variables in scope is possible
WARNING: Warning name, message
INFO: Calls to top-level functions/handlers, like: [2021-05-17 00:06:23] INFO: GET /posts 200 OK
DEBUG: Program setup/initialization info, possibly memory or performance information*

*: more on that later
```

- Thy logs must be **filtrable**: logs are meant to be analyzed. Make them as searchable as possible. Consider formatting them as JSONs, and don’t abuse nesting. Why not? If the JSON is too nested, it becomes very hard to search/analyze, thus defying it's purpose.

For example Elasticsearch won't be able to properly index JSONs with 2 or more levels of nesting. That is, something like this can be indexed:

```
{"timestamp": "2021-05-18T21:09:54Z", "level": "error", "msg": "bad thing happened"}
```

Even something like this:
```
{"timestamp": {"date": "17th May, 2021", "time": "11:30:30am"}, "level": "error", "msg": "bad thing happened"}
```

But do something like this:
```
{"timestamp": {
    "date": "17th May, 2021",
    "time": [11, 30, 30, 124]
    },
 "level": "error",
 "msg": "bad thing happened",
 "context": {
    "some_key_for_multiple_values": []
    }
}
```

And Elastic will treat your deeply nested elements like strings, and the good luck filtering and aggregating these logs. So keep it flat, whenever possible.

Another good format is NCSA Common log format, but if possible, prefer JSON. Why? Most log analysis tools use JSON. Something like NCSA Common log format is better for smaller systems, where you can search your logs with `grep` and friends. Also, don't forget this: _Whatever format you choose, be consistent acros your whole system_

```
Bad log (1): [2021-05-17 12:30:30] ERROR: KeyError // JSON version would be just as bad
Bad log (2): {"datetime": {"date": "17th May, 2021", "time": "11:30:30am"}, "type": "ERROR", "msg": "A KeyError error occured in function some_function"}
Better log: {"timestamp": "2021-05-18T21:09:54Z", "level": "error", "pid": 1201, "traceback": <your traceback as a string>, "msg": "KeyError: 'key_name'"}
```


## Some wisdom on logging ops

So you have well written logs. That's great!! But now you have to decide how to access and analyze them. And the funniest part, these decisions should also be guided by the stage and the scale of your system. In other words, I would advice against a complex infrastructure if you have one app serving a few hundred people.

But let's dive into details.

First, log collection. It should be obvious to you that we want to save our logs somewhere and not just let them print to stderr/stdout. But now have to think about where do we write them, it could be a file, or to syslog, for example, or we could even write them into a TCP or UDP socket, sending them away to some logging server. To be honest, all choices are good.

As long as you don't block the thread where the action happens, you should be fine, otherwise prepare for a performance hit.

So here comes my personal opinion: you should start by logging into a file, and mandatory ensure log file rotation, because you don't want a single 10GB text file. Believe me, you don't. At some point you will also have to think of log compression and possibly log shipping. Log shipping means transfering the logs from where these were created to where these will be analyzed and stored for a long period of time.

![](/_data/LoggingArch.png)

When it comes to log shipping, I would strongly suggest using TCP or HTTP over UDP and other protocols. Why, you may ask? Because first of all, with UDP you might lose logs while transfering them due to (1) no way of retransmiting lost packets, (2) no flow control, which might be the cause of lost packets, but also because with UDP message size is limited to 65KB of data, or even less, depending on network settings, which quite frankly could be not nearly enough. Also, your company firewalls might block this kind of traffic. So, a lot of trouble.

Once you have multiple services, you can think of a centralized logging server, something like an ELK (Elasticsearch, Logstash, Kibana) cluster, with one or a few of Elastic instances in a cluster setup.

Having a centralized logging solution, you will now absolutely need to ship the logs, and having them first written to a file will prove a very nice idea, because now your logs won't be lost in case of network outages, server failure, logging system failure or any of the above mentioned being too slow.

Nice. 

![](https://media.giphy.com/media/k0hKRTq5l9HByWNP1j/giphy.gif)


# Act 1.1: Hey, I think I can make a chatbot to notify me when something blows up

Yup, you can. And if you want to reduce MTTR you most likely should. Just take into account a few things.

First and foremost, if you have the possibility, set up alerting thresholds. You don't want to be notified when something is even slightly off every. single. time. Maybe it's some unique event, no need to bother, while if the issue happens frequently, you better be notified.

Another consideration when it comes to alerting, is the possibility to have **escalation alerting**. Basically, first send an alert via email. If no action was taken, now send it to a chat group of the responsible team. Still no activity? Send it in DM to an engineer, or even to a technical manager.

Finally, just aggregate the stuff, no need for 12, or a hundred, emails/Slack messages of the same issue. Something like one log message and then some text like `X occured 25 times in the last Y seconds` should be good.

When it comes to what tools to use for alerting, well, you have Sentry, also to my knowledge it is possible to setup alerting in Kibana, although I don't know whenever this is a paid option or free, and there are of course other tools.

This is by no means a definitive guide on how to do it, only some things to keep in mind. This whole blog post isn't a definitive guide, if you haven't noticed yet.


# Act 2: My system is slow, I guess I'll log execution time, and # of requests, and ...

... just. Stop. Please. The fact that you **can** do it, doesn't mean you should. Welcome to the world of telemetry and performance monitoring, where you will initially wonder, why not use just logs? I mean, in principle you could do this, but better to have a different infrastructure, so as to not mess everything up.

So, the easiest thing if of course to use `TRACE` level of logging, and most importantly a dedicated infrastructure for performance monitoring. Otherwise, you will put too much pressure on your logging infrastructure.

But then you might start looking towards more restricted type of logs, maybe some binary protocols, given that you will be sending small packets of information very frequently.

Performance monitoring has a bit of a different write and query patterns than log analytics, so different storage is recommended. Queries are simpler mainly showing trends, time series, curent values, or some simple aggregate values, like counts, means, medians and percentiles, and writes are very frequent but with little data, only a few metrics, compared with logging tracebacks and contexts and stuff like that.

That's why for example ELK stack is more common in logging infrastructure, where Elasticsearch can index and analyze even very unstructured data, and stuff like Grafana + Prometheus are more commonly used for performance monitoring. Prometheus, among other things, is time series database, just the right thing to store and quickly query performance metrics.

Also, when it comes to performance analysis, you will want to monitor your system utilization, not just the stuff intrinsinc to your code. If you're using Prometheus, that's easy to do.


# Act 3: My microservice system is slow, but I can't figure out why

First, a likbez (crash-course) on networking and dynamic systems. Agains our intuition, a computer network is a shared resource with a limited capacity. Which basically means if one service is very chatty, it will influence throughput and latency for all the rest. Also given that network are a priori not 100% reliable and we mostly use TCP, we will have a lot of traffic. That's only half the problem though. There's more. 

Our services are dependent upon each other. So if one service is slow, it might influence other services, even ones that are not directly interacting with it. One metaphor to help you think of it is a spider web. When you touch it on one side, it could ripple on the other side. Like a butterfly effect.

So, how do we monitor this? Maybe logs? Or something like performance monitoring from the previous act?

Well, I mean, it's a start, but only logs won't cut it. Because we don't see the full picture, specifically, we don't see the interaction between services, only each individual's performance. Enter **tracing**.

First, a good mental model about tracing is that it's like logging, but with a correlation identifier, which makes it possible to combine said logs into a "trace".
A trace like this now can show us how for example a single request spans multiple services, how much time does each step take and even how much time was spent on communication. All this can help uncover bugs and performance bottlenecks which a simple performance monitoring tool, or just logs, won't be able to do.


TK START

Tracing, why? How?
<!-- If you add correlation IDs to debug logs and pass them between services, now you have a (half-baked) form of tracing. Tracing will help you find bottleneck services, and sometimes even aid you in debugging distributed systems. -->

Hold on, I have logs? Logs vs Traces?

How to structure your traces?

Can you use traces as logs? Yes, but careful now. Traces primary purpose is to uncover performance issues, also sometimes pinpoint the reason a specific operation failed. You could use them as logs, but don't overload them whith information, otherwise your collection, storage and analysis infrastructure will cry.

Jaeger, Zipkin? Architecture. Good setup.

TK END


# Final act: Welcome to observability!!!

Observability, what? 

Observability is the property of a system to be understood. It's a property of how well can one infer the internal state of something from its external outputs.
It’s a spectrum, and depending where on it your system stands, you can use monitoring and alerting more or less efficiently.
In other words, if a system is observable you can understand what is happening within it from its outputs.

We need to design our systems with observability in mind. And with all the stuff outlined above, that should become a doable task.

I prefer to think of observability, with proper incident response of course, as a way to make a system anti-fragile (see the works of Nasim Taleb),
because with every failure and issue that happens, it "learns", on organizational level, to be better. Or one could argue that on the contrary, the system now becomes more fragile, because with every fix we believe more and more that the system is now unkillable, which it never will be.

Pick for yourself, but don't forget to use logging. At least you'll know when and why things go south, and that's something.


# Epilogue

You've made it! Congrats! Now you have some very important knowledge of how to be prepared when manure hits the proverbial fan in production.
This knowledge should help you debug even super-obscure bugs. Of course this isn't going to be easy, plus you now have an entire infrastructure to take care of,
but hey, if this helps reducing time to solve an issue from 1 week (or more) to 1, maybe 2 days, it might be worth it.

I know for a fact that it was worth it for me, time and time again, when it helped me quickly identify edge cases, stupid missconfigurations and performance bottlenecks.


So yeah, that's it for now. Incredibly, it didn't took much time since my last blog post.

Finally, if you’re reading this, I’d like to thank you. Let me know what are your thoughts about it via Twitter, for now, until I plug in some form of comment section. Your feedback is valuable for me.


<!-- https://ferd.ca/erlang-otp-21-s-new-logger.html
https://iamondemand.com/blog/open-source-distributed-tracing-why-you-need-it-how-to-get-started/ -->