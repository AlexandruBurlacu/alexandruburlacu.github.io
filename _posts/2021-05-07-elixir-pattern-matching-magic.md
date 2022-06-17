---
title: Elixir pattern matching magic
published: true
description: Elixir, as a true functional language, allows for some impressive feats using pattern matching. Let's see some of this magic.
tags: functional, elixir, pattern matching, programming, software engineering, erlang
keywords: functional, elixir, pattern matching, programming, software engineering, erlang
layout: post
date:   2021-05-07 22:12:00 +0100
categories: posts
permalink: /posts/2021-05-07-elixir-pattern-matching-magic
comments: true
---

# Prologue

So, a while ago, while preparing for an off-topic lecture about polymorphism and type systems, I recalled an interesting concept called _multiple dispatch_. I won't go into details of what it is, so if you're interested, check these links: [1](http://matthewrocklin.com/blog/work/2014/02/25/Multiple-Dispatch), [2](https://en.wikipedia.org/wiki/Multiple_dispatch), [3](https://eli.thegreenplace.net/2016/a-polyglots-guide-to-multiple-dispatch/).
Anyway, while brushing up my knowledge about multiple dispatch, I found an even more powerful technique, called [_predicate dispatch_](https://en.wikipedia.org/wiki/Predicate_dispatch).

To me, it seemed a lot like what is possible through pattern matching in functional languages. After some research, I asked on SO whenever my assumption was right, [here](https://stackoverflow.com/q/66863443/5428334). TL;DR: no answer as of today(2021/05/07).

<strong>Why am I telling you all this? Because that's how I decided to write an article about how cool pattern matching is, specifically in Elixir, and even if it's not the same as predicate dispatch, it's pretty damn powerful nevertheless.</strong>

So let's get started!

# Basics

I will quickly go through the basics of Elixir pattern matching, before diving into real neat stuff.

In Elixir `=` doesn't just assign some value to some variable, it also matches the left-hand side of the expression with the right-hand side. So as a result, doing something like the code below is entirely possible.

```elixir
iex(0)> x = 2
2
iex(1)> y = 4
4
iex(2)> 4 = y
4
iex(3)> 2 = y
** (MatchError) no match of right hand side value: 4
```

No big deal, right? Wrong! Because of this interesting property, we can do matching on composite data types, for example on lists.

### Lists

In Elixir `[] = []` is a valid expression. But now, we can also write something like:

```elixir
iex(0)> xs = [1, 2]
[1, 2]
iex(1)> [x, y] = xs
[1, 2]
iex(2)> x
1
iex(3)> y
2
```

Starts to get interesting, eh? But wait, there's more!

```elixir
iex(0)> [head | tail] = [1, 2, 3, 4, 5]
[1, 2, 3, 4, 5]
iex(1)> head
1
iex(2)> tail
[2, 3, 4, 5]
```

![](https://media.giphy.com/media/3o6wrzcyVdjrEC5UaY/giphy.gif)

Aaaaand moreeee!!!!

```elixir
iex(0)> [head, next_to_it | tail] = [1, 2, 3, 4, 5]
[1, 2, 3, 4, 5]
iex(1)> head
1
iex(2)> next_to_it
2
iex(3)> tail
[3, 4, 5]
```

![](https://media.giphy.com/media/26AHLBZUC1n53ozi8/giphy.gif)

Noice.


### Tuples

Alright, so pattern matching can do interesting stuff. In Elixir it's so deeply ingrained that it's used for example to signal whenever or not we have an error. For this, pattern matching on tuples is used.

```elixir
iex(0)> {:ok, value} = SomeModule.some_function()
{:ok, "the value"}
iex(1)> # If the function returns something else than {:ok, value}
iex(2)> {:ok, value} = SomeModule.some_function()
** (MatchError) no match of right hand side value: {:ok, value}
```

Elixir has a special control structure to enable more flexible usage of pattern matching, it's called `case`.

```elixir
iex(0)> x = {:ok, "is fine"}
iex(1)> case x do
...(1)>   {:ok, v} -> v
...(1)>   _ -> "nothing at all"
...(1)> end
"is fine"
iex(2)> x = {:err, "not ok"}
iex(3)> case x do
...(3)>   {:ok, v} -> v
...(3)>   _ -> "nothing at all"
...(3)> end
"nothing at all"
```

This concludes the basics part, so now we're gonna dive into more interesting stuff.

# Functions

Elixir can use pattern matching even in function definitions, like in Haskell. And by the way, that's one of the most performant options, usually.

```elixir
defmodule FactorialM do
    def factorial(0), do: 1
    def factorial(1), do: 1
    def factorial(x) do
        x * factorial(x-1)
    end
end
```

Can you spot a problem with this function? Well, what if we pass a floating-point value instead of an integer? Think what would happen, and compare with the answer<sup>1</sup> at the end of the article.

How can you fix it? Enter guards.

```elixir
defmodule FactorialM do
    def factorial(0), do: 1
    def factorial(1), do: 1
    def factorial(x) when is_integer(x) do
        x * factorial(x-1)
    end
    def factorial(_), do: raise RuntimeError, "Input should be integer"
end
```

So now we can also define different paths for code execution depending on whenever or not our guard(s) are satisfied. Guards in Elixir are fairly limited, and hard-ish to extend, for safety reasons. Guards should be pure functions, and even if you try to define them using macros, the compiler still can check whenever they can be distilled down to existing guards and logical operators or not. For more information, see [this documentation page](https://hexdocs.pm/elixir/master/patterns-and-guards.html) and [this little tutorial/blog post](https://keathley.io/blog/elixir-guard-clauses.html) on how to write guards.

Finally, we can combine pattern matching capabilities of functions with those of tuples and lists and implement fairly interesting things. For example a map function.

```elixir
defmodule FairlyInteresting do
    def map([], _func), do: []
    def map([head | tail], func) when is_function(func) do
        [func.(head) | map(tail, func)]
    end
end
```

Also, using pattern matching on tuples in function prototypes is the go-to way of using Elixir's `GenServer`, `GenStage`, and other `Gen`-things. This pattern is inherited from Erlang's OTP and is pretty beautiful if you ask me.

```elixir
defmodule Stack do
    @moduledoc """Taken from: https://hexdocs.pm/elixir/master/GenServer.html"""
    use GenServer

    # Callbacks
    @impl true
    def init(stack) do
        {:ok, stack}
    end

    @impl true
    def handle_call(:pop, _from, [head | tail]) do
        {:reply, head, tail}
    end

    @impl true
    def handle_cast({:push, element}, state) do
        {:noreply, [element | state]}
    end
end
```

# It's getting more interesting

Remember I told you pattern matching can be applied to composite data? Well, it's not just lists, it's also maps, and by extension structs, here:

```elixir
iex(0)> kv = %{key: :value}
{key: :value}
iex(1)> %{key: data} = kv
{key: :value}
iex(2)> data
:value
```

And with structs:

```elixir
iex(0)> defmodule AStruct do
...(0)>     defstruct [:state]
...(0)> end
# I'll omit this for brevety
iex(1)> s = %AStruct{state: 12}
%AStruct{state: 12}
iex(3)> s.state
12
iex(4)> %{state: st} = s # recall, a struct is just syntactic sugar for a map
%AStruct{state: 12}
iex(5)> st
12
iex(6)> %AStruct{state: st} = s
%AStruct{state: 12}
iex(7)> st
12
```

### The as-pattern

What if you need to match a function parameter with some specific structure, but you also need a reference to the entire value. Have you ever heard about **as-patterns**?

```elixir
defmodule FairlyInteresting do
    def merge([], xs), do: xs
    def merge(xs, []), do: xs
    def merge(first=[x|xs], second=[y|ys]) do
        if x < y do
            [x | merge(xs, second)]
        else
            [y | merge(ys, first)]
        end
    end
end

# ...

iex(0)> FairlyInteresting.merge [1, 3, 4, 7], [2, 2, 4, 8, 9]
[1, 2, 2, 3, 4, 4, 7, 8, 9]
```

![](https://media.giphy.com/media/gdKAVlnm3bmKI/giphy.gif)

You still with us? Yes? Good, because the fun part hasn't even started yet.

### Partial functions

Moving on, in Elixir it is possible to define partial functions. Mathematically speaking, a partial function is a function defined only for some values, not for the whole set of values. For example, the division is technically a partial function, because we can't define it when the divisor is 0. We can make a partial function explicit via pattern matching. And it also works for anonymous functions!

```elixir
iex(0)> partial = fn 
...(0)>     {:ok, value} when is_number(value) -> value * 2
...(0)>     {:notok, _} -> :i_mean_its_not_ok
...(0)> end
iex(1)> partial.(12)
# raises a FunctionClauseError
iex(2)> partial.({:ok, 12})
24
```

### The pin (not my card's)

Finally, no discussion about pattern matching in Elixir would be complete without the `^` operator. So what is it?
It is commonly known as the pin operator, and it allows pattern matching without any assignment.

Recall that normally, using `=` we perform both pattern matching __and__ assignment. That is, we check whenever the left-hand side of the expression matches the right-hand side, and if so, all the variables in the expression get assigned to corresponding values.

Ex. `[1, x, y] = [1, 2, 3] # x = 2, y = 3`.

So, using `^` we can pattern match, but not assign. Like this:

```elixir
iex(0)> x = 2
2
iex(1)> ^x = 3
** (MatchError) no match of right hand side value: 3
iex(2)> ^x = 2
2
```

You might ask, where would I use this? Well, what about deciding in runtime what matching criteria are you interested in. For example:

```elixir
iex(0)> status_of_interest = :wip # imagine that it is decided while the program is running
iex(1)> # maybe can even change throughout the program lifetime
iex(2)> partial = fn 
...(2)>     {^status_of_interest, value} when is_number(value) -> value * 2
...(2)>     {:notok, _} -> :i_mean_its_not_ok
...(2)> end
iex(3)> partial.({:ok, 12})
# raises a FunctionClauseError
iex(4)> partial.({:wip, 11})
22
```

# Leveling up

Now we've seen enough, let's combine everything!

```elixir
defmodule Measurement do
    defstruct [:prob, status: :ok]
end

defmodule Measurement.CDF do
    defstruct [:value]
end

defmodule FairlyInteresting do
    @doc "CDF stands for cummulative density function"
    def kinda_cdf([], acc, _func), do: [%Measurement.CDF{value: acc}]

    # So, now we have pattern matching on structs, inside lists,
    #  with as-patterns and guards, isn't it cool?
    def kinda_cdf([%Measurement{prob: t, status: :ok}=head | tail], acc, func)
        when is_function(func, 2) do
        [%Measurement.CDF{value: acc} | kinda_cdf(tail, func.(t, acc), func)]
    end

    def kinda_cdf([_head | tail], acc, func) when is_function(func, 2) do
        kinda_cdf(tail, acc, func)
    end
end

# ...

iex(0)> ms = [%Measurement{prob: 0.11}, %Measurement{prob: 0.07},
...(0)>       %Measurement{prob: 0.31, status: :notok}, %Measurement{prob: 0.21},
...(0)>       %Measurement{prob: 0.17, status: :ok}, %Measurement{prob: 0.08, status: :notok}]
[
  %Measurement{prob: 0.11, status: :ok},
  %Measurement{prob: 0.07, status: :ok},
  %Measurement{prob: 0.31, status: :notok},
  %Measurement{prob: 0.21, status: :ok},
  %Measurement{prob: 0.17, status: :ok},
  %Measurement{prob: 0.08, status: :notok}
]
iex(1)> FairlyInteresting.kinda_cdf ms, 0.0, &(&1+&2)
[
  %Measurement.CDF{value: 0.0},
  %Measurement.CDF{value: 0.11},
  %Measurement.CDF{value: 0.18},
  %Measurement.CDF{value: 0.39},
  %Measurement.CDF{value: 0.56}
]
```

![](https://media.giphy.com/media/3o7aCZDlmQZLe4Q4V2/giphy.gif)

If you're like Patrick right now, I don't blame you, even I was a bit shocked while writing this.

And if that's not enough, we move onto the mindbending stuff. Bear with me.


### Working with bits

<!-- 
https://docs.replit.com/repls/embed -->

Something cool that Erlang and Elixir can do is pattern matching on binary data. Isn't this amazing?

Binary pattern matching in Erlang and Elixir exists because Erlang was initially developed to be used for network and telecom programming, that is implementing software for switches, routers, and servers; developing protocols, and doing this efficiently. Binary matching allows for very concise parsing of binary protocols.

```
    0                   1                   2                   3
    0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
   |          Source Port          |       Destination Port        |
   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
   |                        Sequence Number                        |
   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
   |                    Acknowledgment Number                      |
   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
   | Data |                                                        |
   |Offset|                      data                              |
   |      |                                                        |
   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
                        Some Binary Header Format
```

```elixir
iex(0)> source_port = <<12070 :: 16>>
"/&"
iex(1)> destination_port = <<80 :: 16>>
<<0, 80>>
iex(2)> seq_num = <<12_345_678 :: 32>>
<<0, 188, 97, 78>>
iex(3)> offset = <<0 :: 3>> # you can even specify bit-strings
<<0::size(3)>>
```

Now let's assemble the packet.

```elixir
iex(4)> header = << source_port <> destination_port <> seq_num <> seq_num, offset :: bitstring>>
<<47, 38, 0, 80, 0, 188, 97, 78, 0, 188, 97, 78, 0::size(3)>>
iex(5)> packet = <<header :: bitstring, <<"and here comes the data">> >>
<<47, 38, 0, 80, 0, 188, 97, 78, 0, 188, 97, 78, 12, 45, 204, 132, 13, 12, 174,
  76, 164, 12, 109, 237, 172, 174, 100, 14, 141, 12, 164, 12, 140, 46, 140,
  1::size(3)>>
```

Notice the strange way we assemble the packet. Because we are working on the bit level, not even on the byte level, sometimes concatenation (`<>`) isn't possible.
That's why we use the list-like behaviour of `<<>>`, that is to say this form: `<< 12::3, <<1::2, <<3::3>> >> >>` will be equivalent to this `<<12::3, 1::2, 3::3>>`.

Also, we need to specify that we're dealing with a `bitstring` not a sequence of `bytes`.

And now we match.

```elixir
iex(6)> <<_sp :: 16, _dp :: 16, _seq_num :: 32, ack_num :: 32, _offset :: 3, data :: bitstring>> = packet
<<47, 38, 0, 80, 0, 188, 97, 78, 0, 188, 97, 78, 12, 45, 204, 132, 13, 12, 174,
  76, 164, 12, 109, 237, 172, 174, 100, 14, 141, 12, 164, 12, 140, 46, 140,
  1::size(3)>>
iex(7)> data
"and here comes the data"
```

<!-- ```elixir
# https://hexdocs.pm/elixir/Kernel.SpecialForms.html#%3C%3C%3E%3E/1
# http://erlang.org/doc/programming_examples/bit_syntax.html
# https://bgmarx.com/2015/06/12/binary-pattern-matching-with-elixir/
# https://zohaib.me/binary-pattern-matching-in-elixir/
# https://dev.to/l1x/matching-binary-patterns-11kh
``` -->

Nice, but can we use `^` for more powerful matching?

```elixir
iex(5)> <<^i, 32, ^have, 32, ^a, "n ", apple>> = "I have an apple"
** (MatchError) no match of right hand side value: "I have an apple"
```

Now, this is one thing you can't do. You can't combine `<<>>` and `^` operators. Shame. But it's useful never the less, you'll see in a moment.

### A bit about strings

In Elixir you can even match the text. Nice, isn't it? But text, or strings, are arrays of bytes, so, it's pretty much obvious why can we do it.

```elixir
iex(0)> partial = fn 
...(0)>     {:ok, "he" <> v} -> v
...(0)>     {:still_ok, v <> "ou"} -> v
...(0)>     _ -> :nope
...(0)> end
iex(1)> 
iex(2)> partial = fn # won't compile
...(2)>     {:ok, "he" <> v} -> v
...(2)>     {:still_ok, v <> "ou"} -> v # because of this
...(2)> _ -> :nope
...(2)> end
** (ArgumentError) the left argument of <> operator inside a match should always be a literal binary because its size can't be verified.
```

Well, the capability is very limited, because potentially you could have a very long string, and checking its end potentially could be a very expensive operation.
There's a way tho.

Back to bit sequences. So, because of the Erlang legacy, strings can be treated as sequences of characters, which in turn are just sequences of short unsigned integers. So, if you know the size of the matchable subsequence, you could potentially match even in the middle of the string.

Just in case someone needed it. If you need to match on the part of the string that is in the known middle and you aware of its length then you can use binary matching:
```elixir
iex(1)> <<"I ", v::binary-size(9), "ing">> = "I got a string"
iex(2)> v
"got a str"
```

Strings and bits and bytes and pattern matching in Elixir is a huge topic, so don't worry if you're confused at this moment. You could check [this](https://medium.com/blackode/playing-with-elixir-binaries-strings-dd01a40039d5) post about exactly that if my ramblings didn't make sense ;)

# Epilogue

I hope you like it. I don't know about you, but I like to discover weird powerful things like all the stuff above. I mean, lists and tuples are fine, but to be able to pattern match on bits, that's some Voodoo magic in here.

So yeah, that's it for now. I might write some more about advanced Elixir stuff, most likely related to the actor model. Let's hope it won't take as long as usual.

If you’re reading this, I’d like to thank you and hope all of the above written will be of great help for you, as it was for me. Let me know what are your thoughts about it via Twitter, for now, until I plug in some form of comment section. Your feedback is valuable for me.

1. Oh, yeah, the answer **It will run until killed by the OS, because 1 is not 1.0 in Elixir, nor 0.0 is 0**.

<!-- https://replit.com/@AlexandruBurlac/ElixirPatternMatchingMagic -->

# P.S.

For your efforts, I'd like to reward you with the possibility to run all these examples in a sandbox (`elixir <filename.exs>`). Knock yourself out ;)

<iframe frameborder="0" width="100%" height="700px" src="https://replit.com/@AlexandruBurlac/ElixirPatternMatchingMagic?lite=true"></iframe>
