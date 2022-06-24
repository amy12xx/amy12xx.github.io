---
layout: post
title:  "On Language"
date:   2019-08-09 20:05:00 +0530
categories: NLP
---

> A summary of the Introduction (chapter) to Statistical Natural Language Processing by Manning & Schütze.

<!--more-->

This post is a summary of the Introduction (chapter) to Statistical Natural Language Processing by Manning & Schütze.

### “All Grammars Leak”

It is difficult, if not impossible to create an exact and complete characterization of a natural language. Language is constantly evolving to support its primary purpose of communication.

### Empiricist and Rationalist Approaches

A rationalist’s approach considers language learning as innate, one that we are born with. It is interested in this I-Language (or language of the mind). It is concerned with both a linguistic competence (knowledge of the language structure present in the mind of a native speaker), and linguistic performance in the world.

Empiricist approaches focus on the language as it is occurs in the world.

### Non categorical phenomena (of language)

For the most part, language (constituents) can be placed into categories (for e.g. The belongs to DET (Determiner) part of speech class. But there are several cases where a word might belong to several different categories. Its usage over time may also change.

E.g.1. “while” was typically used to denote time, as in “take a while”. Later uses involve the term as a complementizer; as in “while you were out”.

E.g.2. “kind of/sort of” were used mainly as prepositional phrases. X is a kind of A. Later uses constitute them being used to mean somewhat, as in “I kind of understand”.

Statistical NLP deals mostly with the categorical phenomena of language (driven by its most frequent occurrences/uses).

### Probability & Ambiguity

Human cognition is probabilistic. We make decisions/inferences based on what information we have collectively about an event. Since language is an integral part of cognition, it must also be probabilistic.

“Use theory of meaning” implies the meaning of a word is defined by its uses. (Wittgenstein)

Language is ambiguous. Using a reasonable grammar we will typically have multiple valid parses for a sentence (some parses are more likely than others, including those that are semantically invalid). A grammar will have to trade-off between coverage, and computational complexity (caused by an explosion of parses).

### Zipf’s Law

Zipf’s law establishes a relationship between the frequency of a word in a corpus, and it’s rank.

```
f ∝ 1/r
```

or

```
f . x = k
```

So, the 50th ranked word is 3 times more frequent than the 150th ranked word.

This aligns with Zipf’s theory of the principle of least effort. Both speaker and reader want to minimize the effort of communication (by using a smaller vocabulary of common words)

Also, m ∝ sqrt(f), where m is # of meanings

### Collocations, Concordances

Collocation: A turn of phrase that has meaning or exists beyond its constituents.

Concordance: Key Word in Context (KWIC) program that displays a word/phrase along with its left and right contexts or the syntactic frames of its left/right contexts.
