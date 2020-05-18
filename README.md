# Sentiment Analyzer

## Introduction

This project was built with the purpose of classifying sentiment of hotel reviews that were scraped from review sites. These reviews can be found in the [/Datasets](https://github.com/rajchandak/Sentiment-Analyzer/tree/master/Datasets) directory of this repository. Using 6 features (described below), we use **Binary Logistic Regression** by running **1000 epochs** of **Stochastic Gradient Descent**. The end result is the binary classification of the test set reviews into '**POS**' for positive and '**NEG**' for negative, with an accuracy of ~94%. THe output can be found in the [output](https://github.com/rajchandak/Sentiment-Analyzer/blob/master/output.txt) file.

## Features

I made use of the following 6 features for Logistic Regression:

**1. Count all positive lexicons**: Using a list of positive words, I calculated the number of positive words in each review.

**2. Count all negative lexicons**: Using a list of negative words, I calculated the number of negative words in each review.

**3. Check for the word "no"**

**4. Count all pronouns**: Using a list of pronouns, I calculated the  number of negative words in each review.

**5. Check for "!"**

**6 Count all words in the review**: Words do not include puncutation marks.

## Output

Each review is classified by ID. Here's a sample output:

```
ID-1158	POS
ID-2888	NEG
ID-2980	NEG
ID-1038	POS
ID-1170	POS
ID-1236	NEG
...
```
