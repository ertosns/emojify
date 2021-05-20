# emojify
is the many to one RNN network with LSTM, note that the model is only trained on ten's of samples, despite that the results are satisfying, but fails in case of sentences with contradicting words, look the last sample.

# sample execution:

``` console
nlp@bot$python3 emojify.py &>/dev/null &
studying machine learning is fruitful
😄
natural language processing is modern day wizard
😄
ignorance is the doom of our society
😞
failure is one step closer towards success
😞
```
