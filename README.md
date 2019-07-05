<H1> Kaggle Jigsaw Competition attempt </H1>
The initial idea of this was to use a bidirectional LSTM with some feedforward layers based around some
existing research around the idea of applying bidirectional LSTM's on text. One interesting thing to note
is that we use a cudnn lstm as opposed to a GRU because the GRU seems to cause a minor dip in accuracy.
You can see the initial efforts in the bidirectional lstm file, this was somewhat good but overall I would
say that the preprocessing was lacking and the model itself was just not as good.

The next attempt involved a bidirectional LSTM with an attention layer and 2 pooling layers. It also involved
heavy text preprocessing because I suspect many contractions, punctutation and swear words were confusing the
model. This attempt yielded great results and at this point I was considering stopping because finals suck but I thought it would be interesting to see what sort of results BERT would yield.

I didn't have time to really implement it so here is a notebook from @Morris Park(https://www.kaggle.com/morrispark) who managed to achieve a score of 0.94217 which is significantly higher than my score with my
custom attention + bidirectional LSTM model.

Overall I would say this was a great learning experience and I think the kaggle community is an excellent place to learn more about state of the art systems.

Notebook: (https://www.kaggle.com/morrispark/3bert-1stm-custom-loss-cleansing)
