The initial idea of this was to use a bidirectional LSTM with some feedforward layers based around some
existing research around the idea of applying bidirectional LSTM's on text. One interesting thing to note
is that we use a cudnn lstm as opposed to a GRU because the GRU seems to cause a minor dip in accuracy.
