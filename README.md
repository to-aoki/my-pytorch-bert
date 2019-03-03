# my-pytorch-bert
These codes are [BERT](https://arxiv.org/abs/1810.04805) implementation by PyTorch.

The base of this implementation is [google BERT](https://github.com/google-research/bert) and [pytorch-pretrained-BERT](https://github.com/huggingface/pytorch-pretrained-BERT).
And we add [bert-japanese](https://github.com/yoheikikuta/bert-japanese) as [SentencePiece](https://github.com/google/sentencepiece) Tokenizer.


### Classification result of my-pytorch-bert
- Dataset : [livedoor ニュースコーパス](https://www.rondhuit.com/download.html)  8(training):2(evaluate)
- epoch : 10

1. [Pretrained BERT model and trained SentencePiece model](https://drive.google.com/drive/folders/1Zsm9DD40lrUVu6iAnIuTH2ODIkh-WM-O?usp=sharing) (PyTorch converted).
  ```
                  precision    recall  f1-score   support

              0       0.86      0.94      0.90       173
              1       0.96      0.86      0.91       174
              2       0.99      0.92      0.95       173
              3       0.79      0.90      0.85       103
              4       0.96      0.98      0.97       174
              5       0.89      0.89      0.89       169
              6       0.93      0.98      0.96       174
              7       0.96      0.99      0.98       180
              8       1.00      0.87      0.93       154

      micro avg       0.93      0.93      0.93      1474
      macro avg       0.93      0.93      0.92      1474
   weighted avg       0.93      0.93      0.93      1474

 ```

