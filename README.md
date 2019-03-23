# my-pytorch-bert
These codes are [BERT](https://arxiv.org/abs/1810.04805) implementation by PyTorch.

The base of this implementation is [google BERT](https://github.com/google-research/bert) and [pytorch-pretrained-BERT](https://github.com/huggingface/pytorch-pretrained-BERT).
And we add [bert-japanese](https://github.com/yoheikikuta/bert-japanese) as [SentencePiece](https://github.com/google/sentencepiece) Tokenizer.

### How to convert from TensorFlow model to my model
```
python load_tf_bert.py \
    --config_path=json-file \
    --tfmodel_path=bert-wiki-ja/model.ckpt-1400000 \
    --vocab_num=32000 --output_path=bert-wiki-ja.pt
```
json-file example:
```
{
	"vocab_size": 32000,
	"hidden_size": 768,
	"num_hidden_layers": 12,
	"num_attention_heads": 12,
	"intermediate_size": 3072,
	"attention_probs_dropout_prob": 0.1,
	"hidden_dropout_prob": 0.1,
	"max_position_embeddings": 128,
	"type_vocab_size": 2,
	"initializer_range": 0.02
}
```

### How to Classifier train
```
python run_classifier.py \
 --config_path=config/bert_base.json  \
 --dataset_path=/content/drive/My\ Drive/data/sample_train.tsv \
 --pretrain_path=/content/drive/My\ Drive/pretrain/bert.pt \
 --vocab_path=/content/drive/My\ Drive/data/sample.vocab \
 --sp_model_path=/content/drive/My\ Drive/data/sample.model \
 --save_dir=classifier/  \
 --batch_size=4  \
 --max_pos=512  \
 --lr=2e-5  \
 --warmup_steps=0.1  \
 --epoch=10  \
 --per_save_epoch=1 \
 --mode=train \
 --label_num=9
```

### How to Classifier evaluate
```
python run_classifier.py \
 --config_path=config/bert_base.json \
 --dataset_path=/content/drive/My\ Drive/data/sample_eval.tsv \
 --model_path=/content/drive/My\ Drive/classifier/classifier.pt \
 --vocab_path=/content/drive/My\ Drive/data/sample.vocab \
 --sp_model_path=/content/drive/My\ Drive/data/sample.model \
 --batch_size=4 \
 --max_pos=512 \
 --mode=eval \
 --label_num=9
```

### How to train Sentence Piece
```
python train-sentencepiece.py --config_path=json-file
```
json-file example:
```
{
    "text_dir" : "tests/",
    "prefix" : "tests/sample_text",
    "vocab_size" : 100,
    "ctl_symbols" : "[PAD],[CLS],[SEP],[MASK]"
}
```

### How to pre-train
```
python run_pretrain.py \
 --config_path=config/bert_base.json \
 --dataset_path=/content/drive/My\ Drive/data/sample.txt \
 --vocab_path=/content/drive/My\ Drive/data/sample.vocab \
 --sp_model_path=/content/drive/My\ Drive/data/sample.model \
 --save_dir=pretrain/ \
 --batch_size=4 \
 --max_pos=256 \
 --lr=5e-5 \
 --warmup_steps=0.1 \
 --epoch=20 \
 --per_save_epoch=4 \
 --mode=train
```


### Classification result of my-pytorch-bert
- Dataset : [livedoor ニュースコーパス](https://www.rondhuit.com/download.html)  6(training):2(evaluate)
- epoch : 10

1. [Pretrained BERT model and trained SentencePiece model](https://drive.google.com/drive/folders/1Zsm9DD40lrUVu6iAnIuTH2ODIkh-WM-O?usp=sharing) (PyTorch converted).
 ```
               precision    recall  f1-score   support

            0       0.96      0.92      0.94       178
            1       0.93      0.97      0.95       172
            2       0.98      0.95      0.97       176
            3       0.93      0.92      0.92        95
            4       0.96      0.98      0.97       158
            5       0.91      0.93      0.92       174
            6       0.97      0.98      0.97       167
            7       0.98      1.00      0.99       190
            8       0.98      0.96      0.97       163

    micro avg       0.96      0.96      0.96      1473
    macro avg       0.96      0.96      0.96      1473
weighted avg       0.96      0.96      0.96      1473
```

