# LLMTransformers
LLM project using Roberta model for training.

## Step 1: Creating a tokenizer


```python
#https://huggingface.co/
!pip install transformers[torch]
```
    
    


```python
PATH = './sample_data/'
dados_treino = 'crepusculoDosIdolos.txt'
```


```python
from tokenizers import ByteLevelBPETokenizer

# Initialize a ByteLevelBPETokenizer
tokenizer = ByteLevelBPETokenizer()
# min_frequency minimum amount of occurrences to collect.
tokenizer.train(files=[PATH+dados_treino], vocab_size=52_000, min_frequency=2, special_tokens=[
    "<s>", # start of line.
    "<pad>", # filling empty values.
    "</s>", # end of sentence.
    "<unk>", # unknown character.
    "<mask>", # character that will predict the sentence.
])

```


```python
# Test
tokenizer.encode("Hoje é um novo dia!").ids
```




    [44, 83, 570, 306, 300, 1714, 556, 5]




```python
tokenizer.decode([44, 83, 570, 306, 300, 1714, 556, 5])
```




    'Hoje é um novo dia!'




```python
# at this point, our model already has a tokenizer built from the data
# vocab.json, list of tokens ordered by frequency - converts tokens to IDs
# merges.txt - maps texts to tokens

!rm -r ./sample_data/RAW_MODEL
!mkdir ./sample_data/RAW_MODEL
tokenizer.save_model(PATH+'RAW_MODEL')
```

    rm: cannot remove './sample_data/RAW_MODEL': No such file or directory

    ['./sample_data/RAW_MODEL/vocab.json', './sample_data/RAW_MODEL/merges.txt']



## Step 2. Building our Tokenizer


```python
# Using our tokenizer
# RobertaTokenizer
# https://huggingface.co/docs/transformers/tokenizer_summary

from transformers import RobertaTokenizer
# reinstantiating the tokenizer
# from_pretrained, you already have the vocabulary with the merges built, you just need to put it in the tokenizer.
tokenizer = RobertaTokenizer.from_pretrained(PATH+'RAW_MODEL', max_len=512)
```

## Step 3. Creating our Transformer


```python
from transformers import RobertaConfig
# configure a Roberta type transformer
config = RobertaConfig(
    vocab_size=52_000,
    max_position_embeddings=512,# amount of vectors within the model
    num_attention_heads=12,# attention mechanism. Server to understand the context, the main subject of the sentence.
    num_hidden_layers=6,# amount of hidden layers
    type_vocab_size=1,
)

from transformers import RobertaForMaskedLM
model = RobertaForMaskedLM(config=config) # instantiate the model with Roberta's transformer task, with the defined settings.
```


```python
# how many parameters does our neural network have.
model.num_parameters()
```




    83502880



## Step 4. Creating our tokenized Dataset


```python
# simple way to load a raw file as a Dataset.
from transformers import LineByLineTextDataset
# LineByLineTextDataset will take the training_data, with the tokenizer that was defined, to feed the transformer model dataset.
dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=PATH+dados_treino,
    block_size=128, # amount of the vector that will be transformed.
)

```

    /usr/local/lib/python3.10/dist-packages/transformers/data/datasets/language_modeling.py:119: FutureWarning: This dataset will be removed from the library soon, preprocessing should be handled with the 🤗 Datasets library. You can have a look at this example script for pointers: https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_mlm.py
      warnings.warn(
    


```python
# verification test
dataset.examples[:2]
```




    [{'input_ids': tensor([   0,   69,  276, 1154,  341,  306,  277,  273,   73,  271,  446, 1058,
                18,  352,   35, 1155,  262, 1058,  300,  527, 2240,   35,    2])},
     {'input_ids': tensor([   0,   83,  358, 1142, 3664, 1816,  272,  687, 2688,  781, 4651,  315,
              2377,  271, 2768, 1635,  285,  811, 2375,  527,    2])}]




```python
tokenizer.decode(dataset.examples[7]['input_ids'])
# seventh position
```




    '<s>na escola bélica da vida — o que não me faz morrer me torna mais forte.</s>'



## Step 5. Training the model


```python
'''
Data Collators are strategies for building batches of data
to train the model. It creates lists of samples from the dataset and
allows Pytorch to apply backpropagation appropriately.
Probability = probability of masking tokens from the input
'''
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.1
)
```


```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir=PATH+'RAW_MODEL',
    overwrite_output_dir=True,
    num_train_epochs=1200,# number of times it will go through the training data.
    per_device_train_batch_size=64,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,# prediction losses.
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)
```


```python
trainer.train()
```


```python
trainer.save_model(PATH+'RAW_MODEL')
```

## Step 6. Testing the model


```python
from transformers import pipeline

fill_mask = pipeline(
    "fill-mask",
    model=PATH+'RAW_MODEL',
    tokenizer=PATH+'RAW_MODEL'
)
```


```python
texto = 'Digo que o amor é verdadeiro.'
```


```python
texto = 'O verdadeiro valor da vida.'
```


```python
texto = 'Digo-lhes: '
```
