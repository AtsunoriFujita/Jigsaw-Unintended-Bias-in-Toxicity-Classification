# jigsaw-Unintended-Bias-in-Toxicity-Classification
This respository contains my code for competition in kaggle.


7th Place Solution for [Jigsaw Unintended Bias in Toxicity Classification](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification "Jigsaw Unintended Bias in Toxicity Classification")

Team: [Abhishek Thakur](https://www.kaggle.com/abhishek), [Duy](https://www.kaggle.com/pvduy23), [R0seNb1att](https://www.kaggle.com/frankrosenblatt), [atfujita](https://www.kaggle.com/atsunorifujita)

All models(Team)    
Public LB: 0.94729(3rd)   
Private LB: 0.94660(7th)

##### Note: This repository contains only my models and only train script.


My models   
Public LB: 0.94719   
Private LB: 0.94651

Thanks to Abhishek and Duy's wonderful models and support, I was able to get better results.

### Set up
- Particularly important libraries are listed in requirements.txt


### Models
I created 5 models

- LSTM
  - Based on the [Quora competition model](https://github.com/AtsunoriFujita/Quora-Insincere-Questions-Classification)
  - Architecture: LSTM + GRU + Self Attention + Max pooling
  - Word embeddings: concat glove and fasttext.
  - Optimizer: AdamW
  - Train:
    - max_len = 220
    - n_splits = 10
    - batch_size = 512
    - train_epochs = 7
    - base_lr, max_lr = 0.0005, 0.003
    - Weight Decay = 0.0001
    - Learning schedule: CyclicLR


- BERT
  - The model is based on [yuval reina](https://www.kaggle.com/yuval6967)'s graet kernel(https://www.kaggle.com/yuval6967/toxic-bert-plain-vanila)
  - Changes are loss function and preprocessing.
  - I created 4 BERT models.
    - BERT-Base Uncased
    - BERT-Base Cased
    - BERT-Large Uncased(Whole Word Masking)
    - BERT-Large Cased(Whole Word Masking)
  - Train:
    - max_len = 220
    - train samples = 1.7M, val samples= 0.1M
    - batch_size = 32(Base), 4(Large)
    - accumulation_steps = 1(Base), 16(Large)
    - train_epochs = 2
    - lr = 2e-5


## Worked well
The loss function was very important in this competition.   
In fact, all winners used different loss functions.

My loss function is below.

```python
y_columns = ['target']

y_aux_train = train_df[['target', 'severe_toxicity', 'obscene',
                        'identity_attack', 'insult',
                        'threat',
                        'sexual_explicit'
                        ]]

y_aux_train = y_aux_train.fillna(0)

identity_columns = [
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness']
# Overall
weights = np.ones((len(train_df),)) / 4
# Subgroup
weights += (train_df[identity_columns].fillna(0).values >= 0.5).sum(
    axis=1).astype(bool).astype(np.int) / 4
# Background Positive, Subgroup Negative
weights += (((train_df['target'].values >= 0.5).astype(bool).astype(np.int) +
             (train_df[identity_columns].fillna(0).values < 0.5).sum(
                 axis=1).astype(bool).astype(np.int)) > 1).astype(
    bool).astype(np.int) / 4
# Background Negative, Subgroup Positive
weights += (((train_df['target'].values < 0.5).astype(bool).astype(np.int) +
             (train_df[identity_columns].fillna(0).values >= 0.5).sum(
                 axis=1).astype(bool).astype(np.int)) > 1).astype(
    bool).astype(np.int) / 4

y_train = np.vstack(
    [(train_df['target'].values >= 0.5).astype(np.int), weights]).T

y_train = np.hstack([y_train, y_aux_train])


def custom_loss(data, targets):
    ''' Define custom loss function for weighted BCE on 'target' column '''
    bce_loss_1 = nn.BCEWithLogitsLoss(
        weight=targets[:, 1:2])(data[:, :1], targets[:, :1])
    bce_loss_2 = nn.BCEWithLogitsLoss()(data[:, 1:2], targets[:, 2:3])
    bce_loss_3 = nn.BCEWithLogitsLoss()(data[:, 2:3], targets[:, 3:4])
    bce_loss_4 = nn.BCEWithLogitsLoss()(data[:, 3:4], targets[:, 4:5])
    bce_loss_5 = nn.BCEWithLogitsLoss()(data[:, 4:5], targets[:, 5:6])
    bce_loss_6 = nn.BCEWithLogitsLoss()(data[:, 5:6], targets[:, 6:7])
    bce_loss_7 = nn.BCEWithLogitsLoss()(data[:, 6:7], targets[:, 7:8])
    bce_loss_8 = nn.BCEWithLogitsLoss()(data[:, 7:8], targets[:, 8:9])

    return bce_loss_1 + bce_loss_2 + bce_loss_3 + bce_loss_4 \
           + bce_loss_5 + bce_loss_6 + bce_loss_7 + bce_loss_8
```
