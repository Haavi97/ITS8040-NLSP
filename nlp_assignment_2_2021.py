# -*- coding: utf-8 -*-
import torch.nn.functional as F
import torch.nn as nn
from cachetools import cached, LRUCache, TTLCache
from collections import Counter
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

# Convert the CONLL data to our format
from itertools import chain, groupby


def read_conll(filename):
    result = []
    f = open(filename)
    lines = (str.strip(line) for line in f)
    groups = (grp for nonempty, grp in groupby(lines, bool) if nonempty)

    for group in groups:
        group = list(group)

        obs, lbl = zip(*(ln.rsplit(None, 1) for ln in group))
        lbl = [l.lstrip("B-").lstrip("I-") for l in lbl]
        word = [x.split()[0] for x in obs]

        result.append(list(zip(word, lbl)))
    return result


train_data = read_conll("eng.train")
dev_data = read_conll("eng.testa")

word_counter = Counter()
for sentence in train_data:
    for word, label in sentence:
        word_counter[word] += 1

vocabulary = {}
vocabulary["<unk>"] = 0
vocabulary["<s>"] = 1
vocabulary["</s>"] = 2
for word in word_counter:
    if word_counter[word] > 1:
        vocabulary[word] = len(vocabulary)

label_vocabulary = {}
label_vocabulary["O"] = 0
label_vocabulary["ORG"] = 1
label_vocabulary["LOC"] = 2
label_vocabulary["MISC"] = 3
label_vocabulary["PER"] = 4


@cached(cache={})
def generate_word_feature_candidates(word):
    result = set()
    for i in [2, 3]:
        result.add(word[:i] + "__")
        result.add("__" + word[-i:])
    return result


print(generate_word_feature_candidates("Brexit"))

"""Next, we count the feature occurrences in training data:"""

feature_counter = Counter()

for sentence in train_data:
    for word, label in sentence:
        features = generate_word_feature_candidates(word)
        for feature in features:
            feature_counter[feature] += 1

"""Next, we make a feature-to-int mapping for the features that occur at least 50 times:"""

feature_vocabulary = {}
feature_list = []

feature_threshold = 50
for feature in feature_counter:
    if feature_counter[feature] >= feature_threshold:
        feature_vocabulary[feature] = len(feature_vocabulary)
        feature_list.append(feature)

"""Some info about the kept features:"""

print(len(feature_vocabulary))
print(list(feature_vocabulary.items())[0:20])
print(feature_list[0:20])

"""The next function find the feature IDs for a word. It first generates feature candidates for a word, and then keeps only those that are in our feature vocabulary:"""


@cached(cache={})
def get_word_feature_ids(word):
    feature_candidates = generate_word_feature_candidates(word)
    result = []
    for feature in feature_candidates:
        feature_id = feature_vocabulary.get(feature, -1)
        if feature_id >= 0:
            result.append(feature_id)
    return result


"""There feature IDs for the word 'Brexit':"""

get_word_feature_ids("Brexit")

"""Feature IDs and feature string reprsenatations wor the word 'Tallinn':"""

print([(i, feature_list[i]) for i in get_word_feature_ids("Tallinn")])

"""Now, we include the feature extraction part in our dataset. Note that now, each data item (word observation) will consists of 3 items: words (left word, current word, right word), features for the left, current and right word, and the label (y):"""


class NERDataset(Dataset):
    """Name Classification dataset"""

    def __init__(self, data):
        words = []
        self.features = []
        labels = []
        for sentence in data:
            for i in range(len(sentence)):
                if i > 0:
                    prevw = vocabulary.get(sentence[i-1][0], 0)
                    prevw_features = get_word_feature_ids(sentence[i-1][0])
                else:
                    prevw = vocabulary["<s>"]
                    prevw_features = []
                if i+1 < len(sentence):
                    nextw = vocabulary.get(sentence[i+1][0], 0)
                    nextw_features = get_word_feature_ids(sentence[i+1][0])
                else:
                    nextw = vocabulary["</s>"]
                    nextw_features = []
                words.append((prevw, vocabulary.get(sentence[i][0], 0), nextw))
                self.features.append(
                    (prevw_features,  get_word_feature_ids(sentence[i][0]), nextw_features))

                labels.append(label_vocabulary[sentence[i][1]])
        self.words = torch.from_numpy(np.array(words).astype(int)).long()

        self.y = torch.from_numpy(np.array(labels).astype(int)).long()

    def __len__(self):
        return len(self.words)

    def __getitem__(self, index):
        words = self.words[index]
        feature_matrix = torch.zeros(
            3, len(feature_vocabulary), dtype=torch.uint8)
        for j in range(3):
            for k in self.features[index][j]:
                feature_matrix[j, k] = 1

        y = self.y[index]

        sample = {'words': words, 'features': feature_matrix, 'y': y}
        return sample


train_dataset = NERDataset(train_data)
dev_dataset = NERDataset(dev_data)

"""Let's check how our first data item looks like. Note that 'features' is a 3x1833 tensor -- the rows corresponds to previous, current and next word and columns correspond to features. The tensor consists of mostly zeros, only in places where a certain feature is activated there is a one."""

train_dataset[0]

"""Next part implements our baseline model that only uses word embeddings. Note that the `forward` method actually takes feature tensor as an argument, but it is not used in this model."""


class NERNN(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_size):
        super(NERNN, self).__init__()
        self.embeddings = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=0)
        self.fc1 = nn.Linear(embedding_dim * 3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_size)

    def forward(self, words, features):
        x = self.embeddings(words).view(-1, (embedding_dim * 3))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        y = F.log_softmax(x, dim=1)
        return y


device = 'cpu'
if torch.cuda.is_available():
    device = torch.device('cuda')

print(device)

"""The next part implements training and evaluation. It has also been enhanced so that it feeds the feature tensor to the model."""


def train(model, num_epochs, train_iter, test_iter):

    optimizer = torch.optim.Adam(model.parameters())

    steps = 0
    best_acc = 0
    last_step = 0
    for epoch in range(1, num_epochs+1):
        print("Epoch %d" % epoch)
        model.train()
        for batch in train_iter:
            words, features, y = batch['words'].to(
                device), batch['features'].to(device), batch['y'].to(device)

            optimizer.zero_grad()
            output = model(words, features)

            loss = F.nll_loss(output, y)
            loss.backward()
            optimizer.step()

            steps += 1

        print('  Epoch finished, evaluating...')
        train_acc = evaluate("train", train_iter, model)
        dev_acc = evaluate("test", test_iter, model)


def evaluate(dataset_name, data_iter, model):

    model.eval()
    total_corrects, avg_loss = 0, 0
    # the following disables gradient computation for evaluation, as we don't need it
    with torch.no_grad():
        for batch in data_iter:
            words, features, y = batch['words'].to(
                device), batch['features'].to(device), batch['y'].to(device)

            output = model(words, features)

            # sum up batch loss
            loss = F.nll_loss(output, y, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(y.view_as(pred)).sum().item()

            avg_loss += loss

            total_corrects += correct

    size = len(data_iter.dataset)
    avg_loss /= size
    accuracy = 100.0 * total_corrects/size
    print('  Evaluation on {} - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(dataset_name,
                                                                          avg_loss,
                                                                          accuracy,
                                                                          total_corrects,
                                                                          size))
    return accuracy


"""Let's define some constants that we need for he model:"""

vocab_size = len(vocabulary)
embedding_dim = 50  # dimensionality of word embeddings
hidden_dim = 100   # dim of hidden layer
output_size = len(label_vocabulary)  # number of classes

"""Now, let's train the baseline model:"""

train_iter = DataLoader(train_dataset, batch_size=64, shuffle=True)
dev_iter = DataLoader(dev_dataset, batch_size=64, shuffle=True)
model = NERNN(vocab_size, embedding_dim, hidden_dim, output_size).to(device)
train(model, 5, train_iter, dev_iter)

"""Now comes your part:

## Excercise 1

Implement a model that also uses word features. The word features should first be concatenated (i.e., the features for the left, center and right words should be concatenated into one single tensor), and then fed through a hidden layer, that has output dimensionality defined by `feature_hidden_dim` constructor argument, and uses ReLU nonlinearity. The output from this hidden layer should be concatenated with the word embeddings, and then passed through a common hidden layer (with output dim defined by `hidden_dim`), and the output of this goes to the last layer.
"""


class NERNN_improved(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_features, feature_hidden_dim, output_size):
        super(NERNN_improved, self).__init__()
        # Implement this
        self.embeddings = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=0)
        self.featuresEmb = nn.Linear(num_features * 3, feature_hidden_dim)
        self.fce = nn.Linear(embedding_dim * 3, int(hidden_dim/2))
        self.fcf = nn.Linear(feature_hidden_dim, int(hidden_dim/2))
        self.fc3 = nn.Linear(hidden_dim, output_size)

    def forward(self, words, features):
        # Implement this
        x0 = self.embeddings(words).view(-1, (embedding_dim * 3))
        x0 = F.relu(self.fce(x0))

        flatten_features = torch.flatten(features, start_dim=1)
        x1 = self.featuresEmb(flatten_features.to(torch.float))
        x1 = F.relu(self.fcf(x1))

        x = torch.cat((x0, x1), dim=1)
        x = self.fc3(x)

        y = F.log_softmax(x, dim=1)
        return y


"""Let's train the model. *Please report the dev accuracy after 5 epochs (last line)!*"""

train_iter = DataLoader(train_dataset, batch_size=64, shuffle=True)
dev_iter = DataLoader(dev_dataset, batch_size=64, shuffle=True)
vocab_size = len(vocabulary)
embedding_dim = 50  # dimensionality of word embeddings
hidden_dim = 100   # dim of hidden layer
output_size = len(label_vocabulary)  # number of classes
feature_hidden_dim = 100
model2 = NERNN_improved(vocab_size, embedding_dim, hidden_dim, len(
    feature_vocabulary), feature_hidden_dim, output_size).to(device)
train(model2, 5, train_iter, dev_iter)

"""## Exercise 2

The third model is similar to the second model, but instead of concatenating the features before the hidden layer, the features of the previous, current and next words are each passed though a separate hidden layer with *shared weights*, so that the weights for individual features would be the same, regardless of the word position, and then concatenated with the embeddings.
"""


class NERNN_improved_shared(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_features, feature_hidden_dim, output_size):
        super(NERNN_improved_shared, self).__init__()
        # Implement this
        self.num_features = num_features
        self.embeddings = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=0)
        self.fcf = nn.Linear(num_features, feature_hidden_dim)
        self.fce = nn.Linear(embedding_dim * 3, int(hidden_dim/2))
        self.fcf_hidden = nn.Linear(feature_hidden_dim * 3, int(hidden_dim/2))
        self.fc3 = nn.Linear(hidden_dim, output_size)

    def forward(self, words, features):
        # Implement this
        x0 = self.embeddings(words).view(-1, (embedding_dim * 3))
        x0 = F.relu(self.fce(x0))
        flatten_features = torch.flatten(features, start_dim=1).to(torch.float)
        features_split = torch.split(
            flatten_features, self.num_features, dim=1)

        features_prev = features_split[0]
        features_current = features_split[1]
        features_next = features_split[2]

        x1 = F.relu(self.fcf(features_prev))
        x2 = F.relu(self.fcf(features_current))
        x3 = F.relu(self.fcf(features_next))
        xf = torch.cat((x1, x2, x3), dim=1)
        xf = self.fcf_hidden(xf)

        x = torch.cat((x0, xf), dim=1)
        x = self.fc3(x)

        y = F.log_softmax(x, dim=1)
        return y


"""Let's train the model. *Please report the dev accuracy after 5 epochs (last line)!*"""

train_iter = DataLoader(train_dataset, batch_size=64, shuffle=True)
dev_iter = DataLoader(dev_dataset, batch_size=64, shuffle=True)
feature_hidden_dim = 100
model3 = NERNN_improved_shared(vocab_size, embedding_dim, hidden_dim, len(
    feature_vocabulary), feature_hidden_dim, output_size).to(device)
train(model3, 5, train_iter, dev_iter)


"""## Some hints

You might need to use torch.cat(..., dim=...) to concatenate tensors.

## Exercise 3: interpolating models

You should now have three models: the baseline model (`model`), improved model (`model2`) and the improved model with shared feature weights (`model3`). 

It is often beneficial to combine the predictions of several models. The simplest way to do this is to just compute the average of the probability distributions given by the different models. Note that the average has to be done in linear space, not in the log space. So, if the $y_i$ is the posterior probability distribution of model $i$ (in linear space), then the interpolation is:

$y_{interpolated} = \frac{1}{N} \sum_{i=1..N} y_i$

where $N$ is the number of models.

Your task is to implement model interpolation as a Pytorch module. Template is given below.

Note that to be compatible with other models, the output of the interpolated model should also be in log space.
"""


class ModelInterpolation(nn.Module):

    def __init__(self, models):
        super(ModelInterpolation, self).__init__()
        # Implement this
        self.models = models
        self.n = len(models)

    def forward(self, words, features):
        # Implement this
        ysum = 0
        for model in self.models:
          ysum += torch.exp(model.forward(words, features))
        y = ysum/self.n

        return torch.log(y)


"""When you have completed the implementation, it should be possible to evaluate the interpolation performance like this:"""

model_interpolation = ModelInterpolation([model, model2, model3])
print(evaluate("test", dev_iter, model_interpolation))
