import torch
from torchtext.legacy import data
from torchtext.legacy import datasets

CHARS = data.Field(tokenize=list, init_token="<bos>", eos_token="<eos>", include_lengths=True, batch_first=True)
PHONEMES = data.Field(tokenize=str.split, init_token="<bos>", eos_token="<eos>", include_lengths=True, batch_first=True)

train_dataset, dev_dataset = data.TabularDataset.splits(path=".", train='fre_train.tsv', test='fre_dev.tsv', format='tsv', skip_header=False, fields=[('chars', CHARS), ('phonemes', PHONEMES)])

test_word_list = [l.strip() for l in open("fre_test.unlabelled.tsv")]

"""Let's explore the datasets:"""

print(len(train_dataset))
print(train_dataset[0].chars, train_dataset[0].phonemes)

CHARS.build_vocab(train_dataset)
PHONEMES.build_vocab(train_dataset)

device = 'cpu'
if torch.cuda.is_available():
  device = torch.device('cuda')

print(device)

train_iter, dev_iter = data.BucketIterator.splits((train_dataset, dev_dataset), batch_size=32, device=device, sort=False)

batch = next(iter(train_iter))
print(batch)

import sys
import torch.nn as nn
import torch.nn.functional as F
import time
import random
import math

"""First, the encoder part of the seq2seq model:"""

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers):
        super().__init__()        
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hid_dim, n_layers, batch_first=True)
        
    def forward(self, src):        
        # src: [batch size, seq_len]       
        embedded = self.embedding(src)
        #embedded: [batch_size, seq_len,  emb_dim]
        outputs, hidden = self.rnn(embedded)

        return hidden, outputs

"""Then, the decoder part:"""

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        self.rnn = nn.GRU(emb_dim, hid_dim, n_layers, batch_first=True)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        
        
    def forward(self, input, hidden, encoder_outputs):            
        #input: [batch size]
        #hidden: [batch size, hid_dim]
        #encoder_outputs: [batch size, src_len, hid_dim]
        
        input = input.unsqueeze(1)
        #input: [batch size, 1]        
        embedded = self.embedding(input)
        #embedded: [batch size, 1, emb dim]
        output, hidden = self.rnn(embedded, hidden)        
        prediction = self.fc_out(output.squeeze(1))        
        #prediction: [batch size, output dim]
        
        return prediction, hidden

"""And finally, the seq2seq model:"""

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        
        #src: [batch size, src_len]
        #trg: [batch size, trg_len]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, encoder_outputs = self.encoder(src)
        
        #first input to the decoder is the <sos> tokens
        input = trg[:, 0]
        
        for t in range(1, trg_len):    
            #insert input token embedding, previous hidden and previous cell states
            #receive output tensor (predictions) and new hidden and cell states
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            #place predictions in a tensor holding predictions for each token
            outputs[:, t, :] = output
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            #get the highest predicted token from our predictions
            top1 = output.argmax(1) 
            #print("---", trg[:, t].shape, top1.shape)
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            input = trg[:, t] if teacher_force else top1
        
        return outputs

"""Now, we implement the training routines:"""

def train_one_epoch(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        src = batch.chars[0]
        trg = batch.phonemes[0]
        optimizer.zero_grad()
        output = model(src, trg)
        
        #trg: [trg_len, batch size]
        #output_ [trg_len, batch size, output dim]
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)

        #trg:  [(trg len - 1) * batch size]
        #output: [(trg len - 1) * batch size, output dim]
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.chars[0]
            trg = batch.phonemes[0]
            output = model(src, trg, 0) #turn off teacher forcing
            #trg: [trg len, batch size]
            #output: [trg len, batch size, output dim]
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            #trg: [(trg len - 1) * batch size]
            #output: [(trg len - 1) * batch size, output dim]
            loss = criterion(output, trg)
            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def train(model, num_epochs, clip):
  optimizer = torch.optim.AdamW(model.parameters(), lr=0.0008)
  criterion = nn.CrossEntropyLoss(ignore_index=CHARS.vocab.stoi[CHARS.pad_token])

  for epoch in range(num_epochs):
      start_time = time.time()
      train_loss = train_one_epoch(model, train_iter, optimizer, criterion, clip)
      valid_loss = evaluate(model, dev_iter, criterion)
      end_time = time.time()
      epoch_mins, epoch_secs = epoch_time(start_time, end_time)
      
      print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
      print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
      print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

"""Finally we can train our baseline model:"""

INPUT_DIM = len(CHARS.vocab)
OUTPUT_DIM = len(PHONEMES.vocab)
ENC_EMB_DIM = 500
DEC_EMB_DIM = 50
HID_DIM = 256
N_LAYERS = 4

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS)

model = Seq2Seq(enc, dec, device).to(device)

N_EPOCHS = 100
CLIP = 1

train(model, N_EPOCHS, CLIP)


"""Now, we implement a function that applies the model to a given string:"""

def chars2phonemes(word, model, max_len=50):
    model.eval()
    chars = CHARS.process([word])[0].to(device)
    with torch.no_grad():
        hidden, encoder_outputs = model.encoder(chars)
    trg_indexes = [PHONEMES.vocab.stoi[PHONEMES.init_token]]

    for i in range(max_len):
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
        with torch.no_grad():
            output, hidden = model.decoder(trg_tensor, hidden, encoder_outputs)
        pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token)
        if pred_token == CHARS.vocab.stoi[CHARS.eos_token]:
            break
    
    trg_tokens = [PHONEMES.vocab.itos[i] for i in trg_indexes]
    
    return " ".join(trg_tokens[1:-1])

chars2phonemes("appartement", model)

"""This should be a correct pronunciation ('a p a ʁ t ə m ɑ̃').

Now, we implement scoring functionality that allows to evaluate our model based on *phoneme error rate* (how many phonemes in the output are correct, compared to the reference). We use the implementation in jiwer package. The appropriate function is called `wer` (word error rate) because this is typically used to measure the word error rate of a speech recognition system.
"""


import jiwer
ref = "a b a c"
hyp = "a b a d"
jiwer.wer(ref, hyp)

"""Both substitution, insertion and deletion errors are taken into account when computing WER:"""

ref = "a b a c"
hyp = "a b a a c"
jiwer.wer(ref, hyp)

ref = "a b a c"
hyp = "a b c"
jiwer.wer(ref, hyp)

"""Finally, the function that computes the scoring metrics based on the development set:"""

def per(model):
  refs = [] 
  hyps = []
  for pair in iter(dev_dataset):
    ref = " ".join(pair.phonemes)
    hyp = chars2phonemes("".join(pair.chars), model)
    refs.append(ref)
    hyps.append(hyp)

  per = jiwer.wer(refs, hyps)
  print(f"Phoneme error rate: {per}")

per(model)

"""Finally, we implement a function that applies the model to our test data and saves it is a file called `submission.tsv`."""

def make_submission(model):
  with open("submission.tsv", "w") as f:
    print("Word\tPhonemes", file=f)
    for word in test_word_list:
      hyp = chars2phonemes(word, model)
      print(f"{word}\t{hyp}", file=f)

make_submission(model)


"""## Your assignment

Your task is to improve the performance of the baseline model. You can use any method do this, with the following exceptions:
  * You cannot use some some pretrained model that you found on the Internet that already does this task
  * You cannot use additional French data for training
  * You have to stick to basic Pytorch, and not use libraries that implement whole models for you (you can copy/paste code from there though, as long as you understand what this code is doing)
  * You cannot hand-label additional data
  * You are not allowed to fold the dev set into training data

Some things you can do:

  * Tune basic learning hyperparametes, such as number of epochs, learning rate, optimizer
  * Tune model architecture (number of outputs in the layers, number of layers, etc)
  * Add regularization (dropout, etc)

But the most lucrative thing to try is to implement attention mechanism.  In the baseline model, decoder has to generate a pronunciation solely based on the hidden state passed from the encoder.

You should try to implement attention mechanism between the encoder and decoder.

More specifically, decoder should at each time step generate a new weighted view over the encoder GRU outputs (that are luckily already passed to the decoder's forward method). You should use the query-key-value based attention: output from the decoder's GRU is passed through a (learnable) linear transform that results in a query vector $q$. Another linear transform is applied to encoder outputs that results in key vectors $k_i$. Note that query and keys should have the same dimensionality. Now the query is compared to the keys (using dot product) and the results is softmaxed, resulting in attention weights $w_i$ over the encoder outputs. Encoder outputs are now passed through another linear transform (resulting in $v_i$ ) and summed up to form $v*$, using the computed weights. 


Luckily, most of the attention functionality is implemented in torch's `nn.MultiheadAttention` class. You just have to take care of the linear transforms and some other stuff.

The picture below describes this. Note how how the decoder generates a character "a" based on the output from it's GRU *and* the output $v*$ from the attention mechanism.

You only need modify the Decoder to implement this. 

You have to add three linear transforms (for transforming encoder outputs to key and value vectors and for transforming decoder's GRU output to a query vector) and you also have to concatenate attention output with the decoder's GRU output before passing it to the final fully connected layer.

Note that the linear transforms used to compute key, query and value should not have a bias term.
"""

class DecoderAttn(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, attn_dim):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        self.rnn = nn.GRU(emb_dim, hid_dim, n_layers, batch_first=True)
        # Implement this
        # BEGIN
        # self.fc_out = 
        
        # add attention layer and linear transform layers

        
        # END
        
    def forward(self, input, hidden, encoder_outputs):
             
        #input:  [batch size]
        #hidden:  [batch size, hid_dim]
        #encoder_outputs: [batch size, src_len, hid_dim]
        
        input = input.unsqueeze(1)
        #input: [batch size, 1]
        
        embedded = self.embedding(input)
        #embedded:  [batch size, 1, emb dim]

        output, hidden = self.rnn(embedded, hidden)

        # implement this
        # BEGIN
        # compute v* (attention output)
        # compute prediction, using a fully connected layer that takes as input 
        # both attention output and output from GRU


        # END

        #prediction : [batch size, output dim]
        
        return prediction, hidden

INPUT_DIM = len(CHARS.vocab)
OUTPUT_DIM = len(PHONEMES.vocab)
ENC_EMB_DIM = 500
DEC_EMB_DIM = 50
HID_DIM = 256
ATTN_DIM = 128
N_LAYERS = 2
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS)
dec = DecoderAttn(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, ATTN_DIM)

model_attn = Seq2Seq(enc, dec, device).to(device)

N_EPOCHS = 10
CLIP = 1

train(model_attn, N_EPOCHS, CLIP)

"""If implemented correctly, your phoneme error rate around 0.04 ... 0.08 (actually different training runs give quite different results). That is, attention really works very well in this task, compared to the more basic models.

Another thing that you should try is to use the Transformer architecture, i.e., get rid of the GRU layers in both encoder and decoder completely. You should also then use positional encoding in both encoder and decoder. There are plenty of tutorials and code samples about this.

## Grading

In order to be graded, you have to submit the generated test pronunciations (i.e., the `submission.tsv` file generated by you improved model) to the evaluation leaderboard at http://bark.phon.ioc.ee/am3-leaderboard/. 

In the leaderboard, you have to enter your UniID and upload the generated `submission.tsv` file. *NB!* If you don't feel comfortable that other students see your score, use a nickname (i.e., could be random string only you know), but please be consistent, that is, use only one name across all your submissions. If you use a nickname, let me know what it is when you submit your code.

*Another NB!* The leaderboard is totally unsecure and you don't even have to authenticate yourself, so please don't do stupid things, like uploading under other people's names, etc.

Your grade is determined by the error rate of your best model on test data, compared to the average score of the top 3 students. 

More specifially:
  * First, the average of the top 3 students' models will be computed. Let's call it $score_{top3}$. Let's also save the error rate of the baseline model to $score_{baseline}$.
  * Then, we'll compute how much of the error rate reduction does your best score $score_i$ achieve  over the baseline, when baseline would give you 0% points and $score_{top3}$ 100% points. 
  * That is, your points will be calculated as $\delta * 15$, where 
  $\delta=\frac{score_{baseline} - score_{i}}{score_{baseline} - score_{top3}}$
  * For example: baseline score is 0.18, top3 average  is 0.07, and your score 0.11. This would give you $\delta=\frac{0.18-0.11}{0.18-0.07}=0.63$ and $0.63 \times 15 = 9.6 ≃ 10$ points.
  * Yes, the very top scorers can actually get more than 15 points!


### More rules!

  * You can make *up to 20 submissions* to the leaderboard
  * The `topline` model in the leaderboard corresponds to the model with attention-equipped decoder, as proposed above. It will count as one of the top3 models, if it will be in the final top.
  * The leaderboard $score_{top3}$ will be taken at the time of the assignment deadline. Later submissions that make it to the top 3 will not change it.

You should also submit the jupyter notebook (link) that reproduces your best model. Note that it's expected that multiple runs of the training won't have the same results. 

Please make sure that you share your notebook's link so that "anyone with the link" can see it.
"""

