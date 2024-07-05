import numpy as np
import tqdm
from torch.optim import Adam
import torch.nn as nn

from .Dataset import BERTDataset
from .Vocab import Vocab
from .Model import BERT, BERTLM
from collections import Counter

from torch.utils.data import DataLoader

class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = np.power(d_model, -0.5)

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr

def train(model, train_data_loader, criterion, optim_schedule):
    for i, data in enumerate(train_data_loader):
        data = {key:value for key, value in data.items()}
        next_sentence_out, mask_lm_out = model.forward(data["bert_input"], data["segment_label"])

        next_loss = criterion(next_sentence_out, data["is_next"])

        mask_loss = criterion(mask_lm_out.transpose(1, 2), data["bert_label"])

        loss = next_loss + mask_loss

        optim_schedule.zero_grad()
        loss.backward()
        optim_schedule.step_and_update_lr()


counter = Counter()
with open('./data/eng-fra.txt', "r", encoding="utf-8") as f:
    for line in tqdm.tqdm(f, desc="Loading Dataset"):
        if isinstance(line, list):
            words = line
        else:
            words = line.replace("\n", "").replace("\t", " ").split()

        for word in words:
            counter[word] += 1

vocab = Vocab(counter, specials=["<pad>", "<unk>", "<eos>", "<sos>", "<mask>"])
train_dataset = BERTDataset(corpus_path = './data/eng-fra.txt', vocab = vocab , seq_len = 20)
train_data_loader = DataLoader(train_dataset, batch_size = 64)

criterion = nn.NLLLoss(ignore_index = 0)
bert_hidden = 768
weight_decay = 0.01
warmup_steps = 10000
lr = 1e-4
betas = (0.9, 0.999)
weight_decay = 0.01


bert = BERT(len(vocab))
model = BERTLM(bert, len(vocab))

optim = Adam(model.parameters(), lr = lr, betas = betas, weight_decay = weight_decay)
optim_schedule = ScheduledOptim(optim, bert.hidden, n_warmup_steps=warmup_steps)

train(model = model, train_data_loader = train_data_loader, 
      criterion = criterion, optim_schedule = optim_schedule)

