import tqdm
import torch
import random
from torch.utils.data import Dataset

class BERTDataset(Dataset):
    '''
    Dataset class for preparing data for BERT model training.
    Processes an English-French parallel corpus for training with masked language modeling 
    and next sentence prediction tasks.
    '''
    def __init__(self, corpus_path = './data/eng-fra.txt', vocab = None , seq_len = 20):
        self.vocab = vocab
        self.seq_len = seq_len
        self.corpus_path = corpus_path
        self.lines = []

        # Reopen the file to read the lines
        with open(self.corpus_path , "r", encoding="utf-8") as f:
            for line in tqdm.tqdm(f, desc="Loading Dataset"):

                # Each line is a english-franch pair, for example
                # Hi, guys.	    Salut, les mecs !
                self.lines.append(line.replace('\n', '').split('\t'))

        # self.lines is a list of english-franch pair
        # [[eng1, fra1], [eng2, fra2], ...]

        # self.corpus_lines is a counter of how many data in this dataset
        self.corpus_lines = len(self.lines)

    def get_corpus_line(self, index):
        '''
        Returns the English-French sentence pair at the specified index.
        '''
        return self.lines[index][0], self.lines[index][1]
    
    def get_random_line(self):
        '''
        Returns a random French sentence from the corpus.
        '''
        return self.lines[random.randrange(self.corpus_lines)][1]
    
    def random_sent(self, index):
        '''
        Generates a training instance for next sentence prediction.
        Returns a tuple containing English sentence, 
        a second sentence which is either a correct or random French sentence, 
        and a label indicating if it is the correct next sentence.
        
        For example:
        random_sent(index) will return:
        (eng, corresping fra, 1) with 50% probability (is_next = 1)
        (eng, random fra, 0)     with 50% probability (is_next = 0)
        '''
        t1, t2 = self.get_corpus_line(index)

        # 50% chance to return the correct or a random sentence
        if random.random() > 0.5:
            return t1, t2, 1                        # Correct next sentence
        else:
            return t1, self.get_random_line(), 0     # Random sentence
        
    def random_word(self, sentence):
        '''
        Performs token masking for BERT's masked language modeling task.
        Randomly masks tokens (with a probability of 15%), replacing them with a MASK token, a 
        random token, or leaving them unchanged.

        Returns a list of token indices and a corresponding label list (index of token).

        For example: random_word(["hi", "guys"]) will return:
        Assume hi has index 12, guys has index 9

        tokens will be: [12, 9]          (86.5 % probability)
                        [4, 9] or [9, 4] (some is masked, 12 %)
                        [12, 25]         (a random token, 1.5%) 

        output_label:   [12, 0]            (85 % probability)
                        [12, 9]            (15 % probability)
        '''
        tokens = sentence.split()   # "Hi, guys." -> ["hi", "guys"]
        output_label = []

        for i, token in enumerate(tokens):
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    # tokens[i] = 4
                    tokens[i] = self.vocab.mask_index

                # 10% randomly change token to random token's index
                elif prob < 0.9:
                    # tokens[i] = random token
                    tokens[i] = random.randrange(len(self.vocab))

                # 10% change token to current token's index
                else:
                    # tokens[i] = current token
                    # tokens[i] = self.vocab.stoi.get(token, 1)
                    tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)

                # output_label.append(self.vocab.stoi.get(token, 1))
                output_label.append(self.vocab.stoi.get(token, self.vocab.unk_index))

            else:
                tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
                # 0 is a pad token
                output_label.append(0)

        return tokens, output_label
    
    def __getitem__(self, item):
        # t1 = English sentence's index
        # t2 = corresping Franch sentence with 50% probability (is next = 0)
        #    = random franch sentence with 50% probability (is next = 1)
        t1, t2, is_next_label = self.random_sent(item)

        t1_random, t1_label = self.random_word(t1)
        t2_random, t2_label = self.random_word(t2)

        # [CLS] tag = SOS tag, [SEP] tag = EOS tag

        # t1 = [3] + t1_random + [2]
        # t2 = t2_random + [2]
        t1 = [self.vocab.sos_index] + t1_random + [self.vocab.eos_index]
        t2 = t2_random + [self.vocab.eos_index]

        # t1_label = [0] + t1_label + [0]
        # t2_label = t2_label + [0]
        t1_label = [self.vocab.pad_index] + t1_label + [self.vocab.pad_index]
        t2_label = t2_label + [self.vocab.pad_index]
        
        # segment_label = [1, 1, ..., 2, 2, ...] (1 for eng sentence, 2 for franch sentence)
        segment_label = ([1 for _ in range(len(t1))] + [2 for _ in range(len(t2))])[:self.seq_len]
        bert_input = (t1 + t2)[:self.seq_len]
        bert_label = (t1_label + t2_label)[:self.seq_len]

        # fill sentence with padding at end
        padding = [self.vocab.pad_index for _ in range(self.seq_len - len(bert_input))]
        bert_input.extend(padding), bert_label.extend(padding), segment_label.extend(padding)

        output = {"bert_input": bert_input, "bert_label": bert_label,
                  "segment_label": segment_label, "is_next": is_next_label}
        
        return {key: torch.tensor(value) for key, value in output.items()}
        
    def __len__(self):
        return self.corpus_lines
