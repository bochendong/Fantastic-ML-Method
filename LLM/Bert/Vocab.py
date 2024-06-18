class Vocab(object):
    def __init__(self, counter, specials, max_size = None, min_freq = 1):
        self.freqs = counter
        self.itos = list(specials)

        self.pad_index = 0
        self.unk_index = 1
        self.eos_index = 2
        self.sos_index = 3
        self.mask_index = 4

        for token in specials:
            del counter[token]

        words_and_freqs = sorted(counter.items(), key = lambda tup: tup[0])
        words_and_freqs.sort(key = lambda tup: tup[1], reverse=True)

        for word, freq in words_and_freqs:
            if freq < min_freq or len(self.itos) == max_size:
                break
            self.itos.append(word)

        self.stoi = {token: i for i, token in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)