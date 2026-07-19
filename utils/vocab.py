from collections import Counter

class Vocab:
  def __init__(self, min_freq=2):
    self.word2idx={"<pad>":0 , "<unk>":1}
    self.idx2word={0:"<pad>" , 1:"<unk>"}
    self.min_freq= min_freq

  def build_vocab(self, sentences):
    counter= Counter()
    for sentence in sentences:
      counter.update(sentence)
    index= 2
    for word, freq in counter.items():
      if freq >= self.min_freq:
        self.word2idx[word]= index
        self.idx2word[index]= word
        index+=1

  def encode(self, sentence):
    encoded=[]
    for word in sentence:
      encoded.append(self.word2idx.get(word, 1))
    return encoded

    
