from collections import Counter

class Vocab:
  def __init__(self, min_freq=2):
    self.word2indx={"<pad>":0 , "<unk>":1}
    self.indx2word={0:"<pad" , 1:"<unk>"}
    self.min_freq= min_freq

  def build_vocab(self, sentences):
    counter= Counter()
    for sentence in sentences:
      counter.update(sentence)
    index= 2
    for word, freq in counter.items():
      if freq >= self.min_freq:
        self.word2indx[word]= index
        self.indx2word[index]= word
        index+=1

  def encode(self, sentance):
    encoded=[]
    for word in sentance:
      encoded.append(self.word2indx.get(word, 1))
    return encoded
  
test=[
  ["i", "love", "it"],
  ["it", "was", "amazing"]
]
vocab= Vocab() 
vocab.build_vocab(test)
print(vocab.encode(["i", "love", "it"]))
print(vocab.encode(["it", "was", "amazing"]))
    
    
