from tokenizers.trainers import BpeTrainer
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace

def create_tokenizer():
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(special_tokens=["<unk>", "<s>", "</s>", "<pad>", "<mask>"])
    tokenizer.train(files=["/home/olab/itayitzhak/oscar/oscar_all.txt.train", "/home/olab/itayitzhak/oscar/oscar_all.txt.valid"], trainer=trainer)
    tokenizer.save("data-bin/oscar/oscar_he_tokenizer.json")

#create_tokenizer()

tokenizer = Tokenizer.from_file("data-bin/oscar/oscar_he_tokenizer.json")
output = tokenizer.encode("היי מה קורה? שלום עולם או שלום עולמי?").tokens
print(output)

vocab = tokenizer.get_vocab()
print(vocab)
with open('data-bin/oscar/dict.he.oscar.txt', 'w+') as f:
    for key in vocab.keys():
        f.write(str(key)+' '+str(vocab[key])+'\n')

