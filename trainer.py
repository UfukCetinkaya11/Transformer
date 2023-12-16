from transformer import Transformer
import torch
from datasets import load_dataset
from vocabulary_helper import get_vocabulary

test_dataset = load_dataset("wmt16", "de-en", split="train[:100000]", num_proc=8)

german_sentences = []
english_sentences = []

# make this after knowing the vocabulary
def preprocess_function(example):

    german_sentences.append(example["translation"]["de"].rstrip('\n').lower())
    english_sentences.append(example["translation"]["en"].rstrip('\n').lower())
    return


test_dataset.map(preprocess_function)



START_TOKEN = '<START>'
PADDING_TOKEN = '<PADDING>'
END_TOKEN = '<END>'

german_vocabulary = get_vocabulary(german_sentences, START_TOKEN, PADDING_TOKEN, END_TOKEN)
print(german_vocabulary)
english_vocabulary = get_vocabulary(english_sentences, START_TOKEN, PADDING_TOKEN, END_TOKEN)
print(english_vocabulary)



index_to_german = {k:v for k,v in enumerate(german_vocabulary)}
german_to_index = {v:k for k,v in enumerate(german_vocabulary)}
index_to_english = {k:v for k,v in enumerate(english_vocabulary)}
english_to_index = {v:k for k,v in enumerate(english_vocabulary)}





import numpy as np
PERCENTILE = 97
print( f"{PERCENTILE}th percentile length german: {np.percentile([len(x) for x in german_sentences], PERCENTILE)}" )
print( f"{PERCENTILE}th percentile length english: {np.percentile([len(x) for x in english_sentences], PERCENTILE)}" )




max_sequence_length = 425
def is_valid_tokens(sentence, vocab):
    for token in list(set(sentence)):
        if token not in vocab:
            return False
    return True

def is_valid_length(sentence, max_sequence_length):
    return len(list(sentence)) < (max_sequence_length - 1) # need to re-add the end token so leaving 1 space

valid_sentence_indicies = []
for index in range(len(german_sentences)):
    german_sentence, english_sentence = german_sentences[index], english_sentences[index]
    if is_valid_length(german_sentence, max_sequence_length)  and is_valid_length(english_sentence, max_sequence_length) and is_valid_tokens(german_sentence, german_vocabulary):
        valid_sentence_indicies.append(index)

print(f"Number of sentences: {len(english_sentences)}")
print(f"Number of valid sentences: {len(valid_sentence_indicies)}")


german_sentences = [german_sentences[i] for i in valid_sentence_indicies]
english_sentences = [english_sentences[i] for i in valid_sentence_indicies]



d_model = 512
batch_size = 32
ffn_hidden = 2048
num_heads = 8
drop_prob = 0.1
num_layers = 6
max_sequence_length = 425
german_vocab_size = len(german_vocabulary)

transformer = Transformer(d_model,
                          ffn_hidden,
                          num_heads,
                          drop_prob,
                          num_layers,
                          max_sequence_length,
                          german_vocab_size,
                          english_to_index,
                          german_to_index,
                          START_TOKEN,
                          END_TOKEN,
                          PADDING_TOKEN)


print(transformer)

#tokenizer = SentenceEmbedding(max_sequence_length, d_model, german_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)
#total_tokens = tokenizer.count_tokens_in_dataset(german_sentences, start_token=True, end_token=True)

total_params = sum(p.numel() for p in transformer.parameters())
print(f'Total number of parameters: {total_params}')
#print(f'Total number of tokens in the dataset: {total_tokens}')


from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):

    def __init__(self, english_sentences, german_sentences):
        self.english_sentences = english_sentences
        self.german_sentences = german_sentences

    def __len__(self):
        return len(self.english_sentences)

    def __getitem__(self, idx):
        return self.english_sentences[idx], self.german_sentences[idx]


dataset = TextDataset(english_sentences, german_sentences)

train_loader = DataLoader(dataset, batch_size)
iterator = iter(train_loader)


from torch import nn

criterian = nn.CrossEntropyLoss(ignore_index=german_to_index[PADDING_TOKEN],
                                reduction='none')

# When computing the loss, we are ignoring cases when the label is the padding token
for params in transformer.parameters():
    if params.dim() > 1:
        nn.init.xavier_uniform_(params)

optim = torch.optim.Adam(transformer.parameters(), lr=1e-4)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


NEG_INFTY = -1e9

def create_masks(english_batch, german_batch):
    num_sentences = len(english_batch)
    look_ahead_mask = torch.full([max_sequence_length, max_sequence_length] , True)
    look_ahead_mask = torch.triu(look_ahead_mask, diagonal=1)
    encoder_padding_mask = torch.full([num_sentences, max_sequence_length, max_sequence_length] , False)
    decoder_padding_mask_self_attention = torch.full([num_sentences, max_sequence_length, max_sequence_length] , False)
    decoder_padding_mask_cross_attention = torch.full([num_sentences, max_sequence_length, max_sequence_length] , False)

    for idx in range(num_sentences):
      english_sentence_length, german_sentence_length = len(english_batch[idx]), len(german_batch[idx])
      english_chars_to_padding_mask = np.arange(english_sentence_length + 1, max_sequence_length)
      german_chars_to_padding_mask = np.arange(german_sentence_length + 1, max_sequence_length)
      encoder_padding_mask[idx, :, english_chars_to_padding_mask] = True
      encoder_padding_mask[idx, english_chars_to_padding_mask, :] = True
      decoder_padding_mask_self_attention[idx, :, german_chars_to_padding_mask] = True
      decoder_padding_mask_self_attention[idx, german_chars_to_padding_mask, :] = True
      decoder_padding_mask_cross_attention[idx, :, english_chars_to_padding_mask] = True
      decoder_padding_mask_cross_attention[idx, german_chars_to_padding_mask, :] = True

    encoder_self_attention_mask = torch.where(encoder_padding_mask, NEG_INFTY, 0)
    decoder_self_attention_mask =  torch.where(look_ahead_mask + decoder_padding_mask_self_attention, NEG_INFTY, 0)
    decoder_cross_attention_mask = torch.where(decoder_padding_mask_cross_attention, NEG_INFTY, 0)
    return encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask


transformer.train()
transformer.to(device)
total_loss = 0
num_epochs = 15

for epoch in range(num_epochs):
    print(f"Epoch {epoch}")
    iterator = iter(train_loader)
    for batch_num, batch in enumerate(iterator):
        transformer.train()
        english_batch, german_batch = batch
        encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask = create_masks(english_batch,
                                                                                                              german_batch)
        optim.zero_grad()
        german_predictions = transformer(english_batch,
                                     german_batch,
                                     encoder_self_attention_mask.to(device),
                                     decoder_self_attention_mask.to(device),
                                     decoder_cross_attention_mask.to(device),
                                     enc_start_token=False,
                                     enc_end_token=False,
                                     dec_start_token=True,
                                     dec_end_token=True)
        labels = transformer.decoder.sentence_embedding.batch_tokenize(german_batch, start_token=False, end_token=True)
        loss = criterian(
            german_predictions.view(-1, german_vocab_size).to(device),
            labels.view(-1).to(device)
        ).to(device)
        valid_indicies = torch.where(labels.view(-1) == german_to_index[PADDING_TOKEN], False, True)
        loss = loss.sum() / valid_indicies.sum()
        loss.backward()
        optim.step()
        # train_losses.append(loss.item())
        if batch_num % 100 == 0:
            print(f"Iteration {batch_num} : {loss.item()}")
            print(f"english: {english_batch[0]}")
            print(f"German Translation: {german_batch[0]}")
            german_sentence_predicted = torch.argmax(german_predictions[0], axis=1)
            predicted_sentence = ""
            for idx in german_sentence_predicted:
                if idx == german_to_index[END_TOKEN]:
                    break
                predicted_sentence += index_to_german[idx.item()]
            print(f"German Prediction: {predicted_sentence}")

            transformer.eval()
            german_sentence = ("",)
            english_sentence = ("i love my dog",)
            for word_counter in range(max_sequence_length):
                encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask = create_masks(
                    english_sentence, german_sentence)
                predictions = transformer(english_sentence,
                                          german_sentence,
                                          encoder_self_attention_mask.to(device),
                                          decoder_self_attention_mask.to(device),
                                          decoder_cross_attention_mask.to(device),
                                          enc_start_token=False,
                                          enc_end_token=False,
                                          dec_start_token=True,
                                          dec_end_token=False)
                next_token_prob_distribution = predictions[0][word_counter]
                next_token_index = torch.argmax(next_token_prob_distribution).item()
                next_token = index_to_german[next_token_index]
                german_sentence = (german_sentence[0] + next_token,)
                if next_token == END_TOKEN:
                    break

            print(f"Evaluation translation (i love my dog) : {german_sentence}")
            print("-------------------------------------------")

PATH = r"C:\Users\Ufuk\PycharmProjects\Transformer"
torch.save(transformer.state_dict(), PATH)
