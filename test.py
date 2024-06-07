import nltk
from nltk.tokenize import word_tokenize

# Tải các gói cần thiết của nltk
nltk.download('punkt')

sentence = "<<React1>> so, you know what, that doesn't matter."

# Sử dụng word_tokenize để tách các token và dấu câu
tokens = word_tokenize(sentence)

print(tokens)