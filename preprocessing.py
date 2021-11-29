import re
import nltk as nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt') 
stop_words = set(stopwords.words('english'))
from nltk.stem import PorterStemmer
    
    
def text_preprocessing(sentense):
    ps = PorterStemmer() 
    sen = []
    sen1 = []

    # lowercase
    low_sen = sentense.lower()
  
    # remove specail characters
    clean_sen = re.sub('[^A-Za-z0-9]+', ' ', low_sen)

    # tokenization
    token_sen = nltk.word_tokenize(clean_sen)
  
    # remove stop words
    for i in token_sen:
        if i not in stop_words:
            sen.append(i)

    # stemming
    for x in sen:
        sen1.append(ps.stem(x))

    # join into single sentense
    return " ".join(sen1)