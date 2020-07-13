#!/usr/bin/env python
# coding: utf-8

# In[20]:


from IPython.display import Image
import pandas as pd
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyser = SentimentIntensityAnalyzer()
import re
import pickle
import numpy as np
import pandas as pd

# plotting
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# nltk
from nltk.stem import WordNetLemmatizer, SnowballStemmer

# sklearn
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score


# ### New3

# ### At first place the attached `'balmy-channel-278201-7d1169cc888d.json'` file to your `'C:\\Users\your_document\'` as 'GOOGLE_APPLICATION_CREDENTIALS'. Then input the file_name and path_name of the test_data. It'll automatically evaluate sentiment.

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[1]:


import glob
import cv2
import os, io
from google.cloud import vision_v1
from google.cloud.vision import types
import nltk
from nltk.tokenize import word_tokenize, RegexpTokenizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import json
os.environ['GOOGLE_APPLICATION_CREDENTIALS']= 'Your API Key'
client = vision_v1.ImageAnnotatorClient()

Category=[]
try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract
import os
pytesseract.pytesseract.tesseract_cmd=r"C:\Program Files\Tesseract-OCR\tesseract.exe"
def ocr_core(filename):
    """
    This function will handle the core OCR processing of images.
    """


    #text = pytesseract.image_to_string(Image.open(filename))  # We'll use Pillow's Image class to open the image and pytesseract to detect the string in the image
  
   
    sentiment_dict= analyser.polarity_scores(text) 
    
    
    
   
    # print("sentence was rated as ", sentiment_dict['neg']*100, "% Negative") 
    #   print("sentence was rated as ", sentiment_dict['neu']*100, "% Neutral") 
    #   print("sentence was rated as ", sentiment_dict['pos']*100, "% Positive") 
    

    if sentiment_dict['compound'] >= 0.08 : 
                Category.append('Positive')   
                print('Positive')   
  
    elif (sentiment_dict['compound'] > - 0.08) & (sentiment_dict['compound'] < 0.08): 
            Category.append('Random')
            print('Random')
        
    elif   (sentiment_dict['compound'] <= -0.08):
                Category.append('Negative')
                print('Negative')
  
    #return text
Filename2=[]
for file in glob.glob("Hackerearth/*.jpg"):
    print(file)
    print(detect_text(file))
    Filename2.append(file)
    print(file)


# In[3]:


def clean_text(text):
    text=text.lower().split()
    from nltk.corpus import stopwords
#    stops=set(stopwords.words('english'))
#    text=[w for w in text if not w in stops]
    
    
    text= " ".join(text)
    
    text=re.sub(r'https?://[A-Za-z0-9./]+','url',text)
    text=re.sub(r"[^A-Za-z0-9^,!.\/'+-=]"," ",text)
    text=re.sub(r"what's","what is",text)
    text=re.sub(r"\'s"," is " ,text)
    text=re.sub(r"\'ve",' have ',text)
    text=re.sub(r"n't",' not ',text)
    text=re.sub(r"i'm",'i am',text)
    text=re.sub(r"\'re",' are ',text)
    text=re.sub(r"\'d",' would ',text)
    text=re.sub(r"\'ll",' will ',text)
    text=re.sub(r"\n",'',text)
    text=re.sub(r',',',',text)
    text=re.sub(r'\.','.',text)
    text=re.sub(r'!','!',text)
    text=re.sub(r'\/'," ",text)
    text=re.sub(r'\^',' ^ ',text)
    text=re.sub(r'\=',' = ',text)
    text=re.sub(r"'",' ',text)
    text=re.sub(r'(\d+)(k)',r"\g<1>000",text)
    text=re.sub(r':',' : ',text)
    text=re.sub(r' e g ',' eg ',text)
    text=re.sub(r' b g ',' bg ',text)
    text=re.sub(r' u s ',' american ',text)
    text=re.sub(r'\0s','0',text)
    text=re.sub(r' 9 11 ','911',text)
    text=re.sub(r'[0123456789]','',text)
    text=re.sub(r'e - mail','email',text)
    text=re.sub(r'j k','jk',text)
    text=re.sub(r'\s{2,}',' ',text)
    text=re.sub(r'@[A-Za-z0-9]+','',text)
    text=re.sub(r'(\w)\1{2,}',r'\1\1',text)
    text=re.sub(r'\w(\w)\1{2}','',text)
   
    
    return text

def del_NoAlphaWords(sentence):
    return " ".join([word for word in sentence.split() if word.isalpha()])


# In[10]:


import os, io
import re
from google.cloud import vision
from google.cloud.vision import types
import nltk
from nltk.tokenize import word_tokenize, RegexpTokenizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
Category=[]
analyser = SentimentIntensityAnalyzer()
from textblob import TextBlob
import json
import pandas as pd
import argparse
from nltk.stem.porter import PorterStemmer
from google.cloud import language
#from google.cloud.language import enums
#from google.cloud.language import types






os.environ['GOOGLE_APPLICATION_CREDENTIALS']= 'API key'
client = vision.ImageAnnotatorClient()
from nltk.sentiment.vader import SentimentIntensityAnalyzer
def detect_text(img):
    """Detects text in the file."""
    
    with io.open(img, 'rb') as image_file:
        content = image_file.read()

    image = vision.types.Image(content=content)
    response = client.text_detection(image=image)  # returns TextAnnotation
    df = pd.DataFrame(columns=['description'])
    texts = response.text_annotations
    for text in texts:
            df = df.append(
                dict(
                    
                    description= clean_text  (text.description)
                ),
                ignore_index=True
            )
    
    porter = PorterStemmer()

    try:
        text= (df['description'][0])
        text = porter.stem(text)
    except IndexError:
        text = 'i am neutral'
 #   print (analyze(text))
    
        
  #  print(df['description'])
    print(text)
    if len (text.split())<3:
          text = 'i am neutral'

    sentiment_dict= analyze2(text) 
    if sentiment_dict >= 0.008: 
                    Category.append('Positive')   
                    return('Positive')   

    elif (sentiment_dict > - 0.008) & (sentiment_dict < 0.008): 
            Category.append('Random')
            return('Random')

    elif   (sentiment_dict <= -0.008):
            Category.append('Negative')
            return('Negative')
   #best 0.08 
         

    
    
file='Test1001.jpg'    
path=  "C://Users/Abrar/Hackerearth/"    
detect_text(os.path.join(path,file))


# In[6]:


import json
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson.natural_language_understanding_v1 import Features, SentimentOptions, SyntaxOptions
from ibm_watson import ApiException
def analyze2(text):
    
        authenticator = IAMAuthenticator('API key')
        natural_language_understanding = NaturalLanguageUnderstandingV1(
            version='2019-11-28',
            authenticator=authenticator
        )

        natural_language_understanding.set_service_url('url')
        try:
            response = natural_language_understanding.analyze(
                text=text,

                features=Features(sentiment=SentimentOptions())).get_result()

            texts = response
            texts=pd.DataFrame(texts.items())[1][1]
            texts=pd.DataFrame(texts.items())[1][0]
            x=pd.DataFrame(texts.items())[1][0]
            return x 
        except ApiException:
            sentiment=0.3
            return sentiment
analyze2('i am neutral')


# In[13]:


get_ipython().system('pip install --upgrade google-cloud-language')


# In[7]:


"""Demonstrates how to make a simple call to the Natural Language API."""

import argparse
from google.api_core.exceptions import InvalidArgument

from google.cloud import language_v1
from google.cloud.language_v1 import enums
from google.cloud.language_v1 import types
from google.cloud.language_v1 import language_service_client
os.environ['GOOGLE_APPLICATION_CREDENTIALS']= 'API key'

def print_result(annotations):
    score = annotations.document_sentiment.score
    magnitude = annotations.document_sentiment.magnitude

    for index, sentence in enumerate(annotations.sentences):
        sentence_sentiment = sentence.sentiment.score
        print('{}'.format(
            sentence_sentiment))

    
    return 0


def analyze(text):
    """Run a sentiment analysis request on text within a passed filename."""
    client = language_service_client.LanguageServiceClient()

  #  with open(movie_review_filename, 'r') as review_file:
     # Instantiates a plain text document.
    
  #  content = text.read()
    content=text
    document = language_v1.types.Document(
        content=content,
        type=enums.Document.Type.PLAIN_TEXT,
        language='en'
    )
      #  type='PLAIN_TEXT',
  #  )
    
    try:
        response = client.analyze_sentiment(
                document=document,
                encoding_type='UTF32',
            )
        sentiment = response.document_sentiment
        return (sentiment.score)
    except InvalidArgument:
        sentiment=0.0
        return sentiment
    
  #  annotations = client.analyze_sentiment(document=document)
  # score = annotations.document_sentiment.score
  #  print(print_result(annotations))
    

    

    
analyze("the notion that i should be fine with the status quo even if iam not wholly affected by the status quo is repulsive.")   


# In[8]:


tokens=('fuck you guys @123123' )   
from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()
stemmed = porter.stem(tokens)
print(stemmed)

tokens=('fuck you guys @123123' )    


# In[15]:


sentiment_dict= analyze(text) 
  if sentiment_dict >= 0.05 : 
              Category.append('Positive')   
              return('Positive')   

  elif (sentiment_dict > - 0.05) & (sentiment_dict < 0.05): 
          Category.append('Random')
          return('Random')
      
  elif   (sentiment_dict <= -0.05 ):
              Category.append('Negative')
              return('Negative')
ar,zh,zh-Hant,nl,en,fr,de,id,it,ja,ko,pl,pt,es,th,tr,vi


# In[9]:


from google.cloud import language_v1
from google.cloud.language_v1 import enums

os.environ['GOOGLE_APPLICATION_CREDENTIALS']= 'API key'
def sample_analyze_sentiment(text_content):


    client = language_v1.LanguageServiceClient()
    # Available types: PLAIN_TEXT, HTML
    type_ = enums.Document.Type.PLAIN_TEXT

    # Optional. If not specified, the language is automatically detected.
    # For list of supported languages:
    # https://cloud.google.com/natural-language/docs/languages

    language= 'en'
    document = {"content": text_content, "type": type_, "language": language}

    # Available values: NONE, UTF8, UTF16, UTF32
    encoding_type = enums.EncodingType.UTF8

    response = client.analyze_sentiment(document, encoding_type=encoding_type)
   
    return(response.document_sentiment.score)
    
    # Get sentiment for all sentences in the document
 #   for sentence in response.sentences:
 #       print(u"Sentence text: {}".format(sentence.text.content))
 #       print(u"Sentence sentiment score: {}".format(sentence.sentiment.score))
 #       print(u"Sentence sentiment magnitude: {}".format(sentence.sentiment.magnitude))

    
sample_analyze_sentiment("being gay is not a crime and it is not a sin. stop using god to justify your prejudice. religion is about loving one another. you are  just looking for an excuse to hate. being gay   proud quotes www.geckoandfly.com")


# In[ ]:


b=pd.read_csv('Test.csv')


# In[ ]:


b.head()


# In[ ]:


b['Filename']=Filename2
b['Category']=Category


# In[ ]:


b


# In[ ]:


b.to_csv('Test3.csv')


# In[ ]:





# In[ ]:





# In[ ]:




