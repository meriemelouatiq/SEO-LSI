import streamlit
import gensim
import json
from gensim import similarities
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from gensim.models import LsiModel

from gensim import corpora
from gensim.corpora import Dictionary



def lemmatize_stemming(text):
    stemmer = SnowballStemmer("english")
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
def preprocess2(text):
    result=[]
    for token in gensim.utils.simple_preprocess(text) :
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

with open("proc.txt") as f:
    processed_doc2 = json.load(f)

lsi_model = LsiModel.load("lsi.model")
bow_corpus = corpora.MmCorpus('BoW_corpus.mm')

print(" models imported ")

urls = []
# Open the file and read the content in a list
with open('urls.txt', 'r') as filehandle:
    for line in filehandle:
        # Remove linebreak which is the last character of the string
        curr_place = line[:-1]
        # Add item to the list
        urls.append(curr_place)
streamlit.markdown("<h1 style='text-align: center;background-color: #8e7cc3;color: white;border-radius: 20px;padding: 10px'>Search Engine</h1>", unsafe_allow_html=True)
streamlit.markdown("---")
query = streamlit.text_input("What are you looking for ?")
streamlit.markdown("---")
streamlit.write("<h3 style='text-align: center;font-weight: bold'>Results found :</h3>", unsafe_allow_html=True)
if (query):



    # Create a TF-IDF model from the bag-of-words representation
    tfidf = gensim.models.TfidfModel(bow_corpus)
    corpus_tfidf = tfidf[bow_corpus]

    print(" tf idf calculated ")
    # Define the query
    # query = " what is difference between sars  and corona virus.?"

    # Preprocess the query
    query_processed = preprocess2(query)
    dictionary = Dictionary(processed_doc2)
    print(" dictionary loaded ")
    # Convert the preprocessed query into a bag-of-words representation
    bow_query = dictionary.doc2bow(query_processed)

    # Create a TF-IDF representation of the preprocessed query
    tfidf_query = tfidf[bow_query]

    # Get the LSI representation of the preprocessed query
    lsi_query = lsi_model[tfidf_query]

    # calculate the cosine similarity between the query and each document in the corpus
    index = similarities.MatrixSimilarity(lsi_model[bow_corpus])
    sims = index[lsi_query]

    print(" sims calculated ")
    # sort the results in descending order
    sims_sorted_with_urls = [(index, score, urls[index]) for index, score in sorted(enumerate(sims), key=lambda item: -item[1])]
    sims_sorted_with_urls = sims_sorted_with_urls[0:20]
    for index, score, url in sims_sorted_with_urls:
        streamlit.write(f"{url}")
        streamlit.write(f" {score}")
        streamlit.markdown("---")