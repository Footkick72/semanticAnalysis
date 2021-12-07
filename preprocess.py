# https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html
import gensim.downloader as api
import numpy as np
import json

# list of availiable datasets
# models are downloaded to ~/.gensim-data
# ['fasttext-wiki-news-subwords-300',
#  'conceptnet-numberbatch-17-06-300',
#  'word2vec-ruscorpora-300',
#  'word2vec-google-news-300',
#  'glove-wiki-gigaword-50',
#  'glove-wiki-gigaword-100',
#  'glove-wiki-gigaword-200',
#  'glove-wiki-gigaword-300',
#  'glove-twitter-25',
#  'glove-twitter-50',
#  'glove-twitter-100',
#  'glove-twitter-200',
#  '__testing_word2vec-matrix-synopsis']

# models are stored in ~/.gensim-data


def download_data():
    #wordlist = "accusatory abstruse acerbic admonishing aloof ambivalent analytical ardent authoritarian belligerent benevolent brusque caustic cautionary censorious charismatic complimentary conciliatory condemnatory condescending confrontational contemptuous contentious conversational curt cynical derisive despairing detached didactic diffident disdainful disillusioned dogmatic domineering dubious ebullient effusive elegiac eloquent emphatic enigmatic erudite euphoric exhortatory facetious farcical fatalistic flippant forthright frivolous haughty holier-than-thou impassive incisive incredulous indignant inflated insipid irrelevant jovial laudatory lofty ludicrous meditative melancholic mild moralistic nonchalant objective obsequious ominous patronizing penitent pessimistic polished provocative reasoned reserved reticent reverential sardonic scholarly skeptical sobering subtle supercilious tentative terse vindictive vitriolic wistful wry zealous"
    wordlist = "elegiac dubious complimentary skeptical indignant cautionary zealous authoritarian domineering charismatic patronizing condemnatory condescending aloof detached disillusioned fatalistic incredulous despairing admonishing accusatory derisive conciliatory ardent reverential confrontational laudatory ambivalent conversational belligerent cynical acerbic provocative caustic censorious obsequious exhortatory disdainful contentious penitent"
    wv = api.load("word2vec-google-news-300")
    with open("vectors2.txt", "w") as f:
        vectors = {}
        for word in wordlist.split(" "):
            vector = wv[word]
            print("done with", word)
            vectors[word] = vector.tolist()
        json_string = json.dumps(vectors)
        f.write(json_string)


if __name__ == "__main__":
    download_data()
