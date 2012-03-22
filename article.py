''' article.py '''
import getopt, sys, os, re
import cPickle as pickle
from collections import defaultdict
from json import dumps
from time import time
from math import log, sqrt, pow
from operator import mul, itemgetter
from itertools import imap
import tree
from stopwords import *

# global data structures and variables
DEBUG = False
sys.dont_write_bytecode = True

def main():
    
    try :
        opts, args = getopt.getopt(sys.argv[1:], [], "debug")
    except getopt.GetoptError:
        err()

    if not args:
        err()
    else:
        source_dir = sys.argv[-1]
    
    for opt, arg in opts:
        if opt == "--debug":
            global DEBUG
            DEBUG = True
        
    print "***************************************************************"
    print "Newsgroups with IDFs"
    newsgroups_IDFs = computeDocumentFrequency(source_dir)
    for newsgroup, IDF in newsgroups_IDFs.iteritems():
        print newsgroup
        if DEBUG: 
            print dumps(IDF, sort_keys=True, indent=4)
    
    print "***************************************************************"
    print "hClustering"
    newsgroup_TFIDF = computeNewsGroupCategory(source_dir)
    root = hCluster(newsgroup_TFIDF)
    print tree.printChildren(root)


# Part 1
class article():
    ''' article class '''
    def __init__(self, name, category, text):
        self.name = name
        self.category = category
        self.text = text
        self.text_list = []
        self.TFIDF = defaultdict(int)
        self.TF = defaultdict(int)
        
    def __repr__(self):
        return self.name
        
    def __eq__(self, foo):
        return self.name == foo
        
    # Part 3 
    def computeTFIDF(self, DF, corpus_size):
        ''' compute the TFIDF score of this article to given corpus '''
        TFIDF = {}
        
        for word in self.text_list:
            word_TF = self.TF[word]     # num occurrence of this word in the article
            word_IDF = computeIDF(corpus_size, DF, word)
            TFIDF[word] = word_TF * word_IDF    # update this article's TFIDF
            
        self.TFIDF = TFIDF
        return TFIDF

def computeIDF(corpus_size, DF, word):
    ''' calculate IDF score of a given word (smoothing applied) '''
    if word not in DF: DF[word] = 1
    if DF[word] == 0: DF[word] = 1
    return log((corpus_size / DF[word]) )


# Part 2
def computeDocumentFrequency(directory):
    ''' compute IDF score of each subdirectory of a given directory '''
    
    newsgroups, corpus_size = {}, 0
    
    # 1) map each newsgroup to its articles
    # newsgroup = { "sports": [1,2,3], "tech": {a,b,c}, ...}
    for root, sub_dirs, files in os.walk(directory):
        
        # ignore ".svn" and ".DS_Store" dir and files
        if ".svn" in root or ".DS_Store" in root: continue
        for d in sub_dirs:
            if ".svn" in d or ".DS_Store" in d: continue
                    
        # Each Category
        category = root[len(directory):]
        articles_in_category = []
        
        for filename in files:
            path = os.path.join(root, filename)
            if ".DS_Store" in path: continue
            with open(path, "r") as f:                
                art = article(filename, category, f.read())
                articles_in_category.append(art)
                
        clean(articles_in_category)     # strip non-char and stop-words
        newsgroups[category] = articles_in_category
    
    # 2) constructing...
    # - corpus of all newsgroups
    # - DF for each newsgroup
    newsgroups_IDFs, word_IDF = {}, 0.0
    
    for newsgroup, articles in newsgroups.iteritems():
        newsgroup_doc_freq = defaultdict(int)
        corpus_size = len(articles)
        count_words(articles, newsgroup_doc_freq)
        
        newsgroup_each_IDF = {}
        for art in articles:
            for word in art.text_list:
                # word_IDF = computeIDF(corpus_size, newsgroup_doc_freq, word)
                newsgroup_each_IDF[word] = \
                    computeIDF(corpus_size, newsgroup_doc_freq, word)
        
        newsgroups_IDFs[newsgroup] = newsgroup_each_IDF
    
    return newsgroups_IDFs


def computeNewsGroupCategory(directory):
    ''' compute each newsgroup's TFIDF score of a given directory '''
    
    newsgroup_TFIDF = defaultdict(float)
    
    for root, sub_dirs, files in os.walk(directory):
        
        if ".svn" in root or ".DS_Store" in root: continue
        for d in sub_dirs:
            if ".svn" in d or ".DS_Store" in d: continue    # ignore svn directories
            path = os.path.join(root,d)
            if ".DS_Store" in path: continue
            
            pickled_info = computeTFIDFCategory(path)
            info = pickle.loads(pickled_info)
            newsgroup_TFIDF[d] = info["TFIDF"]
            
    return newsgroup_TFIDF


# Part 4
def computeTFIDFCategory(directory, pickleIt=True):
    ''' compute top n TFIDF scores of a given directory '''
    
    dir_doc_freq, dir_articles, corpus_size = defaultdict(int), [], 0
    
    # 1) create article objects
    for root, sub_dirs, files in os.walk(directory):
        if ".svn" in root or ".DS_Store" in root: continue
        for d in sub_dirs:
            if ".svn" in d or ".DS_Store" in d: continue    # ignore svn directories
        
        # Each Category
        category = root[len(directory):]
        for filename in files:
            path = os.path.join(root, filename)
            if ".DS_Store" in path: continue
            
            with open(path, "r") as f:
                art = article(filename, category, f.read())
                dir_articles.append(art)
                
    # 2) build DF and Corpus
    clean(dir_articles)
    count_words(dir_articles, dir_doc_freq)
    corpus_size = len(dir_articles)
    
    # 3) calculate TFIDF for the directory
    dir_TFIDF = defaultdict(int)
    if DEBUG: print "Calculate TFIDF for ", directory
    
    startTime = time() # timing performance
    for art in dir_articles:
        article_TFIDF = art.computeTFIDF(dir_doc_freq, corpus_size)
        dir_TFIDF = merge(sort_dict_by_value(article_TFIDF), dir_TFIDF)
        
    if DEBUG: print "Elaspse time = ", time() - startTime
    entry = { "DF" : dir_doc_freq,
              "articles" : dir_articles,
              "TFIDF" : sort_dict_by_value(dir_TFIDF)} 
    
    if pickleIt:
        return pickle.dumps(entry)
    else:
        return entry


def classify(art, directory):
    ''' estimate the newsgroup which a given article should belong to '''
    
    cos_dict = defaultdict(float)
    pickled = computeTFIDFCategory(directory)
    dir_TFIDF = pickle.loads(pickled)
    art_TFIDF = art.computeTFIDF(dir_TFIDF['DF'], len(dir_TFIDF['articles']))
    
    for root, sub_dirs, files in os.walk(directory):
        if ".svn" in root or ".DS_Store" in root: continue
        for d in sub_dirs:
            if ".svn" in d or ".DS_Store" in d: continue    # ignore svn directories
            path = os.path.join(root,d)
            pickled_category = computeTFIDFCategory(path)
            category = pickle.loads(pickled_category)
            temp = cosineSimilarity(category["TFIDF"], art_TFIDF)
            cos_dict[d] = temp
    
    max_set = max(cos_dict.iteritems(), key=itemgetter(1))
    return {"class": max_set, "dict" : cos_dict}


def cosineSimilarity(df1, df2):
    ''' compute cosine similarity score of two document frequency dicts '''
    
    list1 = [v for k,v in df1.iteritems()]
    list2 = [v for k,v in df2.iteritems()]
    numerator = sum(imap(mul, list1, list2))
    denominator1 = sqrt(sum([pow(i,2) for i in list1]))
    denominator2 = sqrt(sum([pow(i,2) for i in list2]))
    
    # print "numerator = ",numerator
    # print "denominator1 = ",denominator1,"; denominator2 = ",denominator2
    return numerator / (denominator1 * denominator2)


def hCluster(S):
    ''' build denogram by comparing cosine similarity scores of all categories '''
    
    while len(S) > 1:
        cos_score = 0.0
        selected = (None, None)
        right, left = None, None
        
        # 1) find the two most similar elements e1 and e2 in S using cos()
        for k, v in S.iteritems():
            for k_, v_ in S.iteritems():
                if k_ is not k:
                    temp_score = cosineSimilarity(v, v_)
                    if temp_score > cos_score:
                        selected = (k, k_)
                        cos_score = temp_score
                        right = tree.assertCreateNode(k_)
                        left = tree.assertCreateNode(k)
                        
        # 2) replace them in S with e1Ve2
        parent = str(selected[0]) + " && " + str(selected[1]) 
        if DEBUG:
            print "parent : ", parent
            print "right : ", right
            print "left : ", left
        
        node = tree.Node(parent, right, left)
        S[node] = merge(v, v_)
        del S[selected[0]]
        del S[selected[1]]
        
        if DEBUG:
            print "\n"
            print "Tree: "
            print tree.printChildren(node)
            print "\n\n"
        
    return node


def clean(articles):
    ''' strip non-characters and stop_words '''
    
    for art in articles:
        art.text = re.sub("[^A-Za-z]", " ", art.text)
        art.text_list = [w.lower() for w in art.text.split() if w.lower() not in stop_words]
        art.text = ' '.join([o for o in art.text_list])


def count_words(articles, doc_freq):
    ''' update article TF and document frequency dictionaries '''
    
    for art in articles:
        for w in art.text.split(): art.TF[w] += 1
        for k,v in art.TF.iteritems(): doc_freq[k] += 1


def merge(d1, d2):
    ''' merge and accumulate two dictionaries '''
    
    for k in d1.keys():
        if k in d2: d2[k] = d2[k] + d1[k]
        else: d2[k] = d1[k]
        
    return d2
    
    ''' deprecated - inefficent merge '''
    # return dict( (n, d1.get(n,0) + d2.get(n,0) ) for n in set(d1)|set(d2) )


def sort_dict_by_value(d, decending=True, top_n=1000):
    ''' keeping top n results of given dictionary '''
    
    temp_list = sorted(d.iteritems(), key=itemgetter(1), reverse=decending)[:top_n]
    return dict(temp_list)


def err():
    print "Error when running program. Please see README"
    print "Usage:"
    print "$ python article.py [--debug] [source directory]"
    sys.exit(-1)


if __name__ == '__main__':
    main()
