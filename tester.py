''' tester.py '''
import getopt, sys, random, os, re
import json
import cPickle as pickle
from article import *

# global variables
sys.dont_write_bytecode = True

def main():
    
    try :
        opts, args = getopt.getopt(sys.argv[1:], "", ["num_files="])
    except getopt.GetoptError:
        err()

    if not args:
        err()
    else:
        source_dir = sys.argv[-1]
        
    for opt, arg in opts:
        if opt == "--num_files":
            num_files = int(arg)
    
    # Select n random files
    files = [filename
            for path,dirs,files in os.walk(source_dir) 
            for filename in files
            if not ".svn" in filename 
            and not ".DS_Store" in filename
            and not filename.isalpha()
            and filename.isdigit()]
    
    selected_files = [random.choice(files) for i in range(0, num_files)]
    print selected_files
    
    # print "Top 100 TFIDF scores in all categories in ", source_dir
    pickled = computeTFIDFCategory(source_dir)
    data = pickle.loads(pickled)
    articles_list = data["articles"]

    for f in selected_files :
        
        i = articles_list.index(f)
        art = articles_list[i]
        
        print "classifying articles ", art, " of ", art.category
        
        classified = classify(art, source_dir)
        print "classified as... ", classified["class"]
        print json.dumps(classified["dict"], sort_keys=True, indent=4)
    
if __name__ == "__main__":
    main()
    