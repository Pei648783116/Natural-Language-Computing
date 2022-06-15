#  * This code is provided solely for the personal and private use of students
#  * taking the CSC401 course at the University of Toronto. Copying for purposes
#  * other than this use is expressly prohibited. All forms of distribution of
#  * this code, including but not limited to public repositories on GitHub,
#  * GitLab, Bitbucket, or any other online platform, whether as given or with
#  * any changes, are expressly prohibited.
#  
#  * All of the files in this directory and all subdirectories are:
#  * Copyright (c) 2020 Frank Rudzicz
 

import sys
import argparse
import os
import json
import re
import spacy
import html

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
nlp.add_pipe('sentencizer')


def preproc1(comment , steps=range(1, 6)):
    ''' This function pre-processes a single comment

    Parameters:                                                                      
        comment : string, the body of a comment
        steps   : list of ints, each entry in this list corresponds to a preprocessing step  

    Returns:
        modComm : string, the modified comment 
    '''
    modComm = comment
    if 1 in steps:  
        #modify this to handle other whitespace chars.(This includes the characters space, tab, linefeed, return, formfeed, and vertical tab.)
        #replace newlines with spaces
        modComm = re.sub(r"[\n\s\t\r\v\f]{1,}", " ", modComm)

    if 2 in steps:  # unescape html
        # modComm=str(html.unescape(modComm).encode('ascii', 'strict'))
        modComm=html.unescape(modComm)

    if 3 in steps:  # remove URLs
        modComm = re.sub(r"\b(http:\/\/|https:\/\/|www\.)\S+", "", modComm)
        
    if 4 in steps: #remove duplicate spaces.
        modComm = re.sub(r"\s+", " ", modComm)

    if 5 in steps:
        # TODO: get Spacy document for modComm

        doc = nlp(modComm)
        
        # TODO: use Spacy document for modComm to create a string.
        sentences = ""
        for sent in doc.sents:
            # print(sent.text)
            for token in sent:
                if token.text[0] != "-" and token.lemma_[0] == "-":
                    lemma_new = token.text
                else:
                    lemma_new = token.lemma_

                if token.text.isupper():
                    lemma_new = lemma_new.upper()
                else:
                    lemma_new = lemma_new.lower()

                sentences += lemma_new + "/" + token.tag_ + " "
                # print(lemma_new + "/" + token.tag_ + " ")
            sentences += "\n"
            # print("\n")

        # Make sure to:
        #    * Insert "\n" between sentences.
        #    * Split tokens with spaces.
        #    * Write "/POS" after each token.
            
    
    return sentences


def main(args):
    allOutput = []
    for subdir, dirs, files in os.walk(indir):
        for file in files:
            fullFile = os.path.join(subdir, file)
            print( "Processing " + fullFile)
            data = json.load(open(fullFile))

            # TODO: select appropriate args.max lines
            # TODO: read those lines with something like `j = json.loads(line)`
            index_start = args.ID[0] % len(data)
            lines = 0
            while lines < args.max:
                j = json.loads(data[index_start])
            # TODO: choose to retain fields from those lines that are relevant to you
            # TODO: add a field to each selected line called 'cat' with the value of 'file' (e.g., 'Alt', 'Right', ...)
                replaced = {}
                replaced['id'] = j['id']
            # TODO: process the body field (j['body']) with preproc1(...) using default for `steps` argument
            # TODO: replace the 'body' field with the processed text
                replaced['body'] = preproc1(j['body'])
                replaced['cat'] = file
            # TODO: append the result to 'allOutput'
                allOutput.append(replaced)
                lines += 1
                index_start += 1
                if index_start >= len(data):
                    index_start = 0

    fout = open(args.output, 'w')
    fout.write(json.dumps(allOutput))
    fout.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument('ID', metavar='N', type=int, nargs=1,
                        help='your student ID')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("--max", type=int, help="The maximum number of comments to read from each file", default=10000)
    parser.add_argument("--a1_dir", help="The directory for A1. Should contain subdir data. Defaults to the directory for A1 on cdf.", default='/u/cs401/A1')
    # parser.add_argument("--a1_dir", help="The directory for A1. Should contain subdir data. Defaults to the directory for A1 on cdf.", default='/h/u12/c2/00/lipei11/Desktop/Assignment 1 CSC2511')
    
    args = parser.parse_args()

    if (args.max > 200272):
        print( "Error: If you want to read more than 200,272 comments per file, you have to read them all." )
        sys.exit(1)
    
    indir = os.path.join(args.a1_dir, 'data')
    # indir = os.path.join(args.a1_dir, 'sample_data')
    main(args)
