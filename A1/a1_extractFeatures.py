#  * This code is provided solely for the personal and private use of students
#  * taking the CSC401 course at the University of Toronto. Copying for purposes
#  * other than this use is expressly prohibited. All forms of distribution of
#  * this code, including but not limited to public repositories on GitHub,
#  * GitLab, Bitbucket, or any other online platform, whether as given or with
#  * any changes, are expressly prohibited.
#  
#  * All of the files in this directory and all subdirectories are:
#  * Copyright (c) 2020 Frank Rudzicz

import numpy as np
import argparse
import json
import re
import csv
import string

# Provided wordlists.
FIRST_PERSON_PRONOUNS = {
    'i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours'}
SECOND_PERSON_PRONOUNS = {
    'you', 'your', 'yours', 'u', 'ur', 'urs'}
THIRD_PERSON_PRONOUNS = {
    'he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'they', 'them',
    'their', 'theirs'}
SLANG = {
    'smh', 'fwb', 'lmfao', 'lmao', 'lms', 'tbh', 'rofl', 'wtf', 'bff',
    'wyd', 'lylc', 'brb', 'atm', 'imao', 'sml', 'btw', 'bw', 'imho', 'fyi',
    'ppl', 'sob', 'ttyl', 'imo', 'ltr', 'thx', 'kk', 'omg', 'omfg', 'ttys',
    'afn', 'bbs', 'cya', 'ez', 'f2f', 'gtr', 'ic', 'jk', 'k', 'ly', 'ya',
    'nm', 'np', 'plz', 'ru', 'so', 'tc', 'tmi', 'ym', 'ur', 'u', 'sol', 'fml'}


def extract1(comment):
    ''' This function extracts features from a single comment

    Parameters:
        comment : string, the body of a comment (after preprocessing)

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (only the first 29 are expected to be filled, here)
    '''
    feature = np.zeros(173)
    sentences = comment.split("\n")
    words = []
    for sentence in sentences:
        words += sentence.split(" ")
    tokens = []
    tags = []
    for word in words:
        parts = word.split("/")
        if len(parts) != 2:
            continue
        [token, tag] = word.split("/")

        tokens.append(token)
        tags.append(tag)

    # TODO: Extract features that rely on capitalization.

    upper_num = 0
    for token in tokens:
        if len(token) >= 3 and token.isupper():
            upper_num += 1

    feature[0] = upper_num
    # TODO: Lowercase the text in comment. Be careful not to lowercase the tags. (e.g. "Dog/NN" -> "dog/NN").

    tokens_lower = [token.lower() for token in tokens]
    # TODO: Extract features that do not rely on capitalization.
    firstperson_num = 0
    secondperson_num = 0
    thirdperson_num = 0
    token_lengths = []
    aoa = []
    img = []
    fam = []
    vsum = []
    asum = []
    dsum = []
    
    for token in tokens_lower:
        if token in FIRST_PERSON_PRONOUNS:
            firstperson_num += 1
        if token in SECOND_PERSON_PRONOUNS:
            secondperson_num += 1
        if token in THIRD_PERSON_PRONOUNS:
            thirdperson_num += 1
    
        is_punctuation_only = (len(token.strip(string.punctuation)) == 0)
        if len(token) > 1 and is_punctuation_only:
            feature[8] += 1
        if token in SLANG:
            feature[13] += 1
        if not is_punctuation_only:
            token_lengths.append(len(token))

        # Don't count tokens that do not appear in the dicts
        if token in bgl and token != "":
            b_row = bgl[token]
            aoa.append(int(b_row["AoA (100-700)"]))
            img.append(int(b_row["IMG"]))
            fam.append(int(b_row["FAM"]))
        if token in warringer and token != "":
            w_row = warringer[token]
            vsum.append(float(w_row["V.Mean.Sum"]))
            asum.append(float(w_row["A.Mean.Sum"]))
            dsum.append(float(w_row["D.Mean.Sum"]))


    feature[1] = firstperson_num
    feature[2] = secondperson_num
    feature[3] = thirdperson_num
    feature[15] = 0 if not token_lengths else sum(token_lengths) / len(token_lengths)
    feature[17] = 0 if not aoa else sum(aoa) / len(aoa)
    feature[18] = 0 if not img else sum(img) / len(img)
    feature[19] = 0 if not fam else sum(fam) / len(fam)
    feature[20] = 0 if not aoa else np.std(aoa)
    feature[21] = 0 if not img else np.std(img)
    feature[22] = 0 if not fam else np.std(fam)
    feature[23] = 0 if not vsum else sum(vsum) / len(vsum)
    feature[24] = 0 if not asum else sum(asum) / len(asum)
    feature[25] = 0 if not dsum else sum(dsum) / len(dsum)
    feature[26] = 0 if not vsum else np.std(vsum)
    feature[27] = 0 if not asum else np.std(asum)
    feature[28] = 0 if not dsum else np.std(dsum)

    sentence_lengths = []
    current_sentence_length = 0
    for index, tag in enumerate(tags):
        current_sentence_length += 1
        if tag == "CC":
            feature[4] += 1
        if tag == "VBD":
            feature[5] += 1
        if tag == "VB" and index >= 1 and (
                tokens_lower[index - 1] == "will" or
                tokens_lower[index - 1] == "'ll" or
                tokens_lower[index - 1] == "gonna" or
                (index >= 2 and tokens_lower[index - 2] == "going" and tokens_lower[index - 1] == "to")):
            feature[6] += 1
        if tag == ",":
            feature[7] += 1
        if tag[0:3] == "NNP":
            feature[10] += 1
        elif tag[0:2] == "NN":
            feature[9] += 1
        if tag == "RB" or tag == "RBR" or tag == "RBS":
            feature[11] += 1
        if tag == "WDT" or tag == "WP" or tag == "WP$" or tag == "WRB":
            feature[12] += 1
        if tag == ".":
            sentence_lengths.append(current_sentence_length)
            current_sentence_length = 0

    feature[14] = 0 if not sentence_lengths else sum(sentence_lengths) / len(sentence_lengths)
    feature[16] = len(sentence_lengths)

    return feature

def extract2(feat, comment_class, comment_id):
    ''' This function adds features 30-173 for a single comment.

    Parameters:
        feat: np.array of length 173
        comment_class: str in {"Alt", "Center", "Left", "Right"}
        comment_id: int indicating the id of a comment

    Returns:
        feat : numpy Array, a 173-length vector of floating point features (this 
        function adds feature 30-173). This should be a modified version of 
        the parameter feats.
    '''    
    # print('TODO')

    feature_index = id_to_index[comment_class][comment_id]
    feature = liwc_features[comment_class][feature_index]
    feat[29:173] = feature



def main(args):
    #Declare necessary global variables here.

    # Preload extract1 data sources
    global bgl
    bgl = {}
    with open('/u/cs401/Wordlists/BristolNorms+GilhoolyLogie.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            bgl[row['WORD']] = row
    global warringer
    warringer ={}
    with open('/u/cs401/Wordlists/Ratings_Warriner_et_al.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            warringer[row['Word']] = row

    # Preload extract2 data sources
    global cat_to_feature
    cat_to_feature = {'Left': 0, 'Center': 1, 'Right': 2, 'Alt': 3}
    global id_to_index
    id_to_index = {}
    global liwc_features
    liwc_features = {}
    for cat in cat_to_feature.keys():
        id_to_index[cat] = {}
        liwc_features[cat] = []
        ids_filename = "/u/cs401/A1/feats/" + cat + "_IDs.txt"
        features_filename = "/u/cs401/A1/feats/" + cat + "_feats.dat.npy"
        liwc_features[cat] = np.load(features_filename)
        with open(ids_filename, "r") as ids_file:
            for i, str in enumerate(ids_file):
                id_to_index[cat][str.strip("\n")] = i

    #Load data
    data = json.load(open(args.input))
    feats = np.zeros((len(data), 173+1))

    for i, comment in enumerate(data):
        # TODO: Call extract1 for each datatpoint to find the first 29 features.
        # Add these to feats.
        feature_slice = feats[i, :]
        feature_slice[:-1] = extract1(comment['body'])
        # TODO: Call extract2 for each feature vector to copy LIWC features (features 30-173)
        # into feats. (Note that these rely on each data point's class,
        # which is why we can't add them in extract1).
        extract2(feature_slice[:-1], comment['cat'], comment['id'])
        feature_slice[-1] = cat_to_feature[comment['cat']]

    # print('TODO')
    np.savez_compressed(args.output, feats)

    
if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("-i", "--input", help="The input JSON file, preprocessed as in Task 1", required=True)
    parser.add_argument("-p", "--a1_dir", help="Path to csc401 A1 directory. By default it is set to the cdf directory for the assignment.", default="/u/cs401/A1/")
    args = parser.parse_args()

    main(args)

