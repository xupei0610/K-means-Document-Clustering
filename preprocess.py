#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""This script is for Project 2 of the course CSci 5523 Data Mining.

    This project is aimed to cluster a set of news articles by K-means algorithm.
    The dataset is derived from the "Reuters-21578 Text Categorization Collection Data Set" in the `reuters-2157878' folder.
    This script is in charge of the derivation and had been tested under Python 2.7 and 3.4.

    In the dataset of "Reuters-21578", '.sgm' files are the collection of documents. The format of '.sgm' files is similar to XML files but with lots of invalid characters and without root tag.
    Each article is in the tag `REUTERS' and the topics of an article is in the tag `TOPICS', while each topic is in the tag `D`.
    The content of each article is in the tag `TEXT/BODY'. The articles have no content will be ignored.

    In the content of documents, all letters would be converted into the lower case, and all non-ascii characters would be eliminiated while all pure numerical words will be ignored.
    See the cleanup_text() function for details about how the content of an article would be processed.

    Finally, three groups of files will be generated.
    '.csv' files are the sparse representation of documents in the dataset. It uses the default form of CSV. Each line in it represents a document and contains three elements that are separared by a comma. The first element is an integer that represents the id of a document. The second element is a string composed of a set of numbers that are separated by a comma. Each of the number represents a token the document contains. The third element has the same form with the second one, while each of the number in it represents the occurances of the corresponding token in the document. The same tokens different documents share has the same representation number.
    '.class' file records the actual classification of each document and can be used for clustering evaluation. Each line represents a document and contains two elements separated by a comma. The first element is the id of the document and the second element is the name of the actual category to which the document belongs.
    '.clabel' files are the collection of tokens. Each line is a token and the line number is the number that represents the token in corresponding '.csv' file."""

__author__ = "Pei Xu, xuxx0884@umn.edu"
__copyright__ = "Copyright 2016, Pei Xu"
__license__ = "MIT"
__version__ = "1.0"
__date__ = "19:02:38, Nov. 17th, 2016"

import os
import xml.etree.ElementTree as ET
import sys
import re
import codecs
import json
import csv

# Name of the folder that contains the article dataset
DATASET_FOLDER = 'reuters21578'

# Extension of the data files
DATA_FILE_SUFFIX = '.sgm'

# Article tag
ARTICLE_TAG = 'REUTERS'

# Topics tag
TOPICS_TAG = 'TOPICS/D'

# News ID attribute name
NEWS_ID_ATT = 'NEWID'

# Content tag
CONTENT_TAG = './TEXT/BODY'

# Minimum frequency for a token to be accepted
# token here is a word. Words who appear infrequent will be ignored.
MIN_FREQ = 5

# Candidate paremeters for the N-gram model
# level it empty if no N-gram model needs to be generated
N_GRAM = [3, 5, 7]

# Name of the folder to store the generated files
TARGET_FOLDER = 'tokens_extracted'

# Name of the file that records each topic's topic, used for clustering
# analysis
CLASS_FILE = DATASET_FOLDER

# Suffix of the class file
CLASS_FILE_EXT = '.class'

# Name of the file that contains the bag of words tokens for each article
WORD_BAG_FILE = 'bag'

# Name of the files that contain the N-gram model tokens for each article
N_GRAM_FILE = 'char<n>'

# Suffix of the generated files that contain tokens
TOKEN_FILE_EXT = '.csv'

# N-gram is based on tokens if true, otherwise based on characters
BASED_ON_TOKENS = True


def cleanup_text(text):
    """This function cleans up the given text via
    (1) eliminate any non-ascii characters
    (2) change the character case to lower-case
    (3) replace any non alphanumeric characters with space
    (4) Split the text into tokens, using space as the delimiter
    (5) eliminate any tokens that contain only digits (it should be space after the former processes)
    This function will return a string processed by the first three steps and repeated space would be eliminated.
    # The following functions are commented.
    # Additional work:
    #     (1) the abbreviation like 'u.s.' will be kept as 'u.s' rather than being splitted as 'u' and 's'
    #     (2) the possessive case in the form of xxx's will be kept as xxx

    Args:
        text: This is a string wait to be cleaned up.

    Returns:
        A string after cleanning up.

    """
    # Remove non-ascii characters and lower the case
    tt = text.encode('ascii', 'ignore').decode('utf-8').lower()

    # Replace nonalphanumeric characters with space
    tt = re.sub('\W', ' ', tt)
    # Remove pure digits
    tt = re.sub(
        '^\d+(?=\s)|(?<=\s)\d+(?=\s)|(?<=\s)\d+$', '', tt)

    # # Remove possessive cases in the form of xxxx's
    # tt = re.sub('\'s(?!\w)', '', tt)
    # Replace non-alphanumeric characters with space.
    # # We keep the word like `u.s.' as `u.s' rather than changing it into 'u s'.
    # # Attention: point numbers would be kept as well
    # tt = re.sub('[^\w\.]|\.(?!\w)|(?<!\w)\.', ' ', tt)
    # Remove pure digits including point numbers
    # tt = re.sub(
    #     '^(\d*\.*\d+)(?=\s)|(?<=\s)(\d*\.*\d+)(?=\s)|(?<=\s)(\d*\.*\d+)$', ' ', tt)

    # Remove repeated spaces
    tt = re.sub('\s*$|(?<=\s)\s|^\s', '', tt)

    # Return the processed text
    return tt


def progress_bar(current, total):
    """ A simple shell progess bar.

    Args:
        current: This is the current progress that have been completed.
        total: This is the total processes that need to be done.

    Returns:
        Nothing.
    """
    percent = '{0:.2f}'.format(100 * (current / float(total)))
    block_width = int(round(50 * current / float(total)))
    bar = '#' * block_width + ' ' * (50 - block_width)
    progress = ' ' * (4 - len(str(total))) * 2 + ' ' * \
        (len(str(total)) - len(str(current))) + str(current)
    sys.stdout.write('\r  %s/%d: |%s| %s%%' % (progress, total, bar, percent))
    if current == total:
        sys.stdout.write('\n')
    sys.stdout.flush()

if __name__ == '__main__':

    # Global token table
    tokens_table = {'bag': {}, 'n_gram': {
        n: {} for n in N_GRAM}}

    # Article list
    articles = []

    # Topic list
    topics = {}

    # Dataset path
    dataset_dir = os.path.join(os.getcwd(), DATASET_FOLDER)

    # Filter for invalid characters in the dataset files
    _illegal_xml_re = re.compile("&#([0-9]+);|&#x([0-9a-fA-F]+);")

    # Load and parse each dataset file
    print('Load dataset files...')
    files = list(filter(lambda f: f.endswith(
        DATA_FILE_SUFFIX), os.listdir(dataset_dir)))
    i = 0
    total_files = len(files)
    for file_name in files:
        i = i + 1
        progress_bar(i, total_files)
        # Open file
        try:
            with codecs.open(os.path.join(dataset_dir, file_name), 'r', encoding='ascii', errors='ignore') as f:
                # Load file into a string
                text = f.read()
        except:
            print('Error: cannot open the dataset file ' + file_name)
            raise

        # Filter invalid characters and add a root tag
        text = '<px_tag>' + \
            _illegal_xml_re.sub('', text[text.find('\n'):]) + '</px_tag>'

        # Parse the dataset file as a xml file
        dom = ET.fromstring(text)
        # For each article,
        for art in dom.findall('./' + ARTICLE_TAG):
            # find all of the article's topics
            tags = art.findall('./' + TOPICS_TAG)
            # Deal with the articles who only has exactly one topic
            if len(tags) == 1:
                # Extract the article's topic
                topic = tags[0].text
                # Extract the article's ID
                art_id = int(art.get(NEWS_ID_ATT))
                # Extract the article's content and ignore the article
                # without content
                content_dom = art.find(CONTENT_TAG)
                if content_dom == None:
                    continue
                # Clean up the content
                text = cleanup_text(content_dom.text)
                # Obtaine tokens
                tokens = text.split(' ')
                # Store the article
                articles.append(
                    [art_id, topic, tokens, {n: {} for n in N_GRAM}])
                # Update the golbal token tables
                for t in tokens:
                    if t in tokens_table['bag']:
                        tokens_table['bag'][t] = tokens_table['bag'][t] + 1
                    else:
                        tokens_table['bag'][t] = 1

    # Extract frequent words/tokens
    tokens = {}
    length = 0
    for k in sorted(tokens_table['bag'], key=tokens_table['bag'].get, reverse=True):
        if tokens_table['bag'][k] >= MIN_FREQ:
            tokens[k] = length
            length = length + 1

    # Extra frequent words/tokens for each article; Generate tokens for the
    # N-gram model
    try:
        total_article = len(articles)
        print('Extract tokens...')
        # according to the project requirement, we do not use json here
        with open(os.path.join(TARGET_FOLDER, WORD_BAG_FILE) + TOKEN_FILE_EXT, 'w') as bag_file, open(os.path.join(TARGET_FOLDER, CLASS_FILE + CLASS_FILE_EXT), 'w') as class_file:
            bag_file_writer = csv.writer(bag_file)
            i = 0
            for art in articles:
                i = i + 1
                progress_bar(i, total_article)
                # Extract frequent tokens
                tks = {}
                vaild_tks = list(filter(lambda x: x in tokens, art[2]))
                if len(vaild_tks) == 0:
                    continue
                # Count the frequency of the frequent words/tokens in the
                # document
                for t in vaild_tks:
                    if t in tks:
                        tks[t] = tks[t] + 1
                    else:
                        tks[t] = 1
                # Generate the bag-of-words model
                keys = []
                vals = []
                for k in sorted(list(tks), key=tokens.get):
                    keys.append(tokens[k])
                    vals.append(tks[k])

                bag_file_writer.writerow(
                    [art[0], ','.join(map(str, keys)), ','.join(map(str, vals))])

                if art[1] in topics:
                    topics[art[1]].append(art[0])
                else:
                    topics[art[1]] = [art[0]]
                class_file.write(str(art[0]) + ',' + art[1] + '\n')

                # Extract tokens for the N-gram model
                if BASED_ON_TOKENS == True:
                    # N-gram is based on tokens/words
                    text = vaild_tks
                else:
                    # N-gram is based on characters
                    text = ' '.join(vaild_tks)
                text_len = len(text)
                art.append({})
                for n in N_GRAM:
                    if text_len < n:
                        continue
                    for k in range(text_len - n + 1):
                        if BASED_ON_TOKENS == True:
                            tk = ' '.join(text[k:k + n])
                        else:
                            tk = text[k:k + n]

                        if tk in tokens_table['n_gram'][n]:
                            tokens_table['n_gram'][n][
                                tk] = tokens_table['n_gram'][n][tk] + 1

                            if tk in art[3][n]:
                                indx = art[3][n][tk] = art[3][n][tk] + 1
                                continue
                        else:
                            tokens_table['n_gram'][n][tk] = 1
                        art[3][n][tk] = 1
    except:
        print('Error: cannot write results into the file')
        raise
    print('  ' + str(len(tokens)) + ' tokens were extracted from ' +
          str(total_article) + ' documents.')

    # Store the N-gram model
    N_gram_tokens_table = {}
    for n in N_GRAM:
        print('Generate ' + str(n) + '-grams...')
        N_gram_tokens_table[n] = {}
        length = 0
        for k in sorted(tokens_table['n_gram'][n], key=tokens_table['n_gram'][n].get, reverse=True):
            N_gram_tokens_table[n][k] = length
            length = length + 1
        try:
            with open(os.path.join(TARGET_FOLDER, N_GRAM_FILE.replace('<n>', str(n)) + '.csv'), 'w') as gram_file:
                gram_file_writer = csv.writer(gram_file)
                i = 0
                for art in articles:
                    i = i + 1
                    progress_bar(i, total_article)
                    if len(art[3][n]) == 0:
                        continue
                    keys = []
                    vals = []
                    for k in sorted(list(art[3][n]), key=N_gram_tokens_table[n].get):
                        keys.append(N_gram_tokens_table[n][k])
                        vals.append(art[3][n][k])
                    gram_file_writer.writerow(
                        [art[0], ','.join(map(str, keys)), ','.join(map(str, vals))])

        except:
            print('Error: cannot write results into the file')
            raise

        print('  ' + str(length) + ' ' + str(n) +
              '-grams tokens were extracted.')

    # Store frequent tokens
    try:
        with open(os.path.join(TARGET_FOLDER, WORD_BAG_FILE + '.clabel'), 'w') as token_log:
            token_log.write('\n'.join(tokens))

        for n in N_GRAM:
            with open(os.path.join(TARGET_FOLDER, N_GRAM_FILE.replace('<n>', str(n)) + '.clabel'), 'w') as token_log:
                token_log.write(
                    '\n'.join(N_gram_tokens_table[n].keys()))

    except:
        print('Error: cannot write results into the file')
        raise

    print('All ' + str(3 + 2 * len(N_GRAM)) +
          ' files generated at the folder\n  ' + os.path.abspath(TARGET_FOLDER))
