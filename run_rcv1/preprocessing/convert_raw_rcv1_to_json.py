"""
Read RCV1 data from NIST and store it in a JSON which HuggingFace datasets library can process.

Example command: python convert_raw_rcv1_to_json.py --path_to_rcv1 /n/fs/nlp-asd/asd/asd/Projects/SemanticLabelEmbeddings/data/RCV1/rcv1_small --save_name rcv1_small_all.json
"""

import os
import argparse
from bs4 import BeautifulSoup
from tqdm import tqdm
import json

def process_rcv1(args):
    # Read all the XML files in the folder
    list_of_files = list()
    for (dirpath, dirnames, filenames) in os.walk(args.path_to_rcv1):
        list_of_files += [os.path.join(dirpath, file) for file in filenames]

    # Store list of dicts
    document_list = []

    # Read each XML file and store it in a JSON
    for filename in tqdm(list_of_files):
        if filename.endswith('.xml'):
            flag = True
            bs_data = BeautifulSoup(open(filename, 'r', encoding="ISO-8859-1").read(), "lxml")

            # Construct a dictionary with required fields
            document = dict()
            document['title'] = bs_data.find('title').text
            document['headline'] = bs_data.find('headline').text
            document['newsitem'] = bs_data.find('newsitem').get('itemid')
            document['folder'] = filename.strip().split('/')[-2]

            # Text
            document['text'] = ' '.join([data.text for data in bs_data.find('text').find_all('p')])

            # Codes
            code_types = ['bip:topics:1.0', 'bip:industries:1.0', 'bip:countries:1.0']
            for code_type in code_types:
                if bs_data.find('codes', {'class': code_type}) is None:
                    if code_type == 'bip:topics:1.0':
                        flag = False
                    else:
                        document[code_type] = None
                else:
                    document[code_type] = []
                    for code in bs_data.find('codes', {'class': code_type}).find_all('code'):
                        document[code_type].append(code.get('code'))

            # Append to the dictionary
            if flag:
                document_list.append(document)

    # Store list of dicts as a json
    f = open(os.path.join(args.save_dir, args.save_name), 'w', encoding="ISO-8859-1")

    for document in document_list:
        f.write(str(json.dumps(document)) + '\n')

    f.close()


def main():
    parser = argparse.ArgumentParser()

    # Dataset Arguments
    parser.add_argument("--path_to_rcv1", type=str, default="/n/fs/nlp-asd/asd/asd/Projects/SemanticLabelEmbeddings/data/RCV1/rcv1_small", help="")
    parser.add_argument("--save_dir", type=str, default="", help="")
    parser.add_argument("--save_name", type=str, default="rcv1_all.json", help="")

    args = parser.parse_args()

    process_rcv1(args)

if __name__ == '__main__':
    main()