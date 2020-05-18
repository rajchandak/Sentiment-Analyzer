import string
import math
import re
import csv

list_of_reviews = []
list_of_test_reviews = []
def read_all_values(filename,list_of_reviews):
    # Read the reviews from the txt file.
    with open(filename, encoding="utf-8-sig") as training_set:
        reviews = training_set.readlines()
    
    for i in range(0,len(reviews)):
        # Split each review in the form of a kew value pair.
        # Key is of the form : ID-XXXX.
        # Value is the actual review.
        key = reviews[i][:8].strip()
        # Replace a+ with aplus before striping and lowercasing.
        # Lowercase the reviews.
        value = reviews[i][8:].replace('a+','aplus').strip().lower()
        # Remove all punctuation except '!'.
        for p in string.punctuation:
            if p!='!':
                value = value.replace(p,"")
        # Add the key-value pairs in the dictionary.
        dictionary = {}
        dictionary['ID'] = key
        dictionary['value'] = value
        if filename=="hotelPosT-train.txt":
            dictionary['class'] = 1
        elif filename=="hotelNegT-train.txt":
            dictionary['class'] = 0
        list_of_reviews.append(dictionary)

# Feature extraction for training set
read_all_values("hotelPosT-train.txt",list_of_reviews)
read_all_values("hotelNegT-train.txt",list_of_reviews)
# Feature extraction for test set
read_all_values("testset.txt",list_of_test_reviews)

# Read the positive lexicons from the txt file.
with open("positive-words.txt") as postive_words:
    positive_lexicons = postive_words.readlines()
    
positive_lexicons[0] = 'aplus'
# Read the negative lexicons from the txt file.
with open("negative-words.txt") as negative_words:
    negative_lexicons = negative_words.readlines()

list_of_pronouns = ["i", "me", "mine", "my", "you", "your", "yours", "we", "us", "ours", "ive", "weve", "youve"]

feature_extraction = []
feature_extraction_test = []

def get_features(feature_extraction, list_of_reviews, type_of_dataset):
    for item in list_of_reviews:
        dictionary = {}
        key = item.get('ID')
        value = item.get('value')
        if type_of_dataset=="training":    
            item_class = item.get('class')
        x1,x2,x3,x4,x5,x6 = 0,0,0,0,0,0
    
        # Feature 5 (x5) = Check for "!".
        if '!' in value:
            x5 = 1
            value = value.replace('!','')
        
        # Feature 1 (x1) = Count all positive lexicons.
        for lex in positive_lexicons:  
            # Remove all * and - from the current lexicon.
            lex = lex.replace('*',"").replace('-',"")
            if re.search(r"\b{}\b".format(lex.strip()), value, re.IGNORECASE) is not None:
                x1 += 1
        
        # Feature 2 (x2) = Count all negative lexicons.
        for lex in negative_lexicons:
            # Remove all * and - from the current lexicon.
            lex = lex.replace('*',"").replace('-',"")
            if re.search(r"\b{}\b".format(lex.strip()), value, re.IGNORECASE) is not None:
                x2 += 1
        
        # Feature 3 (x3) = Check for the word "no"
        if re.search(r"\b{}\b".format("no"), value, re.IGNORECASE) is not None:
            x3 = 1
            
        # Feature 4 (x4) = Count all pronouns.
        for pronoun in list_of_pronouns:
            if re.search(r"\b{}\b".format(pronoun.strip()), value, re.IGNORECASE) is not None:
                x4 += 1
        
        # Feature 2 (x2) = Count all words in the review. Words do not include puncutation marks.
        x6 = math.log(len(value.split()))
        x6 = round(x6,2)
            
        dictionary['ID'] = key
        dictionary['x1'] = x1
        dictionary['x2'] = x2
        dictionary['x3'] = x3
        dictionary['x4'] = x4
        dictionary['x5'] = x5
        dictionary['x6'] = x6
        if type_of_dataset=="training":    
            dictionary['class'] = item_class
        
        # feature_extraction is a list of dictionaries.
        feature_extraction.append(dictionary)
    
get_features(feature_extraction, list_of_reviews, "training")
get_features(feature_extraction_test, list_of_test_reviews, "test")

# Write the list of dictionaries- feature_extraction for training set to a csv file.
keys = feature_extraction[0].keys()
with open('feature_extraction-training-data.csv', 'w', encoding='utf-8-sig') as output_file:
    dict_writer = csv.DictWriter(output_file, keys, lineterminator = '\n')
    dict_writer.writerows(feature_extraction)

# Write the list of dictionaries- feature_extraction for test set to a csv file.
keys = feature_extraction[0].keys()
with open('feature_extraction-test-set.csv', 'w', encoding='utf-8-sig') as output_file:
    dict_writer = csv.DictWriter(output_file, keys, lineterminator = '\n')
    dict_writer.writerows(feature_extraction_test)