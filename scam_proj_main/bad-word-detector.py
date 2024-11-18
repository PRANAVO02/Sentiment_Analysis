import sys
import time

import nltk
nltk.download('stopwords')
nltk.download('punkt')



# Ensure that NLTK data is available
try:
    from nltk import wordpunct_tokenize
    from nltk.corpus import stopwords
except ImportError:
    print('[!] You need to install nltk (http://nltk.org/index.html)')
    sys.exit(1)  # Exit the script if nltk is not installed

def calculate_languages_ratios(text):
    languages_ratios = {}
    tokens = wordpunct_tokenize(text)
    words = [word.lower() for word in tokens]

    # Compute per language included in nltk number of unique stopwords appearing in analyzed text
    for language in stopwords.fileids():
        stopwords_set = set(stopwords.words(language))
        words_set = set(words)
        common_elements = words_set.intersection(stopwords_set)

        languages_ratios[language] = len(common_elements)  # language "score"

    return languages_ratios

def detect_language(text):
    ratios = calculate_languages_ratios(text)
    most_rated_language = max(ratios, key=ratios.get)
    return most_rated_language

def load_bad_words(language):
    if language.upper() in ['ENGLISH', 'FRENCH', 'SPANISH', 'GERMAN']:
        try:
            badwords_list = []
            with open('datasets/' + language.lower() + '.csv', 'r') as lang_file:
                for word in lang_file:
                    badwords_list.append(word.lower().strip())
            return badwords_list
        except Exception as e:
            print('Error Occurred while loading bad words file. Error: ' + str(e))
            return set()
    else:
        return set()

def load_file(filename):
    try:
        with open(filename, 'r') as file:
            return file.read()
    except Exception as e:
        print('Error Occurred while loading file. Error: ' + str(e))
        return None

def clean_text(text):
    # Remove punctuation and convert to lowercase
    for key in ['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}']:
        text = text.replace(key, '')
    return text.lower()

if len(sys.argv) < 2:
    print('Usage: python bad-word-detector.py <filename>')
    sys.exit(1)

filename = sys.argv[1]
print('Input File Name: ' + filename)
try:
    text = load_file(filename)
    if text is None:
        raise Exception("Failed to read the file.")
    print('\n')
    time.sleep(2)
    print('-----------------Input Text-----------------')
    print(text)
    print('--------------------------------------------\n')

except Exception as e:
    print('Error Occurred while loading text file. Error: ' + str(e))
    sys.exit(1)  # Exit if there's an error loading the file

language = detect_language(text)

# Force "HINGLISH" to "ENGLISH" as a fallback
if language.upper() == 'HINGLISH':
    language = 'english'

print('\n')
time.sleep(1)
print('----------------------------')
print('Language Detected: ', language.upper())
print('----------------------------')
print('\n')

time.sleep(1)
print('Checking for bad words in ' + language.upper() + ' language...')
print('**********************************************************\n')

try:
    badwords = load_bad_words(language)
    if badwords:
        badwords = set(badwords)
    else:
        badwords = set()
except Exception as e:
    print('Error Occurred in Program - Error: ' + str(e))
    badwords = set()

text_list = text.split('\n')
for line_number, sentence in enumerate(text_list, start=1):
    cleaned_sentence = clean_text(sentence)
    words = cleaned_sentence.split()
    abuses = [word for word in words if word in badwords]
    if abuses:
        time.sleep(0.5)
        print(f'-- Bad Words found at line number: {line_number} --')
        x_words = ', '.join(abuses)
        print(f'Bad Words: {x_words}')
        print('-----------------\n')
