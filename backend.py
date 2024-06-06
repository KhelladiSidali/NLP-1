import xml.etree.ElementTree as ET
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from transformers import pipeline

# Define file paths
input_file = 'C:/Users/fethi/Desktop/nlp project/sample-0001.xml'
output_file = 'C:/Users/fethi/Desktop/nlp project/sample.txt'
translation_output_file = 'C:/Users/fethi/Desktop/nlp project/sample_translated.txt'

# Step 1: Preprocessing the Dataset
def extract_abstract_texts(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    abstracts = []
    for abstract in root.findall(".//AbstractText"):
        abstracts.append(abstract.text)
    return abstracts

def segment_sentences(abstracts):
    sentences = []
    for abstract in abstracts:
        if abstract:
            sentences.extend(sent_tokenize(abstract))
    return sentences

abstracts = extract_abstract_texts(input_file)
sentences = segment_sentences(abstracts[:10])  # Adjust the number 10 based on your capacity

with open(output_file, 'w', encoding='utf-8') as f:
    for sentence in sentences:
        f.write(sentence + '\n')

# Step 2: Part of Speech Tagging
def pos_tagging(sentences):
    pos_tags = [nltk.pos_tag(word_tokenize(sentence)) for sentence in sentences]
    return pos_tags

def compute_transition_matrices(tags):
    unigram_counts = Counter()
    bigram_counts = Counter()
    
    for tag_seq in tags:
        tag_seq = [tag for word, tag in tag_seq]  # Correcting the list comprehension to unpack tuples correctly
        unigram_counts.update(tag_seq)
        bigram_counts.update(nltk.bigrams(tag_seq))
    
    unigram_total = sum(unigram_counts.values())
    bigram_total = sum(bigram_counts.values())
    
    unigram_probs = {tag: count/unigram_total for tag, count in unigram_counts.items()}
    bigram_probs = {bigram: count/bigram_total for bigram, count in bigram_counts.items()}
    
    return unigram_probs, bigram_probs

pos_tags = pos_tagging(sentences)
pos_unigram_probs, pos_bigram_probs = compute_transition_matrices(pos_tags)

print("### Part of Speech Tagging Unigram Probabilities:")
print(pos_unigram_probs)
print("\n### Part of Speech Tagging Bigram Probabilities:")
print(pos_bigram_probs)

# Step 3: Named Entity Recognition
def ner_tagging(sentences):
    ner_tags = []
    for sentence in sentences:
        words = word_tokenize(sentence)
        tagged_words = nltk.pos_tag(words)
        chunks = nltk.ne_chunk(tagged_words)
        for chunk in chunks:
            if hasattr(chunk, 'label'):
                ner_tags.append(chunk.label())
            else:
                ner_tags.append(chunk[1])
    return ner_tags

def compute_ner_transition_matrices(ner_tags):
    unigram_counts = Counter()
    bigram_counts = Counter()
    
    unigram_counts.update(ner_tags)
    bigram_counts.update(nltk.bigrams(ner_tags))
    
    unigram_total = sum(unigram_counts.values())
    bigram_total = sum(bigram_counts.values())
    
    unigram_probs = {tag: count/unigram_total for tag, count in unigram_counts.items()}
    bigram_probs = {bigram: count/bigram_total for bigram, count in bigram_counts.items()}
    
    return unigram_probs, bigram_probs

ner_tags = ner_tagging(sentences)
ner_unigram_probs, ner_bigram_probs = compute_ner_transition_matrices(ner_tags)

print("\n### Named Entity Recognition Unigram Probabilities:")
print(ner_unigram_probs)
print("\n### Named Entity Recognition Bigram Probabilities:")
print(ner_bigram_probs)

# Step 4: Topics Extraction Using TF-IDF for Each Abstract
def extract_topics_for_each_abstract(abstracts, num_topics=3, num_words=5):
    topics_per_abstract = []
    vectorizer = TfidfVectorizer(stop_words='english', max_features=100)

    for abstract in abstracts:
        if abstract:  # Check if the abstract is not None or empty
            tfidf_matrix = vectorizer.fit_transform([abstract])
            feature_names = vectorizer.get_feature_names_out()
            
            # Sum the TF-IDF scores for each term in the abstract
            tfidf_scores = tfidf_matrix.sum(axis=0).A1
            tfidf_scores = dict(zip(feature_names, tfidf_scores))
            
            # Sort terms by their TF-IDF scores in descending order
            sorted_terms = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Extract the top terms as topics for the current abstract
            topics = [term for term, score in sorted_terms[:num_topics]]
            topics_per_abstract.append(topics)
        else:
            topics_per_abstract.append([])  # Handle empty abstracts

    return topics_per_abstract

topics_per_abstract = extract_topics_for_each_abstract(abstracts[:10])  # Adjust the number 10 based on your capacity

print("\n### Topics per Abstract:")
for i, topics in enumerate(topics_per_abstract, 1):
    print(f"Abstract {i}: {topics}")

# Step 5: Translation
def translate_sentences(sentences, target_language='fr'):
    if target_language == 'fr':
        translator = pipeline('translation_en_to_fr', model='Helsinki-NLP/opus-mt-en-fr')
    elif target_language == 'ar':
        translator = pipeline('translation_en_to_ar', model='Helsinki-NLP/opus-mt-en-ar')
    else:
        raise ValueError("Unsupported target language. Use 'fr' for French or 'ar' for Arabic.")
    
    translations = translator(sentences)
    translated_sentences = [t['translation_text'] for t in translations]
    return translated_sentences

# Example: Translate to French
translated_sentences = translate_sentences(sentences, target_language='fr')

with open(translation_output_file, 'w', encoding='utf-8') as f:
    for sentence in translated_sentences:
        f.write(sentence + '\n')

print("\n### Translated Sentences:")
for i, sentence in enumerate(translated_sentences, 1):
    print(f"Sentence {i}: {sentence}")


