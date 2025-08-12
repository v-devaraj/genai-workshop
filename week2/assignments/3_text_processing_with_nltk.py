# Import necessary libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter

# Download necessary NLTK resources (only run once)
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')


# Example text for processing
text = """The enthralling 2-2 draw for the Anderson-Tendulkar trophy between England and India provided a dramatic start to the new World Test Championship cycle. It was an epic contest, each of the five Tests going into the final day, four in fact into the final session, providing some of the best individual and collective performances the five-day format has seen in recent years."""


# Tokenize the text
tokens = word_tokenize(text)

# Remove stopwords
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word.lower() not in stop_words and word.isalpha()]

# POS tagging
pos_tags = nltk.pos_tag(filtered_tokens)

# Count nouns, verbs, adjectives
pos_counts = Counter(tag for word, tag in pos_tags)
num_nouns = sum(count for tag, count in pos_counts.items() if tag.startswith('NN'))
num_verbs = sum(count for tag, count in pos_counts.items() if tag.startswith('VB'))
num_adjectives = sum(count for tag, count in pos_counts.items() if tag.startswith('JJ'))

# Output results
print("Original Tokens:", tokens)
print("Filtered Tokens (No Stopwords):", filtered_tokens)
print("POS Tags:", pos_tags)
print(f"Number of Nouns: {num_nouns}")
print(f"Number of Verbs: {num_verbs}")
print(f"Number of Adjectives: {num_adjectives}")