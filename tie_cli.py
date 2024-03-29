import click
import csv
import nltk
import spacy
import numpy as np
import torch

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from transformers import GPT2Model, GPT2Tokenizer

# Load pre-trained model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')
# Set pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # GPT-2 uses eos_token as pad_token

nltk.download('punkt')
nltk.download('stopwords')

# Load the medium-sized model
nlp = spacy.load('en_core_web_md')

def preprocess_text(text):
    """
    Tokenizes, removes stopwords, and lowercases a given text.
    
    :param text: The text to preprocess.
    :return: A list of cleaned tokens.
    """
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    cleaned_tokens = [token.lower() for token in tokens if token.isalpha() and token.lower() not in stop_words]
    return cleaned_tokens

def load_std_phrases(file_path):
    """
    Loads standardized phrases from a CSV file into a Python dictionary.
    
    :param file_path: Path to the CSV file containing the standardized phrases.
    :return: A dictionary with standardized phrases, where keys are the identifiers (if provided).
    """
    phrases = {}
    try:
        with open(file_path, mode='r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                # Assuming the first column is the identifier and the second is the phrase
                phrases[row[0]] = row[1]
    except FileNotFoundError:
        print(f"The file {file_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    
    return phrases

def get_embedding(text):
    """
    Generates a contextual embedding for the given text using GPT-2.

    :param text: The text to embed.
    :return: A tensor representing the contextual embedding of the input text.
    """
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    # Use the last hidden state
    embeddings = outputs.last_hidden_state
    # Detach the tensor from the computation graph and convert it to a numpy array
    embeddings_numpy = embeddings.mean(dim=1).detach().numpy()
    return embeddings_numpy

def calc_similarity(embedding1, embedding2):
    # Flatten the embeddings to 1D if they are 2D of shape (1, N)
    if embedding1.ndim > 1:
        embedding1 = embedding1.flatten()
    if embedding2.ndim > 1:
        embedding2 = embedding2.flatten()

    # Normalize the embeddings to unit vectors
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    embedding1_norm = embedding1 / norm1
    embedding2_norm = embedding2 / norm2
    
    # Calculate the cosine similarity as the dot product of the normalized vectors
    similarity = np.dot(embedding1_norm, embedding2_norm)
    
    return similarity

from nltk.tokenize import sent_tokenize

def generate_suggestions(input_text, standard_phrases, similarity_threshold=0.85):
    suggestions = []
    # Segment input_text into sentences
    input_sentences = sent_tokenize(input_text)

    for original_sentence in input_sentences:
        input_embedding = get_embedding(original_sentence)
        for standard_phrase in standard_phrases.values():
            phrase_embedding = get_embedding(standard_phrase)
            similarity = calc_similarity(input_embedding, phrase_embedding)
            if similarity > similarity_threshold:
                suggestions.append((original_sentence, standard_phrase, similarity))
    
    # Sort suggestions by similarity score in descending order
    suggestions.sort(key=lambda x: x[2], reverse=True)
    return suggestions

@click.command()
@click.option('--file', '-f', type=click.Path(exists=True), help='Path to a text file to analyze.')
def main(file):
    
    file_path = 'standardised_terms.csv'
    standard_phrases = load_std_phrases(file_path)
    # file = 'text-improvment-engine/sample_text.txt'
    if file:
        try:
            with open(file, 'r') as f:
                input_text = f.read()
        except Exception as e:
            click.echo(f"Failed to read file: {e}")
            return
    else:
        input_text = click.prompt("Please enter the text to analyze", type=str)
    
    suggestions = generate_suggestions(input_text, standard_phrases)
    # if suggestions:
    #     click.echo(f"Top suggestion: {suggestions[0][0]} (Similarity: {suggestions[0][1]:.2f})")
    #     for suggestion, score in suggestions:
    #         click.echo(f"Suggested improvement: '{suggestion}' (Similarity: {score:.2f})")
    # else:
    #     click.echo("No suggestions found above the similarity threshold.")
    if suggestions:
        # If you have suggestions, print them out
        for original_phrase, suggestion, score in suggestions:
            click.echo(f"Original: '{original_phrase}' -> Suggested: '{suggestion}' (Similarity: {score:.2f})")
    else:
        # If there are no suggestions, indicate that no suggestions were found
        click.echo("No suggestions found above the similarity threshold.")


# Example usage
if __name__ == "__main__":
    main()
