import argparse
from transformers import BertModel, BertTokenizer
import torch
import numpy as np

def get_user_input():
    parser = argparse.ArgumentParser(description="Text Improvement Engine")
    parser.add_argument('input_text', type=str, help="Text to be analyzed and improved")
    args = parser.parse_args()
    return args.input_text

standardised_phrases = [
    "Optimal performance", 
    "Utilise resources", 
    "Enhance productivity", 
    "Conduct an analysis",
    "Maintain a high standard",
    "Implement best practices",
    "Ensure compliance",
    "Streamline operations",
    "Foster innovation",
    "Drive growth",
    "Leverage synergies",
    "Demonstrate leadership",
    "Exercise due diligence", 
    "Maximize stakeholder value",
    "Prioritise tasks",
    "Facilitate collaboration",
    "Monitor performance metrics",
    "Execute strategies",
    "Gauge effectiveness",
    "Champion change",
]

def get_standard_phrases():
    return standardised_phrases


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def find_closest_phrases(user_input:list)->list:
    user_input_embedding = get_embedding(user_input)
    suggestions = []
    for phrase in standardised_phrases:
        phrase_embedding = get_embedding(phrase)
        similarity = cosine_similarity(user_input_embedding, phrase_embedding)
        suggestions.append((phrase, similarity))
    suggestions.sort(key=lambda x: x[1], reverse=True)
    return suggestions

def main():
    user_input = get_user_input()
    closest_phrases = find_closest_phrases(user_input)
    print("Suggestions:")
    for phrase, score in closest_phrases[:5]:  # Display top 5 suggestions
        print(f"{phrase} - Similarity Score: {score}")

if __name__ == "__main__":
    main()