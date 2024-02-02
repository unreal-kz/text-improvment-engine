import click
import csv

def load_standard_phrases(file_path):
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

@click.command()
@click.option('--file', '-f', type=click.Path(exists=True), help='Path to a text file to analyze.')
def main(file):
    """
    This CLI tool processes text input for analysis. Users can input text directly or provide a file path.
    """
    if file:
        # Read text from the specified file
        try:
            with open(file, 'r') as f:
                text = f.read()
            click.echo("Text loaded from file:")
        except Exception as e:
            click.echo(f"Failed to read file: {e}")
            return
    else:
        # Prompt for text input if no file is provided
        text = click.prompt("Please enter the text you want to analyze", type=str)
    
    click.echo(text)

# Example usage
if __name__ == "__main__":
    file_path = 'standardised_terms.csv'  # Ensure this is the correct path to your CSV files
    phrases = load_standard_phrases(file_path)
    print(phrases)
