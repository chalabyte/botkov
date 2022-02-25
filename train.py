
import sys, os
import json
import nltk
from nltk.tokenize import wordpunct_tokenize

from ast import literal_eval

tokenizer = nltk.data.load('tokenizers/punkt/PY3/french.pickle')

VERSION="0.2"
K = {"min": 2, "max": 5}
TARGET_SENTENCE_LENGTH = 5

BRAIN_NAME = "brain.json"

def sanitize(item):
    for spaced in ['.', '-', ',', '!', '?', '(', '—', ')', '_', '"', '\'']:
        item = item.replace(spaced, f" {spaced} ")

    return item \
        .replace("🏼", "").replace("ê", "e").replace("à", "a") \
        .replace("“", "\"").replace("”", "\"") \
        .lower() \
        .strip()
    

def construct_transition_prob(brain, lines):
    k = brain["k"]["max"]
    min_k = brain["k"]["min"]

    while k>(min_k-1):
        for line in lines:
            sentences = tokenizer.tokenize(line)
            for sentence in sentences:
                if k > (min_k-1) and len(sentence.split(" ")) < k * 3: continue
                items = ["#START#"] + [item for item in wordpunct_tokenize(sentence) if len(item) > 0] + ["#END#"]
                tokens = [items[i:i+k+1] for i,_ in enumerate(items[:-k]) ]

                for token in tokens:
                    next_token = tuple(token[:-1])
                    prev_token = tuple(token[1:])

                    next_value = token[-1]
                    prev_value = token[0]

                    if not next_token in brain["data"]:
                        brain["data"][next_token] = {"next": {}, "prev": {}}
                    if not prev_token in brain["data"]:
                        brain["data"][prev_token] = {"next": {}, "prev": {}}

                    if not next_value in brain["data"][next_token]["next"]:
                        brain["data"][next_token]["next"][next_value] = 0
                    brain["data"][next_token]["next"][next_value] += 1

                    if not prev_value in brain["data"][prev_token]["prev"]:
                        brain["data"][prev_token]["prev"][prev_value] = 0
                    brain["data"][prev_token]["prev"][prev_value] += 1
        k -= 1

    for e in brain["data"]:
        for subset in brain["data"][e]:
            total = sum([v for _,v in brain["data"][e][subset].items()])
            brain["data"][e][subset] = {k: float(v)/total for k,v in brain["data"][e][subset].items()}

    return brain

def dataset_stats(content, brain_data):
    lines = content.split("\n")

    sum_words_per_line = 0
    for line in lines:
        sum_words_per_line += len(line.split(' '))
    
    keys = brain_data.keys()

    content_size        = sys.getsizeof(content)
    number_of_lines     = len(lines)
    number_of_words     = len(content.split())
    number_of_tokens    = len(keys)
    avg_words_per_line  = sum_words_per_line / number_of_lines
    
    k_token_count = {}
    for elem in keys:
        len_elem = len(elem)
        if not len_elem in k_token_count:
            k_token_count[len_elem] = 0
        k_token_count[len_elem] += 1

    print(("="*15) + " STATS " + ("="*15))
    print(f"Dataset size:       {content_size}")
    print(f"Dataset lines:      {number_of_lines}")
    print(f"Words:              {number_of_words}")
    print(f"Tokens:             {number_of_tokens}")
    print(f"Word avg per line:  {avg_words_per_line:.2f}")
    print(f"k min:              {K['min']}")
    print(f"k max:              {K['max']}")

    for i in range(K["max"] - (K["min"]) + 1):
        idx = i+K["min"]
        print(f"{idx}_token_count:      {k_token_count[idx]}")


def create_brain():
    return {
        "version": VERSION,
        "k": K,
        "target_sentence_length": TARGET_SENTENCE_LENGTH,
        "data": {},
    }

def main():

    if (len(sys.argv) != 2):
        usage()

    dataset_name = sys.argv[1]
    print(f"Getting lines from {dataset_name}...")

    with open(dataset_name, "r") as f:
        content = sanitize(f.read())

    lines = content.split("\n")
    brain = {}

    if os.path.isfile(BRAIN_NAME):
        try:
            with open(BRAIN_NAME, "r") as f:
                data = json.loads(f.read())
            brain = {literal_eval(k): v for k,v in data.items()}
            print(f"Loaded brain {BRAIN_NAME} containing {len(list(brain.keys()))} words.")
        except json.decoder.JSONDecodeError:
            print(f"Error parsing {BRAIN_NAME}. Creating new brain...")
            brain = create_brain()
    else:
        print(f" {BRAIN_NAME} not found. Creating new brain...")
        brain = create_brain()

    print("Constructing transistion probabilities...")
    brain = construct_transition_prob(brain, lines)

    dataset_stats(content, brain["data"])

    print(f"Saving brain to {BRAIN_NAME}...")
    with open(BRAIN_NAME, "w") as f:
        brain["data"] = {str(k): v for k, v in brain["data"].items()}
        f.write(json.dumps(brain))
    print(f"Saved!")


def usage():
    print(f"Usage: python3 {sys.argv[0]} <filename>")
    exit()

if __name__=='__main__':
    main()