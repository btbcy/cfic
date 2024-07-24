# Grounding Language Model with Chunking-Free In-Context Retrieval

This is an unofficial implementation of Chunking-Free In-Context (CFIC) retrieval method based on the paper "Grounding Language Model with Chunking-Free In-Context Retrieval" by Qian et al. (2024).

## Environment

```
pip install -r requirements.txt
```


## Usage

```
python src/main.py [-h] [--article_path ARTICLE_PATH] [--question_path QUESTION_PATH] [--output_path OUTPUT_PATH]
               [--model MODEL] [--cache_dir CACHE_DIR] [--topk TOPK] [--max_length MAX_LENGTH]
```

- `--article_path`: Path to the input article file (plain text)
- `--question_path`: Path to the questions file (JSON format)
- `--output_path`: Path to save the output results (in JSON format). If not specified, results will be displayed on the screen.
- `--model`: Name of the language model (default: TinyLlama/TinyLlama-1.1B-Chat-v1.0)
- `--cache_dir`: Directory to cache the language model and tokenizer
- `--topk`: Number of candidate sentences (default: 3)
- `--max_length`: Maximum length of generated grounding texts (default: 256 tokens)

### Example

```
python src/main.py --article_path data/test_long_knowledge.txt --question_path data/questions.json --output_path data/result.json --max_length 256 --topk 3
```

This command will process the article in `test_long_knowledge.txt`, answer questions from `questions.json`, and save the results in `result.json`. 

```
bash run_cfic.sh /path/to/article.txt /path/to/question.json
```

This command will process the article, answer questions as specified, and print the results to the console.

### Input Formats

- Article: Plain text file containing the document to be queried
- Questions: JSON file containing an array of question objects. Each object should have a `question` field. The `answer` field is optional. (see `data/questions.json.example`)

### Output

The script generates results containing the original questions and the grounding texts for each question. Overlapping passages for the same question are merged into a single passage.

If an output path is specified, the results are saved as a JSON file. Otherwise, they are displayed on the screen.

## Project Structure

- `src/cfic.py`: Main implementation of the CFIC algorithm
- `src/preprocessing.py`: Text cleaning and sentence tokenization utilities
- `src/main.py`: Script to run the CFIC retrieval process

## Reference

1. Qian, H., Liu, Z., Mao, K., Zhou, Y., & Dou, Z. (2024). Grounding Language Model with Chunking-Free In-Context Retrieval. *arXiv preprint arXiv:2402.09760*.

2. https://github.com/qhjqhj00/acl2024_cfic
