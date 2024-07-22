import json
import argparse

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def create_parser():
    parser = argparse.ArgumentParser(description='CFIC grounding text generation')
    parser.add_argument('--article_path', type=str,
                        help='/path/to/article.txt')
    parser.add_argument('--question_path', type=str,
                        help='/path/to/question.json')
    parser.add_argument('--model', type=str,
                        default='TinyLlama/TinyLlama-1.1B-Chat-v1.0')
    parser.add_argument('--cache_dir', type=str, default='./model_cache',
                        help='for llm model and tokenizer')
    parser.add_argument('--topk', type=int, default=3)
    parser.add_argument('--max_length', type=int, default=256)
    return parser


def prepare_data(article_path, question_path):
    with open(article_path, 'r') as f:
        long_knowledge = f.read()

    with open(question_path, 'r') as f:
        question_data = json.load(f)
    questions = [q['question'] for q in question_data]
    return long_knowledge, questions


def prepare_model_and_tokenizer(model_name, cache_dir):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        cache_dir=cache_dir
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir
    )
    model.cuda()
    model.eval()
    return model, tokenizer


def main(args):
    long_knowledge, questions = prepare_data(args.article_path, args.question_path)
    model, tokenizer = prepare_model_and_tokenizer(args.model, args.cache_dir)


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(args)
