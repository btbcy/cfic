#!/bin/bash

ARTICLE_PATH=$1
QUESTION_PATH=$2

python src/main.py --article_path "$ARTICLE_PATH" --question_path "$QUESTION_PATH" 
