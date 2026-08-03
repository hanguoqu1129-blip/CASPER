# CASPER

Code for the paper "CASPER: Character-Anchored Salient Passage Extraction
for Character Profile Generation from Long-Form Fiction" (under review).

Given a full novel and a target character, CASPER selects a compact set of
salient, character-anchored passages with purely extractive steps, then
produces a character profile with a single generative call.

## Setup

```
pip install -r requirements.txt
python -m spacy download en_core_web_sm
cp .env.example .env   # then fill in your OpenAI API key
```

## Data

The `data/` directory is expected to contain three things. Data is not
distributed with this repository; it follows the CroSS benchmark of
Yuan et al. (2024), "Evaluating Character Understanding of Large Language
Models via Character Profiling from Fictional Works"
(https://arxiv.org/abs/2404.12726).

`data/books/<book_id>.txt` — plain text of each novel, one file per book.

`data/mr_items.jsonl` — one JSON object per line with fields `book_id`,
`title`, and `character` (the target character's full name).

`data/character_match_terms.csv` — columns `book_id` and `match_terms`,
where `match_terms` is a semicolon-separated list of full-name aliases for
the target character.

The synthesis step also reads a prompt template from
`prompts/original_character_profiling/profile_once_prompt.txt`. Copy the
single-pass profiling prompt from the CroSS repository
(https://github.com/Joanna0123/character_profiling) into that path.

## Usage

```
python casper.py <book_id> 35000
```

The second argument is the evidence token budget. The selected evidence and
the synthesized profile are printed to stdout.
