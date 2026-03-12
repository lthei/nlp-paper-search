import json
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from collections import defaultdict
from rank_bm25 import BM25Okapi
from preprocess import simple_tokenize


def load_papers(path="data/arxiv_papers.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# title is repeated twice to give it more weight than the abstract
def build_corpus(papers):
    docs = []
    for p in papers:
        docs.append({
            "id":       p["id"],
            "title":    p["title"],
            "abstract": p["abstract"],
            "authors":  p["authors"],
            "year":     p["year"],
            "text":     p["title"] + " " + p["title"] + " " + p["abstract"]
        })
    return docs


# adapted from the lab notebook
def build_inverted_index(docs):
    inverted_index = defaultdict(list)
    document_corpus = {}

    for doc in docs:
        document_corpus[doc["id"]] = doc
        tokens = set(simple_tokenize(doc["text"]))
        for token in tokens:
            inverted_index[token].append(doc["id"])

    print(f"Index built with {len(inverted_index)} unique terms.")
    return dict(inverted_index), document_corpus


if __name__ == "__main__":
    papers = load_papers()
    docs = build_corpus(papers)
    inverted_index, document_corpus = build_inverted_index(docs)

    # quick check
    tokenized_corpus = [simple_tokenize(doc["text"]) for doc in docs]
    bm25 = BM25Okapi(tokenized_corpus)
    results = bm25.get_top_n(simple_tokenize("transformer attention"), docs, n=3)
    for i, res in enumerate(results):
        print(f"Rank {i+1}: {res['title'][:80]}...")