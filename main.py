import os
import fitz  # PyMuPDF
import pandas as pd
import re
from transformers import pipeline
from transformers import AutoTokenizer
from collections import Counter
from csrd_buzzwords import csrd_seeds, csrd_words
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

os.environ["OPENAI_API_KEY"] = (
    "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"
)


def extract_text_from_pdf(file_path):
    """Extracts text from a PDF file using PyMuPDF.
    Args:
        file_path: path to the PDF file
    Returns:
        text: text extracted from the PDF file
    """
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


def split_into_token_chunks(text, max_tokens=512):
    """Splits text into chunks of max_tokens using the tokenizer.
    Args:
        tokenizer: The tokenizer to use for splitting.
        max_tokens: Maximum number of tokens per chunk.
    Returns:
        List of tokenized chunks as text.
    """
    # Tokenize the text into token IDs
    tokens = tokenizer.encode(
        text,
        add_special_tokens=False,
    )
    chunks = [tokens[i : i + max_tokens] for i in range(0, len(tokens), max_tokens)]
    # Decode each chunk back into text
    return [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]


def count_words_in_text(text, terms):
    """Counts occurrences of CSRD terms in the text.
    Args:
        text: text to search for CSRD terms
        terms: list of CSRD terms to count
    Returns:
        int: total count of CSRD terms in the text
    """
    text = text.lower()
    counts = Counter()
    for term in terms:
        counts[term] = len(re.findall(r"\b" + re.escape(term) + r"\b", text))
    return sum(counts.values())


def enforce_token_limit(chunks, max_tokens=512):
    """
    Ensures that each chunk does not exceed the maximum token limit.
    Args:
        chunks: List of text chunks.
        max_tokens: Maximum number of tokens allowed per chunk.
    Returns:
        List of chunks with enforced token limits.
    """
    adjusted_chunks = []
    for chunk in chunks:
        tokenized_chunk = tokenizer.encode(chunk, add_special_tokens=False)
        # Split the chunk if it exceeds the max token limit
        for i in range(0, len(tokenized_chunk), max_tokens):
            sub_chunk = tokenized_chunk[i : i + max_tokens]
            adjusted_chunks.append(
                tokenizer.decode(sub_chunk, skip_special_tokens=True)
            )
    return adjusted_chunks


def semantic_chunker(text, max_tokens=512):
    """
    Splits text into semantic chunks using LangChain's SemanticChunker.
    Args:
        text: The input text to split.
        max_tokens: Maximum number of tokens per chunk.
    Returns:
        List of semantic chunks as text.
    """
    chunker = SemanticChunker(OpenAIEmbeddings())
    chunks = chunker.split_text(text)
    return enforce_token_limit(chunks, max_tokens=max_tokens)


# -------------------------------------------------------------------------------
# Sentiment analysis of multiple PDF reports using HuggingFace Transformers
# -------------------------------------------------------------------------------

# Load HuggingFace sentiment pipeline
sentiment_pipeline = pipeline(
    "sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment"
)
# Load the tokenizer for the sentiment model
tokenizer = AutoTokenizer.from_pretrained(
    "nlptown/bert-base-multilingual-uncased-sentiment"
)

pdf_folder = "reports"
results = []

if os.path.isdir(pdf_folder):
    for filename in os.listdir(pdf_folder):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(pdf_folder, filename)
            print(f"Analysing file: {filename}")

            try:
                text = extract_text_from_pdf(file_path)
                csrd_seed_count = count_words_in_text(text, csrd_seeds)
                csrd_word_count = count_words_in_text(text, csrd_words)

                # Split text into 512-token chunks for HuggingFace model
                chunks = split_into_token_chunks(text, max_tokens=500)
                # chunks = semantic_chunker(text, max_tokens=512)

                sentiments = []
                for i, chunk in enumerate(chunks):
                    result = sentiment_pipeline(chunk)[0]
                    sentiments.append(result)

                star_scores = [
                    int(s["label"].split()[0]) * s["score"] for s in sentiments
                ]
                total_score = sum([s["score"] for s in sentiments])
                avg_stars = (
                    round(sum(star_scores) / total_score, 2) if total_score > 0 else 0
                )

                results.append(
                    {
                        "File": filename,
                        "Average Stars": avg_stars,
                        "CSRD Seeds": csrd_seed_count,
                        "CSRD Reprasentative Terms": csrd_word_count,
                    }
                )

            except Exception as e:
                print(f"Error with {filename}: {e}")
else:
    print(
        f"Folder '{pdf_folder}' not found. Please create the folder and add PDF files."
    )

# Print results in a DataFrame and save to Excel
df = pd.DataFrame(results)
print("\n=== Comparing reports ===")
print(df.sort_values(by=["Average Stars"], ascending=False))
df.to_excel("sentiment_comparison_reports.xlsx", index=False)
