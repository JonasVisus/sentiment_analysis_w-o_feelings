import os
import fitz  # PyMuPDF
import pandas as pd
from transformers import pipeline

# Sentiment analysis of multiple PDF reports using HuggingFace Transformers


def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


# Load HuggingFace sentiment pipeline
sentiment_pipeline = pipeline(
    "sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest"
)

pdf_folder = "reports"
results = []

if os.path.isdir(pdf_folder):
    for filename in os.listdir(pdf_folder):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(pdf_folder, filename)
            print(f"Analysed file: {filename}")

            try:
                text = extract_text_from_pdf(file_path)
                # Optional: Split text into 512-token chunks if very long
                chunks = [text[i : i + 2000] for i in range(0, len(text), 2000)]

                sentiments = []
                for chunk in chunks:
                    result = sentiment_pipeline(chunk[:512])[0]  # Limit to 512 tokens
                    sentiments.append(result)

                # Calculate average scores for each sentiment
                positive_scores = [
                    s["score"] for s in sentiments if s["label"] == "POSITIVE"
                ]
                negative_scores = [
                    s["score"] for s in sentiments if s["label"] == "NEGATIVE"
                ]
                neutral_scores = [
                    s["score"] for s in sentiments if s["label"] == "NEUTRAL"
                ]

                avg_positive = (
                    round(sum(positive_scores) / len(positive_scores), 3)
                    if positive_scores
                    else 0
                )
                avg_negative = (
                    round(sum(negative_scores) / len(negative_scores), 3)
                    if negative_scores
                    else 0
                )
                avg_neutral = (
                    round(sum(neutral_scores) / len(neutral_scores), 3)
                    if neutral_scores
                    else 0
                )

                results.append(
                    {
                        "File": filename,
                        "Positive Score": avg_positive,
                        "Negative Score": avg_negative,
                        "Neutral Score": avg_neutral,
                    }
                )

            except Exception as e:
                print(f"File {filename} not found: {e}")
else:
    print(
        f"Folder '{pdf_folder}' not found. Please create the folder and add PDF files."
    )

# Ergebnisse als DataFrame anzeigen und vergleichen
df = pd.DataFrame(results)
print("\n=== Comparing reports ===")
print(df.sort_values(by=["Positive Score"], ascending=False))

df.to_excel("sentiment_comparison_reports.xlsx", index=False)
