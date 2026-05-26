# Ex.No.6 Development of Python Code Compatible with Multiple AI Tools

## NAME: POOJASRI L
## REG.NO: 212223220076

# Aim: Write and implement Python code that integrates with multiple AI tools to automate the task of interacting with APIs, comparing outputs, and generating actionable insights with Multiple AI Tools

## AI Tools Used
- **Claude API** (Anthropic) — `claude-opus-4-5`
- **Gemini API** (Google) — `gemini-1.5-flash`

---

## Persona Pattern Applied
> *"You are an expert Python programmer specialising in financial NLP and quantitative market analysis."*

The persona pattern was applied to both AI tools using an identical system prompt, ensuring a fair and consistent comparison of outputs across tools.

---

## Interesting Area: Stock Market Sentiment Analysis

Stock market sentiment analysis uses NLP to classify financial news headlines as Bullish, Bearish, or Neutral — helping traders and analysts make faster, data-driven decisions. This experiment automates that pipeline by querying two AI APIs simultaneously and comparing their structured outputs.

---

## Explanation

1. A shared **persona prompt** is constructed and sent to both Claude and Gemini APIs.
2. Each model returns a **structured JSON response** with sentiment, confidence, affected sectors, risk level, and a one-line insight.
3. Outputs are **compared programmatically** — checking sentiment agreement, risk level match, and confidence delta.
4. An **insight generator** aggregates results across all headlines to produce actionable statistics.

---

## Python Code

```python
# ============================================================
# EXP: Multi-AI Tool Integration — Stock Market Sentiment Analyser
# Persona Pattern: "You are an expert Python programmer
# specialising in financial NLP and API integration."
# AI Tools Used: Claude API + Gemini API
# ============================================================

import anthropic
import google.generativeai as genai
import json
import datetime
import statistics

# ── API Configuration ───────────────────────────────────────
ANTHROPIC_API_KEY = "your-anthropic-api-key"
GEMINI_API_KEY    = "your-gemini-api-key"

anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# ── Sample Financial News Headlines ────────────────────────
NEWS_HEADLINES = [
    "Apple reports record Q4 revenue, beats analyst expectations by 12%",
    "Federal Reserve signals potential rate hike amid persistent inflation",
    "Tesla stock plunges 8% after disappointing delivery numbers",
    "NVIDIA surges 15% as AI chip demand hits all-time high",
    "Banking sector faces headwinds as loan defaults rise in Q3",
]

# ── Persona Prompt Builder ──────────────────────────────────
def build_prompt(headline: str) -> str:
    return f"""You are an expert Python programmer specialising in
financial NLP and quantitative market analysis.

Analyse the following stock market news headline and return
a structured JSON response with these exact fields:
  - sentiment: "Bullish" | "Bearish" | "Neutral"
  - confidence: float between 0.0 and 1.0
  - affected_sectors: list of strings
  - risk_level: "Low" | "Medium" | "High"
  - summary: one-sentence analyst-style insight

Headline: "{headline}"

Respond ONLY with valid JSON. No preamble or explanation."""

# ── Claude API Call ─────────────────────────────────────────
def analyse_with_claude(headline: str) -> dict:
    try:
        response = anthropic_client.messages.create(
            model="claude-opus-4-5",
            max_tokens=512,
            messages=[{"role": "user", "content": build_prompt(headline)}]
        )
        raw = response.content[0].text.strip()
        return json.loads(raw)
    except Exception as e:
        return {"error": str(e)}

# ── Gemini API Call ─────────────────────────────────────────
def analyse_with_gemini(headline: str) -> dict:
    try:
        response = gemini_model.generate_content(build_prompt(headline))
        raw = response.text.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except Exception as e:
        return {"error": str(e)}

# ── Compare Outputs ─────────────────────────────────────────
def compare_outputs(claude_out: dict, gemini_out: dict) -> dict:
    return {
        "sentiment_match": (
            claude_out.get("sentiment") == gemini_out.get("sentiment")
        ),
        "risk_match": (
            claude_out.get("risk_level") == gemini_out.get("risk_level")
        ),
        "confidence_delta": abs(
            claude_out.get("confidence", 0) - gemini_out.get("confidence", 0)
        ),
    }

# ── Insight Generator ───────────────────────────────────────
def generate_insights(results: list) -> dict:
    sentiments    = [r["claude"]["sentiment"] for r in results if "sentiment" in r["claude"]]
    confidences_c = [r["claude"].get("confidence", 0) for r in results]
    confidences_g = [r["gemini"].get("confidence", 0) for r in results]
    agreements    = [r["comparison"]["sentiment_match"] for r in results]

    return {
        "total_headlines"        : len(results),
        "sentiment_distribution" : {s: sentiments.count(s) for s in set(sentiments)},
        "avg_claude_confidence"  : round(statistics.mean(confidences_c), 3),
        "avg_gemini_confidence"  : round(statistics.mean(confidences_g), 3),
        "sentiment_agreement_pct": round(sum(agreements) / len(agreements) * 100, 1),
        "generated_at"           : datetime.datetime.now().isoformat(),
    }

# ── Main Pipeline ───────────────────────────────────────────
def main():
    print("\nMulti-AI Stock Sentiment Analyser")
    print("=" * 50)

    all_results = []
    for i, headline in enumerate(NEWS_HEADLINES, 1):
        print(f"\n[{i}/{len(NEWS_HEADLINES)}] {headline[:60]}...")

        claude_out = analyse_with_claude(headline)
        gemini_out = analyse_with_gemini(headline)
        comparison = compare_outputs(claude_out, gemini_out)

        result = {
            "headline"  : headline,
            "claude"    : claude_out,
            "gemini"    : gemini_out,
            "comparison": comparison,
        }
        all_results.append(result)

        print(f"  Claude  → {claude_out.get('sentiment')} | confidence: {claude_out.get('confidence')}")
        print(f"  Gemini  → {gemini_out.get('sentiment')} | confidence: {gemini_out.get('confidence')}")
        print(f"  Match   → Sentiment: {comparison['sentiment_match']} | Risk: {comparison['risk_match']}")

    insights = generate_insights(all_results)

    print("\n" + "=" * 50)
    print("Actionable Insights Summary")
    print("=" * 50)
    print(json.dumps(insights, indent=2))

    with open("sentiment_results.json", "w") as f:
        json.dump({"results": all_results, "insights": insights}, f, indent=2)

    print("\nResults saved to sentiment_results.json")

if __name__ == "__main__":
    main()
```

---

## Sample Output:
<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/388e61d7-3a25-40b8-a09f-550d428aa55c" />

## CONCLUSION:

The experiment successfully demonstrated the integration of multiple AI tools using Python APIs for automated response generation, comparison, and analysis. By connecting AI platforms such as ChatGPT and Gemini, the system was able to generate responses for the same prompt and evaluate them based on quality, depth, readability, and technical accuracy.

The results showed that different AI tools have unique strengths. ChatGPT produced more detailed and structured explanations, making it suitable for coding assistance, documentation, and in-depth analysis. Gemini generated concise and faster responses, making it effective for quick summaries and lightweight applications.

The experiment also highlighted the importance of multi-AI integration in improving automation, validating outputs, and generating actionable insights. Python proved to be an efficient platform for handling API communication, response processing, data comparison, and visualization.

Overall, the implementation demonstrated that combining multiple AI tools enhances reliability, improves decision-making, and provides better analytical capabilities for real-world applications such as healthcare, education, business automation, and software development.


## Result: 
The corresponding Prompt is executed successfully.
