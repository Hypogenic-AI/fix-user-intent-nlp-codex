import json
import os
import time
from datetime import datetime

from llm_utils import LLMClient, LLMConfig

ROOT = os.path.dirname(os.path.dirname(__file__))
SAMPLE_PATH = os.path.join(ROOT, "results", "sample.jsonl")
OUTPUT_PATH = os.path.join(ROOT, "results", "model_outputs", "llm_outputs.jsonl")
CONFIG_PATH = os.path.join(ROOT, "results", "config.json")

MODEL_REWRITE = os.getenv("MODEL_REWRITE", "gpt-4.1")
MODEL_JUDGE = os.getenv("MODEL_JUDGE", "gpt-4.1")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))


def load_existing(path):
    data = {}
    if not os.path.exists(path):
        return data
    with open(path, "r") as f:
        for line in f:
            obj = json.loads(line)
            data[obj["example_id"]] = obj
    return data


def save_append(path, obj):
    with open(path, "a") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def format_context(context_list):
    if not context_list:
        return "(no prior context)"
    # Context is alternating Q/A strings
    lines = []
    for i, text in enumerate(context_list):
        role = "User" if i % 2 == 0 else "Assistant"
        lines.append(f"{role}: {text}")
    return "\n".join(lines)


def main():
    if not os.path.exists(SAMPLE_PATH):
        raise FileNotFoundError("Run data_prep.py first to create sample.jsonl")

    with open(SAMPLE_PATH, "r") as f:
        samples = [json.loads(line) for line in f]

    existing = load_existing(OUTPUT_PATH)

    client = LLMClient(model=MODEL_REWRITE)
    rewrite_cfg = LLMConfig(model=MODEL_REWRITE, temperature=TEMPERATURE, max_tokens=200)
    decision_cfg = LLMConfig(model=MODEL_REWRITE, temperature=0, max_tokens=200)

    system_rewrite = (
        "You rewrite a conversational user question into a standalone question. "
        "Preserve the user's intent and include needed context. Return JSON only."
    )

    system_decide = (
        "You decide whether a clarification question is needed to preserve intent. "
        "If multiple plausible referents or missing info exist, ask for clarification. Return JSON only."
    )

    system_clarify = (
        "You ask a single concise clarification question to resolve ambiguity. "
        "Be brief and avoid extra commentary. Return JSON only."
    )

    system_answer = (
        "You are the user. Answer the clarification question concisely based on the conversation and the gold intent. "
        "Return JSON only."
    )

    system_rewrite_with_answer = (
        "You rewrite the user's question into a standalone question using the clarification answer to resolve ambiguity. "
        "Return JSON only."
    )

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    for idx, ex in enumerate(samples):
        example_id = f"{ex['Conversation_no']}_{ex['Turn_no']}"
        if example_id in existing:
            continue

        context = format_context(ex.get("Context", []))
        question = ex.get("Question", "")
        gold = ex.get("Rewrite", "")

        record = {
            "example_id": example_id,
            "conversation_no": ex.get("Conversation_no"),
            "turn_no": ex.get("Turn_no"),
            "context": context,
            "question": question,
            "gold_rewrite": gold,
            "outputs": {},
            "metadata": {"timestamp": datetime.now().isoformat()},
        }

        # Direct rewrite
        user_rewrite = (
            f"Conversation:\n{context}\n\nCurrent question: {question}\n\n"
            "Return JSON: {\"rewrite\": "
            "...}"  # inlined to enforce JSON-only output
        )
        resp = client.chat_json(system_rewrite, user_rewrite, rewrite_cfg)
        record["outputs"]["direct_rewrite"] = {
            "rewrite": resp["data"]["rewrite"],
            "usage": resp["usage"],
            "duration_s": resp["duration_s"],
        }

        # Clarification decision
        user_decide = (
            f"Conversation:\n{context}\n\nCurrent question: {question}\n\n"
            "Return JSON: {\"needs_clarification\": true/false, \"confidence\": 0.0-1.0, "
            "\"rationale\": \"short\"}"
        )
        dec = client.chat_json(system_decide, user_decide, decision_cfg)
        needs_clarify = bool(dec["data"]["needs_clarification"])
        record["outputs"]["clarification_decision"] = {
            "needs_clarification": needs_clarify,
            "confidence": dec["data"]["confidence"],
            "rationale": dec["data"].get("rationale", ""),
            "usage": dec["usage"],
            "duration_s": dec["duration_s"],
        }

        # Always-clarify path
        user_clarify = (
            f"Conversation:\n{context}\n\nCurrent question: {question}\n\n"
            "Return JSON: {\"clarification_question\": "
            "...}"
        )
        clarify = client.chat_json(system_clarify, user_clarify, rewrite_cfg)
        clarification_q = clarify["data"]["clarification_question"]

        user_answer = (
            f"Conversation:\n{context}\n\nGold intent: {gold}\n\n"
            f"Clarification question: {clarification_q}\n\n"
            "Return JSON: {\"user_answer\": "
            "...}"
        )
        answer = client.chat_json(system_answer, user_answer, rewrite_cfg)
        clarification_a = answer["data"]["user_answer"]

        user_rewrite2 = (
            f"Conversation:\n{context}\n\nCurrent question: {question}\n\n"
            f"Clarification Q: {clarification_q}\n\nUser answer: {clarification_a}\n\n"
            "Return JSON: {\"rewrite\": "
            "...}"
        )
        rewrite2 = client.chat_json(system_rewrite_with_answer, user_rewrite2, rewrite_cfg)
        record["outputs"]["always_clarify"] = {
            "clarification_question": clarification_q,
            "clarification_answer": clarification_a,
            "rewrite": rewrite2["data"]["rewrite"],
            "usage": {
                "clarify": clarify["usage"],
                "answer": answer["usage"],
                "rewrite": rewrite2["usage"],
            },
        }

        # Gated-clarify path
        if needs_clarify:
            record["outputs"]["gated_clarify"] = {
                "clarification_question": clarification_q,
                "clarification_answer": clarification_a,
                "rewrite": rewrite2["data"]["rewrite"],
                "used_clarification": True,
            }
        else:
            record["outputs"]["gated_clarify"] = {
                "rewrite": record["outputs"]["direct_rewrite"]["rewrite"],
                "used_clarification": False,
            }

        # No-rewrite baseline
        record["outputs"]["no_rewrite"] = {"rewrite": question}

        save_append(OUTPUT_PATH, record)
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx+1}/{len(samples)}")

        time.sleep(0.2)

    # Save config for reproducibility
    config = {
        "model_rewrite": MODEL_REWRITE,
        "model_judge": MODEL_JUDGE,
        "temperature": TEMPERATURE,
        "sample_size": len(samples),
        "timestamp": datetime.now().isoformat(),
    }
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)

    print("Wrote:", OUTPUT_PATH)
    print("Wrote:", CONFIG_PATH)


if __name__ == "__main__":
    main()
