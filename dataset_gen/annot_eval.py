import json
import yaml
import argparse
from difflib import SequenceMatcher


#similarity of sentence
def text_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def jaccard(list1: list, list2: list) -> float:
    set1, set2 = set(list1), set(list2)
    if not set1 and not set2:
        return 1.0
    return len(set1 & set2) / len(set1 | set2)

def compare_structured(val_dict: dict, res_dict: dict,  
                       free_text_keys=None) -> dict:

    if free_text_keys is None:
        free_text_keys = {"fluidBehavior", "fluidColor", "flowDirection"}

    scores = {}
    all_keys = set(val_dict) | set(res_dict)
    for key in all_keys:
        v = val_dict.get(key, None)
        r = res_dict.get(key, None)

        if isinstance(v, list) or isinstance(r, list):
            scores[key] = jaccard(v or [], r or [])

        elif key in free_text_keys and isinstance(v, str) and isinstance(r, str):
            scores[key] = text_similarity(v, r)

        else:
            scores[key] = 1.0 if v == r else 0.0

    scores["__record_average__"] = sum(scores.values()) / len(scores)
    return scores

def compare_simple(val_str: str, res_str: str) -> float:
    return text_similarity(val_str, res_str)

def score_json(input_path: str, output_path: str, mode: str):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = []
    for idx, entry in enumerate(data):
        value_text = ""
        for msg in entry.get("conversations", []):
            if msg.get("from") == "gpt":
                value_text = msg.get("value", "")
                break

        result_text = entry.get("result", "")

        if mode == "simple":
            sim = compare_simple(value_text, result_text)
            results.append({"index": idx, "similarity": sim})

        else:  # structured
            val_yaml = yaml.safe_load(value_text)
            res_yaml = yaml.safe_load(result_text)

            field_scores = compare_structured(val_yaml, res_yaml)
            results.append({"index": idx, **field_scores})

    if mode == "simple":
        overall = sum(r["similarity"] for r in results) / len(results)
    else:
        overall = sum(r["__record_average__"] for r in results) / len(results)

    out = {
        "mode": mode,
        "overall_average": overall,
        "records": results
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"[{mode}] scored {len(results)} entries.")
    print(f"Overall average similarity: {overall:.4f}")
    print(f"Results written to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute similarity between 'value' and 'result' in JSON")
    parser.add_argument("input_json", default = "video.json")
    parser.add_argument("output_json", default = "annot_score.json")
    parser.add_argument(
        "--mode", choices=["simple", "structured"], default="simple")

    args = parser.parse_args()
    score_json(args.input_json, args.output_json, args.mode)
