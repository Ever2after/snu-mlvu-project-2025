import argparse
import json
import string
from collections import defaultdict
import os
from eval_metric import *
from tqdm import tqdm

def parse_args():  
    parser = argparse.ArgumentParser(description='Evaluate QA metrics on jsonl results')
    parser.add_argument('--resultPath', '-r', required=True, type=str)
    parser.add_argument('--name', '-n', required=True, type=str)
    parser.add_argument('--metrics', '-m', default='em,f1',
                        help='Comma-separated list of metrics: em,f1,rouge,bleu,meteor')
    parser.add_argument('--outputDir', '-o', default='./eval',
                        help='Directory to save the evaluation result')
    parser.add_argument('--detailed', '-d', default = True,
                    help='Whether to save per-item detailed scores (jsonl)')
    return parser.parse_args()


def save_detailed(detailed_records, out_path):
    with open(out_path, 'w', encoding='utf-8') as fw:
        for rec in detailed_records:
            fw.write(json.dumps(rec, ensure_ascii=False) + '\n')

def save_summary(summary_scores, num_examples, out_path):
    summary = {
        "num_examples": num_examples,
        "average": summary_scores
    }
    with open(out_path, 'w', encoding='utf-8') as fw:
        json.dump(summary, fw, ensure_ascii=False, indent=2)


# result : id query result answer
def main(args):
    metrics = [m.strip() for m in args.metrics.split(',')]
    if not set(metrics).issubset(['em', 'f1', 'rouge', 'bleu', 'meteor', 'binary', 'multi-choice' ]):
        print("Check the metrics")
        return

    results = []
    with open(args.resultPath, 'r', encoding='utf-8') as f:
        results = json.load(f)

    detailed_records = []
    summary_agg: dict[str, list[float]] = defaultdict(list)
    for result in tqdm(results, desc="Evaluating results in {}".format(args.name)):
        ans = None
        for conv in result.get('conversations', []):
            if conv.get('from') == 'gpt':
                ans = conv.get('value', '')
                break
        pred = result["result"]
        for m in metrics:
            func = METRIC_FUNCS.get(m)
            try:
                score = func(pred, ans)
            except Exception as e:
                score = 0.0
            result[m] = score
            summary_agg[m].append(score)
        detailed_records.append(result)
    
    os.makedirs(args.outputDir, exist_ok=True)
    os.makedirs(os.path.join(args.outputDir, args.name), exist_ok=True)
    summary_path  = os.path.join(args.outputDir, f'{args.name}/eval_summary.json')

    num_examples = len(results)
    summary_scores = {
        m: (sum(summary_agg[m]) / num_examples if num_examples > 0 else 0.0)
        for m in metrics
    }

    save_summary(summary_scores, num_examples, summary_path)
    if args.detailed:
        detailed_path = os.path.join(args.outputDir, f'{args.name}/eval_detailed.jsonl')
        save_detailed(detailed_records, detailed_path)
    

if __name__ == "__main__":
    args = parse_args()
    main(args)