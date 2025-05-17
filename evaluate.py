import argparse
import json
import string
from collections import defaultdict
import os
from eval_metric import *

def parse_args():  
    parser = argparse.ArgumentParser(description='Evaluate QA metrics on jsonl results')
    parser.add_argument('--resultDir', '-r', required=True, default= './output', help='Directory to results.jsonl')
    parser.add_argument('--name', '-n', required=True, default= 'test2_results', type=str, help='result file to evaluate')
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
    if not set(metrics).issubset(['em', 'f1', 'rouge', 'bleu', 'meteor']):
        print("Check the metrics")
        return

    results = []
    with open(os.path.join(args.resultDir, f'{args.name}.jsonl'), 'r') as f:
        for line in f:
            results.append(json.loads(line))

    detailed_records = []
    summary_agg: dict[str, list[float]] = defaultdict(list)
    for result in results:
        pred = result["result"]
        ans = result["answer"]
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
    summary_path  = os.path.join(args.outputDir, f'{args.name}_eval_summary.json')

    num_examples = len(results)
    summary_scores = {
        m: (sum(summary_agg[m]) / num_examples if num_examples > 0 else 0.0)
        for m in metrics
    }

    save_summary(summary_scores, num_examples, summary_path)
    if args.detailed:
        detailed_path = os.path.join(args.outputDir, f'{args.name}_eval_detailed.jsonl')
        save_detailed(detailed_records, detailed_path)
    

if __name__ == "__main__":
    args = parse_args()
    main(args)