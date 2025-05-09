import argparse
import os
import json
from model import Model

def parse_args():
    parser = argparse.ArgumentParser(description="Run the script with specified parameters.")
    
    parser.add_argument('--dataDir', type=str, default='./data', help='Directory containing the data')
    parser.add_argument('--outputDir', type=str, default='./output', help='Directory to save the output')
    parser.add_argument('--dataset', default='test', choices=['test', 'test2'], help='Dataset names')
    parser.add_argument('--model', default='qwen2.5-vl-7b', choices=['qwen2.5-vl-3b', 'qwen2.5-vl-7b', 'llava-ov-chat', 'internvl3-2b', 'internvl3-8b',
                                                                     'video-llama3-7b', 'llava-next-video-7b', 'internvideo2_5_8b',
                                                                     'gpt-4o', 'gpt-4o-mini', 'gpt-4.1', 'gpt-4.1-mini', 'gpt-4.1-nano',
                                                                     'gemini-2.0-flash', 'gemini-2.0-pro', 'gemini-2.5-flash', 'gemini-2.5-pro'], help='Model name')

    parser.add_argument('--max_new_tokens', type=int, default=512, help='Maximum number of new tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.1, help='Temperature for sampling')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p sampling')

    return parser.parse_args()

def main(args):
    # load dataset
    queries = []
    with open(os.path.join(args.dataDir, f'{args.dataset}.jsonl'), 'r') as f:
        for line in f:
            queries.append(json.loads(line))
    # load the model
    model = Model(args.model)
    # inference
    results = []
    for query in queries:
        kwargs = {'max_new_tokens': args.max_new_tokens, 'temperature': args.temperature, 'top_p': args.top_p}
        result = model.generate(query, os.path.join(args.dataDir, args.dataset), **kwargs)
        query['result'] = result
        results.append(query)
    # save results
    os.makedirs(args.outputDir, exist_ok=True)
    with open(os.path.join(args.outputDir, f'{args.dataset}_results.jsonl'), 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

if __name__ == "__main__":
    args = parse_args()
    main(args)
