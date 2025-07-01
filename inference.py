import argparse
import os
import json
from model import Model
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Run the script with specified parameters.")
    
    parser.add_argument('--dataDir', type=str, default='./data', help='Directory containing the data')
    parser.add_argument('--dataset', type=str, default='free_fall', help='Dataset to use')
    parser.add_argument('--datasetPath', type=str, default='./data/qwen2.5-vl/free_fall/qa_test.json', help='Path to the dataset file')
    parser.add_argument('--outputDir', type=str, default='./output', help='Directory to save the output results')
    parser.add_argument('--model', default='qwen2.5-vl-7b', choices=['qwen2.5-vl-3b', 'qwen2.5-vl-3b-sft', 'qwen2.5-vl-7b', 'qwen2.5-vl-7b-sft', 'llava-ov-chat', 'internvl3-2b', 'internvl3-8b',
                                                                     'video-llama3-7b', 'llava-next-video-7b', 'internvideo2_5_8b',
                                                                     'gpt-4o', 'gpt-4o-mini', 'gpt-4.1', 'gpt-4.1-mini', 'gpt-4.1-nano',
                                                                     'gemini-2.0-flash', 'gemini-2.0-pro', 'gemini-2.5-flash', 'gemini-2.5-pro'], help='Model name')
    parser.add_argument('--modelPath', type=str, default='./models/checkpoint', help='Path to the model checkpoint')
    
    parser.add_argument('--gen_context', action='store_true', help='Whether to generate context for the query')
    parser.add_argument('--refer_context', action='store_true', help='Whether to refer to context in the query')

    parser.add_argument('--max_new_tokens', type=int, default=512, help='Maximum number of new tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.1, help='Temperature for sampling')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p sampling')
    parser.add_argument('--fps', type=int, default=30, help='frames per second for video')
    parser.add_argument('--max_frames', type=int, default=8, help='maximum number of frames to sample from the video')

    return parser.parse_args()

def main(args):    
    file_path = args.datasetPath
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist. Please check the dataset path.")
    with open(file_path, 'r', encoding='utf-8') as f:
        queries = json.load(f)

    # load the model
    model = Model(args.model, args.modelPath)
    
    # inference
    results = []
    
    print(args.gen_context, args.refer_context)

    for query in tqdm(queries, desc="Processing queries"):
        kwargs = {'max_new_tokens': args.max_new_tokens, 'temperature': args.temperature, 'top_p': args.top_p,
                  'fps': args.fps, 'max_frames': args.max_frames, 'gen_context': args.gen_context, 'refer_context': args.refer_context}
        
        if args.gen_context and query['q_type'] in ['fluid_amount', 'fluid_viscosity']:
            query1 = query.copy()
            query1['video'] = query1['video_list'][0]
            query2 = query.copy()
            query2['video'] = query2['video_list'][1]

            result1 = model.generate(query1, args.dataDir, **kwargs)
            result2 = model.generate(query2, args.dataDir, **kwargs)

            result = f"Left scene context: {result1}\nRight scene context: {result2}"

        else:
            result = model.generate(query, args.dataDir, **kwargs)
    
        if args.gen_context:
            query['context'] = result
        else:
            query['result'] = result
            
        results.append(query)

    # save results
    model_name = args.modelPath.split('/')[-1] if 'sft' in args.model else args.model

    os.makedirs(args.outputDir, exist_ok=True)
    os.makedirs(os.path.join(args.outputDir, args.dataset), exist_ok=True)
    with open(os.path.join(args.outputDir, args.dataset, f'{model_name}.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Results saved to {os.path.join(args.outputDir, args.dataset, f'{model_name}.json')}")

if __name__ == "__main__":
    args = parse_args()
    main(args)
