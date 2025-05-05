from transformers import AutoTokenizer, AutoProcessor, AutoModel
import os
import torch
import sys

class Model:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model, self.tokenizer, self.processor = self.load_model()
    
    def load_model(self):
        if 'qwen2.5-vl' in self.model_name:
            from transformers import Qwen2_5_VLForConditionalGeneration
            MODEL_ID = f"qwen/{self.model_name}-instruct"
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                MODEL_ID, 
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto")
            processor = AutoProcessor.from_pretrained(MODEL_ID)
            return model, None, processor
        elif 'llava-ov-chat' in self.model_name:
            from llava.model.builder import load_pretrained_model
            MODEL_ID = "lmms-lab/llava-onevision-qwen2-7b-ov-chat"
            model_name = "llava_qwen"
            device_map = "auto"
            tokenizer, model, image_processor, max_length = load_pretrained_model(MODEL_ID, None, model_name, device_map=device_map)
            model.eval()
            return model, tokenizer, image_processor
        elif 'internvl3' in self.model_name:
            from internvl_utils import split_model
            MODEL_ID = f'opengvlab/{self.model_name}-instruct'
            device_map = split_model(MODEL_ID)
            model = AutoModel.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                use_flash_attn=True,
                trust_remote_code=True,
                device_map=device_map).eval()
            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, use_fast=False)
            return model, tokenizer, None
        elif 'gpt' in self.model_name:
            from openai import OpenAI
            client = OpenAI()
            return client, None, None
        elif 'gemini' in self.model_name:
            return None, None, None
        else:
            raise ValueError(f"Unsupported model name: {self.model_name}")
        

    def generate(self, query, data_path, **kwargs):
        if 'qwen2.5-vl' in self.model_name:
            from qwen_vl_utils import process_vision_info
            messages = []
            if 'images' in query.keys():
                messages = [{
                    "role": "user",
                    "content": [{
                        "type": "image",
                        "image": f"file://{os.path.abspath(os.path.join(data_path, image_path))}"
                    } for image_path in query['images']]
                }]
            elif 'videos' in query.keys():
                messages = [{
                    "role": "user",
                    "content": [{
                        "type": "video",
                        "video": f"file://{os.path.abspath(os.path.join(data_path, video_path))}"
                    } for video_path in query['videos']]
                }]
            else:
                messages = [{
                    "role": "user",
                    "content": []
                }]
            messages[0]['content'].append({
                "type": "text",
                "text": query['text']
            })
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            if 'images' in query.keys():
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
            elif 'videos' in query.keys():
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    fps=kwargs.get("fps", 30),
                    padding=True,
                    return_tensors="pt"
                )
            else:
                inputs = self.processor(
                    text=[text],
                    padding=True,
                    return_tensors="pt",
                )
            inputs = inputs.to("cuda")
            # Inference
            kwargs = {k: v for k, v in kwargs.items() if k != "fps"}
                
            generated_ids = self.model.generate(**inputs, **kwargs)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            return output_text[0]
        elif 'llava-ov-chat' in self.model_name:
            from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
            from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
            from llava.conversation import conv_templates, SeparatorStyle
            from PIL import Image
            import copy
               
            device = 'cuda'
            
            images = [Image.open(os.path.abspath(os.path.join(data_path, image_path))).convert("RGB") for image_path in query['images']]
            image_tensor = process_images(images, self.processor, self.model.config)
            image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

            conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
            question = " ".join([DEFAULT_IMAGE_TOKEN for _ in query['images']]) + "\n" + query['text']
            conv = copy.deepcopy(conv_templates[conv_template])
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt_question = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
            image_sizes = [image.size for image in images]

            cont = self.model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                **kwargs,
            )
            text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
            return text_outputs[0]
        elif 'internvl3' in self.model_name:
            if 'images' in query.keys():
                from internvl_utils import load_image
                pixel_values_ = [load_image(os.path.abspath(os.path.join(data_path, image_path)), max_num=12).to(torch.bfloat16).cuda() for image_path in query['images']]  
                pixel_values = torch.cat(pixel_values_, dim=0)
                num_patches_list = [pixel_values.size(0) for pixel_values in pixel_values_]     
                question = "\n".join([f'Image-{idx+1}: <image>' for idx in range(len(query['images']))]) + "\n" + query['text']
                generation_config = dict(**kwargs, do_sample=True)
                response, history = self.model.chat(self.tokenizer, pixel_values, question, generation_config,
                            num_patches_list=num_patches_list,
                            history=None, return_history=True)
                return response
            elif 'videos' in query.keys():
                from internvl_utils import load_video
                video_path = os.path.join(data_path, query['videos'][0])
                pixel_values, num_patches_list = load_video(video_path, num_segments=8, max_num=1)
                pixel_values = pixel_values.to(torch.bfloat16).cuda()
                video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
                question = video_prefix + query['text']
                # Frame1: <image>\nFrame2: <image>\n...\nFrame8: <image>\n{question}
                generation_config = dict(**kwargs, do_sample=True)
                response, history = self.model.chat(self.tokenizer, pixel_values, question, generation_config,
                                            num_patches_list=num_patches_list, history=None, return_history=True)
                return response
            else:
                generation_config = dict(**kwargs, do_sample=True)
                response, history = self.model.chat(self.tokenizer, None, question, generation_config, history=history, return_history=True)
                return response
        elif 'gpt' in self.model_name:
            from utils import encode_image
            input = [{'role': 'user', 'content': [
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{encode_image(os.path.abspath(os.path.join(data_path, image_path)))}"
                } for image_path in query['images']
            ] + [
                {'type': 'input_text', 'text': query['text']}
            ]}]
            response = self.model.responses.create(
                model=self.model_name,
                input=input,
                max_output_tokens=kwargs.get("max_new_tokens", 512),
                temperature=kwargs.get("temperature", 0.1)
            )
            return response.output_text
        elif 'gemini' in self.model_name:
            from utils import encode_image
            import requests
            GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

            MODEL_ID = self.model_name
            if MODEL_ID == 'gemini-2.5-flash':
                MODEL_ID = 'gemini-2.5-flash-preview-04-17'
            if MODEL_ID == 'gemini-2.5-pro':
                MODEL_ID = 'gemini-2.5-pro-preview-03-25'

            url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_ID}:generateContent?key={GEMINI_API_KEY}"
            contents = [{"parts":[
                {
                    "inline_data": {
                        "mime_type":"image/jpeg",
                        "data": encode_image(os.path.abspath(os.path.join(data_path, image_path)))
                    }
                } for image_path in query['images']
            ] + [{
                    "text": query['text']
                }
            ]}]
            generationConfig = {
                "maxOutputTokens": kwargs.get("max_new_tokens", 512),
                "temperature": kwargs.get("temperature", 0.1),
                "topP": kwargs.get("top_p", 0.9),
            }
            response = requests.post(url, json={ "contents": contents, "generationConfig": generationConfig })
            if response.status_code == 200:
                return response.json()['candidates'][0]['content']['parts'][0]['text']
            else:
                print("Error:", response.status_code, response.text)
                return None
        else:
            raise ValueError(f"Unsupported model name: {self.model_name}")  

if __name__ == "__main__":
    import time
    model_name = 'qwen2.5-vl-3b'
    data_path = "./data/test"
    query1 = {
        "text": "Describe each image.",
        "images": ["test1.jpg", "test2.jpg"],
    }
    query2 = {
        "text": "Describe the video.",
        "videos": ["test1_360_640_24fps.mp4"],
    }
    # Load model
    model = Model(model_name)
    start = time.time()
    output1 = model.generate(query1, data_path, max_new_tokens=512, temperature=0.1)
    print("Time taken for image query:", time.time() - start)
    print("Output for image query:", output1)
    start = time.time()
    output2 = model.generate(query2, data_path, max_new_tokens=512, temperature=0.1)
    print("Time taken for video query:", time.time() - start)
    print("Output for video query:", output2)