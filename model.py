from transformers import AutoTokenizer, AutoProcessor, AutoModel, AutoModelForCausalLM
import os
import torch
import sys
from utils import generate_kwargs, extract_question, encode_file, extract_frames
import av
import numpy as np

class Model:
    def __init__(self, model_name, model_path=None):
        self.model_name = model_name
        self.model_path = model_path
        self.model, self.tokenizer, self.processor = self.load_model()
    
    def load_model(self):
        if 'qwen2.5-vl' in self.model_name:
            from transformers import Qwen2_5_VLForConditionalGeneration
            if 'sft' in self.model_name:
                print("Loading SFT model...")
                MODEL_ID = self.model_path
                processor = AutoProcessor.from_pretrained("qwen/" + self.model_name.split('-sft')[0] + "-instruct")
            else:
                MODEL_ID = f"qwen/{self.model_name}-instruct"
                processor = AutoProcessor.from_pretrained(MODEL_ID)
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                MODEL_ID, 
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto")
            model.eval()
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
        
        #video specialized
        elif 'video-llama3' in self.model_name:
            MODEL_ID = "DAMO-NLP-SG/VideoLLaMA3-7B"
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                attn_implementation="flash_attention_2",
            )
            processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
            model.eval()
            return model, None, processor
        elif 'llava-next-video' in self.model_name:
            from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration
            MODEL_ID = "llava-hf/LLaVA-NeXT-Video-7B-hf"
            model = LlavaNextVideoForConditionalGeneration.from_pretrained(
                MODEL_ID, 
                torch_dtype=torch.bfloat16, 
                low_cpu_mem_usage=True,
                use_flash_attention_2=True
            ).to("cuda")
            processor = LlavaNextVideoProcessor.from_pretrained(MODEL_ID)
            model.eval()
            return model, None, processor
        elif 'internvideo2' in self.model_name:
            MODEL_ID = 'OpenGVLab/InternVideo2_5_Chat_8B'
            model = AutoModel.from_pretrained(
                MODEL_ID,
                low_cpu_mem_usage=True,
                use_flash_attn=False,
                trust_remote_code=True).half().cuda().to(torch.bfloat16)
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
        q = extract_question(query)
        # if kwargs.get("refer_context", False) and 'context' in query.keys():
        #     # context가 부정확 할 수 도 있으므로, 최종적으로는 video를 직접 보고 답변
        #     q += "\nThese contexts may not be accurate, please refer to the video directly.\n"
        q += "\nAnswer format should be 'Final answer: {idx}. {answer}"
        
        if 'image' in query.keys() and isinstance(query['image'], str):
            query['image'] = [query['image']]
        if 'video' in query.keys() and isinstance(query['video'], str):
            query['video'] = [query['video']]
            
        if kwargs.get("gen_context", False):
            q = "Describe the video."

        if 'qwen2.5-vl' in self.model_name:
            from qwen_vl_utils import process_vision_info
            messages = []
            if 'image' in query.keys():
                messages = [{
                    "role": "user",
                    "content": [{
                        "type": "image",
                        "image": f"file://{os.path.abspath(os.path.join(data_path, image_path))}"
                    } for image_path in query['image']]
                }]
            elif 'video' in query.keys():
                messages = [{
                    "role": "user",
                    "content": [{
                        "type": "video",
                        "video": f"file://{os.path.abspath(os.path.join(data_path, video_path))}"
                    } for video_path in query['video']]
                }]
            else:
                messages = [{
                    "role": "user",
                    "content": []
                }]
            if kwargs.get("refer_context", False) and 'context' in query.keys():
                messages[0]['content'].append({
                    "type": "text",
                    "text": query['context']
                })
            messages[0]['content'].append({
                "type": "text",
                "text": q
            })
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            if 'image' in query.keys():
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
            elif 'video' in query.keys():
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
            generated_ids = self.model.generate(**inputs, **generate_kwargs(**kwargs))
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
            
            images = [Image.open(os.path.abspath(os.path.join(data_path, image_path))).convert("RGB") for image_path in query['image']]
            image_tensor = process_images(images, self.processor, self.model.config)
            image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

            conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
            if kwargs.get("refer_context", False) and 'context' in query.keys():
                question = " ".join([DEFAULT_IMAGE_TOKEN for _ in query['image']]) + "\n" + query['context'] + "\n" + q
            else:
                question = " ".join([DEFAULT_IMAGE_TOKEN for _ in query['image']]) + "\n" + q
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
                **generate_kwargs(**kwargs),
            )
            text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
            return text_outputs[0]
    
        elif 'internvl3' in self.model_name:
            if 'image' in query.keys():
                from internvl_utils import load_image
                pixel_values_ = [load_image(os.path.abspath(os.path.join(data_path, image_path)), max_num=12).to(torch.bfloat16).cuda() for image_path in query['images']]  
                pixel_values = torch.cat(pixel_values_, dim=0)
                num_patches_list = [pixel_values.size(0) for pixel_values in pixel_values_]
                if kwargs.get("refer_context", False) and 'context' in query.keys():
                    q = query['context'] + "\n" + q     
                question = "\n".join([f'Image-{idx+1}: <image>' for idx in range(len(query['image']))]) + "\n" + q
                generation_config = dict(**generate_kwargs(**kwargs), do_sample=True)
                response, history = self.model.chat(self.tokenizer, pixel_values, question, generation_config,
                            num_patches_list=num_patches_list,
                            history=None, return_history=True)
                return response
            elif 'video' in query.keys():
                from internvl_utils import load_video
                video_path = os.path.join(data_path, query['video'][0])
                num_segments = kwargs.get("max_frames", 8)
                pixel_values, num_patches_list = load_video(video_path, num_segments=num_segments, max_num=1)
                pixel_values = pixel_values.to(torch.bfloat16).cuda()
                video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
                if kwargs.get("refer_context", False) and 'context' in query.keys():
                    q = query['context'] + "\n" + q
                question = video_prefix + q
                # Frame1: <image>\nFrame2: <image>\n...\nFrame8: <image>\n{question}
                generation_config = dict(**generate_kwargs(**kwargs), do_sample=True)
                response, history = self.model.chat(self.tokenizer, pixel_values, question, generation_config,
                                            num_patches_list=num_patches_list, history=None, return_history=True)
                return response
            else:
                generation_config = dict(**generate_kwargs(**kwargs), do_sample=True)
                response, history = self.model.chat(self.tokenizer, None, question, generation_config, history=history, return_history=True)
                return response

        #video specialized
        #cannot input video due to ffmpeg error (llama3)
        elif 'video-llama3' in self.model_name:
            from PIL import Image
            if 'image' in query.keys():
                visuals = [
                    Image.open(os.path.join(data_path, img_p)).convert("RGB")
                    for img_p in query['image']
                ]
                content = []
                for img in visuals:
                    content.append({"type":"image", "image": img})
                content.append({
                    "type":"text",
                    "text": q
                })
                conversation = [
                    {"role":"user", "content": content}
                ]
            elif 'video' in query.keys():
                conversation = [{
                    "role": "user",
                    "content": [{
                        "type": "video",
                        "video": {
                            "video_path": os.path.abspath(os.path.join(data_path, vp)),
                            "fps":        kwargs.get("fps", 30),
                            "max_frames": kwargs.get("max_frames", 8),
                        }
                    } for vp in query['video']] + [
                        {"type": "text", "text": q}
                    ]
                }]
            inputs = self.processor(
                conversation=conversation,
                return_tensors="pt",
            )
            inputs = {k: (v.cuda() if isinstance(v, torch.Tensor) else v)
                    for k, v in inputs.items()}
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
            output_ids = self.model.generate(**inputs, **generate_kwargs(**kwargs))
            response = self.processor.batch_decode(
                output_ids, skip_special_tokens=True
            )[0].strip()
            return response
        
        elif 'llava-next-video' in self.model_name:
            from utils import extract_assistant_response
            if 'image' in query.keys():
                from PIL import Image
                conversation = [{
                    "role": "user",
                    "content": [{
                        "type": "image",
                    } for i in range(len(query['image']))]
                }]
            elif 'video' in query.keys():
                from utils import read_video_pyav
                conversation = [{
                    "role": "user",
                    "content": [{
                        "type": "video",
                    } for i in range(len(query['video']))]
                }]
            if 'context' in query:
                conversation[0]['content'].append({
                    "type": "text",
                    "text": query['context']
                })
            conversation[0]['content'].append({
                "type": "text", 
                "text": q
            })
            prompt = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
            )
            if 'image' in query.keys():
                path = []
                for image_path in query['image']:
                    path.append(os.path.abspath(os.path.join(data_path, image_path)))
                raw_images = [Image.open(p) for p in path]
                inputs = self.processor(
                    text=prompt,
                    images=raw_images,
                    return_tensors="pt",
                ).to("cuda", torch.bfloat16)
                output = self.model.generate(**inputs, do_sample=True, **generate_kwargs(**kwargs))
                response = self.processor.decode(output[0][2:], skip_special_tokens=True)
                return extract_assistant_response(response)
            elif 'video' in query.keys():
                clips = []
                for video_path in query['video']:
                    container = av.open(f"file://{os.path.abspath(os.path.join(data_path, video_path))}")
                    total_frames = container.streams.video[0].frames
                    num_frames = kwargs.get("max_frames", 8)
                    indices = np.linspace(
                        0, total_frames - 1, num=num_frames, dtype=int)
                    clip = read_video_pyav(container, indices)  # shape: [T, C, H, W]
                    clips.append(clip)
                inputs_video = self.processor(text=prompt, videos=clips, padding=True, return_tensors="pt").to(self.model.device)
                output = self.model.generate(**inputs_video, do_sample=True, **generate_kwargs(**kwargs))
                response = self.processor.decode(output[0][2:], skip_special_tokens=True)
                return extract_assistant_response(response)
        #ONLY VIDEO
        elif 'internvideo2' in self.model_name:
            if 'video' not in query.keys():
                raise ValueError("No video input found.")
            from internvl_utils import load_video
            num_segments = kwargs.get("max_frames", 8)
            generation_config = dict(**generate_kwargs(**kwargs), do_sample=True)
            video_path = os.path.join(data_path, query['video'][0])
            with torch.no_grad():
                pixel_values, num_patches_list = load_video(video_path, num_segments=num_segments, max_num=1)
                pixel_values = pixel_values.to(torch.bfloat16).to(self.model.device)
                video_prefix = "".join([f"Frame{i+1}: <image>\n" for i in range(len(num_patches_list))])
                if kwargs.get("refer_context", False) and 'context' in query.keys():
                    q = query['context'] + "\n" + q
                question = video_prefix + q
                output1 = self.model.chat(self.tokenizer, pixel_values, question, generation_config, num_patches_list=num_patches_list, history=None, return_history=False)
                return output1


        elif 'gpt' in self.model_name:
            from utils import encode_image, encode_file, extract_frames
            input = [ {"role": "system", "content": "You are analyzing video frames sampled at regular intervals. The user will upload these frames in chronological order, but the frame numbers may not be consecutive."},
                {'role': 'user', 'content': []}]
            for image_path in query.get('image', []):
                input[1]['content'].append({
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{encode_file(os.path.join(data_path, image_path))}"
                })
            max_frames = kwargs.get("max_frames", 8)
            for video_path in query.get('video', []):
                abs_path = os.path.join(data_path, video_path)
                for frame_b64 in extract_frames(abs_path, max_frames=max_frames):
                    input[1]['content'].append({
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{frame_b64}"
                    })
            if kwargs.get("refer_context", False) and 'context' in query:
                input[1]['content'].append({
                    "type": "input_text",
                    "text": query['context']
                })
            input[1]['content'].append({
                "type": "input_text",
                "text": q
            })

            response = self.model.responses.create(
                model=self.model_name,
                input=input,
                max_output_tokens=kwargs.get("max_new_tokens", 512),
                temperature=kwargs.get("temperature", 0.1)
            )
            return response.output_text
        elif 'gemini' in self.model_name:
            from utils import encode_image, extract_frames
            import requests
            GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

            MODEL_ID = self.model_name
            if MODEL_ID == 'gemini-2.5-flash':
                MODEL_ID = 'gemini-2.5-flash-preview-04-17'
            if MODEL_ID == 'gemini-2.5-pro':
                MODEL_ID = 'gemini-2.5-pro-preview-03-25'
            if MODEL_ID == 'gemini-2.0-pro':
                MODEL_ID = 'gemini-2.0-pro-exp-02-05'

            url = (
                f"https://generativelanguage.googleapis.com/v1beta/"
                f"models/{MODEL_ID}:generateContent?key={GEMINI_API_KEY}"
            )
            
            if kwargs.get("refer_context", False) and 'context' in query:
                q = query['context'] + "\n" + q

            parts = [
                {
                    "text": (
                        "You are analyzing video frames sampled at regular intervals. "
                        "Below I will send you frames in chronological order; "
                        "please note that their frame numbers are not consecutive."
                    )
                },
                {
                    "text": q
                }
            ]

            for image_path in query.get('image', []):
                img_b64 = encode_image(os.path.abspath(os.path.join(data_path, image_path)))
                parts.append({
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": img_b64
                    }
                })

            max_frames = kwargs.get("max_frames", 8)
            for video_path in query.get('video', []):
                abs_path = os.path.join(data_path, video_path)
                for frame_b64 in extract_frames(abs_path, max_frames=max_frames):
                    parts.append({
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": frame_b64
                        }
                    })

            contents = [{
                "role": "user",
                "parts": parts
            }]

            generationConfig = {
                "maxOutputTokens": kwargs.get("max_new_tokens", 512),
                "temperature":    kwargs.get("temperature", 0.1),
                "topP":           kwargs.get("top_p", 0.9),
            }

            response = requests.post(
                url,
                json={ "contents": contents, "generationConfig": generationConfig }
            )
            
            # print(response.json())

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
        "image": ["test1.jpg", "test2.jpg"],
    }
    query2 = {
        "text": "Describe the video.",
        "video": ["test1_360_640_24fps.mp4"],
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
