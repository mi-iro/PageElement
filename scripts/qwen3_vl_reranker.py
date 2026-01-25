import torch
import numpy as np
import logging
import base64
import os
from io import BytesIO

from PIL import Image
from scipy import special
from typing import List, Optional, Union
from qwen_vl_utils import process_vision_info
# 仅在本地模式下需要导入 transformers 模型
try:
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
except ImportError:
    pass # 允许在没有transformers的环境下运行remote模式

# 引入 OpenAI SDK 用于 vLLM 调用
from openai import OpenAI

logger = logging.getLogger(__name__)

MAX_LENGTH = 8192
IMAGE_BASE_FACTOR = 16
IMAGE_FACTOR = IMAGE_BASE_FACTOR * 2
MIN_PIXELS = 4 * IMAGE_FACTOR * IMAGE_FACTOR  # 4 tokens
MAX_PIXELS = 1280 * IMAGE_FACTOR * IMAGE_FACTOR  # 1280 tokens
MAX_RATIO = 200

FRAME_FACTOR = 2
FPS = 1
MIN_FRAMES = 2
MAX_FRAMES = 64
MIN_TOTAL_PIXELS = 1 * FRAME_FACTOR * MIN_PIXELS  # 1 frames
MAX_TOTAL_PIXELS = 4 * FRAME_FACTOR * MAX_PIXELS  # 4 frames


def sample_frames(frames, num_segments, max_segments):
    duration = len(frames)
    frame_id_array = np.linspace(0, duration - 1, num_segments, dtype=int)
    frame_id_list = frame_id_array.tolist()
    last_frame_id = frame_id_list[-1]

    sampled_frames = []
    for frame_idx in frame_id_list:
        try:
            single_frame_path = frames[frame_idx]
        except:
            break
        sampled_frames.append(single_frame_path)
    # Pad with last frame if total frames less than num_segments
    while len(sampled_frames) < num_segments:
        sampled_frames.append(frames[last_frame_id])
    return sampled_frames[:max_segments]

# 辅助函数：将图片转为base64，用于API传输
def encode_image_to_base64(image_input):
    if isinstance(image_input, str):
        if image_input.startswith("http"):
            return image_input # URL直接返回
        with open(image_input.split('file://')[-1], "rb") as image_file:
            return f"data:image/jpeg;base64,{base64.b64encode(image_file.read()).decode('utf-8')}"
    elif isinstance(image_input, Image.Image):
        buffered = BytesIO()
        image_input.save(buffered, format="JPEG")
        return f"data:image/jpeg;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"
    return image_input

class Qwen3VLReranker():
    def __init__(
        self,
        model_name_or_path: str,
        vllm_api_base: Optional[str] = None, # 新增：vLLM服务地址
        vllm_api_key: str = "EMPTY",         # 新增：vLLM API Key
        max_length: int = MAX_LENGTH,
        min_pixels: int = MIN_PIXELS,
        max_pixels: int = MAX_PIXELS,
        total_pixels: int = MAX_TOTAL_PIXELS,
        fps: float = FPS,
        num_frames: int = MAX_FRAMES,
        max_frames: int = MAX_FRAMES,
        default_instruction: str = "Given a search query, retrieve relevant candidates that answer the query.",
        **kwargs,
    ):
        self.max_length = max_length
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels
        self.fps = fps
        self.num_frames = num_frames
        self.max_frames = max_frames
        self.default_instruction = default_instruction

        # 判断运行模式
        if vllm_api_base:
            self.mode = "remote"
            self.client = OpenAI(
                api_key=vllm_api_key,
                base_url=vllm_api_base,
            )
            self.model_name = model_name_or_path # 远程模式下，这只是作为一个请求参数
            logger.info(f"Initialized Qwen3VLReranker in REMOTE mode connecting to {vllm_api_base}")
            
            # 即使在远程模式，我们也需要知道 'yes' 和 'no' 对应的 token 字符串
            # 这里硬编码 Qwen Tokenizer 的常见行为，或者你可以选择加载一个 tokenizer
            self.token_true_str = "yes" 
            self.token_false_str = "no"
            
        else:
            self.mode = "local"
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            lm = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name_or_path,
                trust_remote_code=True, **kwargs
            ).to(self.device)

            self.model = lm.model
            self.processor = AutoProcessor.from_pretrained(
                model_name_or_path, trust_remote_code=True,
                padding_side='left'
            )
            self.model.eval()

            token_true_id = self.processor.tokenizer.get_vocab()["yes"]
            token_false_id = self.processor.tokenizer.get_vocab()["no"]
            self.score_linear = self.get_binary_linear(lm, token_true_id, token_false_id)
            self.score_linear.eval()
            self.score_linear.to(self.device).to(self.model.dtype)
            logger.info(f"Initialized Qwen3VLReranker in LOCAL mode with model {model_name_or_path}")

    def get_binary_linear(self, model, token_yes, token_no):
        lm_head_weights = model.lm_head.weight.data
        weight_yes = lm_head_weights[token_yes]
        weight_no = lm_head_weights[token_no]
        D = weight_yes.size()[0]
        linear_layer = torch.nn.Linear(D, 1, bias=False)
        with torch.no_grad():
            linear_layer.weight[0] = weight_yes - weight_no
        return linear_layer

    @torch.no_grad()
    def compute_scores_local(self, inputs):
        batch_scores = self.model(**inputs).last_hidden_state[:, -1]
        scores = self.score_linear(batch_scores)
        scores = torch.sigmoid(scores).squeeze(-1).cpu().detach().tolist()
        return scores

    def compute_scores_remote(self, messages_list):
        """
        使用 vLLM API 计算分数。
        原理：让模型生成 1 个 token，并请求 logprobs。
        Score = Sigmoid(LogProb(yes) - LogProb(no))
        """
        scores = []
        for messages in messages_list:
            # 转换消息格式以适应 OpenAI API
            openai_messages = self._convert_to_openai_format(messages)
            
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=openai_messages,
                    max_tokens=1,
                    logprobs=True,
                    top_logprobs=20 # 获取前20个可能的token，以确保包含yes/no
                )
                
                # 获取第一个生成token的logprobs
                top_logprobs = response.choices[0].logprobs.content[0].top_logprobs
                
                prob_yes = -9999.0
                prob_no = -9999.0
                
                # 查找 yes 和 no 的 logprob
                # 注意：Tokenizer 可能会在单词前加空格，这里做简单处理
                for token_obj in top_logprobs:
                    token_str = token_obj.token.lower().strip()
                    if token_str == self.token_true_str:
                        prob_yes = token_obj.logprob
                    elif token_str == self.token_false_str:
                        prob_no = token_obj.logprob
                
                # 如果没有出现在 top_logprobs 中，保持极小值
                
                # 计算得分: Sigmoid(yes - no)
                # Logit difference is roughly equivalent to LogProb difference in this context
                diff = prob_yes - prob_no
                score = special.expit(diff)
                scores.append(score.item())
                
            except Exception as e:
                logger.error(f"Error calling vLLM API: {e}")
                scores.append(0.0) # 失败回退
                
        return scores

    def _convert_to_openai_format(self, internal_messages):
        """
        将内部格式 (type: 'image', 'video' 等) 转换为 OpenAI API 格式
        """
        openai_msgs = []
        for msg in internal_messages:
            role = msg['role']
            content_list = []
            
            # System message usually just text
            if role == "system":
                # OpenAI API 对 system message 只支持 text (通常)
                text_content = ""
                for item in msg['content']:
                    if item['type'] == 'text':
                        text_content += item['text']
                openai_msgs.append({"role": "system", "content": text_content})
                continue

            # User message
            for item in msg['content']:
                if item['type'] == 'text':
                    content_list.append({"type": "text", "text": item['text']})
                
                elif item['type'] == 'image':
                    # 处理图片
                    img_data = item['image']
                    url = encode_image_to_base64(img_data)
                    content_list.append({
                        "type": "image_url", 
                        "image_url": {"url": url}
                    })
                
                elif item['type'] == 'video':
                    # 处理视频
                    # Qwen-VL 在 vLLM 中通常将视频处理为一系列图片帧
                    # 或者如果有原生的video支持，根据vLLM版本调整
                    # 这里我们将视频帧展开为 image_url 列表
                    video_files = item['video']
                    for frame in video_files:
                         # 移除 'file://' 前缀如果存在
                        clean_path = frame.replace('file://', '') if isinstance(frame, str) else frame
                        url = encode_image_to_base64(clean_path)
                        content_list.append({
                            "type": "image_url",
                            "image_url": {"url": url}
                        })
            
            openai_msgs.append({"role": role, "content": content_list})
        return openai_msgs

    # ... [保留原有的 truncate_tokens_optimized, tokenize, format_mm_content, format_mm_instruction 方法] ...
    # 为了节省篇幅，这里假设保留原有的这些辅助方法不变。
    # 实际上，我们需要保留它们，因为 process 方法仍然依赖它们来构建 prompt 结构。
    
    def truncate_tokens_optimized(self, tokens, max_length, special_tokens):
        # ... (保持原代码不变) ...
        if len(tokens) <= max_length:
            return tokens
        special_tokens_set = set(special_tokens)
        num_special = sum(1 for token in tokens if token in special_tokens_set)
        num_non_special_to_keep = max_length - num_special
        final_tokens = []
        non_special_kept_count = 0
        for token in tokens:
            if token in special_tokens_set:
                final_tokens.append(token)
            elif non_special_kept_count < num_non_special_to_keep:
                final_tokens.append(token)
                non_special_kept_count += 1
        return final_tokens

    def tokenize(self, pairs: list, **kwargs):
        # ... (保持原代码不变) ...
        max_length = self.max_length
        text = self.processor.apply_chat_template(pairs, tokenize=False, add_generation_prompt=True)
        try:
            images, videos, video_kwargs = process_vision_info(
                pairs, image_patch_size=16, 
                return_video_kwargs=True, 
                return_video_metadata=True
            )
        except Exception as e:
            logger.error(f"Error in processing vision info: {e}")
            images = None
            videos = None
            video_kwargs = {'do_sample_frames': False}
            text = self.processor.apply_chat_template(
                [{'role': 'user', 'content': [{'type': 'text', 'text': 'NULL'}]}], 
                add_generation_prompt=True, tokenize=False
            )
        
        if videos is not None:
            videos, video_metadatas = zip(*videos)
            videos, video_metadatas = list(videos), list(video_metadatas)
        else:
            video_metadatas = None
        inputs = self.processor(
            text=text,
            images=images,
            videos=videos,
            video_metadata=video_metadatas,
            truncation=False,
            padding=False,
            do_resize=False,
            **video_kwargs
        )
        for i, ele in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = self.truncate_tokens_optimized(
                inputs['input_ids'][i][:-5], max_length,
                self.processor.tokenizer.all_special_ids
            ) + inputs['input_ids'][i][-5:]
        temp_inputs = self.processor.tokenizer.pad(
            {'input_ids': inputs['input_ids']}, padding=True,
            return_tensors="pt", max_length=self.max_length
        )
        for key in temp_inputs:
            inputs[key] = temp_inputs[key]
        return inputs

    def format_mm_content(self, text, image, video, prefix='Query:', fps=None, max_frames=None):
        # ... (保持原代码不变) ...
        content = []
        content.append({'type': 'text', 'text': prefix})
        if not text and not image and not video:
            content.append({'type': 'text', 'text': "NULL"})
            return content

        if video:
            video_content = None
            video_kwargs = { 'total_pixels': self.total_pixels }
            if isinstance(video, list):
                video_content = video
                if self.num_frames is not None or self.max_frames is not None:
                    # 注意：这里调用了未定义的 self._sample_frames，原代码似乎漏掉了这个方法定义
                    # 假设使用外部定义的 sample_frames
                    video_content = sample_frames(video_content, self.num_frames, self.max_frames)
                video_content = [
                    ('file://' + ele if isinstance(ele, str) else ele) 
                    for ele in video_content
                ]
            elif isinstance(video, str):
                video_content = video if video.startswith(('http://', 'https://')) else 'file://' + video
                video_kwargs = {'fps': fps or self.fps, 'max_frames': max_frames or self.max_frames,}
            else:
                raise TypeError(f"Unrecognized video type: {type(video)}")

            if video_content:
                content.append({
                    'type': 'video', 'video': video_content,
                    **video_kwargs
                })

        if image:
            image_content = None
            if isinstance(image, Image.Image):
                image_content = image
            elif isinstance(image, str):
                image_content = image if image.startswith(('http', 'oss')) else 'file://' + image
            else:
                raise TypeError(f"Unrecognized image type: {type(image)}")

            if image_content:
                content.append({
                    'type': 'image', 'image': image_content,
                    "min_pixels": self.min_pixels,
                    "max_pixels": self.max_pixels
                })

        if text:
            content.append({'type': 'text', 'text': text})
        return content

    def format_mm_instruction(self, query_text, query_image, query_video, doc_text, doc_image, doc_video, instruction=None, fps=None, max_frames=None):
        # ... (保持原代码不变) ...
        inputs = []
        inputs.append({
            "role": "system",
            "content": [{
                "type": "text",
                "text": "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
            }]
        })
        if isinstance(query_text, tuple):
            instruct, query_text = query_text
        else:
            instruct = instruction
        contents = []
        contents.append({
            "type": "text",
            "text": '<Instruct>: ' + instruct
        })
        query_content = self.format_mm_content(
            query_text, query_image, query_video, prefix='<Query>:', 
            fps=fps, max_frames=max_frames
        )
        contents.extend(query_content)
        doc_content = self.format_mm_content(
            doc_text, doc_image, doc_video, prefix='\n<Document>:', 
            fps=fps, max_frames=max_frames
        )
        contents.extend(doc_content)
        inputs.append({
            "role": "user",
            "content": contents
        })
        return inputs

    def process(self, inputs) -> List[float]:
        instruction = inputs.get('instruction', self.default_instruction)

        query = inputs.get("query", {})
        documents = inputs.get("documents", [])
        if not query or not documents:
            return []

        # 1. 格式化所有 Query-Document 对
        pairs = [self.format_mm_instruction(
            query.get('text', None),
            query.get('image', None),
            query.get('video', None),
            document.get('text', None),
            document.get('image', None),
            document.get('video', None),
            instruction=instruction,
            fps=inputs.get('fps', self.fps),
            max_frames=inputs.get('max_frames', self.max_frames)
        ) for document in documents]

        # 2. 根据模式分发
        if self.mode == "local":
            final_scores = []
            # 这是一个简单的Batch处理循环，实际使用可能需要分Batch
            for pair in pairs:
                tokenized_inputs = self.tokenize([pair])
                tokenized_inputs = tokenized_inputs.to(self.model.device)
                scores = self.compute_scores_local(tokenized_inputs)
                final_scores.extend(scores)
            return final_scores
        
        elif self.mode == "remote":
            # 远程模式直接传递 pairs 列表
            return self.compute_scores_remote(pairs)
        
if __name__ == '__main__':
    model_path = "/mnt/shared-storage-user/mineru3-share/wangzhengren/JIT-RAG/assets/Qwen/Qwen3-VL-Reranker-8B"
    # 1. 本地模式 (原有方式)
    reranker_local = Qwen3VLReranker(model_name_or_path=model_path)

    # 2. vLLM 服务模式
    reranker_remote = Qwen3VLReranker(
        model_name_or_path="Qwen3-VL-Reranker-8B", # 这里主要是作为API请求中的 model 参数名
        vllm_api_base="http://localhost:8002/v1",
        vllm_api_key="EMPTY"
    )

    # 使用方式一致
    inputs = {
        "query": {"image": "/mnt/shared-storage-user/mineru3-share/wangzhengren/JIT-RAG/assets/images/singapore.jpg"},
        "documents": [{"image": "/mnt/shared-storage-user/mineru3-share/wangzhengren/JIT-RAG/assets/images/beijing.jpg"}, {"text": "singapore with plants"}]
    }
    print(reranker_local.process(inputs), reranker_remote.process(inputs))