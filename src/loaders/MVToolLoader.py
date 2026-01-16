import os
import json
import re
import sys
from typing import List, Dict, Any, Optional

# Adjust path to ensure we can import from src and scripts
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.loaders.base_loader import BaseDataLoader, StandardSample, PageElement

class MVToolLoader(BaseDataLoader):
    """
    MVToolBench 数据集加载器。
    该数据集主要用于单图 VQA 任务，包含带有 BBox 的 Ground Truth。
    """
    def __init__(self, data_root: str):
        """
        :param data_root: 包含 mvtoolbench_full.json 的根目录路径
        """
        super().__init__(data_root)
        # 尝试定位 json 文件，默认在 root 下，也可以兼容直接传入文件路径
        if data_root.endswith(".json") and os.path.isfile(data_root):
            self.json_path = data_root
            self.image_root = os.path.dirname(data_root) # 假设图片是相对路径或绝对路径
        else:
            self.json_path = os.path.join(data_root, "mvtoolbench_full.json")
            self.image_root = data_root

    def _parse_assistant_content(self, content: str) -> Dict[str, Any]:
        """
        解析 Assistant 回复中的 JSON 代码块。
        示例格式: ```json{"evidence": "...", "bbox": [...]}```
        """
        try:
            # 使用非贪婪匹配提取 ```json 和 ``` 之间的内容
            match = re.search(r'```json(.*?)```', content, re.DOTALL)
            if match:
                json_str = match.group(1).strip()
                return json.loads(json_str)
            else:
                # 如果没有代码块，尝试直接解析或返回原始内容作为 text
                return {"evidence": content, "bbox": []}
        except json.JSONDecodeError:
            print(f"Warning: Failed to decode JSON content: {content[:50]}...")
            return {"evidence": content, "bbox": []}

    def load_data(self) -> None:
        """加载 MVToolBench JSON 数据并转换为 StandardSample 格式。"""
        if not os.path.exists(self.json_path):
            raise FileNotFoundError(f"MVToolBench data file not found: {self.json_path}")
        
        print(f"Loading MVToolBench data from: {self.json_path}")
        with open(self.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        count = 0
        for item in data:
            # 1. 基础信息提取
            qid = item.get("id", str(count))
            image_list = item.get("images", [])
            # 这里的 image_path 通常是绝对路径或相对于 dataset 的路径
            # 如果是相对路径，可能需要结合 self.image_root 拼接，这里暂按原样读取
            main_image_path = image_list[0] if image_list else ""
            
            # 2. 解析对话 (Query 和 Answer)
            conversations = item.get("conversations", [])
            query_text = ""
            gold_answer = ""
            gold_bbox = []
            
            for turn in conversations:
                role = turn.get("role", "")
                content = turn.get("content", "")
                
                if role == "user":
                    # 去除 <image> 标签，获取纯文本 Query
                    query_text = content.replace("<image>\n", "").replace("<image>", "").strip()
                
                elif role == "assistant":
                    # 解析包含 evidence 和 bbox 的 JSON
                    parsed_response = self._parse_assistant_content(content)
                    gold_answer = parsed_response.get("evidence", "")
                    gold_bbox = parsed_response.get("bbox", [])

            # 3. 构建 PageElement (作为 Ground Truth 元素)
            gold_elements = []
            if gold_bbox:
                element = PageElement(
                    bbox=gold_bbox,
                    type="text", # MVTool 主要是文本识别/VQA
                    content=gold_answer,
                    corpus_id=main_image_path,
                    crop_path=None
                )
                gold_elements.append(element)

            # 4. 构建 StandardSample
            # 注意：MVTool 是 VQA 任务，data_source 指向具体图像，而不是索引
            sample = StandardSample(
                qid=qid,
                query=query_text,
                dataset="mvtool",
                data_source=main_image_path, 
                gold_answer=gold_answer,
                gold_elements=gold_elements,
                gold_pages=image_list
            )
            self.samples.append(sample)
            count += 1
            
        print(f"✅ Successfully loaded {count} MVTool samples.")

if __name__ == "__main__":
    # 测试代码
    # 假设当前目录下有 mvtoolbench_full.json
    root_dir = "/mnt/shared-storage-user/mineru3-share/jiayu/newBench/dataOri/MVToolBench/mvtoolbench_benchmark"
    
    loader = MVToolLoader(data_root=root_dir)
    try:
        loader.load_data()
        if len(loader.samples) > 0:
            s = loader.samples[0]
            print(f"\nSample 0 ID: {s.qid}")
            print(f"Query: {s.query}")
            print(f"Image: {s.data_source}")
            print(f"Gold Answer: {s.gold_answer}")
            if s.gold_elements:
                print(f"Gold BBox: {s.gold_elements[0].bbox}")
    except FileNotFoundError as e:
        print(e)