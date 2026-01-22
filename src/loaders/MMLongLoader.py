import os
import json
import re
import sys
import ast
import asyncio
import uuid
import torch
from PIL import Image
from typing import List, Dict, Any, Optional
import collections

# Adjust path to ensure we can import from src and scripts
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.loaders.base_loader import BaseDataLoader, StandardSample, PageElement
from src.agents.ElementExtractor import ElementExtractor
from scripts.qwen3_vl_reranker import Qwen3VLReranker

class MMLongLoader(BaseDataLoader):
    """
    MMLongBench-Doc 数据集加载器。
    用于加载 MMLongBench-Doc 中的 DocVQA 任务数据。
    """
    
    # --- Singleton Storage for Reranker ---
    _reranker_instance = None

    def __init__(self, data_root: str, extractor: Optional[ElementExtractor] = None, reranker_model_path: str = None):
        """
        :param data_root: MMLongBench-Doc 的根目录
        :param extractor: ElementExtractor 实例
        :param reranker_model_path: Reranker 模型权重的本地路径 (用于懒加载)
        """
        super().__init__(data_root)
        self.extractor = extractor
        self.reranker_model_path = reranker_model_path
        
        # 根据提供的目录结构设置路径
        self.json_path = os.path.join(data_root, "data", "samples.json")
        self.doc_dir = os.path.join(data_root, "data", "documents")

    @classmethod
    def get_reranker(cls, model_path: str):
        """
        单例模式获取 Reranker 实例。
        只有在第一次调用时才会加载模型到显存。
        """
        if cls._reranker_instance is None:
            if not model_path or not os.path.exists(model_path):
                print(f"Warning: Reranker model path is invalid: {model_path}")
                return None
                
            print(f"⚡ Initializing Reranker Singleton from {model_path} ...")
            try:
                cls._reranker_instance = Qwen3VLReranker(
                    model_name_or_path=model_path,
                    torch_dtype=torch.float16 
                )
                print("✅ Reranker loaded successfully.")
            except Exception as e:
                print(f"❌ Failed to load Reranker: {e}")
                return None
                
        return cls._reranker_instance

    def _parse_assistant_content(self, content: str) -> Dict[str, Any]:
        """
        解析 Assistant 回复中的 JSON 代码块。
        """
        try:
            match = re.search(r'```json(.*?)```', content, re.DOTALL)
            if match:
                json_str = match.group(1).strip()
                return json.loads(json_str)
            else:
                return {"evidence": content, "bbox": []}
        except json.JSONDecodeError:
            print(f"Warning: Failed to decode JSON content: {content[:50]}...")
            return {"evidence": content, "bbox": []}

    def evaluate(self, beta: float = 1.0) -> Dict[str, float]:
        """
        执行评估：计算 QA 指标、页面检索指标和元素提取指标。
        整合了 eval.py 中的 Page Accuracy, IoU Min, IoU EM 等高级指标。
        
        评估结果将存储在每个 sample.extra_info['metrics'] 中。
        依赖 sample.extra_info 包含 'final_answer' 和 'retrieved_elements'。
        
        Returns:
            Dict[str, float]: 整个数据集的平均指标
        """
        total_metrics = collections.defaultdict(float)
        counts = collections.defaultdict(int)

        for sample in self.samples:
            if sample.extra_info is None:
                sample.extra_info = {}
                continue
            
            metrics_result = {}
            
            # 1. 计算 QA 指标 (Text Generation)
            pred_answer = sample.extra_info.get('final_answer', "") # 获取预测结果
            if sample.gold_answer and pred_answer:
                qa_score = self._compute_qa_metrics(pred_answer, sample.gold_answer)
                metrics_result['qa'] = qa_score
                total_metrics['qa_f1'] += qa_score['f1']
                total_metrics['qa_em'] += qa_score['em']
                counts['qa'] += 1
            
            # 尝试获取预测的 elements (兼容 dict 列表或 PageElement 对象列表)
            raw_elements = sample.extra_info.get('retrieved_elements', [])
            pred_elements = []
            for el in raw_elements:
                if isinstance(el, dict):
                    # 过滤掉非 PageElement 字段以防止 TypeError
                    valid_keys = PageElement.__annotations__.keys()
                    filtered_el = {k: v for k, v in el.items() if k in valid_keys}
                    pred_elements.append(PageElement(**filtered_el))
                elif isinstance(el, PageElement):
                    pred_elements.append(el)

            # 2. 计算 页面检索 指标 (Page Retrieval)
            if sample.gold_pages:
                page_score = self._compute_page_metrics(pred_elements, sample.gold_pages)
                metrics_result['page'] = page_score
                total_metrics['page_recall'] += page_score['recall']
                total_metrics['page_precision'] += page_score['precision']
                counts['page'] += 1

            # 存储回 sample
            sample.extra_info['metrics'] = metrics_result

        # --- 汇总平均值 ---
        avg_results = {}
        
        # QA
        if counts['qa'] > 0:
            avg_results['avg_qa_f1'] = total_metrics['qa_f1'] / counts['qa']
            avg_results['avg_qa_em'] = total_metrics['qa_em'] / counts['qa']
        
        # Page Retrieval
        if counts['page'] > 0:
            avg_results['avg_page_recall'] = total_metrics['page_recall'] / counts['page']
            avg_results['avg_page_precision'] = total_metrics['page_precision'] / counts['page']
            
        return avg_results

    def load_data(self) -> None:
        """根据新的 samples.json 格式加载数据。"""
        if not os.path.exists(self.json_path):
            raise FileNotFoundError(f"MMLongBench data file not found: {self.json_path}")
        
        print(f"Loading MMLongBench data from: {self.json_path}")
        with open(self.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        count = 0
        for item in data:
            qid = str(count)
            doc_filename = item.get("doc_id", "")
            main_doc_path = os.path.join(self.doc_dir, doc_filename) if doc_filename else ""
            
            query_text = item.get("question", "")
            gold_answer = item.get("answer", "")
            
            evidence_pages_str = item.get("evidence_pages", "[]")
            gold_pages = []
            try:
                pages_list = ast.literal_eval(evidence_pages_str)
                if isinstance(pages_list, list):
                    gold_pages = [f"page_{str(p)}.png" for p in pages_list]
            except Exception as e:
                gold_pages = []

            extra_info = {
                "doc_type": item.get("doc_type"),
                "evidence_sources": item.get("evidence_sources"),
                "answer_format": item.get("answer_format")
            }

            sample = StandardSample(
                qid=qid,
                query=query_text,
                dataset="mmlongbench-doc",
                data_source=main_doc_path, 
                gold_answer=gold_answer,
                gold_elements=[],
                gold_pages=gold_pages,
                extra_info=extra_info
            )
            self.samples.append(sample)
            count += 1
            
        print(f"✅ Successfully loaded {count} MMLongBench samples.")

    def _pdf_to_images(self, pdf_path: str) -> Dict[int, str]:
        """将 PDF 转换为图片序列，并保存到缓存目录。"""
        if not os.path.exists(pdf_path):
             print(f"Warning: PDF not found at {pdf_path}")
             return {}

        pdf_name = os.path.basename(pdf_path).rsplit('.', 1)[0]
        workspace_dir = os.path.abspath(os.path.join(os.getcwd(), "workspace", "pdf_cache"))
        cache_dir = os.path.join(workspace_dir, pdf_name)
        os.makedirs(cache_dir, exist_ok=True)
        
        image_map = {}
        
        # 1. Check cache
        existing_files = [f for f in os.listdir(cache_dir) if f.endswith('.png')]
        if existing_files:
            temp_map = {}
            for f in existing_files:
                match = re.match(r"page_(\d+)\.png", f)
                if match:
                    idx = int(match.group(1))
                    temp_map[idx] = os.path.join(cache_dir, f)
            if temp_map:
                print(f"Using cached images for {pdf_name} ({len(temp_map)} pages)")
                return temp_map

        # 2. Convert if no cache
        print(f"Converting PDF to images: {pdf_path}")
        try:
            from pdf2image import convert_from_path
            images = convert_from_path(pdf_path, dpi=200)
            for i, img in enumerate(images):
                page_num = i + 1 
                save_name = f"page_{page_num}.png"
                save_path = os.path.join(cache_dir, save_name)
                img.save(save_path, "PNG")
                image_map[page_num] = save_path
            print(f"Converted {len(image_map)} pages.")
        except ImportError:
            print("Error: `pdf2image` library is not installed.")
            return {}
        except Exception as e:
            print(f"Error converting PDF {pdf_path}: {e}")
            return {}
        return image_map

    def rerank(self, query: str, pages: List[PageElement]) -> List[PageElement]:
        """
        执行重排序。内部通过 get_reranker() 获取单例。
        """
        # 获取单例模型
        reranker = self.get_reranker(self.reranker_model_path)
        if not reranker or not pages:
            return pages
            
        print(f"Reranking {len(pages)} pages...")
        documents_input = [{"image": page.crop_path} for page in pages]
        rerank_input = {
            "instruction": "Given a search query, retrieve relevant candidates that answer the query.",
            "query": {"text": query},
            "documents": documents_input,
            "fps": 1.0 
        }
        
        try:
            scores = reranker.process(rerank_input)
            if len(scores) != len(pages):
                print(f"Warning: Reranker returned {len(scores)} scores for {len(pages)} pages.")
                return pages

            for page, score in zip(pages, scores):
                page.retrieval_score = score
                
            sorted_pages = sorted(pages, key=lambda x: x.retrieval_score, reverse=True)
            return sorted_pages
        except Exception as e:
            print(f"Error during reranking: {e}")
            return pages

    def pipeline(self, query: str, image_paths: List[str] = None,  top_k: int = 5) -> List[PageElement]:
        """
        Logic Updated:
        Lazy load and run reranker ONLY if len(pages) > top_k.
        """
        if self.extractor is None:
            print("Error: ElementExtractor is not initialized in MMLongLoader.")
            return []

        if not image_paths:
            return []

        # --- 1. Process PDF to Images ---
        processed_image_paths = []
        for path in image_paths:
            if path.lower().endswith('.pdf'):
                page_map = self._pdf_to_images(path)
                sorted_pages = sorted(page_map.keys())
                for p_num in sorted_pages:
                    processed_image_paths.append(page_map[p_num])
            else:
                processed_image_paths.append(path)
        
        if not processed_image_paths:
            return []

        # --- 2. Construct Candidate Elements ---
        candidate_pages = []
        for img_path in processed_image_paths:
            elem = PageElement(
                bbox=[0, 0, 1000, 1000],
                type="page_image",
                content="",
                corpus_id=img_path,
                crop_path=img_path
            )
            candidate_pages.append(elem)

        # --- 3. Conditional Lazy Reranking ---
        target_pages = candidate_pages
        
        # 仅当 页面数量 > Top_K 且 配置了模型路径 时，才触发重排
        if self.reranker_model_path and len(candidate_pages) > top_k:
            print(f"Page Count ({len(candidate_pages)}) > Top_K ({top_k}). Triggering Rerank...")
            ranked_pages = self.rerank(query, candidate_pages)
            target_pages = ranked_pages[:top_k]
        else:
            if len(candidate_pages) <= top_k:
                print(f"Page Count ({len(candidate_pages)}) <= Top_K ({top_k}). Skipping Rerank.")
            else:
                print(f"No Reranker configured. Taking first {top_k} pages.")
            target_pages = candidate_pages[:top_k]

        # --- 4. Element Extraction (Agent) ---
        workspace_dir = os.path.abspath(os.path.join(os.getcwd(), "workspace", "crops"))
        os.makedirs(workspace_dir, exist_ok=True)

        extracted_elements = []

        for page in target_pages:
            img_path = page.crop_path
            
            try:
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None

                if loop and loop.is_running():
                    print("Warning: Async loop running. Skipping extraction.")
                    continue
                else:
                    agent_output = asyncio.run(self.extractor.run_agent(
                        user_text=query,
                        image_paths=[img_path]  
                    ))
                
                if not agent_output:
                    continue

                predictions = agent_output.get("predictions", [])
                if not predictions:
                    continue

                last_message = predictions[-1]
                content = last_message.get("content", "")

                extracted_data = []
                try:
                    json_match = re.search(r'```json(.*?)```', content, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(1).strip()
                    else:
                        start = content.find('[')
                        end = content.rfind(']')
                        if start != -1 and end != -1:
                            json_str = content[start:end+1]
                        else:
                            json_str = "[]"
                    extracted_data = json.loads(json_str)
                except Exception:
                    extracted_data = []

                if isinstance(extracted_data, dict):
                    extracted_data = [extracted_data]

                if isinstance(extracted_data, list):
                    current_page_image = None
                    img_w, img_h = 0, 0
                    try:
                        current_page_image = Image.open(img_path)
                        img_w, img_h = current_page_image.size
                    except Exception:
                        pass

                    for item in extracted_data:
                        bbox = item.get("bbox", [0, 0, 0, 0])
                        evidence = item.get("evidence", "")
                        
                        if not isinstance(bbox, list) or len(bbox) != 4:
                            bbox = [0, 0, 0, 0]
                        
                        current_crop_path = img_path 
                        
                        if current_page_image and bbox != [0, 0, 0, 0]:
                            try:
                                x1, y1, x2, y2 = bbox
                                x1 = int(x1 / 1000 * img_w)
                                y1 = int(y1 / 1000 * img_h)
                                x2 = int(x2 / 1000 * img_w)
                                y2 = int(y2 / 1000 * img_h)
                                
                                x1 = max(0, x1); y1 = max(0, y1); x2 = min(img_w, x2); y2 = min(img_h, y2)
                                
                                if x2 > x1 and y2 > y1:
                                    cropped_img = current_page_image.crop((x1, y1, x2, y2))
                                    filename = f"{os.path.basename(img_path).split('.')[0]}_{uuid.uuid4().hex[:8]}.jpg"
                                    save_path = os.path.join(workspace_dir, filename)
                                    cropped_img.save(save_path)
                                    current_crop_path = save_path 
                            except Exception:
                                pass

                        element = PageElement(
                            bbox=bbox,
                            type="evidence",
                            content=evidence,
                            corpus_id=img_path.split('/')[-1], 
                            crop_path=current_crop_path 
                        )
                        if hasattr(page, 'retrieval_score'):
                            element.retrieval_score = page.retrieval_score
                        extracted_elements.append(element)

            except Exception as e:
                print(f"Error during agent execution on {img_path}: {e}")

        return extracted_elements

if __name__ == "__main__":
    # Test code
    root_dir = "/mnt/shared-storage-user/mineru3-share/wangzhengren/PageElement/MMLongBench-Doc"
    
    # Pass path instead of instance
    reranker_path = "/mnt/shared-storage-user/mineru3-share/wangzhengren/JIT-RAG/assets/Qwen/Qwen3-VL-Reranker-8B"

    loader = MMLongLoader(data_root=root_dir, reranker_model_path=reranker_path)
    try:
        loader.load_data()
        if len(loader.samples) > 0:
            s = loader.samples[0]
            print(f"\nSample 0 ID: {s.qid}, Doc: {s.data_source}")
            
            from src.agents.utils import ImageZoomOCRTool
            tool = ImageZoomOCRTool(work_dir="./workspace")
            extractor = ElementExtractor(
                base_url="http://localhost:8001/v1", 
                api_key="sk-123456", 
                model_name="MinerU-Agent-CK300",
                tool=tool
            )
            loader.extractor = extractor
            
            # This should trigger reranker ONLY if pdf pages > 2 (for test)
            if s.data_source.endswith(".pdf"):
                print("Testing Pipeline...")
                results = loader.pipeline(s.query, image_paths=[s.data_source], top_k=2)
                print(f"Extracted {len(results)} elements.")
                for res in results:
                    print(f" - Content: {res.content} \n - Crop: {res.crop_path}")
                s.extra_info['retrieved_elements'] = results
                loader.evaluate()
    except Exception as e:
        print(f"Test failed: {e}")