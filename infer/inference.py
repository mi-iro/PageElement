import re
import json
import asyncio
from typing import List, Dict, Optional
from utils import *
import os
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
import ast

class VisualDocAgent:
    def __init__(self, base_url: str, model_name: str,tool):
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key="no-key-required"
        )
        self.model_name = model_name
        self.tool = tool

    def build_multimodal_user_message(
        self,
        text: str,
        image_paths: Optional[List[str]] = None
    ) -> Dict:
        content = []

        if image_paths:
            #print(image_paths)
            for path in image_paths:
                #print(path)
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": local_image_to_data_url(path,resize=False)
                    }
                })

        if text:
            content.append({
                "type": "text",
                "text": text
            })

        return {
            "role": "user",
            "content": content
        }

    def build_multimodal_tool_message(
        self,
        text: Optional[str] = None,
        image_path: Optional[str] = None
    ) -> Dict:
        content = []
        content.append({
                "type": "text",
                "text": "<tool_response>\n"
            })

        if image_path:
            content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": local_image_to_data_url(image_path,resize=False)
                    }
                })

        if text:
            content.append({
                "type": "text",
                "text": f"\n{text}"
            })

        content.append({
                "type": "text",
                "text": "\n</tool_response>"
            })

        return {
            "role": "user",
            "content": content
        }
    def extract_tool_call(self,text: str):
        m = TOOL_CALL_RE.search(text)
        if not m:
            return None

        raw = m.group(1).strip()

        # 找第一个 dict（从第一个 { 到最后一个 }）
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None

        payload = raw[start:end + 1]

        try:
            return json.loads(payload)
        except Exception:
            try:
                return ast.literal_eval(payload)
            except Exception:
                return None

        return None

    # ---------- tool mock ----------

    async def _handle_tool_call(self, tool_call,image_path,step,max_rounds) -> str:
        """
        当前是测试用 mock
        后面你可以在这里真正调用 image_zoom_and_ocr_tool
        """
        if step >= max_rounds-2:
            return [False, "You have used up all the available uses of `image_zoom_and_ocr_tool`, please return you final response without use tool."]
        result_list = await self.tool.call(tool_call,image_path)
        if result_list[0]==False:
            result_list = await self.tool.call(tool_call,image_path)
            if result_list[0]==False:#重试一次
                return [False, "`image_zoom_and_ocr_tool` is wrong, you can try it again."]
        if len(result_list)==2:
            return [True, result_list[1]]
        if len(result_list)==3:
            return [True, result_list[1], f"{result_list[2]}"]

    # ---------- agent loop ----------

    async def run_agent(
        self,
        user_text: str,
        image_paths: Optional[List[str]] = None,
        max_rounds: int = 10,
        uid: int=1
    ):  
        #print(image_paths)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            self.build_multimodal_user_message(user_text, image_paths),
        ]
        output = {"id":uid,"predictions":[{"role": "system", "content": SYSTEM_PROMPT},{"role": "user", "content": "<image>"+user_text}],
        "images":[image_paths[0]]}
        #print(user_text)
        for step in range(max_rounds):
            resp = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=1.0
            )

            content = resp.choices[0].message.content
            #print(f"[*************Round************* {step}]")
            #print(f"[****Model****]\n{content}\n\n")
            tool_call = self.extract_tool_call(content)
            
            # 没有 tool call → agent 结束
            if tool_call is None:
                #print(content)
                #print(1)
                messages.append({"role": "assistant", "content": content})
                output["predictions"].append({"role": "assistant", "content": content})
                return output

            messages.append({"role": "assistant", "content": content})
            output["predictions"].append({"role": "assistant", "content": content})
            #print(tool_text)
            #print(f"\n[****Tool****]\n")
            tool_response_list =await self._handle_tool_call(tool_call,image_paths[0],step,max_rounds)
            if tool_response_list[0]==False:
                #print(tool_response_list[1])
                messages.append(
                    self.build_multimodal_tool_message(text=tool_response_list[1])
                )
                output["predictions"].append({"role": "user", "content":  "<tool_response>\n"+tool_response_list[1]+"\n</tool_response>"})

            if tool_response_list[0]==True and len(tool_response_list)==2:
                #print(tool_response_list[1])
                messages.append(
                    self.build_multimodal_tool_message(image_path=tool_response_list[1])
                )
                output["predictions"].append({"role": "user", "content":  "<tool_response>\n<image>\n</tool_response>"})
                output["images"].append(tool_response_list[1])

            if tool_response_list[0]==True and len(tool_response_list)==3:
                #print(tool_response_list[2])
                messages.append(
                    self.build_multimodal_tool_message(image_path=tool_response_list[1],text=tool_response_list[2])
                )
                output["predictions"].append({"role": "user", "content":  "<tool_response>\n<image>\n"+tool_response_list[2]+"\n</tool_response>"})
                output["images"].append(tool_response_list[1])

        print(f"[WARN] uid={uid} exceeded max_rounds, drop this sample")
        return None

    # ---------- batch API (主入口) ----------

    async def run_batch(
        self,
        inputs: List[Dict],
        concurrency: int = 16,
        timeout: float = 200,   # 单条 item 超时时间（秒）
    ):
        sem = asyncio.Semaphore(concurrency)

        async def worker(item):
            async with sem:
                try:
                    return await asyncio.wait_for(
                        self.run_agent(
                            uid=item["id"],
                            user_text=item["conversations"][0]["content"][7:],
                            image_paths=[item["images"][0]],
                        ),
                        timeout=timeout,
                    )
                except asyncio.TimeoutError:
                    # 超时：直接跳过
                    # logger.warning(f"Timeout on item {item['id']}")
                    return None
                except Exception as e:
                    # 其他异常也兜住，防止 batch 崩
                    # logger.exception(f"Error on item {item['id']}: {e}")
                    return None

        tasks = [worker(i) for i in inputs]
        results = await tqdm_asyncio.gather(*tasks)

        # 过滤掉超时 / 失败的样本
        results = [r for r in results if r is not None]

        return results

if __name__ == "__main__":
    agent =VisualDocAgent(
        base_url="http://localhost:8000/v1",
        #base_url="http://35.220.164.252:3888/v1",
        model_name="/mnt/shared-storage-user/mineru2-shared/madongsheng/saves/Qwen3-VL-8B-Instruct/full/demond_0203_f_base/checkpoint-400",
        #model_name =  "/mnt/shared-storage-user/mineru2-shared/madongsheng/saves/Qwen3-VL-8B-Instruct/full/demond_0131/checkpoint-400",
        tool=ImageZoomOCRTool(work_dir="/mnt/shared-storage-user/mineru2-shared/madongsheng/zoom")
    )
    #file_path = "/mnt/shared-storage-user/madongsheng/Agent_0128/rl/codevision_rl_with_ids.json"
    #file_path = "/mnt/shared-storage-user/madongsheng/Agent_0128/rl/run0_step2_new_step5_removed_sample500.json"
    #file_path = "/mnt/shared-storage-user/madongsheng/Agent_0128/rl/step5_merged.json"
    #file_path = "/mnt/shared-storage-user/madongsheng/Agent_0128/rl/codevision_rl_2000.json"
    #file_path = "/mnt/shared-storage-user/mineru2-shared/jiayu/data/FinRAGBench-V/data/citation_labels/citation_labels_new/finragbench_bbox_test.json"
    #file_path = "/mnt/shared-storage-user/madongsheng/Agent_0128/neg_hard/finrag_neg/neg_0203.json"
    #file_path = "/mnt/shared-storage-user/mineru3-share/jiayu/newBench/dataOri/MVToolBench/mvtoolbench_benchmark/mvtoolbench_full.json"
    #file_path = "/mnt/shared-storage-user/madongsheng/Agent_0128/test_set/run0_step2_new_step5_sample500.json"
    file_path = "/mnt/shared-storage-user/madongsheng/Agent_0128/rl_dataset/combined.json"
    # 读取 JSON 文件
    with open(file_path, "r", encoding="utf-8") as f:
        inputs = json.load(f)  # 假设是列表

    batch_size = 1000
    base_dir = "/mnt/shared-storage-user/madongsheng/Agent_0128/inference/sft_ckpt400_rl_candidate_new_0204"
    os.makedirs(base_dir, exist_ok=True)

    num_batches = (len(inputs) + batch_size - 1) // batch_size  # 向上取整

    for batch_idx in range(7,num_batches):
        batch_inputs = inputs[batch_idx*batch_size : (batch_idx+1)*batch_size]
        #print(batch_inputs[0])
        for run_idx in range(8):  # 每批次跑三次
            results = asyncio.run(agent.run_batch(batch_inputs))

            # 文件名：batch_批次号_run_轮次.json
            output_file = os.path.join(base_dir, f"batch_{batch_idx}_run_{run_idx}.json")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            print(f"Saved batch {batch_idx} run {run_idx} results ({len(batch_inputs)} items) to {output_file}")
