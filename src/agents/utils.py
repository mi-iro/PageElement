import re
import os
import uuid
import json
import time
from io import BytesIO
import mimetypes
import base64

from math import ceil, floor, cos, sin, radians
from typing import List, Union, Tuple

import requests
from PIL import Image
from mineru_vl_utils import MinerUClient


TOOL_CALL_RE = re.compile(
        r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
        re.DOTALL
    )

@staticmethod
def local_image_to_data_url(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    mime, _ = mimetypes.guess_type(path)
    if mime is None:
        mime = "image/png"

    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime};base64,{b64}"




class ImageZoomOCRTool:

    description = 'Zoom in on a specific region of an image by cropping it based on a bounding box (bbox), optionally rotate it or perform OCR.'
    parameters = {
        'type': 'object',
        'properties': {
            'label': {
                'type': 'string',
                'description': 'The name or label of the object in the specified bounding box'
            },
            'bbox': {
                'type':
                    'array',
                'items': {
                    'type': 'number'
                },
                'minItems':
                    4,
                'maxItems':
                    4,
                'description':
                    'The bbox specified as [x1, y1, x2, y2] in 0-1000 coordinates, relative to the page image from the user.'
            },
            'angle': {
                'type': 'number',
                'description': 'The angle to rotate the image (counter-clockwise) after cropping. Default is 0.',
                'default': 0
            },
            'do_ocr': {
                'type': 'boolean',
                'description': 'Whether OCR the processed image. OCR returns results with bboxes relative to the page image from user. Default is False.',
                'default': False
            }
        },
        'required': ['bbox', 'label']
    }

    def __init__(self,work_dir ,mineru_server_url="http://10.102.250.36:8000/",mineru_model_path="/root/checkpoints/MinerU2.5-2509-1.2B/"):
        self.mineru_model_path = "/root/checkpoints/MinerU2.5-2509-1.2B/"
        self.mineru_server_url = "http://10.102.250.36:8000/"
        self.mineru_client = None
        self.work_dir=work_dir

    def _get_mineru_client(self):
        if self.mineru_client is None:
            if MinerUClient is None:
                raise ImportError("MinerUClient module is not installed.")
            self.mineru_client = MinerUClient(
                model_name=self.mineru_model_path,
                backend="http-client",
                server_url=self.mineru_server_url.rstrip('/')
            )
        return self.mineru_client

    # Image resizing functions (copied from qwen-vl-utils)
    def round_by_factor(self, number: int, factor: int) -> int:
        """Returns the closest integer to 'number' that is divisible by 'factor'."""
        return round(number / factor) * factor

    def ceil_by_factor(self, number: int, factor: int) -> int:
        """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
        return math.ceil(number / factor) * factor

    def floor_by_factor(self, number: int, factor: int) -> int:
        """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
        return math.floor(number / factor) * factor

    def smart_resize(self,
                     height: int,
                     width: int,
                     factor: int = 32,
                     min_pixels: int = 56 * 56,
                     max_pixels: int = 12845056) -> tuple[int, int]:
        """Smart resize image dimensions based on factor and pixel constraints"""
        h_bar = max(factor, self.round_by_factor(height, factor))
        w_bar = max(factor, self.round_by_factor(width, factor))
        if h_bar * w_bar > max_pixels:
            beta = math.sqrt((height * width) / max_pixels)
            h_bar = self.floor_by_factor(height / beta, factor)
            w_bar = self.floor_by_factor(width / beta, factor)
        elif h_bar * w_bar < min_pixels:
            beta = math.sqrt(min_pixels / (height * width))
            h_bar = self.ceil_by_factor(height * beta, factor)
            w_bar = self.ceil_by_factor(width * beta, factor)
        return h_bar, w_bar

    def map_point_back(self, x, y, 
                       final_size: Tuple[int, int], 
                       rotated_size: Tuple[int, int], 
                       crop_size: Tuple[int, int], 
                       crop_offset: Tuple[int, int], 
                       rotation_angle: float,
                       original_size: Tuple[int, int]) -> Tuple[int, int]:
        """
        将MinerU识别后的点坐标(在final_image上的0-1000相对坐标)
        映射回原图(original_image)的0-1000相对坐标
        """
        # 1. Convert relative (0-1000) on final image to absolute pixels on final image
        abs_x = x / 1000.0 * final_size[0]
        abs_y = y / 1000.0 * final_size[1]

        # 2. Undo Resize (Mapping from final_size back to rotated_size)
        scale_x = final_size[0] / rotated_size[0]
        scale_y = final_size[1] / rotated_size[1]
        
        abs_x = abs_x / scale_x
        abs_y = abs_y / scale_y

        # 3. Undo Rotation (Mapping from rotated_size back to crop_size)
        # 旋转中心是图像中心
        cx_rot, cy_rot = rotated_size[0] / 2.0, rotated_size[1] / 2.0
        cx_crop, cy_crop = crop_size[0] / 2.0, crop_size[1] / 2.0

        # 平移到旋转中心
        dx = abs_x - cx_rot
        dy = abs_y - cy_rot

        # 逆旋转 (PIL rotate is counter-clockwise, so inverse is clockwise / negative angle)
        # 公式: x' = x cos(-θ) - y sin(-θ)
        #       y' = x sin(-θ) + y cos(-θ)
        rad = radians(-rotation_angle)
        cos_a = cos(rad)
        sin_a = sin(rad)

        rot_x = dx * cos_a - dy * sin_a
        rot_y = dx * sin_a + dy * cos_a

        # 平移回 Crop 中心
        orig_crop_x = rot_x + cx_crop
        orig_crop_y = rot_y + cy_crop

        # 4. Undo Crop (Add offset)
        final_abs_x = orig_crop_x + crop_offset[0]
        final_abs_y = orig_crop_y + crop_offset[1]

        # 5. Normalize to 0-1000 relative to original image
        norm_x = min(1000, max(0, int(final_abs_x / original_size[0] * 1000)))
        norm_y = min(1000, max(0, int(final_abs_y / original_size[1] * 1000)))

        return norm_x, norm_y

    def safe_crop_bbox(self, left, top, right, bottom, img_width, img_height):
        """Only clamp bbox to image bounds, without resizing or expanding it."""
        left = max(0, min(left, img_width))
        top = max(0, min(top, img_height))
        right = max(0, min(right, img_width))
        bottom = max(0, min(bottom, img_height))
        # Ensure valid order
        if left >= right:
            right = left + 1
        if top >= bottom:
            bottom = top + 1
        # Clamp again in case of degenerate
        right = min(right, img_width)
        bottom = min(bottom, img_height)
        return [left, top, right, bottom]

    async def call(self, tool_call: dict, image_path) -> List[any]:
        try:
            params = tool_call["arguments"]
            bbox = params['bbox']
            angle = params.get('angle', 0)
            do_ocr = params.get('do_ocr', False)
        except Exception as e:
            print(f'{e}')
            return [False, f'Error: Invalid tool_call params']

        os.makedirs(self.work_dir, exist_ok=True)

        try:
            image = Image.open(image_path)
        except Exception as e:
            print(f'{e}')
            return [False, f'Error: Invalid input image']

        try:
            # 1. Convert relative bbox (0-1000) to absolute pixels
            img_width, img_height = image.size
            rel_x1, rel_y1, rel_x2, rel_y2 = bbox
            abs_x1 = rel_x1 / 1000.0 * img_width
            abs_y1 = rel_y1 / 1000.0 * img_height
            abs_x2 = rel_x2 / 1000.0 * img_width
            abs_y2 = rel_y2 / 1000.0 * img_height

            # 2. ONLY clamp to image bounds — no resizing or padding!
            validated_bbox = self.safe_crop_bbox(abs_x1, abs_y1, abs_x2, abs_y2, img_width, img_height)
            left, top, right, bottom = validated_bbox

            # Record crop info for coordinate mapping
            crop_offset = (left, top)
            crop_size = (right - left, bottom - top)

            # 3. Crop the image (even if very small)
            cropped_image = image.crop((left, top, right, bottom))

            # 4. Rotate the image
            rotated_image = cropped_image.rotate(angle, expand=True)
            rotated_size = rotated_image.size

            # 5. Apply smart resize to the final image for model input
            # Note: smart_resize expects (height, width), but PIL uses (width, height)
            new_h, new_w = self.smart_resize(
                height=rotated_size[1],
                width=rotated_size[0],
                factor=32,
                min_pixels=32 * 32,  # Allow very small images to be upscaled
                max_pixels=12845056
            )
            final_image = rotated_image.resize((new_w, new_h), resample=Image.BICUBIC)
            final_size = final_image.size

            # Save processed image
            output_filename = f'{uuid.uuid4()}.png'
            output_path = os.path.abspath(os.path.join(self.work_dir, output_filename))
            final_image.save(output_path)

            if not do_ocr:
                return [True,output_path]

            # 6. OCR with MinerU
            mineru_result = []
            ocr_text_output = ""

            try:
                client = self._get_mineru_client()
                img_for_ocr = Image.open(output_path).convert("RGB")

                max_retries = 3
                raw_mineru_result = None
                for attempt in range(max_retries):
                    try:
                        raw_mineru_result = await client.aio_two_step_extract(img_for_ocr)
                        if raw_mineru_result:
                            break
                    except Exception as e:
                        if attempt == max_retries - 1:
                            print(f"MinerU failed after {max_retries} attempts: {e}")
                        time.sleep(1)

                if raw_mineru_result:
                    if len(raw_mineru_result) == 1 and raw_mineru_result[0].get('type') == 'image':
                        # 对输入区域调用OCR
                        ocr_text_output = await client.aio_content_extract(img_for_ocr)
                        raw_mineru_result[0]['content'] = ocr_text_output
                    
                    transformed_results = []
                    for item in raw_mineru_result:
                        raw_bbox = item.get('bbox', [])
                        if not raw_bbox or len(raw_bbox) != 4:
                            continue

                        # Normalize MinerU bbox to 0-1000 on final_image
                        if all(0 <= x <= 1.0 for x in raw_bbox):
                            # Assume [x1, y1, x2, y2] in 0-1
                            x1, y1, x2, y2 = [c * 1000 for c in raw_bbox]
                        else:
                            # Assume already in 0-1000
                            x1, y1, x2, y2 = raw_bbox

                        # Map back to original image coordinates
                        orig_x1, orig_y1 = self.map_point_back(
                            x1, y1, final_size, rotated_size, crop_size, crop_offset, angle, image.size
                        )
                        orig_x2, orig_y2 = self.map_point_back(
                            x2, y2, final_size, rotated_size, crop_size, crop_offset, angle, image.size
                        )

                        new_item = item.copy()
                        new_item['bbox'] = [
                            min(orig_x1, orig_x2),
                            min(orig_y1, orig_y2),
                            max(orig_x1, orig_x2),
                            max(orig_y1, orig_y2)
                        ]
                        transformed_results.append(new_item)

                    mineru_result = transformed_results
                    ocr_text_output = json.dumps(mineru_result, ensure_ascii=False)
                else:
                    ocr_text_output = "MinerU returned empty result."

            except Exception as e:
                print(f"MinerU Processing Error: {e}")
                ocr_text_output = f"MinerU Error: {str(e)}"

            return [
                True,
                output_path,
                f"OCR Result (Mapped to original coords): {ocr_text_output}"
            ]

        except Exception as e:
            obs = f'Tool Execution Error: {str(e)}'
            return [False,obs]