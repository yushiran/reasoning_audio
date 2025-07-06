#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import time
import requests
from tqdm import tqdm
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import torch
from langdetect import detect, LangDetectException

# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 高级标签列表
CATEGORY_LIST = [
    'aeroway', 'amenity', 'building', 'highway', 'landuse',
    'leisure', 'natural', 'public_transport', 'railway', 'tourism', 'waterway'
]

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# 加载 M2M100 模型与分词器
MODEL_PATH = r"C:\Users\wuyanru\Desktop\Thesis_Geospatial_ESC-master\m2m100_418M"
try:
    tokenizer = M2M100Tokenizer.from_pretrained(MODEL_PATH)
    model = M2M100ForConditionalGeneration.from_pretrained(MODEL_PATH).to(device)
except OSError:
    print(f"错误 {MODEL_PATH} 加载 M2M100 模型。")
    exit(1)


# 翻译函数
def translate_m2m100(texts, src_lang="auto", tgt_lang="en", batch_size=32):
    if not texts or not any(texts):
        return [""] * len(texts) if texts else []

    detected_src_lang = src_lang
    if src_lang == "auto":
        first_text = next((text for text in texts if text and text.strip()), None)
        if first_text:
            try:
                detected_src_lang = detect(first_text)
            except LangDetectException:
                return texts
        else:
            return [""] * len(texts)  # 全是空文本

    tokenizer.src_lang = detected_src_lang
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        if not any(batch):
            results.extend([""] * len(batch))
            continue
        try:
            encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            generated = model.generate(**encoded, forced_bos_token_id=tokenizer.get_lang_id(tgt_lang))
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
            results.extend(decoded)
        except Exception:
            results.extend(batch)  # 翻译失败则返回原始文本
    return results


# 构建 Overpass 查询语句
def build_overpass_query(lat1, lon1, lat2, lon2):
    filters = [f'["{key}"]' for key in CATEGORY_LIST]
    block = "\n".join(
        f'  node{f}({lat1},{lon1},{lat2},{lon2});\n' \
        f'  way{f}({lat1},{lon1},{lat2},{lon2});\n' \
        f'  relation{f}({lat1},{lon1},{lat2},{lon2});'
        for f in filters
    )
    return f"[out:json][timeout:90];(\n{block}\n);out center tags;"


# 查询 POI 数据
def fetch_poi(lat1, lon1, lat2, lon2):
    query = build_overpass_query(lat1, lon1, lat2, lon2)
    try:
        resp = requests.post(OVERPASS_URL, data={'data': query}, timeout=120)
        resp.raise_for_status()
        return resp.json().get('elements', [])
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            time.sleep(60)
        return []
    except (requests.exceptions.Timeout, requests.exceptions.RequestException, json.JSONDecodeError):
        return []


# 提取并去重 POI 文本
def extract_poi_text(tags: dict) -> list:
    seen = set()
    poi_texts = []
    for key in CATEGORY_LIST:
        if key in tags:
            value = str(tags[key]).replace('"', "'")
            text = f'"{key}": "{value}"'
            if text not in seen:
                seen.add(text)
                poi_texts.append(text)
    return poi_texts


# 主流程
def main(input_file, output_file):
    all_output_data = []

    try:
        with open(input_file, 'r', encoding='utf-8') as fin:
            lines = fin.readlines()
        if not lines:
            print(f"输入文件 '{input_file}' 为空。")
            return
    except IOError as e:
        print(f"读取输入文件 '{input_file}' 失败: {e}。程序退出。")
        return

    for line_content in tqdm(lines, desc="处理POI数据"):
        try:
            rec = json.loads(line_content)
        except json.JSONDecodeError:
            continue

        if not all(key in rec for key in ['lat1', 'lon1', 'lat2', 'lon2', 'text']):
            continue

        lat1, lon1, lat2, lon2 = rec['lat1'], rec['lon1'], rec['lat2'], rec['lon2']
        current_id = rec.get('text', 'unknown_id')

        elems = fetch_poi(lat1, lon1, lat2, lon2)

        poi_texts_raw = []
        if elems:
            for el in elems:
                tags = el.get('tags', {})
                texts = extract_poi_text(tags)
                poi_texts_raw.extend(texts)

        poi_texts_raw_unique = list(dict.fromkeys(poi_texts_raw))  # 去重

        poi_texts_en = []
        if poi_texts_raw_unique:
            poi_texts_en = translate_m2m100(poi_texts_raw_unique, src_lang='auto', tgt_lang='en', batch_size=32)
            if not poi_texts_en and poi_texts_raw_unique:
                poi_texts_en = poi_texts_raw_unique

        output_data = {
            'id': current_id,
            'bbox': [lat1, lon1, lat2, lon2],
            'poi_texts': poi_texts_en,
            'segments': rec.get('audio_spans', [])
        }
        all_output_data.append(output_data)

    try:
        with open(output_file, 'w', encoding='utf-8') as fout:
            json.dump(all_output_data, fout, ensure_ascii=False, indent=2)
        print(f"✅ 数据处理完成，结果已保存到 {output_file}")
    except IOError as e:
        print(f"写入输出文件 '{output_file}' 失败: {e}。")


if __name__ == '__main__':
    main('../freesound_data_for_poi.txt', '../outputs/poi_features_cleaned.json')
