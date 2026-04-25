import os
from datasets import load_dataset
import json

def download_data():
    print("正在连接 HuggingFace 下载 FineWeb-Edu 高质量语料...")
    # 工业级做法：使用 streaming=True 避免内存撑爆，直接流式写入磁盘
    dataset = load_dataset('HuggingFaceFW/fineweb-edu', name='sample-10BT', split='train', streaming=True)
    
    output_path = 'data/raw/fineweb_edu_raw.jsonl'
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, entry in enumerate(dataset):
            if i >= 500000: # 50万条高质量数据对于 7B MoE 的起步测试非常正规
                break
            f.write(json.dumps({'text': entry['text']}, ensure_ascii=False) + '\n')
            if i % 10000 == 0:
                print(f'已成功导出 {i} 条语料到 {output_path}')

if __name__ == "__main__":
   download_data()