import os
import pandas as pd
from pathlib import Path

def calculate_usage():
    base_dir = Path(__file__).parent.resolve()
    usage_file = base_dir / "data" / "api_usage.csv"
    
    if not usage_file.exists():
        print(f"找不到使用量紀錄檔 ({usage_file})，目前可能尚未執行過 API 呼叫。")
        return
        
    try:
        df = pd.read_csv(usage_file)
        
        if len(df) == 0:
            print("使用量紀錄檔為空。")
            return
            
        print("====================== API 使用量統計 ======================")
        print(f"總呼叫次數: {len(df)} 次")
        
        total_prompt = df['prompt_tokens'].sum()
        total_candidate = df['candidate_tokens'].sum()
        total = df['total_tokens'].sum()
        
        print(f"\n總計")
        print(f"  - Prompt Tokens (輸入): {total_prompt:,}")
        print(f"  - Candidate Tokens (輸出): {total_candidate:,}")
        print(f"  - Total Tokens (總和): {total:,}")
        
        # 簡單估算成本 (以 Gemini 1.5/2.5 Flash Pay-as-you-go 大致標準費率估算)
        # 輸入約 $0.075 / 1M tokens, 輸出約 $0.30 / 1M tokens
        est_cost = (total_prompt / 1_000_000) * 0.075 + (total_candidate / 1_000_000) * 0.50
        print(f"\n估算成本 (以 Gemini Flash 標準費率)")
        print(f"  - 約 USD ${est_cost:.5f}")
        
        print("\n依模組統計 (總 Tokens)")
        module_stats = df.groupby('module')[['prompt_tokens', 'candidate_tokens', 'total_tokens']].sum()
        print(module_stats.to_string())
        
        print("\n============================================================")
        
    except Exception as e:
        print(f"計算使用量時發生錯誤: {e}")

if __name__ == "__main__":
    calculate_usage()
