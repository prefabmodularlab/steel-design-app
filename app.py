import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# 1. 페이지 설정 (반드시 코드 맨 윗줄)
st.set_page_config(page_title="철골 구조 자동 설계", layout="wide")

# =========================================================
# 2. 핵심 로직 및 함수 정의
# =========================================================

def load_data_safe(file):
    """CSV 파일을 안전하게 읽어오는 함수"""
    try:
        # 헤더가 8번째 줄(Index 7)에 있다고 가정
        df = pd.read_csv(file, header=7)
        df.columns = [str(c).strip() for c in df.columns]
        
        if 'H' not in df.columns:
            return None, "CSV 파일 형식이 올바르지 않습니다. 'H' 열이 없습니다."

        # 숫자 변환 헬퍼
        def to_num(series):
            return pd.to_numeric(series.astype(str).str.replace(',', '').str.strip(), errors='coerce')

        clean_df = pd.DataFrame()
        
        # 이름(호칭)
        if len(df.columns) > 2:
            clean_df['Name'] = df.iloc[:, 2]
        else:
            clean_df['Name'] = "Unknown"

        # 데이터 매핑
        clean_df['H'] = to_num(df['H'])
        clean_df['B'] = to_num(df['B'])
        clean_df['t1'] = to_num(df['t1'])
        clean_df['t2'] = to_num(df['t2'])
        clean_df['A'] = to_num(df['A']) * 100   # cm2 -> mm2
        clean_df['W'] = to_num(df['W'])         # kg/m
        clean_df['Ix'] = to_num(df['Ix']) * 10000 # cm4 -> mm4
        
        if 'Zx' in df.columns:
            clean_df['Zx'] = to_num(df['Zx']) * 1000
        elif 'Sx' in df.columns:
            clean_df['Zx'] = to_num(df['Sx']) * 1000
        else:
            clean_df['Zx'] = 0

        clean_df = clean_df[clean_df['H'].notnull()].reset_index(drop=True)
        
        if clean_df.empty:
            return None, "유효한 데이터가 없습니다."
            
        return clean_df, None

    except Exception as e:
        return None, f"데이터 로드 중 오류: {e}"

def find_best_section(df, Mu, Vu, L, max_defl_ratio=360):
    """최적 휨 부
