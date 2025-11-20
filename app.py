import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ==========================================
# 1. 페이지 설정
# ==========================================
st.set_page_config(page_title="철골 구조 자동 설계", layout="wide")

# ==========================================
# 2. 클래스 및 함수 정의
# ==========================================

class SteelDB:
    def __init__(self, uploaded_file):
        self.data = self.load_data(uploaded_file)

    def load_data(self, file):
        try:
            # 헤더 위치 지정 (8번째 줄)
            df = pd.read_csv(file, header=7)
            df.columns = [str(c).strip() for c in df.columns]
            
            if 'H' not in df.columns:
                st.error("❌ [데이터 오류] CSV 파일에 'H' 열이 없습니다.")
                return pd.DataFrame()

            # 숫자 변환 유틸리티
            def to_num(series):
                return pd.to_numeric(series.astype(str).str.replace(',', '').str.strip(), errors='coerce')

            clean_df = pd.DataFrame()
            
            if len(df.columns) > 2:
                clean_df['Name'] = df.iloc[:, 2]
            else:
                clean_df['Name'] = "Unknown"

            clean_df['H'] = to_num(df['H'])
            clean_df['B'] = to_num(df['B'])
            clean_df['t1'] = to_num(df['t1'])
            clean_df['t2'] = to_num(df['t2'])
            clean_df['A'] = to_num(df['A']) * 100
            clean_df['W'] = to_num(df['W'])
            clean_df['Ix'] = to_num(df['Ix']) * 10000
            
            if 'Zx' in df.columns:
                clean_df['Zx'] = to_num(df['Zx']) * 1000
            elif 'Sx' in df.columns:
                clean_df['Zx'] = to_num(df['Sx']) * 1000
            else:
                clean_df['Zx'] = 0

            clean_df = clean_df[clean_df['H'].notnull()].reset_index(drop=True)
            return clean_df

        except Exception as e:
            st.error(f"❌ 데이터 로드 중 오류: {e}")
            return pd.DataFrame()

    def get_optimized_section(self, Mu, Vu, L, max_deflection_ratio=360):
        Fy = 275
        E = 205000
        Phi_b = 0.9
