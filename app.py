import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# 페이지 기본 설정 (와이드 모드)
st.set_page_config(page_title="철골 구조 자동 설계", layout="wide")

# ==========================================
# 1. 클래스 및 함수 정의 (Core Logic)
# ==========================================

class SteelDB:
    def __init__(self, uploaded_file):
        self.data = self.load_data(uploaded_file)

    def load_data(self, file):
        try:
            # RH.csv 파일 구조에 맞춰 Header를 8번째 줄(Index 7)로 지정
            df = pd.read_csv(file, header=7)
            
            # 컬럼 이름의 공백 제거
            df.columns = [str(c).strip() for c in df.columns]
            
            if 'H' not in df.columns:
                st.error("❌ 오류: CSV 파일에서 'H' 열을 찾을 수 없습니다.")
                return pd.DataFrame()
            
            # 숫자 변환 유틸리티 (쉼표, 공백 제거)
            def to_num(series):
                return pd.to_numeric(series.astype(str).str.replace(',', '').str.strip(), errors='coerce')

            # H값이 숫자인 행만 필터링
            df['H'] = to_num(df['H'])
            df = df[df['H'].notnull()].copy()

            clean_df = pd.DataFrame()
            
            # 호칭(Name) 가져오기
            clean_df['Name'] = df.iloc[:, 2] 
            
            # 데이터 매핑
            clean_df['H'] = df['H']
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

            return clean_df.reset_index(drop=True)

        except Exception as e:
            st.error(f"데이터 로드 중 오류가 발생했습니다: {e}")
            return pd.DataFrame()

    def get_optimized_section(self, Mu, Vu, L, max_deflection_ratio=360):
        Fy = 275 
        E = 205000
        Phi_b = 0.9
        
        valid_sections = []
        
        for _, row in self.data.iterrows():
            # 1. 휨 강도 검토
            Mn = row['Zx'] * Fy
            if Mu > Phi_b * Mn: continue 

            # 2. 처짐 검토
            delta = (5 * Mu * L**2) / (48 * E * row['Ix'])
            if delta > (L / max_deflection_ratio): continue 
            
            valid_sections.append(row)

        if not valid_sections: return None
        return pd.DataFrame(valid_sections).sort_values(by='W').iloc[0]

    def get_column_section(self, Pu, L_unbraced):
        Fy = 275
        E = 205000
        Phi_c = 0.9
        
        valid_sections = []
        for _, row in self.data.iterrows():
            Iy_est = row['Ix'] * 0.3 
            Pe = (3.14159**2 * E * Iy_est) / (L_unbraced**2)
            Pn = min(0.
