import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ==========================================
# 1. 페이지 설정 (무조건 맨 윗줄)
# ==========================================
st.set_page_config(page_title="철골 구조 자동 설계", layout="wide")

# ==========================================
# 2. 클래스 및 함수 정의
# ==========================================

class SteelDB:
    def __init__(self, uploaded_file):
        self.data = self.load_data(uploaded_file)

    def load_data(self, file):
        # 1. CSV 읽기 (헤더 위치 Index 7 = 8번째 줄)
        df = pd.read_csv(file, header=7)
        df.columns = [str(c).strip() for c in df.columns]
        
        if 'H' not in df.columns:
            return pd.DataFrame() # 빈 데이터프레임 반환

        # 숫자 변환 유틸리티
        def to_num(series):
            return pd.to_numeric(series.astype(str).str.replace(',', '').str.strip(), errors='coerce')

        clean_df = pd.DataFrame()
        
        # 이름 가져오기 (컬럼 수 확인)
        if len(df.columns) > 2:
            clean_df['Name'] = df.iloc[:, 2]
        else:
            clean_df['Name'] = "Unknown"

        # 데이터 매핑
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

        # 유효 데이터만 남김
        clean_df = clean_df[clean_df['H'].notnull()].reset_index(drop=True)
        return clean_df

    def get_optimized_section(self, Mu, Vu, L, max_deflection_ratio=360):
        Fy = 275
        E = 205000
        Phi_b = 0.9
        
        valid_sections = []
        for _, row in self.data.iterrows():
            # 휨 강도
            Mn = row['Zx'] * Fy
            if Mu > Phi_b * Mn: continue 
            # 처짐
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
            # 약축 좌굴 고려
            Iy_est = row['Ix'] * 0.3
            Pe = (3.14159**2 * E * Iy_est) / (L_unbraced**2)
            Pn = min(0.7 * Pe, row['A'] * Fy)

            if Pu <= Phi_c * Pn:
                valid_sections.append(row)
        
        if not valid_sections: return None
        return pd.DataFrame(valid_sections).sort_values(by='W').iloc[0]


def calculate_structure(db, t_slab, spacing, ll_kpa):
    L_X, L_Y, H_COL = 10000, 10000, 5000
    
    wd_total = (t_slab * 24e-6) + 1.5e-3
    wl_total = ll_kpa * 1e-3
    wu_area = 1.2 * wd_total + 1.6 * wl_total 

    # 1. 작은보 (Y방
