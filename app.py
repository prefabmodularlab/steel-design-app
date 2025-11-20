import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# 1. 페이지 기본 설정 (와이드 모드)
st.set_page_config(page_title="철골 구조 자동 설계", layout="wide")

# ==========================================
# 2. 클래스 및 함수 정의 (Core Logic)
# ==========================================

class SteelDB:
    def __init__(self, uploaded_file):
        self.data = self.load_data(uploaded_file)

    def load_data(self, file):
        """
        CSV 파일을 읽어와서 구조 계산에 필요한 형태로 정리합니다.
        쉼표 제거, 공백 제거, 헤더 위치 지정을 수행하여 오류를 방지합니다.
        """
        try:
            # 헤더 위치 8번째 줄 (Index 7)
            df = pd.read_csv(file, header=7)
            df.columns = [str(c).strip() for c in df.columns]
            
            if 'H' not in df.columns:
                st.error("❌ 오류: CSV 파일 형식이 맞지 않습니다. 'H' 열을 찾을 수 없습니다.")
                return pd.DataFrame()

            # 숫자 변환 유틸리티 (쉼표, 공백 처리)
            def to_num(series):
                return pd.to_numeric(series.astype(str).str.replace(',', '').str.strip(), errors='coerce')

            clean_df = pd.DataFrame()
            clean_df['Name'] = df.iloc[:, 2] # 호칭
            
            # 주요 제원 변환
            clean_df['H'] = to_num(df['H'])
            clean_df['B'] = to_num(df['B'])
            clean_df['t1'] = to_num(df['t1'])
            clean_df['t2'] = to_num(df['t2'])
            clean_df['A'] = to_num(df['A']) * 100   # mm2
            clean_df['W'] = to_num(df['W'])         # kg/m
            clean_df['Ix'] = to_num(df['Ix']) * 10000 # mm4
            
            if 'Zx' in df.columns:
                clean_df['Zx'] = to_num(df['Zx']) * 1000 # mm3
            elif 'Sx' in df.columns:
                clean_df['Zx'] = to_num(df['Sx']) * 1000
            else:
                clean_df['Zx'] = 0

            # 유효 데이터 필터링
            clean_df = clean_df[clean_df['H'].notnull()].reset_index(drop=True)
            return clean_df

        except Exception as e:
            st.error(f"데이터 로드 중 오류 발생: {e}")
            return pd.DataFrame()

    def get_optimized_section(self, Mu, Vu, L, max_deflection_ratio=360):
        Fy = 275  # MPa
        E = 205000 # MPa
        Phi_b = 0.9
        
        valid_sections = []
        for _, row in self.data.iterrows():
            # 1. 휨 강도 (Mu <= Phi * Mn)
            Mn = row['Zx'] * Fy
            if Mu > Phi_b * Mn: continue 

            # 2. 처짐 (Delta <= L/ratio)
            # 등분포 하중 기준 약산식 적용
            delta = (5 * Mu * L**2) / (48 * E * row['Ix'])
            if delta > (L / max_deflection_ratio): continue 
            
            valid_sections.append(row)

        if not valid_sections: return None
        # 가장 가벼운 부재 반환
        return pd.DataFrame(valid_sections).sort_values(by='W').iloc[0]

    def get_column_section(self, Pu, L_unbraced):
        Fy = 275
        E = 205000
        Phi_c = 0.9
        
        valid_sections = []
        for _, row in self.data.iterrows():
            # 기둥 설계 (약축 좌굴 고려)
            Iy_est = row['Ix'] * 0.3 
            Pe = (3.14159**2 * E * Iy_est) / (L_unbraced**2)
            Pn = min(0.7 * Pe, row['A'] * Fy) # 약산식

            if Pu <= Phi_c * Pn:
                valid_sections.append(row)
        
        if not valid_sections: return None
        return pd.DataFrame(valid_sections).sort_values(by='W').iloc[0]


def calculate_structure(db, t_slab, spacing, ll_kpa):
    # --- 구조 제원 ---
    L_X = 10000 # mm (거더 길이)
    L_Y = 10000 # mm (작은보/테두리보 길이)
    H_COL = 5000 # mm
    
    # --- 하중 산정 ---
    wd_total = (t_slab * 24e-6) + 1.5e-3 # N/mm2 (DL)
    wl_total = ll_kpa * 1e-3 # N/mm2 (LL)
    wu_area = 1.2 * wd_total + 1.6 * wl_total # 계수하중 (N/mm2)

    # 1. 작은보 (Small Beam) - 거더 사이에 배치 (Y방향)
    # 분담폭 = spacing
    w_sb_lin = wu_area * spacing # N/mm
    Mu_sb = (w_sb_lin * L_Y**2) / 8
    Vu_sb = (w_sb_lin * L_Y) / 2
    sb_mem = db.get_optimized_section(Mu_sb, Vu_sb, L_Y)

    # 2. 테두리보 (Edge Beam) - 양 끝단 약축 연결 (Y방향)
    # 분담폭 = spacing / 2 (외곽이므로 절반) + 벽체 하중 가정(약간)
    w_eb_lin = wu_area * (spacing / 2) + 2.0 # 벽체 등 2.0 N/mm 가정
    Mu_eb = (w_eb_lin * L_Y**2) / 8
    Vu_eb = (w_eb_lin * L_Y) / 2
    eb_mem = db.get_optimized_section(Mu_eb, Vu_eb, L_Y)

    # 3. 거더 (Girder) - 기둥 강축 연결 (X방향)
    # 작은보와 테두리보의 반력을 받음. 등분포 치환하여 계산.
    # 분
