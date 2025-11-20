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
    """최적 휨 부재 선정"""
    Fy = 275; E = 205000; Phi_b = 0.9
    candidates = []
    for _, row in df.iterrows():
        Mn = row['Zx'] * Fy
        if Mu > Phi_b * Mn: continue # 강도 부족
        
        delta = (5 * Mu * L**2) / (48 * E * row['Ix'])
        if delta > (L / max_defl_ratio): continue # 처짐 과다
        
        candidates.append(row)
    
    if not candidates: return None
    return pd.DataFrame(candidates).sort_values(by='W').iloc[0]

def find_column_section(df, Pu, L_unbraced):
    """최적 기둥 부재 선정"""
    Fy = 275; E = 205000; Phi_c = 0.9
    candidates = []
    for _, row in df.iterrows():
        Iy_est = row['Ix'] * 0.3 # 약축 가정
        Pe = (3.14159**2 * E * Iy_est) / (L_unbraced**2)
        Pn = min(0.7 * Pe, row['A'] * Fy)
        
        if Pu <= Phi_c * Pn:
            candidates.append(row)
            
    if not candidates: return None
    return pd.DataFrame(candidates).sort_values(by='W').iloc[0]

def draw_3d_model(Lx, Ly, H, spacing, res):
    """3D 모델링 시각화"""
    fig = go.Figure()
    def add_line(x, y, z, color, name, width=5):
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines',
            line=dict(color=color, width=width), name=name, showlegend=False))

    # 기둥
    cols_x, cols_y = [0, Lx, Lx, 0], [0, 0, Ly, Ly]
    for i in range(4):
        add_line([cols_x[i], cols_x[i]], [cols_y[i], cols_y[i]], [0, H], 'red', 'Column', 8)
    # 거더 (X방향)
    add_line([0, Lx], [0, 0], [H, H], 'blue', 'Girder', 6)
    add_line([0, Lx], [Ly, Ly], [H, H], 'blue', 'Girder', 6)
    # 테두리보 (Y방향 끝)
    add_line([0, 0], [0, Ly], [H, H], 'orange', 'Edge Beam', 5)
    add_line([Lx, Lx], [0, Ly], [H, H], 'orange', 'Edge Beam', 5)
    # 작은보 (Y방향 내부)
    curr_x = spacing
    while curr_x < Lx - 100:
        add_line([curr_x, curr_x], [0, Ly], [H, H], 'green', 'Small Beam', 3)
        curr_x += spacing
    # 슬래브
    fig.add_trace(go.Mesh3d(x=[0, Lx, Lx, 0], y=[0, 0, Ly, Ly], z=[H
