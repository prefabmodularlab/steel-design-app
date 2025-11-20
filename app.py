import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# 1. 페이지 설정 (반드시 코드 맨 윗줄에 위치해야 함)
st.set_page_config(page_title="철골 구조 자동 설계", layout="wide")

# ---------------------------------------------------------
# 함수 정의 구간 (복잡한 로직을 단순화함)
# ---------------------------------------------------------

def load_data_safe(file):
    """CSV 파일을 안전하게 읽어오는 함수"""
    try:
        # 헤더가 8번째 줄(Index 7)에 있다고 가정
        df = pd.read_csv(file, header=7)
        
        # 컬럼명 공백 제거
        df.columns = [str(c).strip() for c in df.columns]
        
        # 파일 형식 체크
        if 'H' not in df.columns:
            return None, "CSV 파일 형식이 올바르지 않습니다. 'H' 열이 없습니다."

        # 숫자 변환 헬퍼 함수 (쉼표, 공백 제거)
        def to_num(series):
            return pd.to_numeric(series.astype(str).str.replace(',', '').str.strip(), errors='coerce')

        clean_df = pd.DataFrame()
        
        # 이름(호칭) 가져오기 - 보통 3번째 열(Index 2)
        if len(df.columns) > 2:
            clean_df['Name'] = df.iloc[:, 2]
        else:
            clean_df['Name'] = "Unknown"

        # 데이터 매핑 및 변환
        clean_df['H'] = to_num(df['H'])
        clean_df['B'] = to_num(df['B'])
        clean_df['t1'] = to_num(df['t1'])
        clean_df['t2'] = to_num(df['t2'])
        clean_df['A'] = to_num(df['A']) * 100   # cm2 -> mm2
        clean_df['W'] = to_num(df['W'])         # kg/m
        clean_df['Ix'] = to_num(df['Ix']) * 10000 # cm4 -> mm4
        
        # Zx(소성) 없으면 Sx(탄성) 사용
        if 'Zx' in df.columns:
            clean_df['Zx'] = to_num(df['Zx']) * 1000
        elif 'Sx' in df.columns:
            clean_df['Zx'] = to_num(df['Sx']) * 1000
        else:
            clean_df['Zx'] = 0

        # 유효 데이터 필터링
        clean_df = clean_df[clean_df['H'].notnull()].reset_index(drop=True)
        
        if clean_df.empty:
            return None, "데이터 정제 후 남은 행이 없습니다. 숫자가 올바른지 확인해주세요."
            
        return clean_df, None

    except Exception as e:
        return None, f"데이터 로드 중 치명적 오류: {e}"

def find_best_section(df, Mu, Vu, L, max_defl_ratio=360):
    """주어진 하중조건을 만족하는 최적(최소중량) 부재 찾기"""
    Fy = 275     # MPa
    E = 205000   # MPa
    Phi_b = 0.9
    
    # 조건을 만족하는 후보군 찾기
    candidates = []
    for _, row in df.iterrows():
        # 1. 휨 강도 체크
        Mn = row['Zx'] * Fy
        if Mu > Phi_b * Mn:
            continue
            
        # 2. 처짐 체크
        delta = (5 * Mu * L**2) / (48 * E * row['Ix'])
        if delta > (L / max_defl_ratio):
            continue
            
        candidates.append(row)
    
    if not candidates:
        return None
        
    # 무게 순 정렬 후 가장 가벼운 것 반환
    return pd.DataFrame(candidates).sort_values(by='W').iloc[0]

def find_column_section(df, Pu, L_unbraced):
    """기둥 부재 찾기"""
    Fy = 275
    E = 205000
    Phi_c = 0.9
    
    candidates = []
    for _, row in df.iterrows():
        # 약축 좌굴 고려
        Iy_est = row['Ix'] * 0.3
        Pe = (3.14159**2 * E * Iy_est) / (L_unbraced**2)
        Pn = min(0.7 * Pe, row['A'] * Fy)
        
        if Pu <= Phi_c * Pn:
            candidates.append(row)
            
    if not candidates:
        return None
        
    return pd.DataFrame(candidates).sort_values(by='W').iloc[0]

def draw_3d_model(Lx, Ly, H, spacing, res):
    """3D 모델링 시각화"""
    fig = go.Figure()
