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

    # 1. 작은보 (Y방향)
    w_sb = wu_area * spacing
    Mu_sb = (w_sb * L_Y**2) / 8
    Vu_sb = (w_sb * L_Y) / 2
    sb_mem = db.get_optimized_section(Mu_sb, Vu_sb, L_Y)

    # 2. 테두리보 (Y방향)
    w_eb = wu_area * (spacing / 2) + 2.0 
    Mu_eb = (w_eb * L_Y**2) / 8
    Vu_eb = (w_eb * L_Y) / 2
    eb_mem = db.get_optimized_section(Mu_eb, Vu_eb, L_Y)

    # 3. 거더 (X방향)
    w_g = wu_area * (L_Y / 2) + 1.5
    Mu_g = (w_g * L_X**2) / 8
    Vu_g = (w_g * L_X) / 2
    girder_mem = db.get_optimized_section(Mu_g, Vu_g, L_X)

    # 4. 기둥
    Pu_c = (Vu_g + Vu_eb) * 1.1
    col_mem = db.get_column_section(Pu_c, H_COL)

    num_sb = int(L_X / spacing) - 1
    if num_sb < 0: num_sb = 0
    
    total_weight = 0
    if all([sb_mem is not None, eb_mem is not None, girder_mem is not None, col_mem is not None]):
        total_weight = (num_sb * L_Y/1000 * sb_mem['W']) + \
                       (2 * L_Y/1000 * eb_mem['W']) + \
                       (2 * L_X/1000 * girder_mem['W']) + \
                       (4 * H_COL/1000 * col_mem['W'])

    return {
        "sb": sb_mem, "eb": eb_mem, "girder": girder_mem, "col": col_mem,
        "Mu_sb": Mu_sb, "Mu_eb": Mu_eb, "Mu_g": Mu_g, "Pu_c": Pu_c,
        "num_sb": num_sb, "total_weight": total_weight, "wu": wu_area
    }

def draw_3d_plotly(Lx, Ly, H, spacing, res):
    fig = go.Figure()
    
    def add_line(x, y, z, color, name, width=5):
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z, mode='lines',
            line=dict(color=color, width=width), name=name, showlegend=False
        ))

    # 기둥
    cols_x, cols_y = [0, Lx, Lx, 0], [0, 0, Ly, Ly]
    for i in range(4):
        add_line([cols_x[i], cols_x[i]], [cols_y[i], cols_y[i]], [0, H], 'red', 'Column', 8)

    # 거더
    add_line([0, Lx], [0, 0], [H, H], 'blue', 'Girder', 6)
    add_line([0, Lx], [Ly, Ly], [H, H], 'blue', 'Girder', 6)

    # 테두리보
    add_line([0, 0], [0, Ly], [H, H], 'orange', 'Edge Beam', 5)
    add_line([Lx, Lx], [0, Ly], [H, H], 'orange', 'Edge Beam', 5)

    # 작은보
    curr_x = spacing
    while curr_x < Lx - 100:
        add_line([curr_x, curr_x], [0, Ly], [H, H], 'green', 'Small Beam', 3)
        curr_x += spacing

    # 슬래브
    fig.add_trace(go.Mesh3d(x=[0, Lx, Lx, 0], y=[0, 0, Ly, Ly], z=[H, H, H, H], 
                            opacity=0.2, color='gray', name='Slab'))

    fig.update_layout(scene=dict(aspectmode='data'), height=600, margin=dict(t=0,b=0,l=0,r=0))
    return fig

# ==========================================
# 3. 메인 UI 및 실행 로직 (들여쓰기 제거함)
# ==========================================

# --- 사이드바 ---
with st.sidebar:
    st.header("1. 설계 조건 입력")
    st.markdown("---")
    
    uploaded_file = st.file_uploader("H형강 DB(csv) 업로드", type=['csv'])
    
    st.subheader("제원 설정")
    t_slab = st.number_input("슬래브 두께 (mm)", 100, 300, 150, 10)
    spacing = st.number_input("작은보 간격 (mm)", 1000, 5000, 2500, 100)
    ll_load = st.number_input("활하중 (kN/m²)", 1.0, 10.0, 2.5, 0.1)
    
    if st.button("설계 실행 (Run)", type="primary"):
        st.session_state['run'] = True

# --- 메인 화면 ---
st.title("🏗️ 철골 구조 시스템 자동 설계")
st.markdown("거더(X축) - 테두리보(Y축) - 작은보(Y축) 시스템 최적화")

# [Step 1] 파일 확인 (없으면 중단)
if uploaded_file is None:
    st.info("👈 왼쪽 사이드바에서 **CSV 파일(RH.csv)**을 업로드해주세요.")
    st.stop()

# [Step 2] 데이터 로드
db = SteelDB(uploaded_file)
if db.data.empty:
    st.error("❌ 데이터 로드 실패: CSV 파일을 읽었으나 내용이 비어있거나 형식이 맞지 않습니다.")
    st.stop()
else:
    with st.expander("✅ 데이터 로드 성공 (내용 확인)"):
        st.dataframe(db.data.head())

# [Step 3] 실행 버튼 확인
if 'run' not in st.session_state or not st.session_state['run']:
    st.info("👈 설정을 확인하고 **[설계 실행]** 버튼을 눌러주세요.")
    st.stop()

# [Step 4] 구조 계산 수행
res = calculate_structure(db,
