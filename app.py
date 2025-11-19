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
            # 업로드된 파일 읽기 (헤더 위치 조정)
            df = pd.read_csv(file, header=6) 
            df.columns = [str(c).strip() for c in df.columns]
            
            # H값이 있는 행만 필터링
            df = df[pd.to_numeric(df['H'], errors='coerce').notnull()].copy()

            clean_df = pd.DataFrame()
            clean_df['Name'] = df.iloc[:, 1] 
            clean_df['H'] = pd.to_numeric(df.iloc[:, 2])
            clean_df['B'] = pd.to_numeric(df.iloc[:, 3])
            clean_df['t1'] = pd.to_numeric(df.iloc[:, 4])
            clean_df['t2'] = pd.to_numeric(df.iloc[:, 5])
            clean_df['A'] = pd.to_numeric(df.iloc[:, 7]) * 100 
            clean_df['W'] = pd.to_numeric(df.iloc[:, 8]) 
            clean_df['Ix'] = pd.to_numeric(df.iloc[:, 9]) * 10000 
            clean_df['Zx'] = pd.to_numeric(df.iloc[:, 14]) * 1000 

            return clean_df.reset_index(drop=True)
        except Exception as e:
            st.error(f"데이터 로드 오류: {e}")
            return pd.DataFrame()

    def get_optimized_section(self, Mu, Vu, L, max_deflection_ratio=360):
        Fy = 275 
        E = 205000
        Phi_b = 0.9
        
        valid_sections = []
        for _, row in self.data.iterrows():
            Mn = row['Zx'] * Fy
            if Mu > Phi_b * Mn: continue # 휨 강도

            delta = (5 * Mu * L**2) / (48 * E * row['Ix'])
            if delta > (L / max_deflection_ratio): continue # 처짐
            
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
            Pn = min(0.7 * Pe, row['A'] * Fy)

            if Pu <= Phi_c * Pn:
                valid_sections.append(row)
        
        if not valid_sections: return None
        return pd.DataFrame(valid_sections).sort_values(by='W').iloc[0]

def calculate_structure(db, t_slab, spacing, ll_kpa):
    # 상수
    L_X, L_Y, H_COL = 10000, 10000, 5000
    
    # 하중 산정
    wd_total = (t_slab * 24e-6) + 1.5e-3 # N/mm2
    wl_total = ll_kpa * 1e-3
    wu_area = 1.2 * wd_total + 1.6 * wl_total

    # 1. 작은보 (Beam)
    w_beam_lin = wu_area * spacing
    Mu_beam = (w_beam_lin * L_X**2) / 8
    Vu_beam = (w_beam_lin * L_X) / 2
    beam_mem = db.get_optimized_section(Mu_beam, Vu_beam, L_X)

    # 2. 큰보 (Girder)
    w_girder_lin = wu_area * (L_Y / 2) + 1.0 
    Mu_girder = (w_girder_lin * L_X**2) / 8
    Vu_girder = (w_girder_lin * L_X) / 2
    girder_mem = db.get_optimized_section(Mu_girder, Vu_girder, L_X)

    # 3. 기둥 (Column)
    Pu_col = wu_area * (L_X/2 * L_Y/2) * 1.1
    col_mem = db.get_column_section(Pu_col, H_COL)

    # 물량 산출
    if beam_mem is not None:
        num_beams = int(L_Y / spacing) - 1
        if num_beams < 0: num_beams = 0
        w_total = (num_beams * L_X/1000 * beam_mem['W']) + \
                  (2 * L_X/1000 * girder_mem['W']) + \
                  (4 * H_COL/1000 * col_mem['W'])
    else:
        w_total = 0
        num_beams = 0

    return {
        "beam": beam_mem, "girder": girder_mem, "col": col_mem,
        "Mu_b": Mu_beam, "Mu_g": Mu_girder, "Pu_c": Pu_col,
        "num_beams": num_beams, "total_weight": w_total,
        "wu": wu_area
    }

def draw_3d_plotly(Lx, Ly, H, spacing, res):
    fig = go.Figure()
    
    # Style function
    def add_line(x, y, z, color, name, width=5):
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z, mode='lines',
            line=dict(color=color, width=width), name=name, showlegend=False
        ))

    # 1. Columns (Red)
    cols_x = [0, Lx, Lx, 0]
    cols_y = [0, 0, Ly, Ly]
    for i in range(4):
        add_line([cols_x[i], cols_x[i]], [cols_y[i], cols_y[i]], [0, H], 'red', 'Column')

    # 2. Girders (Blue)
    add_line([0, Lx], [0, 0], [H, H], 'blue', 'Girder')
    add_line([0, Lx], [Ly, Ly], [H, H], 'blue', 'Girder')

    # 3. Beams (Green)
    curr_y = spacing
    while curr_y < Ly - 100:
        add_line([0, Lx], [curr_y, curr_y], [H, H], 'green', 'Beam', width=3)
        curr_y += spacing

    # 4. Slab (Transparent Surface)
    fig.add_trace(go.Mesh3d(
        x=[0, Lx, Lx, 0], y=[0, 0, Ly, Ly], z=[H, H, H, H],
        opacity=0.2, color='gray', name='Slab'
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title='X (mm)', yaxis_title='Y (mm)', zaxis_title='Z (mm)',
            aspectmode='data'
        ),
        margin=dict(r=0, l=0, b=0, t=0),
        height=500
    )
    return fig

# ==========================================
# 2. 메인 UI 구성 (Input vs Report)
# ==========================================

# --- [좌측: Input Frame] ---
with st.sidebar:
    st.header("1. 설계 조건 입력 (Input)")
    st.markdown("---")
    
    # 파일 업로드
    uploaded_file = st.file_uploader("H형강 DB 파일(csv) 업로드", type=['csv'])
    
    st.subheader("구조 제원 설정")
    t_slab = st.number_input("바닥 슬래브 두께 (mm)", min_value=100, max_value=300, value=150, step=10)
    spacing = st.number_input("작은보(Beam) 간격 (mm)", min_value=1000, max_value=5000, value=2500, step=100)
    ll_load = st.number_input("활하중 (kN/m²)", min_value=1.0, max_value=10.0, value=2.5, step=0.1)
    
    st.info("💡 팁: 작은보 간격을 조절하여 가장 경제적인(가벼운) 설계를 찾아보세요.")
    
    if st.button("설계 실행 (Run Design)"):
        st.session_state['run'] = True
    else:
        if 'run' not in st.session_state:
            st.session_state['run'] = False

# --- [우측: Report Frame] ---
st.title("🏗️ 자동화 철골 구조 설계 시스템")
st.markdown("단층 철골 구조물($10m \\times 10m$)의 최적 부재 선정 및 계산서 자동 생성")

if uploaded_file is not None and st.session_state['run']:
    # DB 로드 및 계산 수행
    db = SteelDB(uploaded_file)
    if not db.data.empty:
        res = calculate_structure(db, t_slab, spacing, ll_load)
        
        if res['beam'] is None:
            st.error("❌ 설계 실패: 하중이 너무 커서 DB 내 적절한 부재를 찾을 수 없습니다.")
        else:
            # Tab 구성
            tab1, tab2, tab3 = st.tabs(["📄 구조계산서", "📊 물량산출서", "🧊 3D 모델링"])
            
            # Tab 1: 구조계산서
            with tab1:
                st.subheader("1. 설계 하중 산정")
                st.latex(r"w_u = 1.2 \times DL + 1.6 \times LL")
                st.write(f" - 설계 등분포 하중 ($w_u$): **{res['wu']*1000:.2f} kN/m²**")
                
                st.divider()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("### 🔹 작은보 (Beam)")
                    st.success(f"**{res['beam']['Name']}**")
                    st.write(f"소요 모멘트: {res['Mu_b']/1e6:.1f} kN·m")
                    st.write(f"단위 중량: {res['beam']['W']} kg/m")
                    
                with col2:
                    st.markdown("### 🔹 큰보 (Girder)")
                    st.success(f"**{res['girder']['Name']}**")
                    st.write(f"소요 모멘트: {res['Mu_g']/1e6:.1f} kN·m")
                    st.write(f"단위 중량: {res['girder']['W']} kg/m")
                    
                with col3:
                    st.markdown("### 🔹 기둥 (Column)")
                    st.success(f"**{res['col']['Name']}**")
                    st.write(f"소요 축하중: {res['Pu_c']/1e3:.1f} kN")
                    st.write(f"단위 중량: {res['col']['W']} kg/m")

            # Tab 2: 물량산출서
            with tab2:
                st.subheader("총 강재 소요량 산출")
                
                bom_data = {
                    "구분": ["작은보 (Beam)", "큰보 (Girder)", "기둥 (Column)"],
                    "규격": [res['beam']['Name'], res['girder']['Name'], res['col']['Name']],
                    "개수 (EA)": [res['num_beams'], 2, 4],
                    "단위중량 (kg/m)": [res['beam']['W'], res['girder']['W'], res['col']['W']],
                    "길이/개소 (m)": [10.0, 10.0, 5.0]
                }
                df_bom = pd.DataFrame(bom_data)
                df_bom["총 중량 (kg)"] = df_bom["개수 (EA)"] * df_bom["단위중량 (kg/m)"] * df_bom["길이/개소 (m)"]
                
                st.dataframe(df_bom, use_container_width=True)
                
                total_ton = res['total_weight'] / 1000
                st.metric(label="총 철골 물량 (Total Weight)", value=f"{total_ton:.3f} Ton")
                
                if total_ton < 5.5:
                    st.balloons()
                    st.success("매우 경제적인 설계입니다!")

            # Tab 3: 3D 모델링
            with tab3:
                st.subheader("구조 프레임 3D 시각화")
                fig = draw_3d_plotly(10000, 10000, 5000, spacing, res)
                st.plotly_chart(fig, use_container_width=True)
                st.caption(f"적용 부재 - Beam: {res['beam']['Name']} | Column: {res['col']['Name']}")

    else:
        st.warning("CSV 파일을 확인해주세요.")
else:
    # 초기 화면 (파일 업로드 전)
    st.info("👈 왼쪽 사이드바에 CSV 파일을 업로드하고 설계 조건을 입력하세요.")
    st.markdown("""
    ### 사용 방법
    1. **Input:** 왼쪽 메뉴에서 H형강 DB 파일(`RH.csv`)을 업로드합니다.
    2. **Setting:** 슬래브 두께와 보 간격, 활하중을 설정합니다.
    3. **Run:** '설계 실행' 버튼을 누릅니다.
    4. **Output:** 오른쪽 화면에서 계산서와 3D 모델을 확인합니다.
    """)