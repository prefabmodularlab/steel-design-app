import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# 1. 페이지 설정 (반드시 맨 윗줄)
st.set_page_config(page_title="철골 구조 자동 설계", layout="wide")

# ==========================================
# 2. 클래스 및 함수 정의
# ==========================================

class SteelDB:
    def __init__(self, uploaded_file):
        self.data = self.load_data(uploaded_file)

    def load_data(self, file):
        try:
            # 헤더 8번째 줄 (Index 7)
            df = pd.read_csv(file, header=7)
            
            # 컬럼명 공백 제거
            df.columns = [str(c).strip() for c in df.columns]
            
            if 'H' not in df.columns:
                return pd.DataFrame() # 빈 데이터프레임

            # 숫자 변환 (쉼표, 공백 제거)
            def to_num(series):
                return pd.to_numeric(series.astype(str).str.replace(',', '').str.strip(), errors='coerce')

            clean_df = pd.DataFrame()
            
            # 호칭(Name)
            if len(df.columns) > 2:
                clean_df['Name'] = df.iloc[:, 2]
            else:
                clean_df['Name'] = "Unknown"

            # 물성치 매핑
            clean_df['H'] = to_num(df['H'])
            clean_df['B'] = to_num(df['B'])
            clean_df['t1'] = to_num(df['t1'])
            clean_df['t2'] = to_num(df['t2'])
            clean_df['A'] = to_num(df['A']) * 100   # cm2 -> mm2
            clean_df['W'] = to_num(df['W'])         # kg/m
            clean_df['Ix'] = to_num(df['Ix']) * 10000 # cm4 -> mm4
            
            # Zx 처리
            if 'Zx' in df.columns:
                clean_df['Zx'] = to_num(df['Zx']) * 1000
            elif 'Sx' in df.columns:
                clean_df['Zx'] = to_num(df['Sx']) * 1000
            else:
                clean_df['Zx'] = 0

            # 유효 데이터 필터링
            clean_df = clean_df[clean_df['H'].notnull()].reset_index(drop=True)
            return clean_df

        except Exception as e:
            st.error(f"데이터 로드 오류: {e}")
            return pd.DataFrame()

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
            
            # [이 부분이 잘렸던 곳입니다]
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

    # 1. 작은보 (Small Beam) - Y방향, 내부
    w_sb = wu_area * spacing
    Mu_sb = (w_sb * L_Y**2) / 8
    Vu_sb = (w_sb * L_Y) / 2
    sb_mem = db.get_optimized_section(Mu_sb, Vu_sb, L_Y)

    # 2. 테두리보 (Edge Beam) - Y방향, 양 끝단
    w_eb = wu_area * (spacing / 2) + 2.0 
    Mu_eb = (w_eb * L_Y**2) / 8
    Vu_eb = (w_eb * L_Y) / 2
    eb_mem = db.get_optimized_section(Mu_eb, Vu_eb, L_Y)

    # 3. 거더 (Girder) - X방향
    w_g = wu_area * (L_Y / 2) + 1.5
    Mu_g = (w_g * L_X**2) / 8
    Vu_g = (w_g * L_X) / 2
    girder_mem = db.get_optimized_section(Mu_g, Vu_g, L_X)

    # 4. 기둥 (Column)
    Pu_c = (Vu_g + Vu_eb) * 1.1
    col_mem = db.get_column_section(Pu_c, H_COL)

    # 물량 산출
    num_sb = int(L_X / spacing) - 1
    if num_sb < 0: num_sb = 0
    
    w_total = 0
    if all([sb_mem is not None, eb_mem is not None, girder_mem is not None, col_mem is not None]):
        w_total = (num_sb * L_Y/1000 * sb_mem['W']) + \
                  (2 * L_Y/1000 * eb_mem['W']) + \
                  (2 * L_X/1000 * girder_mem['W']) + \
                  (4 * H_COL/1000 * col_mem['W'])

    return {
        "sb": sb_mem, "eb": eb_mem, "girder": girder_mem, "col": col_mem,
        "Mu_sb": Mu_sb, "Mu_eb": Mu_eb, "Mu_g": Mu_g, "Pu_c": Pu_c,
        "num_beams": num_sb, "total_weight": w_total,
        "wu": wu_area
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

    fig.update_layout(scene=dict(aspectmode='data'), height=500, margin=dict(t=0,b=0,l=0,r=0))
    return fig

# ==========================================
# 3. 메인 UI 및 실행 로직
# ==========================================

with st.sidebar:
    st.header("1. 설계 조건 입력")
    uploaded_file = st.file_uploader("H형강 DB(csv) 업로드", type=['csv'])
    
    st.subheader("제원 설정")
    t_slab = st.number_input("슬래브 두께 (mm)", 100, 300, 150, 10)
    spacing = st.number_input("작은보 간격 (mm)", 1000, 5000, 2500, 100)
    ll_load = st.number_input("활하중 (kN/m²)", 1.0, 10.0, 2.5, 0.1)
    
    if st.button("설계 실행 (Run)", type="primary"):
        st.session_state['run'] = True

st.title("🏗️ 철골 구조 시스템 자동 설계")
st.markdown("거더(X) - 테두리보(Y) - 작은보(Y) 시스템")

# Step 1: 파일 확인
if uploaded_file is None:
    st.info("👈 왼쪽 사이드바에서 **CSV 파일(RH.csv)**을 업로드해주세요.")
    st.stop()

# Step 2: 데이터 로드
db = SteelDB(uploaded_file)
if db.data.empty:
    st.error("❌ 데이터 로드 실패. 파일 형식을 확인해주세요.")
    st.stop()

# Step 3: 실행 버튼 확인
if 'run' not in st.session_state or not st.session_state['run']:
    st.info("👈 설정을 확인하고 **[설계 실행]** 버튼을 눌러주세요.")
    st.stop()

# Step 4: 계산 및 출력
res = calculate_structure(db, t_slab, spacing, ll_load)

if res['sb'] is None: st.error("❌ 작은보 선정 실패"); st.stop()
if res['eb'] is None: st.error("❌ 테두리보 선정 실패"); st.stop()
if res['girder'] is None: st.error("❌ 거더 선정 실패"); st.stop()
if res['col'] is None: st.error("❌ 기둥 선정 실패"); st.stop()

st.balloons()

tab1, tab2, tab3 = st.tabs(["📄 구조계산서", "📊 물량산출서", "🧊 3D 모델링"])

with tab1:
    st.subheader("설계 하중 및 부재 선정 결과")
    st.info(f"설계 등분포 하중 ($w_u$): **{res['wu']*1000:.2f} kN/m²**")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 🔹 거더 (Girder)")
        st.success(f"**{res['girder']['Name']}**")
        st.write(f"Mu: {res['Mu_g']/1e6:.1f} kN·m | W: {res['girder']['W']} kg/m")
        
        st.markdown("#### 🔹 작은보 (Small Beam)")
        st.success(f"**{res['sb']['Name']}**")
        st.write(f"Mu: {res['Mu_sb']/1e6:.1f} kN·m | W: {res['sb']['W']} kg/m")

    with c2:
        st.markdown("#### 🔹 테두리보 (Edge Beam)")
        st.success(f"**{res['eb']['Name']}**")
        st.write(f"Mu: {res['Mu_eb']/1e6:.1f} kN·m | W: {res['eb']['W']} kg/m")

        st.markdown("#### 🔹 기둥 (Column)")
        st.success(f"**{res['col']['Name']}**")
        st.write(f"Pu: {res['Pu_c']/1e3:.1f} kN | W: {res['col']['W']} kg/m")

with tab2:
    st.subheader("총 철골 물량 (BOM)")
    bom_data = {
        "구분": ["거더", "테두리보", "작은보", "기둥"],
        "규격": [res['girder']['Name'], res['eb']['Name'], res['sb']['Name'], res['col']['Name']],
        "수량(EA)": [2, 2, res['num_beams'], 4],
        "단위중량(kg/m)": [res['girder']['W'], res['eb']['W'], res['sb']['W'], res['col']['W']],
        "길이(m)": [10.0, 10.0, 10.0, 5.0]
    }
    df_bom = pd.DataFrame(bom_data)
    df_bom["총중량(kg)"] = df_bom["수량(EA)"] * df_bom["단위중량(kg/m)"] * df_bom["길이(m)"]
    
    st.dataframe(df_bom, use_container_width=True)
    total_ton = res['total_weight'] / 1000
    st.metric("총 철골 소요량", f"{total_ton:.3f} Ton")

with tab3:
    st.subheader("3D Wireframe View")
    st.caption("Blue: Girder | Orange: Edge Beam | Green: Small Beam")
    fig = draw_3d_plotly(10000, 10000, 5000, spacing, res)
    st.plotly_chart(fig, use_container_width=True)
