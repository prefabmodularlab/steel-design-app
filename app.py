import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ==========================================
# 1. 페이지 설정 (가장 윗부분에 있어야 함)
# ==========================================
st.set_page_config(page_title="철골 구조 자동 설계", layout="wide")

# ==========================================
# 2. 클래스 및 함수 정의 (Core Logic)
# ==========================================

class SteelDB:
    def __init__(self, uploaded_file):
        self.data = self.load_data(uploaded_file)

    def load_data(self, file):
        """
        CSV 파일을 읽고 정제하는 함수
        - 헤더 위치 보정 (Row 8 -> header=7)
        - 쉼표(,) 및 공백 제거
        - 유효한 데이터만 필터링
        """
        try:
            # 1. CSV 읽기 (헤더 위치 지정)
            df = pd.read_csv(file, header=7)
            df.columns = [str(c).strip() for c in df.columns]
            
            # 2. 필수 컬럼 확인
            if 'H' not in df.columns:
                st.error("❌ [데이터 오류] CSV 파일에 'H' 열이 없습니다. 올바른 규격 파일을 업로드해주세요.")
                return pd.DataFrame()

            # 3. 숫자 변환 유틸리티 (쉼표, 공백 제거)
            def to_num(series):
                return pd.to_numeric(series.astype(str).str.replace(',', '').str.strip(), errors='coerce')

            # 4. 데이터 정제 및 매핑
            clean_df = pd.DataFrame()
            
            # 호칭(Name)은 보통 3번째 컬럼(Index 2)에 위치함
            if len(df.columns) > 2:
                clean_df['Name'] = df.iloc[:, 2]
            else:
                clean_df['Name'] = "Unknown"

            # 주요 제원 변환
            clean_df['H'] = to_num(df['H'])
            clean_df['B'] = to_num(df['B'])
            clean_df['t1'] = to_num(df['t1'])
            clean_df['t2'] = to_num(df['t2'])
            clean_df['A'] = to_num(df['A']) * 100   # cm2 -> mm2
            clean_df['W'] = to_num(df['W'])         # kg/m
            clean_df['Ix'] = to_num(df['Ix']) * 10000 # cm4 -> mm4
            
            # Zx (소성단면계수) 처리 - 없으면 Sx(탄성) 사용, 둘 다 없으면 0
            if 'Zx' in df.columns:
                clean_df['Zx'] = to_num(df['Zx']) * 1000 # cm3 -> mm3
            elif 'Sx' in df.columns:
                clean_df['Zx'] = to_num(df['Sx']) * 1000
            else:
                clean_df['Zx'] = 0

            # H값이 유효한(숫자인) 행만 남김
            clean_df = clean_df[clean_df['H'].notnull()].reset_index(drop=True)
            
            return clean_df

        except Exception as e:
            st.error(f"❌ [데이터 로드 중 예외 발생] {e}")
            return pd.DataFrame()

    def get_optimized_section(self, Mu, Vu, L, max_deflection_ratio=360):
        """
        조건을 만족하는 최소 중량 부재 선정
        """
        Fy = 275  # MPa (SS275)
        E = 205000 # MPa
        Phi_b = 0.9
        
        valid_sections = []
        for _, row in self.data.iterrows():
            # 1. 휨 강도 검토
            Mn = row['Zx'] * Fy
            if Mu > Phi_b * Mn: continue 

            # 2. 처짐 검토 (등분포 하중 기준 약산)
            delta = (5 * Mu * L**2) / (48 * E * row['Ix'])
            if delta > (L / max_deflection_ratio): continue 
            
            valid_sections.append(row)

        if not valid_sections: return None
        # 무게(W) 오름차순 정렬 후 가장 가벼운 것 리턴
        return pd.DataFrame(valid_sections).sort_values(by='W').iloc[0]

    def get_column_section(self, Pu, L_unbraced):
        """
        기둥 부재 선정 (약축 좌굴 고려)
        """
        Fy = 275
        E = 205000
        Phi_c = 0.9
        
        valid_sections = []
        for _, row in self.data.iterrows():
            Iy_est = row['Ix'] * 0.3 # 약축 관성모멘트 약산 (Ix의 30%)
            Pe = (3.14159**2 * E * Iy_est) / (L_unbraced**2)
            Pn = min(0.7 * Pe, row['A'] * Fy) # 설계강도 약산식

            if Pu <= Phi_c * Pn:
                valid_sections.append(row)
        
        if not valid_sections: return None
        return pd.DataFrame(valid_sections).sort_values(by='W').iloc[0]


def calculate_structure(db, t_slab, spacing, ll_kpa):
    """
    구조 해석 및 부재 선정 메인 함수
    시스템: 거더(X), 테두리보(Y), 작은보(Y)
    """
    # 제원
    L_X = 10000 # mm (거더 길이)
    L_Y = 10000 # mm (빔 길이)
    H_COL = 5000 # mm
    
    # 하중
    wd_total = (t_slab * 24e-6) + 1.5e-3 # N/mm2
    wl_total = ll_kpa * 1e-3
    wu_area = 1.2 * wd_total + 1.6 * wl_total # 계수하중

    # 1. 작은보 (Small Beam) - Y방향, 거더 사이 배치
    w_sb = wu_area * spacing # N/mm
    Mu_sb = (w_sb * L_Y**2) / 8
    Vu_sb = (w_sb * L_Y) / 2
    sb_mem = db.get_optimized_section(Mu_sb, Vu_sb, L_Y)

    # 2. 테두리보 (Edge Beam) - Y방향, 양 끝단
    # 분담폭 절반 + 벽체하중(2.0kN/m 가정)
    w_eb = wu_area * (spacing / 2) + 2.0 
    Mu_eb = (w_eb * L_Y**2) / 8
    Vu_eb = (w_eb * L_Y) / 2
    eb_mem = db.get_optimized_section(Mu_eb, Vu_eb, L_Y)

    # 3. 거더 (Girder) - X방향, 기둥 강축 연결
    # 작은보 반력을 등분포로 치환하여 계산 (전체 하중의 절반 부담)
    w_g = wu_area * (L_Y / 2) + 1.5 # 자중 포함
    Mu_g = (w_g * L_X**2) / 8
    Vu_g = (w_g * L_X) / 2
    girder_mem = db.get_optimized_section(Mu_g, Vu_g, L_X)

    # 4. 기둥 (Column)
    # 거더 반력 + 테두리보 반력 + 자중 할증
    Pu_c = (Vu_g + Vu_eb) * 1.1
    col_mem = db.get_column_section(Pu_c, H_COL)

    # 물량 산출
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
    """
    3D 시각화 함수 (Plotly)
    """
    fig = go.Figure()
    
    def add_line(x, y, z, color, name, width=5):
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z, mode='lines',
            line=dict(color=color, width=width), name=name, showlegend=False
        ))

    # 기둥 (Red)
    cols_x, cols_y = [0, Lx, Lx, 0], [0, 0, Ly, Ly]
    for i in range(4):
        add_line([cols_x[i], cols_x[i]], [cols_y[i], cols_y[i]], [0, H], 'red', 'Column', 8)

    # 거더 (Blue, X-Dir)
    add_line([0, Lx], [0, 0], [H, H], 'blue', 'Girder', 6)
    add_line([0, Lx], [Ly, Ly], [H, H], 'blue', 'Girder', 6)

    # 테두리보 (Orange, Y-Dir, Edges)
    add_line([0, 0], [0, Ly], [H, H], 'orange', 'Edge Beam', 5)
    add_line([Lx, Lx], [0, Ly], [H, H], 'orange', 'Edge Beam', 5)

    # 작은보 (Green, Y-Dir, Inner)
    curr_x = spacing
    while curr_x < Lx - 100:
        add_line([curr_x, curr_x], [0, Ly], [H, H], 'green', 'Small Beam', 3)
        curr_x += spacing

    # 슬래브 (Surface)
    fig.add_trace(go.Mesh3d(x=[0, Lx, Lx, 0], y=[0, 0, Ly, Ly], z=[H, H, H, H], 
                            opacity=0.2, color='gray', name='Slab'))

    fig.update_layout(scene=dict(aspectmode='data'), height=600, margin=dict(t=0,b=0,l=0,r=0))
    return fig

# ==========================================
# 3. 메인 UI 및 실행 로직 (Debugging Mode)
# ==========================================

# --- 사이드바: 입력 ---
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

# --- 메인 화면: 출력 ---
st.title("🏗️ 철골 구조 시스템 자동 설계")
st.markdown("거더(X축) - 테두리보(Y축) - 작은보(Y축) 시스템 최적화")

# [Step 1] 파일 체크
if uploaded_file is None:
    st.info("👈 **[Step 1]** 왼쪽 사이드바에서 **CSV 파일(RH.csv)**을 업로드해주세요.")
    st.stop()

# [Step 2] 데이터 로드 체크
try:
    db = SteelDB(uploaded_file)
    if db.data.empty:
        st.error("❌ 데이터 로드 실패: 파일 내용이 비어있거나 형식이 맞지 않습니다.")
        st.stop()
    else:
        with st.expander("✅ 데이터 로드 성공 (클릭하여 내용 확인)"):
            st.dataframe(db.data.head())
except Exception as e:
    st.error(f"❌ 치명적 오류 발생: {e}")
    st.stop()

# [Step 3] 실행 버튼 체크
if 'run' not in st.session_state or not st.session_state['run']:
    st.info("👈 **[Step 2]** 설정을 확인하고 **[설계 실행]** 버튼을 눌러주세요.")
    st.stop()

# [Step 4] 구조 계산 및 결과 출력
try:
    res = calculate_structure(db, t_slab, spacing, ll_load)
    
    # 부재 선정 실패 체크
    if res['sb'] is None: st.error("❌ [설계 실패] 작은보(Small Beam) 선정 불가 (하중 과다)")
    elif res['eb'] is None: st.error("❌ [설계 실패] 테두리보(Edge Beam) 선정 불가")
    elif res['girder'] is None: st.error("❌ [설계 실패] 거더(Girder) 선정 불가")
    elif res['col'] is None: st.error("❌ [설계 실패] 기둥(Column) 선정 불가")
    
    else:
        # 성공 시 화면 출력
        st.balloons()
        
        tab1, tab2, tab3 = st.tabs(["📄 구조계산서", "📊 물량산출서", "🧊 3D 모델링"])
        
        with tab1:
            st.subheader("설계 하중 및 부재 선정 결과")
            st.info(f"설계 등분포 하중 ($w_u$): **{res['wu']*1000:.2f} kN/m²**")
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### 🔹 거더 (Girder / X-Dir)")
                st.success(f"**{res['girder']['Name']}**")
                st.caption(f"W: {res['girder']['W']} kg/m | Mu: {res['Mu_g']/1e6:.1f} kN·m")

                st.markdown("#### 🔹 작은보 (Small Beam / Y-Dir)")
                st.success(f"**{res['sb']['Name']}**")
                st.caption(f"W: {res['sb']['W']} kg/m | Mu: {res['Mu_sb']/1e6:.1f} kN·m")
            
            with c2:
                st.markdown("#### 🔹 테두리보 (Edge Beam / Y-Dir)")
