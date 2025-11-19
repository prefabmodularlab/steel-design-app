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
        오류 방지를 위해 쉼표 제거, 공백 제거, 헤더 위치 지정을 수행합니다.
        """
        try:
            # [핵심 수정 1] 헤더 위치를 8번째 줄(Index 7)로 지정 (사용자 파일 형식 맞춤)
            df = pd.read_csv(file, header=7)
            
            # 컬럼 이름의 앞뒤 공백 제거
            df.columns = [str(c).strip() for c in df.columns]
            
            # 파일 형식 검증 ('H' 컬럼 존재 여부)
            if 'H' not in df.columns:
                st.error("❌ 오류: CSV 파일 형식이 맞지 않습니다. 'H' 열을 찾을 수 없습니다.")
                return pd.DataFrame()

            # [핵심 수정 2] "1,021 " 같은 문자열을 숫자로 변환하는 함수 정의
            def to_num(series):
                # 1. 문자로 변환 -> 2. 쉼표(,) 제거 -> 3. 공백 제거 -> 4. 숫자로 변환
                return pd.to_numeric(series.astype(str).str.replace(',', '').str.strip(), errors='coerce')

            # 데이터 정리 시작
            clean_df = pd.DataFrame()
            
            # 호칭(Name)은 별도 헤더가 없는 경우가 많으므로 위치(3번째 열)로 가져옴
            clean_df['Name'] = df.iloc[:, 2]
            
            # 주요 제원 숫자로 변환 (쉼표 오류 해결)
            clean_df['H'] = to_num(df['H'])
            clean_df['B'] = to_num(df['B'])
            clean_df['t1'] = to_num(df['t1'])
            clean_df['t2'] = to_num(df['t2'])
            
            # 단위 변환 및 매핑
            clean_df['A'] = to_num(df['A']) * 100   # cm2 -> mm2
            clean_df['W'] = to_num(df['W'])         # kg/m
            clean_df['Ix'] = to_num(df['Ix']) * 10000 # cm4 -> mm4
            
            # 소성단면계수(Zx) 처리 (없으면 탄성단면계수 Sx 사용)
            if 'Zx' in df.columns:
                clean_df['Zx'] = to_num(df['Zx']) * 1000 # cm3 -> mm3
            elif 'Sx' in df.columns:
                clean_df['Zx'] = to_num(df['Sx']) * 1000
            else:
                clean_df['Zx'] = 0

            # H값이 유효한(숫자인) 행만 남기고 필터링
            clean_df = clean_df[clean_df['H'].notnull()].reset_index(drop=True)

            return clean_df

        except Exception as e:
            st.error(f"데이터 로드 중 알 수 없는 오류가 발생했습니다: {e}")
            return pd.DataFrame()

    def get_optimized_section(self, Mu, Vu, L, max_deflection_ratio=360):
        # 재료 물성치 (SS275)
        Fy = 275  # MPa
        E = 205000 # MPa
        Phi_b = 0.9 # 휨 강도감소계수
        
        valid_sections = []
        
        # 모든 부재를 순회하며 검토
        for _, row in self.data.iterrows():
            # 1. 휨 강도 검토 (Mu <= Phi * Mn)
            Mn = row['Zx'] * Fy
            if Mu > Phi_b * Mn: continue 

            # 2. 처짐 검토 (Delta <= L/360) - 등분포하중 가정 약산식
            # Delta = 5 * w * L^4 / (384 * E * I) = 5 * M * L^2 / (48 * E * I)
            delta = (5 * Mu * L**2) / (48 * E * row['Ix'])
            if delta > (L / max_deflection_ratio): continue 
            
            valid_sections.append(row)

        if not valid_sections: return None
        
        # 조건을 만족하는 부재 중 '단위중량(W)'이 가장 작은 것 반환
        return pd.DataFrame(valid_sections).sort_values(by='W').iloc[0]

    def get_column_section(self, Pu, L_unbraced):
        Fy = 275
        E = 205000
        Phi_c = 0.9
        
        valid_sections = []
        for _, row in self.data.iterrows():
            # 기둥 좌굴 검토 (약축 Ix의 30% 가정)
            Iy_est = row['Ix'] * 0.3 
            Pe = (3.14159**2 * E * Iy_est) / (L_unbraced**2)
            
            # 설계강도 Pn (탄성좌굴의 70%로 제한하여 안전측 검토)
            Pn = min(0.7 * Pe, row['A'] * Fy)

            if Pu <= Phi_c * Pn:
                valid_sections.append(row)
        
        if not valid_sections: return None
        return pd.DataFrame(valid_sections).sort_values(by='W').iloc[0]


def calculate_structure(db, t_slab, spacing, ll_kpa):
    # 구조물 제원
    L_X, L_Y, H_COL = 10000, 10000, 5000 # mm
    
    # 하중 산정 (단위: N, mm)
    # 고정하중 = (두께 * 콘크리트단위중량) + 마감하중
    wd_total = (t_slab * 24e-6) + 1.5e-3 # N/mm2
    # 활하중
    wl_total = ll_kpa * 1e-3 # N/mm2
    # 계수하중 (1.2D + 1.6L)
    wu_area = 1.2 * wd_total + 1.6 * wl_total 

    # 1. 작은보 (Beam) 설계 - 등분포 하중
    w_beam_lin = wu_area * spacing # N/mm
    Mu_beam = (w_beam_lin * L_X**2) / 8
    Vu_beam = (w_beam_lin * L_X) / 2
    beam_mem = db.get_optimized_section(Mu_beam, Vu_beam, L_X)

    # 2. 큰보 (Girder) 설계 - 등분포 치환 + 자중 가정
    w_girder_lin = wu_area * (L_Y / 2) + 1.0 
    Mu_girder = (w_girder_lin * L_X**2) / 8
    Vu_girder = (w_girder_lin * L_X) / 2
    girder_mem = db.get_optimized_section(Mu_girder, Vu_girder, L_X)

    # 3. 기둥 (Column) 설계 - 축하중
    Pu_col = wu_area * (L_X/2 * L_Y/2) * 1.1 # 자중 10% 할증
    col_mem = db.get_column_section(Pu_col, H_COL)

    # 물량 산출 (BOM)
    if beam_mem is not None and girder_mem is not None and col_mem is not None:
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
    
    def add_line(x, y, z, color, name, width=5):
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z, mode='lines',
            line=dict(color=color, width=width), name=name, showlegend=False
        ))

    # 1. 기둥 (Columns) - 빨간색
    cols_x = [0, Lx, Lx, 0]
    cols_y = [0, 0, Ly, Ly]
    for i in range(4):
        add_line([cols_x[i], cols_x[i]], [cols_y[i], cols_y[i]], [0, H], 'red', 'Column')

    # 2. 큰보 (Girders) - 파란색
    add_line([0, Lx], [0, 0], [H, H], 'blue', 'Girder')
    add_line([0, Lx], [Ly, Ly], [H, H], 'blue', 'Girder')

    # 3. 작은보 (Beams) - 초록색
    curr_y = spacing
    while curr_y < Ly - 100: # 오차 고려하여 범위 설정
        add_line([0, Lx], [curr_y, curr_y], [H, H], 'green', 'Beam', width=3)
        curr_y += spacing

    # 4. 슬래브 (반투명 회색 면)
    fig.add_trace(go.Mesh3d(
        x=[0, Lx, Lx, 0], y=[0, 0, Ly, Ly], z=[H, H, H, H],
        opacity=0.2, color='gray', name='Slab'
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title='X (mm)', yaxis_title='Y (mm)', zaxis_title='Z (mm)',
            aspectmode='data' # 실제 비율 유지
        ),
        margin=dict(r=0, l=0, b=0, t=0),
        height=500
    )
    return fig

# ==========================================
# 3. 메인 UI 구성 (Streamlit)
# ==========================================

# --- [좌측 사이드바: Input] ---
with st.sidebar:
    st.header("1. 설계 조건 입력 (Input)")
    st.markdown("---")
    
    # 파일 업로드
    uploaded_file = st.file_uploader("H형강 DB 파일(csv) 업로드", type=['csv'])
    
    st.subheader("구조 제원 설정")
    t_slab = st.number_input("바닥 슬래브 두께 (mm)", min_value=100, max_value=300, value=150, step=10)
    spacing = st.number_input("작은보(Beam) 간격 (mm)", min_value=1000, max_value=5000, value=2500, step=100)
    ll_load = st.number_input("활하중 (kN/m²)", min_value=1.0, max_value=10.0, value=2.5, step=0.1)
    
    # 버튼 상태 관리
    if st.button("설계 실행 (Run Design)", type="primary"):
        st.session_state['run'] = True

# --- [우측 메인 화면: Report] ---
st.title("🏗️ 자동화 철골 구조 설계 시스템")
st.markdown("1-Bay($10m \\times 10m$) 단층 철골 구조 최적 설계 프로젝트")

if uploaded_file is not None:
    # 버튼이 눌렸거나 이미 실행된 상태일 때
    if 'run' in st.session_state and st.session_state['run']:
        
        # DB 로드
        db = SteelDB(uploaded_file)
        
        # 데이터가 정상적으로 로드되었을 때만 계산 진행
        if not db.data.empty:
            res = calculate_structure(db, t_slab, spacing, ll_load)
            
            # 부재 선정 실패 여부 확인
            if res['beam'] is None or res['girder'] is None or res['col'] is None:
                st.error("❌ 설계 실패: 하중이 너무 커서 DB 내의 부재로는 안전성을 확보할 수 없습니다. (하중을 줄이거나 DB를 보강하세요)")
            else:
                # 탭 생성
                tab1, tab2, tab3 = st.tabs(["📄 구조계산서", "📊 물량산출서", "🧊 3D 모델링"])
                
                # [Tab 1] 구조계산서
                with tab1:
                    st.subheader("1. 설계 하중 산정")
                    st.latex(r"w_u = 1.2 \times DL + 1.6 \times LL")
                    st.info(f"설계 등분포 하중 ($w_u$): **{res['wu']*1000:.2f} kN/m²**")
                    
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

                # [Tab 2] 물량산출서
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
                    
                    if total_ton < 6.0:
                        st.balloons()
                        st.success("매우 경제적인 설계입니다!")

                # [Tab 3] 3D 모델링
                with tab3:
                    st.subheader("구조 프레임 3D 시각화")
                    fig = draw_3d_plotly(10000, 10000, 5000, spacing, res)
                    st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.info("👈 왼쪽 사이드바에서 설정을 마치고 [설계 실행] 버튼을 눌러주세요.")
else:
    st.warning("👈 먼저 왼쪽 사이드바에서 CSV 파일(RH.csv)을 업로드해주세요.")
