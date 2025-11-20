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
        
    # 슬래브 (Slab)
    fig.add_trace(go.Mesh3d(
        x=[0, Lx, Lx, 0], y=[0, 0, Ly, Ly], z=[H, H, H, H], 
        opacity=0.2, color='gray', name='Slab'
    ))
    
    fig.update_layout(scene=dict(aspectmode='data'), height=600, margin=dict(l=0,r=0,t=0,b=0))
    return fig

# =========================================================
# 3. 메인 실행 구간 (UI)
# =========================================================

with st.sidebar:
    st.header("1. 설계 조건 입력")
    uploaded_file = st.file_uploader("H형강 DB(csv) 업로드", type=['csv'])
    st.subheader("제원 설정")
    t_slab = st.number_input("슬래브 두께 (mm)", 100, 300, 150, 10)
    spacing = st.number_input("작은보 간격 (mm)", 1000, 5000, 2500, 100)
    ll_load = st.number_input("활하중 (kN/m²)", 1.0, 10.0, 2.5, 0.1)
    run_btn = st.button("설계 실행 (Run)", type="primary")

st.title("🏗️ 철골 구조 시스템 자동 설계")

if uploaded_file is None:
    st.info("👈 왼쪽에서 **CSV 파일(RH.csv)**을 먼저 업로드해주세요.")
    st.stop()

db_data, error_msg = load_data_safe(uploaded_file)
if error_msg:
    st.error(f"❌ {error_msg}"); st.stop()

if not run_btn:
    st.info("👈 설정을 확인하고 **[설계 실행]** 버튼을 눌러주세요."); st.stop()

# ---------------------------------------------------------
# 4. 구조 계산 및 상세 보고서 생성
# ---------------------------------------------------------
try:
    # 상수 및 하중
    Lx, Ly, H_col = 10000, 10000, 5000
    wd = (t_slab * 24e-6) + 1.5e-3
    wl = ll_load * 1e-3
    wu = 1.2 * wd + 1.6 * wl

    # (1) 작은보 설계
    w_sb = wu * spacing
    M_sb = (w_sb * Ly**2) / 8
    V_sb = (w_sb * Ly) / 2
    sb_mem = find_best_section(db_data, M_sb, V_sb, Ly)

    # (2) 테두리보 설계
    w_eb = wu * (spacing/2) + 2.0
    M_eb = (w_eb * Ly**2) / 8
    V_eb = (w_eb * Ly) / 2
    eb_mem = find_best_section(db_data, M_eb, V_eb, Ly)

    # (3) 거더 설계
    w_g = wu * (Ly/2) + 1.5
    M_g = (w_g * Lx**2) / 8
    V_g = (w_g * Lx) / 2
    girder_mem = find_best_section(db_data, M_g, V_g, Lx)

    # (4) 기둥 설계
    Pu_c = (V_g + V_eb) * 1.1
    col_mem = find_column_section(db_data, Pu_c, H_col)

    # 결과 검증
    if any(x is None for x in [sb_mem, eb_mem, girder_mem, col_mem]):
        st.error("❌ 일부 부재 선정 실패! 하중을 줄이거나 DB를 확인하세요."); st.stop()

    st.balloons()
    tab1, tab2, tab3 = st.tabs(["📄 상세 구조계산서", "📊 물량산출서", "🧊 3D 모델링"])

    # --- [Tab 1] 상세 보고서 (Markdown) ---
    with tab1:
        st.header("1. 설계 하중 산정 근거")
        st.markdown(f"""
        - **고정하중(DL):** 콘크리트($24kN/m^3$) $\\times$ {t_slab}mm + 마감($1.5kN/m^2$) = **{wd*1000:.2f} kN/m²**
        - **활하중(LL):** 용도별 하중 적용 = **{wl*1000:.2f} kN/m²**
        - **계수하중(Wu):** $1.2 \\times DL + 1.6 \\times LL$ = **{wu*1000:.2f} kN/m²**
        """)
        st.markdown("---")
        
        # 부재별 상세 계산서 출력 함수
        def print_beam_calc(title, member, Mu, Vu, L_mm):
            # 재계산 (검토용)
            Fy, E = 275, 205000
            Phi_Mn = 0.9 * member['Zx'] * Fy / 1e6 # kNm
            Mu_kNm = Mu / 1e6
            ratio_M = (Mu_kNm / Phi_Mn) * 100
            
            # 처짐 재계산
            delta = (5 * Mu * L_mm**2) / (48 * E * member['Ix'])
            allow = L_mm / 360
            
            with st.container():
                st.subheader(f"📘 {title} 설계 ({member['Name']})")
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**1) 부재 제원 및 하중**")
                    st.write(f"- 부재명: {member['Name']}")
                    st.write(f"- 단면계수($Z_x$): {member['Zx']:.0f} $mm^3$")
                    st.write(f"- 단면2차모멘트($I_x$): {member['Ix']:.0f} $mm^4$")
                    st.write(f"- 계수 모멘트($M_u$): **{Mu_kNm:.2f} kN·m**")
                with c2:
                    st.markdown("**2) 안전성 검토 (Ratio)**")
                    st.write(f"- 설계강도($\\phi M_n$): {Phi_Mn:.2f} kN·m")
                    st.write(f"- **검토결과:** {ratio_M:.1f}% < 100% (OK)")
                    st.write(f"- 처짐($\\delta$): {delta:.1f}mm (허용 {allow:.1f}mm)")
                    if delta < allow: st.success("✅ 안전성 및 사용성 만족")
                    else: st.error("❌ 처짐 초과")
                st.markdown("---")

        print_beam_calc("작은보 (Small Beam)", sb_mem, M_sb, V_sb, Ly)
        print_beam_calc("테두리보 (Edge Beam)", eb_mem, M_eb, V_eb, Ly)
        print_beam_calc("거더 (Girder)", girder_mem, M_g, V_g, Lx)
        
        # 기둥 보고서
        st.subheader(f"📘 기둥 (Column) 설계 ({col_mem['Name']})")
        Fy, E = 275, 205000
        Iy_est = col_mem['Ix'] * 0.3
        Pe = (3.14159**2 * E * Iy_est) / (H_col**2)
        Phi_Pn = 0.9 * min(0.7 * Pe, col_mem['A'] * Fy) / 1e3 # kN
        Pu_kN = Pu_c / 1e3
        ratio_P = (Pu_kN / Phi_Pn) * 100
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**1) 부재 제원**")
            st.write(f"- 부재명: {col_mem['Name']}")
            st.write(f"- 단면적($A$): {col_mem['A']:.0f} $mm^2$")
            st.write(f"- 소요 축하중($P_u$): **{Pu_kN:.2f} kN**")
        with c2:
            st.markdown("**2) 안전성 검토**")
            st.write(f"- 좌굴고려 강도($\\phi P_n$): {Phi_Pn:.2f} kN")
            st.write(f"- **검토결과:** {ratio_P:.1f}% < 100% (OK)")
            st.success("✅ 기둥 안전성 만족")

    # --- [Tab 2] 물량산출서 ---
    with tab2:
        num_sb = int(Lx / spacing) - 1
        if num_sb < 0: num_sb = 0
        
        data = [
            ["거더 (Girder)", girder_mem['Name'], 2, 10.0, girder_mem['W']],
            ["테두리보 (Edge)", eb_mem['Name'], 2, 10.0, eb_mem['W']],
            ["작은보 (Small)", sb_mem['Name'], num_sb, 10.0, sb_mem['W']],
            ["기둥 (Column)", col_mem['Name'], 4, 5.0, col_mem['W']]
        ]
        df_bom = pd.DataFrame(data, columns=["구분", "규격", "수량(EA)", "길이(m)", "단위중량(kg/m)"])
        df_bom["총중량(kg)"] = df_bom["수량(EA)"] * df_bom["길이(m)"] * df_bom["단위중량(kg/m)"]
        
        st.dataframe(df_bom, use_container_width=True)
        st.metric("총 철골 소요량", f"{df_bom['총중량(kg)'].sum()/1000:.3f} Ton")

    # --- [Tab 3] 3D 모델링 ---
    with tab3:
        st.subheader("3D 구조 시각화 (Wireframe)")
        st.caption("🔵거더(X) | 🟠테두리보(Y) | 🟢작은보(Y) | 🔴기둥")
        fig = draw_3d_model(Lx, Ly, H_col, spacing, None)
        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"❌ 실행 중 오류가 발생했습니다: {e}")
