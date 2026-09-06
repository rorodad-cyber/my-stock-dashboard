import datetime
from io import StringIO
import warnings
from bs4 import BeautifulSoup
import FinanceDataReader as fdr
import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st
import numpy as np
import platform

warnings.filterwarnings('ignore')

# 1. 깃허브(리눅스) 및 모바일 호환을 위한 폰트 자동 설정
system_name = platform.system()
if system_name == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif system_name == 'Darwin': # Mac OS
    plt.rcParams['font.family'] = 'AppleGothic'
else: # 깃허브(Streamlit Cloud) 등 Linux 환경
    plt.rcParams['font.family'] = 'NanumGothic'

plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 데이터 크롤링 함수들
# ==========================================
@st.cache_data(ttl=3600)
def get_golden_cross_stocks():
    url = 'https://finance.naver.com/sise/item_gold.naver'
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        res = requests.get(url, headers=headers)
        soup = BeautifulSoup(res.content, 'html.parser')
        stocks = []
        table = soup.find('table', {'class': 'type_5'})
        if table:
            rows = table.find_all('tr')
            for row in rows:
                a_tag = row.find('a', class_='tltle')
                if a_tag:
                    name = a_tag.text.strip()
                    code = a_tag['href'].split('code=')[-1]
                    stocks.append({'Name': name, 'Code': code})
        return pd.DataFrame(stocks)
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=86400)
def get_all_krx_stocks():
    df = fdr.StockListing('KRX')
    return df[['Name', 'Code']]

def get_company_summary(stock_code):
    url = f'https://finance.naver.com/item/main.naver?code={stock_code}'
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        res = requests.get(url, headers=headers)
        soup = BeautifulSoup(res.content, 'html.parser')
        summary_info = soup.find('div', class_='summary_info')
        if summary_info:
            paragraphs = summary_info.find_all('p')
            if paragraphs:
                return '\n\n'.join([p.text.strip() for p in paragraphs])
        return '해당 기업의 사업 요약 정보를 찾을 수 없습니다.'
    except Exception:
        return '기업 정보를 가져오는 데 실패했습니다.'

def get_financial_data(stock_code):
    headers = {'User-Agent': 'Mozilla/5.0'}
    url = f'https://finance.naver.com/item/main.naver?code={stock_code}'
    try:
        res = requests.get(url, headers=headers)
        soup = BeautifulSoup(res.content, 'html.parser')
        dfs = pd.read_html(StringIO(str(soup)))
        for df in dfs:
            if not df.empty and any('매출액' in str(val) for val in df.iloc[:, 0].values):
                df.set_index(df.columns[0], inplace=True)
                df_annual = df['최근 연간 실적'].T
                cols_to_keep = ['매출액', '영업이익', 'PER(배)']
                available_cols = [col for col in cols_to_keep if col in df_annual.columns]
                df_res = df_annual[available_cols].copy()
                for col in df_res.columns:
                    df_res[col] = pd.to_numeric(df_res[col].astype(str).str.replace(',', ''), errors='coerce')
                return df_res
    except Exception:
        pass
    return None

def get_technical_data(stock_code):
    today = datetime.datetime.today()
    start_date = (today - datetime.timedelta(days=365)).strftime('%Y-%m-%d')
    try:
        df = fdr.DataReader(stock_code, start_date, today.strftime('%Y-%m-%d'))
        if len(df) < 50:
            return None
        # 기본 차트용 지표
        df['MA20'] = df['Close'].rolling(20).mean()
        df['Envelope_Lower'] = df['MA20'] * 0.85
        # 돌파 차트용 지표
        df['MA50'] = df['Close'].rolling(50).mean()
        df['High20'] = df['High'].shift(1).rolling(20).max()
        return df
    except Exception:
        return None

# ==========================================
# ★ 전문가용: 5가지 돌파 전략 심층 분석 함수
# ==========================================
def analyze_breakout_strategies_detailed(df):
    if df is None or len(df) < 50:
        return None, 0

    df_strat = df.copy()
    
    df_strat['ma_50'] = df_strat['Close'].rolling(window=50).mean()
    df_strat['high_20'] = df_strat['High'].shift(1).rolling(window=20).max()
    
    df_strat['tr0'] = df_strat['High'] - df_strat['Low']
    df_strat['tr1'] = abs(df_strat['High'] - df_strat['Close'].shift(1))
    df_strat['tr2'] = abs(df_strat['Low'] - df_strat['Close'].shift(1))
    df_strat['tr'] = df_strat[['tr0', 'tr1', 'tr2']].max(axis=1)
    df_strat['atr_10'] = df_strat['tr'].rolling(window=10).mean()

    latest = df_strat.iloc[-1]
    prev = df_strat.iloc[-2]
    d2 = df_strat.iloc[-3]
    
    close_price = latest['Close']
    
    details = []
    score = 0
    
    # 1. 50일선 돌파
    ma50_val = latest['ma_50']
    if (close_price > ma50_val) and (prev['Close'] <= prev['ma_50']):
        details.append(("1. 50일선 돌파", True, f"오늘 종가({close_price:,.0f}원)가 50일선({ma50_val:,.0f}원)을 뚫고 올라왔습니다."))
        score += 1
    else:
        details.append(("1. 50일선 돌파", False, f"종가({close_price:,.0f}원) 기준, 50일선({ma50_val:,.0f}원)을 새롭게 돌파하지 못했습니다."))
        
    # 2. 변동성 조절 돌파
    atr_ratio = (latest['atr_10'] / close_price) * 100 
    dynamic_period = max(5, min(int(atr_ratio * 5), 40))
    dyn_ma = df_strat['Close'].rolling(window=dynamic_period).mean().iloc[-1]
    prev_dyn_ma = df_strat['Close'].rolling(window=dynamic_period).mean().iloc[-2]
    if (close_price > dyn_ma) and (prev['Close'] <= prev_dyn_ma):
        details.append(("2. ATR 변동성 돌파", True, f"종가({close_price:,.0f}원)가 동적저항선({dyn_ma:,.0f}원, {dynamic_period}일)을 돌파했습니다."))
        score += 1
    else:
        details.append(("2. ATR 변동성 돌파", False, f"종가({close_price:,.0f}원)가 동적저항선({dyn_ma:,.0f}원)을 돌파하지 못했습니다."))
        
    # 3. 전고점 돌파
    high20_val = latest['high_20']
    if close_price > high20_val:
        details.append(("3. 20일 전고점 돌파", True, f"최근 20일 최고가({high20_val:,.0f}원)를 오늘 종가가 넘어섰습니다!"))
        score += 1
    else:
        details.append(("3. 20일 전고점 돌파", False, f"최근 20일 최고가({high20_val:,.0f}원)의 저항을 뚫지 못했습니다."))
        
    # 4. ORB (시가 레인지) 돌파
    prev_range = prev['High'] - prev['Low']
    orb_threshold = latest['Open'] + (prev_range * 0.3)
    if close_price > orb_threshold:
        details.append(("4. ORB(시가) 돌파", True, f"장중 강한 매수세로 시가기준 저항선({orb_threshold:,.0f}원)을 뚫었습니다."))
        score += 1
    else:
        details.append(("4. ORB(시가) 돌파", False, f"시가돌파 기준선({orb_threshold:,.0f}원)을 넘지 못했습니다."))
        
    # 5. 인사이드 데이 돌파
    is_inside_day = (prev['High'] < d2['High']) and (prev['Low'] > d2['Low'])
    if is_inside_day and (close_price > prev['High']):
        details.append(("5. 인사이드데이 돌파", True, f"어제 응축된 변동성(고점 {prev['High']:,.0f}원)이 오늘 상향 폭발했습니다!"))
        score += 1
    else:
        details.append(("5. 인사이드데이 돌파", False, "전일 변동성 수축(인사이드 데이) 후 상향 돌파 패턴이 아닙니다."))

    return details, score


# ==========================================
# 2. 차트 그리기 함수 모음
# ==========================================
# (1번 섹션용) 일반 기술적 분석 차트
def draw_technical_chart(df, stock_name):
    fig, ax = plt.subplots(figsize=(10, 5))
    close_val = df['Close'].iloc[-1]
    stop_val = df['MA20'].iloc[-1] * 0.98 if df['Close'].iloc[-1] > df['MA20'].iloc[-1] else df['Low'].iloc[-1] * 0.97
    loss_pct = ((stop_val - close_val) / close_val) * 100

    ax.plot(df.index, df['Close'], label='종가', color='black', linewidth=1.5)
    ax.plot(df.index, df['MA20'], label='20일선', color='magenta', linestyle='-.')
    ax.plot(df.index, df['Envelope_Lower'], label='엔벨로프 하단(-15%)', color='green', linestyle=':')

    last_date = df.index[-1]
    ax.scatter(last_date, close_val, color='red', marker='*', s=200, label='현재위치', zorder=5)
    ax.axhline(stop_val, color='red', linestyle='--', alpha=0.7, label='참고 손절선', linewidth=1.2)

    ax.text(last_date, stop_val, f' 손절가: {int(stop_val):,}원 ({loss_pct:.1f}%)', color='white', fontsize=10, fontweight='bold', verticalalignment='center', bbox=dict(boxstyle='round,pad=0.3', facecolor='crimson', alpha=0.85, edgecolor='none'))
    ax.text(last_date, close_val, f' 현재가: {int(close_val):,}원', color='black', fontsize=9, fontweight='bold', verticalalignment='bottom')

    ax.set_title(f'[{stock_name}] 기본 차트 (매수/손절 타점)', fontsize=15, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    return fig

# (2번 섹션용) 재무 실적 차트
def draw_fundamental_chart(df, stock_name):
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={'height_ratios': [2, 1]})
    x_labels = df.index.get_level_values(0).astype(str) if isinstance(df.index, pd.MultiIndex) else df.index.astype(str)

    ax1.bar(x_labels, df['매출액'], color='#4a90e2', alpha=0.7, label='매출액(억원)', width=0.4)
    ax1.set_ylabel('매출액 (억원)', color='#4a90e2', fontweight='bold')
    ax1.set_title(f'[{stock_name}] 기업 실적 및 PER 분석', fontsize=15, fontweight='bold', pad=15)

    ax2 = ax1.twinx()
    ax2.plot(x_labels, df['영업이익'], color='#d0021b', marker='o', linewidth=2.5, markersize=8, label='영업이익(억원)')
    ax2.set_ylabel('영업이익 (억원)', color='#d0021b', fontweight='bold')

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')
    ax1.grid(axis='y', alpha=0.3)

    if 'PER(배)' in df.columns:
        ax3.plot(x_labels, df['PER(배)'], color='#f5a623', marker='s', linewidth=2, markersize=8, label='PER (배)')
        ax3.set_ylabel('PER (배)', fontweight='bold')
        ax3.grid(axis='both', alpha=0.3)
        ax3.legend(loc='upper left')
        for i, txt in enumerate(df['PER(배)']):
            if pd.notna(txt):
                ax3.annotate(f'{txt:.1f}', (x_labels[i], df['PER(배)'].iloc[i]), textcoords='offset points', xytext=(0, 10), ha='center', fontsize=9)
    else:
        ax3.axis('off')

    plt.tight_layout()
    return fig

# (3번 섹션용) 돌파 검증용 차트
def draw_technical_chart_pro(df, stock_name):
    fig, ax = plt.subplots(figsize=(10, 5))
    close_val = df['Close'].iloc[-1]
    
    ax.plot(df.index, df['Close'], label='종가', color='black', linewidth=2.0)
    ax.plot(df.index, df['MA50'], label='50일선 (추세선)', color='blue', linestyle='--')
    ax.plot(df.index, df['High20'], label='20일 전고점 (저항선)', color='orange', linestyle=':')
    
    last_date = df.index[-1]
    ax.scatter(last_date, close_val, color='red', marker='*', s=200, label='현재위치', zorder=5)
    ax.text(last_date, df['MA50'].iloc[-1], f' 50일선: {int(df["MA50"].iloc[-1]):,}', color='blue', fontsize=9, verticalalignment='top')
    ax.text(last_date, close_val, f' 현재가: {int(close_val):,}원', color='black', fontsize=10, fontweight='bold', verticalalignment='bottom')

    ax.set_title(f'[{stock_name}] 돌파 저항선 검증 차트', fontsize=15, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    return fig

# ==========================================
# 3. Streamlit 웹 화면 UI 구성
# ==========================================
st.set_page_config(page_title='종합 주식 분석 대시보드', layout='wide')

st.title('📈 종합 주식 분석 대시보드 (멀티 뷰)')
st.markdown('다양한 전문가의 관점을 한 곳에 모았습니다. **기본 차트, 재무제표, 그리고 돌파 전략**을 순서대로 교차 검증하세요.')
st.divider()

if 'selected_name' not in st.session_state:
    st.session_state.selected_name = None
if 'selected_code' not in st.session_state:
    st.session_state.selected_code = None
if 'analyze_clicked' not in st.session_state:
    st.session_state.analyze_clicked = False

tab1, tab2 = st.tabs(['🔥 A. 네이버 골든크로스 종목 선택', '🔍 B. 직접 종목명 검색'])

with tab1:
    with st.spinner('골든크로스 종목 리스트를 불러오는 중입니다...'):
        df_gold = get_golden_cross_stocks()
    if not df_gold.empty:
        gold_name = st.selectbox('👇 네이버 금융 골든크로스 종목 중 선택', df_gold['Name'].tolist())
        if st.button('📊 골든크로스 종목 분석하기', type='primary', key='btn_gold'):
            st.session_state.selected_name = gold_name
            st.session_state.selected_code = df_gold.loc[df_gold['Name'] == gold_name, 'Code'].values[0]
            st.session_state.analyze_clicked = True

with tab2:
    manual_name = st.text_input('👇 분석할 종목명을 정확히 입력하세요 (예: 삼성전자, 카카오, 현대차)')
    if st.button('📊 검색한 종목 분석하기', type='primary', key='btn_manual'):
        if manual_name:
            with st.spinner('종목 코드를 찾는 중...'):
                df_krx = get_all_krx_stocks()
                match = df_krx[df_krx['Name'] == manual_name]
                if not match.empty:
                    st.session_state.selected_name = manual_name
                    st.session_state.selected_code = match['Code'].values[0]
                    st.session_state.analyze_clicked = True
                else:
                    st.error(f"'{manual_name}' 종목을 찾을 수 없습니다.")

# ==========================================
# 결과 출력부 (1.기본기술적 / 2.기본적 / 3.돌파분석 순서 배치)
# ==========================================
if st.session_state.analyze_clicked and st.session_state.selected_name and st.session_state.selected_code:
    st.divider()
    st.markdown(f'## 💡 [{st.session_state.selected_name}] 다각도 분석 리포트')

    with st.spinner("데이터를 수집하고 분석 중입니다..."):
        company_summary = get_company_summary(st.session_state.selected_code)
        df_tech = get_technical_data(st.session_state.selected_code)
        df_fin = get_financial_data(st.session_state.selected_code)

    st.info(f'**🏢 비즈니스 모델 요약:**\n\n{company_summary}')

    # --- 상단 섹션: 1. 기존 기술적 차트 / 2. 기본적(재무) 분석 ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader('🎯 1. 일반 기술적 분석 (매수/손절)')
        if df_tech is not None:
            fig_tech_basic = draw_technical_chart(df_tech.tail(120), st.session_state.selected_name)
            # 2. 핸드폰 화면에 꽉 차게 비율을 맞추도록 use_container_width 적용
            st.pyplot(fig_tech_basic, use_container_width=True)
        else:
            st.warning('차트 데이터가 부족합니다.')

    with col2:
        st.subheader('📑 2. 기본적 분석 (실적/PER)')
        if df_fin is not None and not df_fin.empty:
            fig_fin = draw_fundamental_chart(df_fin, st.session_state.selected_name)
            st.pyplot(fig_fin, use_container_width=True)
        else:
            st.warning('네이버 금융에서 기본적 재무 데이터를 불러올 수 없습니다.')

    st.divider()

    # --- 하단 섹션: 3. 돌파 전략 심층 분석 ---
    st.subheader('🚀 3. 5가지 왕건이 돌파 전략 심층 분석')
    st.markdown("오늘 당장 강한 수급과 함께 저항선을 돌파했는지 수치와 전용 차트로 검증합니다.")
    
    col3, col4 = st.columns([1.2, 1]) 
    
    with col3:
        if df_tech is not None:
            details, b_score = analyze_breakout_strategies_detailed(df_tech)
            
            if details is not None:
                st.write(f"**현재 달성률: {b_score}/5**")
                st.progress(b_score / 5.0)
                
                formatted_data = []
                for name, is_met, detail in details:
                    formatted_data.append({
                        "전략명": name,
                        "판정": "🟢 돌파" if is_met else "🔴 미달",
                        "수치 해설": detail
                    })
                res_df = pd.DataFrame(formatted_data)
                st.dataframe(res_df, use_container_width=True, hide_index=True)
            else:
                st.warning("데이터가 부족하여 분석할 수 없습니다.")
                
    with col4:
        if df_tech is not None:
            fig_tech_pro = draw_technical_chart_pro(df_tech.tail(120), st.session_state.selected_name)
            st.pyplot(fig_tech_pro, use_container_width=True)
            st.caption("※ 파란색 점선(50일선)이나 주황색 점선(전고점)을 확실히 뚫었는지 눈으로 확인하세요.")
