import datetime
from io import StringIO
import warnings
from bs4 import BeautifulSoup
import FinanceDataReader as fdr
import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# 폰트 설정 (Streamlit Cloud 전용 나눔고딕)
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 데이터 크롤링 함수들 (자동 인코딩 감지 적용)
# ==========================================
@st.cache_data(ttl=3600)
def get_golden_cross_stocks():
    """네이버 금융 골든크로스 종목 수집 (인코딩 자동 감지)"""
    url = 'https://finance.naver.com/sise/item_gold.naver'
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        res = requests.get(url, headers=headers)
        # 📌 res.text 대신 res.content(바이트)를 넘겨서 BeautifulSoup이 완벽하게 인코딩을 자동 판독하게 함
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
    """전체 한국 증시 종목코드 수집 (검색용)"""
    df = fdr.StockListing('KRX')
    return df[['Name', 'Code']]


def get_company_summary(stock_code):
    """기업 비즈니스 모델 요약 크롤링 (인코딩 자동 감지)"""
    url = f'https://finance.naver.com/item/main.naver?code={stock_code}'
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        res = requests.get(url, headers=headers)
        # 📌 여기서도 자동 판독 적용 (글자 깨짐 원천 차단)
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
    """네이버 금융 재무제표 수집 (인코딩 자동 감지된 텍스트로 표 해독)"""
    headers = {'User-Agent': 'Mozilla/5.0'}
    url = f'https://finance.naver.com/item/main.naver?code={stock_code}'
    try:
        res = requests.get(url, headers=headers)
        soup = BeautifulSoup(res.content, 'html.parser')
        
        # 📌 BeautifulSoup이 깔끔하게 해독한 한글(str(soup))을 Pandas에 전달하여 표 인식 오류 0%
        dfs = pd.read_html(StringIO(str(soup)))

        for df in dfs:
            if not df.empty and any('매출액' in str(val) for val in df.iloc[:, 0].values):
                df.set_index(df.columns[0], inplace=True)
                df_annual = df['최근 연간 실적'].T
                cols_to_keep = ['매출액', '영업이익', 'PER(배)']
                available_cols = [col for col in cols_to_keep if col in df_annual.columns]
                df_res = df_annual[available_cols].copy()
                
                for col in df_res.columns:
                    df_res[col] = pd.to_numeric(
                        df_res[col].astype(str).str.replace(',', ''), errors='coerce'
                    )
                return df_res
    except Exception:
        pass
    return None


def get_technical_data(stock_code):
    """기술적 분석 (과거 1년 차트 데이터) 수집"""
    today = datetime.datetime.today()
    start_date = (today - datetime.timedelta(days=365)).strftime('%Y-%m-%d')
    try:
        df = fdr.DataReader(stock_code, start_date, today.strftime('%Y-%m-%d'))
        if len(df) < 20:
            return None
        df['MA20'] = df['Close'].rolling(20).mean()
        df['Envelope_Lower'] = df['MA20'] * 0.85
        return df
    except Exception:
        return None

# ==========================================
# 2. 차트 그리기 함수 모음
# ==========================================
def draw_technical_chart(df, stock_name):
    fig, ax = plt.subplots(figsize=(10, 5))

    close_val = df['Close'].iloc[-1]
    stop_val = (
        df['MA20'].iloc[-1] * 0.98
        if df['Close'].iloc[-1] > df['MA20'].iloc[-1]
        else df['Low'].iloc[-1] * 0.97
    )
    loss_pct = ((stop_val - close_val) / close_val) * 100

    ax.plot(df.index, df['Close'], label='종가', color='black', linewidth=1.5)
    ax.plot(df.index, df['MA20'], label='20일선', color='magenta', linestyle='-.')
    ax.plot(df.index, df['Envelope_Lower'], label='엔벨로프 하단(-15%)', color='green', linestyle=':')

    last_date = df.index[-1]
    ax.scatter(last_date, close_val, color='red', marker='*', s=200, label='현재위치', zorder=5)
    ax.axhline(stop_val, color='red', linestyle='--', alpha=0.7, label='참고 손절선', linewidth=1.2)

    ax.text(
        last_date, stop_val, f' 손절가: {int(stop_val):,}원 ({loss_pct:.1f}%)',
        color='white', fontsize=10, fontweight='bold', verticalalignment='center',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='crimson', alpha=0.85, edgecolor='none')
    )
    ax.text(last_date, close_val, f' 현재가: {int(close_val):,}원', color='black', fontsize=9, fontweight='bold', verticalalignment='bottom')

    ax.set_title(f'[{stock_name}] 기술적 분석 차트 (매수/손절 타점)', fontsize=15, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')

    return fig

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

# ==========================================
# 3. Streamlit 웹 화면 UI 구성
# ==========================================
st.set_page_config(page_title='종합 주식 분석 대시보드', layout='wide')

st.title('📈 주식 종합 분석 대시보드')
st.markdown('관심 종목의 **비즈니스 모델**, **기술적 차트(진입/손절가)**, **기본적 재무(실적/PER)**를 한눈에 확인하세요.')
st.divider()

tab1, tab2 = st.tabs(['🔥 1. 네이버 골든크로스 종목 선택', '🔍 2. 직접 종목명 검색'])

selected_name = None
selected_code = None
analyze_clicked = False

with tab1:
    with st.spinner('골든크로스 종목 리스트를 불러오는 중입니다...'):
        df_gold = get_golden_cross_stocks()
    if not df_gold.empty:
        gold_name = st.selectbox('👇 네이버 금융 골든크로스 종목 중 선택', df_gold['Name'].tolist())
        if st.button('📊 골든크로스 종목 분석하기', type='primary', key='btn_gold'):
            selected_name = gold_name
            selected_code = df_gold.loc[df_gold['Name'] == gold_name, 'Code'].values[0]
            analyze_clicked = True

with tab2:
    manual_name = st.text_input('👇 분석할 종목명을 정확히 입력하세요 (예: 삼성전자, 카카오, 현대차)')
    if st.button('📊 검색한 종목 분석하기', type='primary', key='btn_manual'):
        if manual_name:
            with st.spinner('종목 코드를 찾는 중...'):
                df_krx = get_all_krx_stocks()
                match = df_krx[df_krx['Name'] == manual_name]
                if not match.empty:
                    selected_name = manual_name
                    selected_code = match['Code'].values[0]
                    analyze_clicked = True
                else:
                    st.error(f"'{manual_name}' 종목을 찾을 수 없습니다. 오타가 없는지 확인해 주세요.")
        else:
            st.warning('종목명을 빈칸에 입력해 주세요.')

if analyze_clicked and selected_name and selected_code:
    st.divider()
    st.markdown(f'### 💡 [{selected_name}] 종합 분석 결과')

    with st.spinner(f"'{selected_name}' 기업 정보 및 차트 데이터를 가져오는 중..."):
        company_summary = get_company_summary(selected_code)
        df_tech = get_technical_data(selected_code)
        df_fin = get_financial_data(selected_code)

    st.info(f'**🏢 {selected_name}은(는) 무엇으로 돈을 버는 기업일까요?**\n\n{company_summary}')

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.subheader('🎯 1. 기술적 분석 (매수/손절 타점)')
        if df_tech is not None:
            fig_tech = draw_technical_chart(df_tech.tail(120), selected_name)
            st.pyplot(fig_tech)
        else:
            st.warning('기술적 분석을 위한 과거 데이터가 부족합니다.')

    with chart_col2:
        st.subheader('📑 2. 기본적 분석 (실적/PER)')
        if df_fin is not None and not df_fin.empty:
            fig_fin = draw_fundamental_chart(df_fin, selected_name)
            st.pyplot(fig_fin)
            with st.expander('원문 데이터 표 보기'):
                st.dataframe(df_fin.style.format('{:,.1f}', na_rep='-'), use_container_width=True)
        else:
            st.warning('네이버 금융에서 기본적 재무 데이터를 불러올 수 없습니다.')
