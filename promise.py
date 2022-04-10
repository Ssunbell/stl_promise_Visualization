import streamlit as st
import streamlit.components.v1 as components
from collections import Counter

import preprocessing
import execute

# root_path = '/Users/junho/Desktop/server/data/'
root_path = './data/'
pre = preprocessing
tokens = pre.tokens
lines_token = pre.lines_token
promise140 = pre.promise140

candidates = ['전체', '기호 1번', '기호 2번', '기호 3번', '기호 4번', '기호 5번', '기호 6번', '기호 7번',
              '기호 8번', '기호 9번', '기호 10번', '기호 11번', '기호 12번', '기호 13번', '기호 14번']


sp = execute.SearchPromise()
vk = execute.VisualizeKeywords()
km = execute.Kmeans_Visualization()
chap3_LSA = execute.CategorizePromise()
sp = execute.SearchPromise()

st.sidebar.header('메뉴')
chapter = st.sidebar.selectbox('선택', ['시작', '후보자 공약 분석', '주제 분류', '클러스터링', '후보자/공약 추천'])


if chapter == '시작':
    st.title('🔥 제20대 대선 정책 공약 시각화 🔥')
    st.markdown('> 2020년에 열린 제20대 대통령 선거 후보자 14명의 정책 공약을 시각화하였습니다.')
    st.markdown('#### 🔴 후보자 공약 분석')
    st.markdown('1. 후보자별 사용 단어 비교')
    st.markdown('2. 공약 상관관계 비교')
    st.markdown('3. 주요정당 정보')
    st.markdown('#### 🟠 주제 분류')
    st.markdown('1. LSA')
    st.markdown('2. LDA')
    st.markdown('#### 🟡 클러스터링')
    st.markdown('1. 계층 클러스터링')
    st.markdown('2. 부분 클러스터링')
    st.markdown('#### 🟢 후보자/공약 추천')
    st.markdown('1. 머신러닝')
    st.markdown('2. 클러스터링')
    st.markdown('3. 코사인 유사도')
    st.markdown('### 👈 메뉴를 클릭하여 다양한 시각화 결과를 확인해보세요😆')

# 시각화 - 후보자별 그래프, 클러스터링맵, 지역공약
elif chapter == '후보자 공약 분석':
    st.title('🔥 후보자 공약 분석 🔥')
    st.markdown('`정책공약` `지역공약` `워드클라우드` `상관관계`')
    sub_chapter = st.sidebar.selectbox('선택', ['선택','후보자별 사용 단어 비교', '공약 상관관계 비교', '주요 정당 정보'])
    if sub_chapter == '선택':
        st.markdown('##### 단순 시각화')
        st.markdown('❗ 가장 기본적이고 직관적인 접근으로 대선공약집의 공약 순위와 지역별 공약을 시각화했습니다.')
        st.markdown('❗ 유사도를 통해 각 후보가 어떤 정책을 추진 혹은 폐지할지 키워드를 분석해 보았고, 각 후보 간 단어 사용에 대한 상관관계를 시각화했습니다.')
        st.write('👈  선택을 눌러 원하는 정보를 확인해 보세요😆')
    if sub_chapter == '후보자별 사용 단어 비교':
        st.subheader('💡 단어 분석')
        st.markdown('📌 각 후보 그래프의 위쪽 두 개의 그래프는 단어 수 기반으로 작성되었습니다.')
        st.markdown('📌 아래쪽 두 개의 그래프는 word2vec을 이용했고, 각 후보가 추진 혹은 폐지하고 싶은 정책, 공약이 무엇인지 키워드의 유사도를 이용해 유추해 보았습니다.')
        st.markdown('#### 👇 관심있는 후보를 선택할 수 있습니다.')
        candidate = st.selectbox('', candidates)
        i = candidates.index(candidate)
        if candidate != '전체':
            counts = Counter(tokens[i-1])
            vk.show_graphs(counts, i-1)
        else:
            counts_for_all = Counter(sum(tokens, []))
            vk.show_graphs(counts_for_all)
    elif sub_chapter == '공약 상관관계 비교':
        st.subheader('💡 공약 상관관계 비교')
        st.markdown('📌 각 후보자 공약집 전체에 대해서 count 기반 상위 500단어에 대한 상관계수와 그에 대한 덴드로그램 입니다.')
        st.markdown('📌 단어 맵핑에대한 상관관계이므로 값이클수록 같은 단어의 사용횟수의 차이가 작습니다.')
        st.markdown('📌 대각선을 제외한 가장 높은 상관계수는 1,3번후보의 값이고 1번후보의 상관계수는 전체적으로 높습니다.')
        st.markdown('📌 계층적 클러스터링에서 특징적인 두 개의 그룹이 드러납니다.')
        st.markdown('> 🎈 2, 4, 9, 13')
        st.markdown('> 🎈 1, 3, 7, 12')
        vk.corr_map()
    elif sub_chapter == '주요 정당 정보':
        # 그래프 표시
        st.subheader('💡 주제 단어 분류')
        st.markdown('📌 키워드를 선정한 뒤, 각 후보자의 정책 공약에 얼마나 사용되었는지 비교해 보았습니다.')
        st.markdown('📌 그래프는 기호 1, 2, 3, 4번 후보자들의 키워드 누적 사용 횟수와 비율을 보여줍니다.')
        st.markdown('📌 그래프를 통해 각 후보자가 어떤 키워드에 중점을 두고 있는지 유추해볼 수 있습니다.')
        vk.stacked_category()
        st.markdown('___')
        st.subheader('💡 지역 공약 살펴보기')
        st.markdown('📌 기호 1, 2, 3번 후보자들의 지역 공약 자료집을 활용하여 시각화해 보았습니다.')
        st.markdown('📌 지역 공약 자료집을 활용하여 지역별로 핵심 공약을 분류한 다음, 지도 위에 표현해 보았습니다.')
        st.markdown('📌 지역별 인구수를 나타냄으로써 해당 지역의 유권자 수 차이를 확인할 수 있습니다.')
        with open(f"{root_path}folium.html", 'r', encoding='utf-8') as f:
            html_string = f.read() 
        components.html(html_string, width= 800, height= 720)


# 주제 분류 - LSA, pyLDAvis
elif chapter == '주제 분류':
    st.title('🔥 주제 분류 🔥')
    sub_chapter = st.sidebar.selectbox('방법', ['선택','LSA', 'LDA'])
    st.markdown('`토픽모델링` `잠재의미` `LSA` `LDA`')
    if sub_chapter == '선택':
        st.markdown('##### 토픽 모델링')
        st.markdown('###### ❗ 레이블이 없는 데이터 특성상 공약들이 단어를 통해 주제가 어떻게 묶이는지 분석하기 위해 LSA를 이용해서 주제를 유추하였고, 유추한 레이블을 통해 각 후보의 정책분포를 시각화했습니다.')
        st.write('👈 선택을 눌러 원하는 정보를 확인해 보세요😆')
    elif sub_chapter == 'LSA':
        st.subheader('💡 LSA: Latent Semantic Analysis')
        st.markdown('📌 [LSA](https://wikidocs.net/24949)는 특잇값 분해(SVD)를 활용하여 문서에 숨어 있는 주제를 찾아내는 기법입니다.')
        st.markdown('📌 각 정책 공약과 가장 관련성이 높은 주제를 할당하였습니다.')
        st.markdown('📌 여기서 관련성은 각 정책 공약이 어떤 잠재 의미군(주제)에 속하는 용어들을 많이 포함하고 있는지, 그 비중을 의미합니다.')
        st.markdown('📌 주제는 주제별 키워드를 확인하여 다음과 같이 설정하였습니다.')
        st.markdown('> ###### 🔹 평등/안정 🔹 정치/행정/교육 🔹 부동산 🔹 국방/통일/외교 🔹 경제/환경/기술 🔹 미래/청년')
        chap3_LSA.show_graphs_all()
        st.markdown('___')
        st.subheader('💡후보자 비교')
        st.markdown('#### 👇 관심있는 후보를 선택할 수 있습니다.')
        number = st.select_slider('',
        options=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        value=4)
        chap3_LSA.show_pie_chart_selective(number)
        chap3_LSA.show_stacked_plot_selective(number)

    else:
        st.subheader('💡 LDA: Latent Dirichlet Allocation')
        st.markdown('📌 [LDA](https://wikidocs.net/30708)는 LSA와 비슷하지만 다른점은 주제들이 디리클레 분포(Dirichlet distribution)를 따른다고 가정하여 주제를 분석하는 기법입니다.')
        st.markdown('📌 왼쪽 그래프는 후보의 공약을 LDA를 적용하여 주제에 따라 분류한 것을 시각화한 결과입니다.')
        st.markdown('📌 오른쪽 바 그래프는 각 군집에 따라 특징적인 단어와 빈도수를 나타냅니다.')
        st.markdown('📌 오른쪽 위의 λ값이 0에 가까울수록 그 군집만의 특징적인 단어가 나타나고, 1에 가까울수록 전체 공약집에서 가장 많은 빈도수를 차지한 단어가 해당 군집에서 차지하는 정도를 나타냅니다.')
        st.markdown('📌 왼쪽 위에 숫자를 입력하면 해당 군집에 대한 정보를 보여줍니다. LDA를 이용한 사후 분석 결과 최적의 군집 개수는 7개이며, 각 군집에 대한 주제는 다음과 같습니다.')
        st.markdown('> ###### 1️⃣ 평화/안보  2️⃣ 고용/일자리  3️⃣ 주택/부동산  4️⃣ 선거  5️⃣ 법률/제도  6️⃣ 사회보장서비스  7️⃣ 산업')
        with open(f'{root_path}lda.html', 'r') as f:
            html_string = f.read()
        components.html(html_string,width=1300,height=800)

# 클러스터링
elif chapter == '클러스터링':
    st.title('🔥 클러스터링 🔥')
    st.markdown('`덴드로그램` `K-means` `실루엣 계수` `엘보우 방법`')
    sub_chapter = st.sidebar.selectbox('방법', ['선택','계층 클러스터링','군집 갯수 평가', '부분 클러스터링'])
    if sub_chapter == '선택':
        st.markdown('##### 군집 분석')
        st.markdown('###### ❗ 후보들의 공약을 레이블이 없는 형태 그대로 분석하기 위해 비지도학습 클러스터링인 [K-means](https://ko.wikipedia.org/wiki/K-평균_알고리즘)를 사용하여 유사한 후보끼리 군집을 나누어 후보 집단을 형성한 결과를 시각화했습니다.')
        st.write('👈 선택을 눌러 원하는 정보를 확인해 보세요😆')
    elif sub_chapter == '계층 클러스터링':
        st.subheader('💡 계층 클러스터링')
        st.markdown('📌 군집의 개수를 계층을 만들어서 군집을 분류하는 방법을 계층 클러스터링이라 합니다.')
        st.markdown('📌 scipy를 이용했고, 덴드로그램을 그래프 위에 임의의 점선을 그어 4개의 클러스터로 구분해보았습니다.')
        st.markdown('📌 입력 데이터는 공약 제목을 pca로 차원 축소한 값이고, 와드연결법을 이용한 덴드로그램 입니다.')
        st.markdown('📌 그 결과, 후보자들의 정책 공약은 다음과 같이 4개의 군집으로 구분할 수 있습니다.')
        st.markdown('> 🎈 1')
        st.markdown('> 🎈 2, 3, 5, 7, 12')
        st.markdown('> 🎈 4, 8, 9, 10, 11, 13, 14')
        st.markdown('> 🎈 6')
        st.pyplot(km.show_dendrogram())
    elif sub_chapter == '군집 갯수 평가':
        st.subheader('💡 군집 갯수 평가 ')
        st.markdown('📌 군집 개수 K를 정하는 다양한 평가 방법이 있는데, 저희는 실루엣 방법과 엘보우 방법을 사용하였습니다.')
        st.markdown('📌 실루엣 계수가 1에 가까울수록 두 군집 간 거리가 멀어 군집이 잘 분류된 것을 의미합니다.')
        st.markdown('📌 실루엣 시각화는 색상이 들어간 막대가 각 군집을 의미하며, 빨간색 점선은 위에서 구한 각 군집 갯수의 실루엣 계수입니다.')
        st.markdown('📌 이 군집들이 빨간 점선인 평균 실루엣 계수를 많이 넘을수록 균일하게 잘 분류된 것을 의미합니다.')
        st.markdown('📌 엘보우 방법은 오차들의 거리(SSE)를 이용하여 최적의 군집 개수를 정합니다. 검은색 점선의 위치가 최적 군집 갯수입니다.')
        st.markdown('> ##### 아래의 결과를 종합해서, 군집 갯수를 4로 선택했습니다.')
        km.sil_score()
        st.pyplot(km.silhouette_visualizer())
        st.pyplot(km.kelbow_visualizer())
    elif sub_chapter == '부분 클러스터링':
        st.subheader('💡 부분 클러스터링')
        st.markdown('📌 여기서 제공하는 [UMAP](https://arxiv.org/abs/1802.03426) 시각화는 후보의 공약 제목 문서를 위치 벡터값으로 변환하는 Doc2Vec방식을 적용했습니다.')
        st.markdown('📌 퍼지 이론과 위상수학을 이용하여 고차원 데이터에 적합한 UMAP방법을 적용했습니다.')
        st.markdown('📌 UMAP은 문서의 위치가 확률적으로 가장 가까운 후보끼리 군집화된 결과를 제공합니다.')
        st.markdown('#### 👇 군집 갯수를 제어할 수 있습니다.')
        number = st.select_slider('',
        options=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        value=4)
        st.markdown('> 저희의 분석 결과 최적의 군집 개수는 4 입니다.')
        st.pyplot(km.UMAP_show(number))
        

# 추천
elif chapter == '후보자/공약 추천':
    st.title('🔥 후보자/공약 추천 🔥')
    st.markdown('`추천시스템` `머신러닝` `클러스터링` `코사인유사도`')
    sub_chapter = st.sidebar.selectbox('방법', ['선택','머신러닝', '클러스터링', '코사인 유사도'])
    if sub_chapter == '선택':
        st.markdown('##### 추천 시스템')
        st.markdown('###### ❗ 후보 추천은 지도학습과 비지도학습 그리고 유사도 분석기법을 사용하여 사용자 입력값을 받고 입력값에 가장 적합한 공약집을 추천합니다.')
        st.write('👈 선택을 눌러 원하는 정보를 확인해 보세요😆')

    elif sub_chapter == '머신러닝':
        st.subheader('💡 머신러닝을 이용한 후보자 추천')
        st.markdown('#### 👇 사용자의 관심사를 입력하면 그 관심사와 일치하는 후보를 추천합니다.')
        target = st.text_input('', value='인공 지능')
        if target:
            for i in execute.ML().pred(target):
                if i == '적용된 키워드가 없으므로 다시 검색해 주세요.':
                    st.warning('적용된 키워드가 없습니다 다시 입력해주세요.😅')
                else:
                    st.success('검색이 완료되었습니다.')
                    st.write('추천된 공약집은', i, '입니다.😆')

    elif sub_chapter == '클러스터링':
        st.subheader('💡 클러스터링을 이용한 후보자 추천')
        st.markdown('#### 👇 사용자의 관심사를 입력하면 사용자와 비슷한 후보 집단을 추천해줍니다.')
        user_input = st.text_input('', value = '저는 취업을 준비하고 있는 학생입니다. 학생들이 일자리를 구할 수 있도록 다양한 교육 기회와 지원 정책에 관심이 많습니다.')
        st.markdown('#### 👇 군집 갯수를 제어할 수 있습니다.')
        number = st.select_slider('',
        options=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        value=4)
        ku = execute.User_Kmeans(user_input)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(ku.UMAP_show(number))

    elif sub_chapter == '코사인 유사도':
        st.subheader('💡 코사인 유사도를 이용한 공약 추천')
        st.markdown('#### 👇 [코사인 유사도](https://ko.wikipedia.org/wiki/%EC%BD%94%EC%82%AC%EC%9D%B8_%EC%9C%A0%EC%82%AC%EB%8F%84)를 이용하여 관련 정책 공약을 추천해 드립니다.')
        user_input = st.text_input('', value = 'ex) 폭력과 차별 없는 세상을 만들어 주세요!!')
        sp.show_similar_promise(user_input)