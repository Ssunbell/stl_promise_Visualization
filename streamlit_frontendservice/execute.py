import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.python.keras import models
from tensorflow.python.keras import layers
from tensorflow.python.keras import optimizers
from tensorflow.python.keras import losses
from tensorflow import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from PIL import Image
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from konlpy.tag import Okt
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import seaborn as sns
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from gensim.models.doc2vec import TaggedDocument
import gensim
from umap import umap_
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.cluster import hierarchy as shc
from yellowbrick.cluster import SilhouetteVisualizer, KElbowVisualizer
from wordcloud import WordCloud
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
import platform
from ast import literal_eval
import streamlit as st

okt = Okt()

import preprocessing
pre = preprocessing

root_path = './data/'
candidates = ['기호 1번', '기호 2번', '기호 3번', '기호 4번', '기호 5번', '기호 6번', '기호 7번',
              '기호 8번', '기호 9번', '기호 10번', '기호 11번', '기호 12번', '기호 13번', '기호 14번', '전체']


class VisualizeKeywords():
    def __init__(self):
        self.mask_path = pre.mask_path
        self.font_location = root_path + 'NanumGothic.ttf'
        self.font_name = font_manager.FontProperties(fname=self.font_location, size = 10)
        
        lines_token_df = pd.read_csv(root_path+'lines_token.csv', converters={'0': literal_eval})
        lines_token = lines_token_df['0'].values.tolist()
        self.lines_token = lines_token

        sns.set_style('whitegrid')

    def show_graphs(self, text_counts, num=None):
        tags = text_counts.most_common(100)
    
        mask = np.array(Image.open(self.mask_path))
        
        wc = WordCloud(font_path=self.font_location, background_color='white', 
                  colormap='twilight_shifted', max_font_size=100, mask=mask)
        
        cloud = wc.generate_from_frequencies(dict(tags))
    
        fig = plt.figure(figsize=(25, 13))
        plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.35)
    
        fig.add_subplot(2, 2, 1)
        plt.title(f'정책 공약 키워드', size=20, fontproperties=self.font_name)
        plt.axis('off')
        plt.imshow(cloud)

        tags20 = dict(text_counts.most_common(20))
    
        fig.add_subplot(2, 2, 2)
        plt.xlabel('주요 단어', fontproperties=self.font_name)
        plt.ylabel('빈도', fontproperties=self.font_name)
    
        tags20_df = pd.DataFrame(tags20.values(), index=tags20.keys(), columns=['빈도'])
        
        plt.title(f'정책 공약 키워드 top 20', size=20, fontproperties=self.font_name)
     
        sns.set_palette(reversed(sns.color_palette('Purples', 20)), 20)
        
        ax1 = sns.barplot(x=tags20_df.index, y=tags20_df['빈도'], data=tags20_df)
        for bar in ax1.patches:
            bar.set_width(0.5)
        
        plt.xticks(rotation='45', fontproperties=self.font_name) 

        if num:
            sents_clear = self.lines_token[num-1]
  
            keys = ['추진', '폐지']
            model = Word2Vec(sents_clear, min_count=4, seed=65)
    
            for i in range(len(keys)):
                try:
                    df = pd.DataFrame(model.wv.most_similar(keys[i]))
                    df.set_index(0)
                except:
                    st.warning(f"정책 공약에 '{keys[i]}'이/가 존재하지 않거나 그 수가 너무 적습니다.")
                else:
                    fig.add_subplot(2, 2, i + 3)
                    ax2 = sns.barplot(y=df[0], x=df[1])
                    plt.yticks(fontproperties=self.font_name) 
                    
                    for bar in ax2.patches:
                        bar.set_height(0.5)
                        
                    plt.xlabel('유사도', fontproperties=self.font_name)
                    plt.ylabel('단어', fontproperties=self.font_name)
                    plt.title(f"단어 유사도: {keys[i]}", size=20, fontproperties=self.font_name)
        
        st.pyplot(fig)

    def stacked_category(self):
        temp = []
        for i in range(4):
            key = {'경제': 0, '복지': 0, '정치': 0, '보건': 0, '환경': 0, '문화': 0,
                   '관광': 0, '노동': 0, '교육': 0, '산업': 0, '안보': 0, '국방': 0,
                   '북한': 0}
            for j in pre.tokens[i]:
                if j in key:
                    key[j] += 1

            temp.append(key)
        df = pd.DataFrame(temp)
        df.index = ['기호1', '기호2', '기호3', '기호4']
        df = df.transpose()

        df_per = df.copy()
        for i in range(len(df_per.index)):
            df_per.loc[df_per.index[i]] /= sum(df_per.loc[df_per.index[i]])

        df_per = round(df_per, 3)

        fig = plt.figure(figsize=(30, 8))
        colors = ['royalblue', 'crimson', 'gold', 'sandybrown']

        ax1 = fig.add_subplot(1, 2, 1)
        df.plot(kind='bar', stacked=True, color=colors, ax=ax1)
        plt.xticks(rotation='45', fontproperties=self.font_name)
        plt.legend(frameon=True, shadow=True)
        plt.title('후보자별 키워드 등장 횟수', fontsize=20, fontproperties=self.font_name)

        ax2 = fig.add_subplot(1, 2, 2)
        df_per.plot(kind='bar', stacked=True, color=colors, ax=ax2)
        plt.xticks(rotation='45', fontproperties=self.font_name)
        plt.title('후보자별 키워드 등장 비율', fontsize=20, fontproperties=self.font_name)
        plt.legend().remove()

        i = 0
        j = 0
        for p in ax2.patches:
            if i >= len(df.index):
                i = 0
                j += 1
            left, bottom, width, height = p.get_bbox().bounds
            if height == 0:
                i += 1
                continue
            ax2.annotate("%d%%" % (df_per[df.columns[j]][i] * 100), xy=(left + width / 2, bottom + height / 2),
                         ha='center', va='center')
            i += 1

        st.pyplot(fig)

    def corr_map(self, max_features=500):  # max features = 1000 등 높은값 에러 이슈 rename하지 않은코드는 정상작동
        cv = CountVectorizer(max_features=max_features)
        tdm = cv.fit_transform(pre.tf_sentences)
        df = pd.DataFrame(data=tdm.toarray(), columns=cv.get_feature_names_out(), index=list(range(1, 15)))
        df = df.transpose()

        fig = sns.clustermap(df.corr(),
                       annot=True,
                       cmap='RdYlBu_r',
                       vmin=-1, vmax=1,
                       figsize=(12, 10))

        st.pyplot(fig)
# 2.4
# pyLDAvis 처럼 html 세이브 후 promise 에서 똑같이 작성.
# html save
'''class GetLocalPromise():
    def __init__(self, population_path, jason_path, regional_promise_path):
        self.province = ['서울특별시', '부산광역시', '대구광역시', '인천광역시', '광주광역시',
                         '대전광역시', '울산광역시', '세종특별자치시', '경기도', '강원도',
                         '충청북도', '충청남도', '전라북도', '전라남도', '경상북도', '경상남도',
                         '제주특별자치도']
        
        self.df_loc = pd.DataFrame([[37.5657, 126.9769],[36.4802, 127.2892],[37.4561, 126.7059],[36.3505, 127.3848],
                                    [35.8714, 128.6014],[35.5374, 129.3105],[35.1800, 129.0768],[35.1589, 126.8546],
                                    [37.2747, 127.0096],[37.8854, 127.7298],[36.5184, 126.8000],[36.8000, 127.7000],
                                    [36.5760, 128.5056],[35.2383, 128.6924],[35.8198, 127.1081],[34.8162, 126.4629],[33.4996, 126.5312]],
                                    index = ['서울특별시','세종특별시','인천광역시','대전광역시',
                                             '대구광역시','울산광역시','부산광역시','광주광역시',
                                             '경기도','강원도','충청남도','충청북도',
                                             '경상북도','경상남도','전라북도','전라남도','제주특별자치도'],
                                    columns = ['위도', '경도'])
        
        self.population_path = population_path
        self.jason_path = jason_path
        self.files = regional_promise_path
        
        self.promise_no1 = self.promise_no1(self.files[0])
        self.promise_no2 = self.promise_no2(self.files[1])
        self.promise_no3 = self.promise_no3(self.files[2])
        
    def two_columns_pdf(self, file):    
        x0 = 0    
        x1 = 0.5  
        y0 = 0  
        y1 = 1  

        all_content = []
        with pdfplumber.open(file) as pdf:
            for i, page in enumerate(pdf.pages):
                width = page.width
                height = page.height

                left_bbox = (x0*float(width), y0*float(height), x1*float(width), y1*float(height))
                page_crop = page.crop(bbox=left_bbox)
                left_text = page_crop.extract_text()

                left_bbox = (0.5*float(width), y0*float(height), 1*float(width), y1*float(height))
                page_crop = page.crop(bbox=left_bbox)
                right_text = page_crop.extract_text()
                page_context = '\n'.join([left_text, right_text])
                all_content.append(page_context)
                
        return all_content
    
    def promise_no1(self, file_name):
        file_name = file_name
        texts = self.two_columns_pdf(file_name)
        
        ans = []
        for doc in texts:
            loc = []
            for sen in re.split('[0-1]\d\d\d',doc):
                loc.append(sen.replace('\n    ', '').lstrip('\n'))
            ans.append(loc)
            
        promises = []
        for i in ans:
            temp = []
            for j in i:
                temp.append(j.split('\n')[0].strip().split('•')[0])
            promise = '<ol start="1"><br><li>' + '<br><br><li>'.join(temp[1:])
            promises.append(promise)
            
        df = pd.DataFrame(promises, index=self.province, columns=['공약'])
        df['지역'] = '<h3>' + df.index + '</h3>'
        df = pd.merge(df, self.df_loc, left_on=df.index, right_on=self.df_loc.index)
        
        return df
    
    def promise_no2(self, file_name):
        pdf = pdfplumber.open(file_name)
        
        promises = []
        for page in pdf.pages:
            lines = page.extract_text().split('\n')
            for line in lines:
                if line[0] != '■':
                    temp = []
                else:
                    temp.append(line[2:])
                    if len(temp) == 7:
                        promise = '<ol start="1"><br><li>' + '<br><br><li>'.join(temp)
                        promises.append(promise)
                        
        df = pd.DataFrame(promises, index=self.province, columns=['공약'])
        df['지역'] = '<h3>' + df.index + '</h3>'
        df = pd.merge(df, self.df_loc, left_on=df.index, right_on=self.df_loc.index)
    
        return df
                        
    def promise_no3(self, file_name):
        file_name = file_name
        texts = self.two_columns_pdf(file_name)
        
        ans = []
        for doc in texts:
            loc = []
            for sen in re.split('\d\.',doc):
                loc.append(sen)
            ans.append(loc)
        
        promises = []
        for i in ans:
            temp = []
            for j in i:
                temp.append(j.split('\n')[0].lstrip())
            promise = '<ol start="1"><br><li>' + '<br><br><li>'.join(temp[1:])
            promises.append(promise)
            
        df = pd.DataFrame(promises, index=self.province, columns=['공약'])
        df['지역'] = '<h3>' + df.index + '</h3>'
        df = pd.merge(df, self.df_loc, left_on=df.index, right_on=self.df_loc.index)
        
        return df
    
    def promise_map(self):
        with open(self.jason_path, mode='rt', encoding='utf-8') as f:
            geo_data = json.loads(f.read())
        f.close()
    
        file_path = self.population_path
        df_p = pd.read_csv(file_path)  
        df_p.columns = ['지역', '인구']
        
        korea_map = folium.Map(location=[35.8714, 127.6014],tiles='cartodbpositron',width=800,height=750, zoom_start = 7, max_bounds = True,
                               zoom_control = False, scrollWheelZoom=False, dragging=True,
                               min_lat = 32, max_lat = 43, min_lon =123, max_lon = 133)

        korea_map.add_child(folium.GeoJson(geo_data, control = False))

        folium.Choropleth(
                          geo_data=geo_data,
                          data=df_p,
                          columns=('지역', '인구'),
                          key_on='feature.properties.CTP_KOR_NM',
                          fill_color='YlOrRd',
                          legend_name='인구수',
                          control = False
                          ).add_to(korea_map)


        fg = folium.FeatureGroup(name='후보전체', control = False)
        korea_map.add_child(fg)

        g1 = plugins.FeatureGroupSubGroup(fg, '이재명')
        korea_map.add_child(g1)
        g2 = plugins.FeatureGroupSubGroup(fg, '윤석열', show = False)
        korea_map.add_child(g2)
        g3 = plugins.FeatureGroupSubGroup(fg, '심상정', show = False)
        korea_map.add_child(g3)

        df_list = [self.promise_no1, self.promise_no2, self.promise_no3]
        g_list = [g1, g2, g3]
        loc_adjustments = [-0.05, 0, 0.03]
        colors = ['blue', 'red', 'beige']
        
        for i in range(0, 3):
            df = df_list[i]
            g = g_list[i]
            loc_adj = loc_adjustments[i]
            color = colors[i]
            for province, promise, lat, lng in zip(df.지역, df.공약, df.위도, df.경도):
                iframe = folium.IFrame(province + promise)
                popup = folium.Popup(iframe, min_width=600, max_width=1000)
                folium.Marker([lat + loc_adj,lng + loc_adj], popup=popup, icon=folium.Icon(color=color, icon='info-sign')).add_to(g)       

        folium.LayerControl(collapsed=False).add_to(korea_map)

        korea_map.save('folium.html')

gp = GetLocalPromise(population_path, jason_path, regional_promise_path)
gp.promise_map()'''


# chapter3
# 3.1
class CategorizePromise():
    def __init__(self):
        self.font_location = root_path + 'NanumGothic.ttf'
        self.font_name = font_manager.FontProperties(fname=self.font_location, size = 10)

        sns.set_style('whitegrid')

        self.df = self.promise_reshaped()
        self.df_tokens = self.promise_tokenized()
        (self.tfidf_vectorizer, self.svd_model) = self.model_for_categorize()
        self.df_result = self.show_categorized_result()

    def promise_reshaped(self):
        df_array = np.array(pre.top10).reshape(140, 1)
        df_reshaped = pd.DataFrame(df_array, columns=['정책 공약'])
        df_reshaped['기호'] = [f'기호 {index // 10 + 1}번' for index in df_reshaped.index]

        return df_reshaped

    def promise_tokenized(self):
        tokens = []
        for text in self.df.values.tolist():
            text = okt.nouns(str(text))
            token = ' '.join(word for word in text if word in pre.word_idx)
            tokens.append(token)

        df_reshaped_tokens = pd.DataFrame(tokens, columns=['정책 공약_tokens'])

        return df_reshaped_tokens

    def model_for_categorize(self):
        tfidf_vectorizer = TfidfVectorizer()
        X_tfidf = tfidf_vectorizer.fit_transform(self.df_tokens['정책 공약_tokens'])

        svd_model = TruncatedSVD(n_components=6, algorithm='randomized', n_iter=3, random_state=1)
        svd_model.fit(X_tfidf)

        return tfidf_vectorizer, svd_model

    def get_topics(self):
        components = self.svd_model.components_
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        for idx, topic in enumerate(components):
            print(f'Topic {idx + 1}:', [(feature_names[i], topic[i].round(5)) for i in topic.argsort()[:-6: -1]])

    def show_categorized_result(self):
        pipe = Pipeline([('tfidf', self.tfidf_vectorizer), ('svd_model', self.svd_model)])

        self.df['주제_idx'] = 0
        self.df['주제'] = 0
        self.df['관련성'] = 0

        topics = ['평등/권리보장', '정치/행정/교육', '부동산', '국방/통일/외교', ' 경제/환경/기술', '미래/청년']

        for idx in self.df_tokens.index:
            t = pipe.transform([self.df_tokens['정책 공약_tokens'].iloc[idx]])
            self.df['주제_idx'].iloc[idx] = np.argmax(t) + 1
            self.df['주제'].iloc[idx] = topics[np.argmax(t)]
            self.df['관련성'].iloc[idx] = f'{np.max(t) * 100:.3f}%'

        return self.df

    def show_graphs_all(self):
        fig = plt.figure(figsize=(20, 8))
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.35)

        sns.set_palette('Set2', 6)

        df = self.df_result
        plt.suptitle(f'전체 정책 공약 분류', fontsize=40, va='bottom', fontproperties=self.font_name)

        plt.subplot(1, 2, 1)
        ax = sns.countplot(x=df['주제'])
        plt.xticks(fontproperties=self.font_name)
        plt.xlabel('주제', fontproperties=self.font_name)
        for bar in ax.patches:
            bar.set_width(0.5)

        plt.subplot(1, 2, 2)
        df_groupby = df.groupby('주제').count()

        colors = ['mediumaquamarine', 'yellowgreen', 'gold', 'lightsteelblue', 'plum', 'darksalmon']

        plt.pie(df_groupby['주제_idx'],
                autopct='%.1f%%',
                colors=colors,
                wedgeprops=dict(width=0.6))
        plt.legend(labels=df_groupby.index, loc='upper right', prop=self.font_name)
        plt.show()
        st.pyplot(fig)

    def show_pie_chart_selective(self, num):
        fig = plt.figure(figsize=(20, 80))

        for i in range(num):
            df = self.df_result[i * 10: (i + 1) * 10]
            df_groupby = df.groupby('주제').count()

            plt.subplot(7, 2, i + 1)
            plt.pie(df_groupby['주제_idx'],
                    autopct='%.1f%%',
                    wedgeprops=dict(width=0.6))
            plt.xticks(fontproperties=self.font_name)
            plt.legend(labels=df_groupby.index, prop=self.font_name)
            plt.title(f'기호 {i + 1}번', size=20, fontproperties=self.font_name)
        plt.show()
        st.pyplot(fig)

    def show_stacked_plot_selective(self, num):
        df_selective = self.df_result[:num * 10]

        df = pd.crosstab(df_selective['주제'], df_selective['기호'])
        plt.xticks(fontproperties=self.font_name)

        df_per = df.copy()

        for i in range(len(df_per.index)):
            df_per.loc[df_per.index[i]] /= sum(df_per.loc[df_per.index[i]])

        df_per = round(df_per, 3)

        colors = ['royalblue', 'crimson', 'gold', 'sandybrown', 'green', 'firebrick', 'red',
                  'olivedrab', 'cadetblue', 'yellowgreen', 'forestgreen', 'tomato', 'darkseagreen', 'steelblue']

        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(1, 1, 1)

        df.plot(kind='bar', stacked=True, color=colors, ax=ax)
        plt.xticks(rotation=0, fontproperties=self.font_name)
        plt.xlabel('주제', fontproperties=self.font_name)
        plt.legend(frameon=True, shadow=True, loc=0, prop=self.font_name)

        i = 0
        j = 0
        for p in ax.patches:
            if i > 5:
                i = 0
                j += 1
            left, bottom, width, height = p.get_bbox().bounds
            if height == 0:
                i += 1
                continue
            ax.annotate("%.1f%%" % (df_per[df.columns[j]][i] * 100), xy=(left + width / 2, bottom + height / 2),
                        ha='center', va='center')
            i += 1

        plt.show()
        st.pyplot(fig)
# 3.2
# html save
'''
import pyLDAvis
from pyLDAvis import gensim_models
import gensim
from gensim.models.ldamodel import LdaModel

# 데이터프레임, 전체, 후보자별, 누적 그래프
st.subheader('LDA')
st.write('> LDA 설명')
#pyLDAvis
lines_token = pre.lines_token

clean_text = [list(' '.join(line) for line in candidate_document) for candidate_document in lines_token]
processed_text = sum(lines_token, [])
dictionary = gensim.corpora.Dictionary(processed_text)
dictionary.filter_extremes(no_below=3, no_above=0.05)
corpus = [dictionary.doc2bow(text) for text in processed_text]

lda_model = LdaModel(corpus, id2word=dictionary, num_topics=7, passes=30, random_state=42)
lda_visualization = gensim_models.prepare(lda_model, corpus, dictionary, sort_topics=False)

pyLDAvis.display(lda_visualization)
pyLDAvis.save_html(lda_visualization,'/Users/junho/Desktop/server/data/lda.html')
'''

# chapter4
# 4.1
class Kmeans_Visualization:
    def __init__(self):
        self.top10_token = sum(pre.top10_token,[])
        self.image_path = pre.image_path
        
        self.new_df = self.df_vectorize()
        (pca, pca_df) = self.pca()
        self.pca = pca
        self.pca_df = pca_df
        
        self.x = self.Doc2Vec(self.top10_token)
        self.clusterable_embedding = self.umap()

        self.x2 = [50, 50, 70, -60, -70, -40, 40, -70, 0, 0,
                -70, 70, 0, -75]
        self.y2 = [-60, 70, 0, 0, 0, -60, 60, 60, -70, 90,
                70, 0, -70, -60]
        
    def show_dendrogram(self):
        data = self.pca_df
        
        fig = plt.figure(figsize=(15, 10))
        fig.suptitle('Dendrogram', fontsize=20)
        
        shc.set_link_color_palette(['r', 'g'])
        
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.axhline(y=7, ls='--', c='k')

        dend = shc.dendrogram(shc.linkage(data, method='ward'), 
                              labels=data.index + 1,
                              color_threshold=5)
    
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.axhline(y=7, ls='--', c='k')
        dend = shc.dendrogram(shc.linkage(data, method='ward'),
                              truncate_mode='level',
                              p=2,
                              labels=data.index + 1,
                              color_threshold=0)
        
        return fig
    
    def df_vectorize(self):
        vectorize = CountVectorizer()
        new_df = pd.DataFrame(vectorize.fit_transform(self.top10_token).toarray(), columns=vectorize.get_feature_names_out())
        
        return new_df
    
    def pca(self):
        pc = PCA(n_components=2)
        pca = pc.fit_transform(self.new_df)
        pca_df = pd.DataFrame(data=pca, columns=['x', 'y'])
        
        return pca, pca_df
    
    def Doc2Vec(self, top10_token):
        train_documents_top10 = [TaggedDocument(words=list(text.split(' ')), tags=[i]) for i, text in enumerate(top10_token)]
        model_top10 = gensim.models.doc2vec.Doc2Vec(train_documents_top10, vector_size=100, min_count=1, window = 5, seed = 10)
        model_top10.build_vocab(train_documents_top10)
        model_top10.train(train_documents_top10, total_examples=model_top10.corpus_count, epochs=10, start_alpha=0.002, end_alpha=-0.016)
        x = model_top10.dv.vectors
        
        return x
    
    def umap(self):
        clusterable_embedding = umap_.UMAP(
            n_neighbors=30,
            min_dist=0.0,
            n_components=2,
            random_state=10,
        ).fit_transform(self.x).astype('float64')
       
        return clusterable_embedding
    
    def sil_score(self):
        range_n_clusters = range(2, 11)
        
        for n_clusters in range_n_clusters:
            clusterer = KMeans(n_clusters=n_clusters)
            preds = clusterer.fit_predict(self.clusterable_embedding)
            score = silhouette_score(self.clusterable_embedding, preds)
            score = np.round(float(score), 4)
            st.write(f'n_clusters={n_clusters}, sihouette score={score}')
    
    def silhouette_visualizer(self):
        fig, ax = plt.subplots(3, 2, figsize=(15, 15))

        for i in range(2, 8):
            km = KMeans(n_clusters=i, n_init=10, max_iter=300, random_state=10)
            q, mod = divmod(i, 2)

            visualizer = SilhouetteVisualizer(km, colors='yellowbrick', ax=ax[q-1][mod]).fit(self.clusterable_embedding)
            visualizer.finalize()
            ax[q-1][mod].set_title(f'n_clusters={i}')
            ax[q-1][mod].xaxis.set_visible(False)
        
        return fig
        
    def kelbow_visualizer(self):
        fig, ax = plt.subplots()
        
        visualizer = KElbowVisualizer(KMeans(random_state=10), k=(2, 14), ax=ax).fit(self.clusterable_embedding)
        visualizer.finalize()
        
        return fig

    def UMAP_show(self, n=4):
        fig = plt.figure(figsize = (12, 12), facecolor = 'ghostwhite')
        ax = fig.add_subplot()
        ax.axis("off")
        km = KMeans(init='k-means++', n_clusters=n, n_init=10, random_state = 10).fit(self.clusterable_embedding)

        h = 0.01

        x_min, x_max = self.clusterable_embedding[:, 0].min() - 1, self.clusterable_embedding[:, 0].max() + 1
        y_min, y_max = self.clusterable_embedding[:, 1].min() - 1, self.clusterable_embedding[:, 1].max() + 1
        x_vals, y_vals = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        output = km.predict(np.c_[x_vals.ravel(), y_vals.ravel()])
        output = output.reshape(x_vals.shape)
        
        plt.imshow(output, interpolation='nearest', extent=(x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()),
        cmap='Pastel2', aspect='auto', origin='lower')
        plt.scatter(
        km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
        s=1000, marker='X',
        c='#6667AB', edgecolor='white',
        alpha=0.7,
        label='centroids'
        )
        candidate = (np.where(km.labels_ == km.labels_[-1])[0] + 1).astype('str').tolist()
        candidate = ','.join(candidate[:len(candidate)-1])
        sns.scatterplot(self.clusterable_embedding[:, 0], self.clusterable_embedding[:, 1],hue=km.labels_, palette='gist_rainbow', s=300, alpha=0.6).set_title('Cluster UMAP', fontsize=15)

        for i in range(len(self.top10_token)):
            if i == 14:
                st.markdown(f'##### ❗ USER의 관심사를 포함하는 후보군은 후보 {candidate}이 포함된 그룹 {km.labels_[i]} 입니다. 😆')
                plt.annotate('USER', (self.clusterable_embedding[i,0], self.clusterable_embedding[i,1]), fontsize=20, color = 'red')
            else:plt.annotate(i + 1, (self.clusterable_embedding[i,0], self.clusterable_embedding[i,1]), fontsize=14)
            
        if len(self.top10_token) == 14:
            ax1 = fig.add_subplot()
            ax1.axis("off")
            for i, path in enumerate(self.image_path):
                img = mpimg.imread(path)
                imagebox = OffsetImage(img, zoom = 0.33)
                ab = AnnotationBbox(imagebox, (self.clusterable_embedding[i, 0], self.clusterable_embedding[i, 1]), frameon = True, xybox=(self.x2[i], self.y2[i]),
                                    xycoords='data', boxcoords="offset points",
                                    arrowprops=dict(arrowstyle="->"))
                ax.add_artist(ab)
                ax.set_zorder(2)
        
        plt.legend()
        
        return fig

# chapter5
# 5.1
from tensorflow import random
class ML():
    def __init__(self):
        # self.clear_sentences,self.top10 = self.make_clear_sentences()
        pass

    def vetorize_seq(self, seqs, dim=10000):
        results = np.zeros((len(seqs), dim))
        for i, seq in enumerate(seqs):
            try:
                results[i, seq] += 1
            except:
                pass
            # print()
        return results

    def setting_data(self, list, dim, word_idx):
        for i in list:
            for j in range(len(i)):
                try:
                    i[j] = word_idx[i[j]]
                except:
                    i[j] = 0
        for i in list:
            for j in range(len(i)):
                if i[j] > dim:
                    i[j] = 0
        return self.vetorize_seq(list, dim)

    def preprocessing_ml_data(self, label_con, word_idx, dim):
        # dim = len(word_idx) 3900-
        train_data = []
        train_label = []
        for i in range(1, label_con + 1):
            temp_data = pre.lines_token[i - 1]
            temp_label = [i - 1] * len(temp_data)
            train_data.extend(temp_data)
            train_label.extend(temp_label)
        train_data = self.setting_data(train_data, dim, word_idx)
        train_label = np.array(train_label)
        return train_data, train_label

    def ml(self, label_con, train_data, train_label):

        random.set_seed(50)

        model = models.Sequential()
        model.add(layers.Dense(300, activation='relu'))
        model.add(layers.Dense(50, activation='relu'))

        model.add(layers.Dense(label_con, activation='softmax'))

        # opt = optimizers.Adam(learning_rate=0.01)
        opt = optimizers.adam_v2.Adam(learning_rate=0.01)
        loss = losses.sparse_categorical_crossentropy
        model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])

        model.fit(train_data, train_label, epochs=5, batch_size=50)  #
        return model

    def show_similar_candidate(self, target_string, model, dim):
        target = model.predict(self.setting_data([okt.nouns(target_string)], dim, pre.word_idx))
        target = target.squeeze()
        idx = []
        acc = []
        for i in range(len(target)):
            if target[i] > 0.10:
                idx.append(i + 1)
                acc.append(round(target[i] * 100, 2))
        return idx, acc

    def tf_idf_score(self):

        tf_sentences = pre.tf_sentences
        stopwords = pre.stopwords
        t_vec = TfidfVectorizer(max_features=3000, stop_words=stopwords)
        tdm = t_vec.fit_transform(tf_sentences)
        t_vec.get_feature_names_out()
        tf_rank = pd.DataFrame({
            '단어': t_vec.get_feature_names_out(),
            'tf-idf': tdm.sum(axis=0).flat
        })

        return tf_rank.sort_values(by='tf-idf', ascending=False)

    def pred(self,target_string):
        key = okt.nouns(target_string)
        idx, acc = mlc.show_similar_candidate(target_string, model, dim)
        candidate_dict = {idx[i]: acc[i] for i in range(len(idx))}

        for i in key:
            if i not in pre.word_idx.keys():
                key.pop(key.index(i))
        temp = []
        if not key:
            return ['적용된 키워드가 없으므로 다시 검색해 주세요.']
        else:
            # print(f'적용된 키워드: {key}')
            for i in idx:
                x = 0
                cnt = 0
                print()
                for j in range(1, 11):
                    if x == 0:
                        temp.append(f'기호{i}번, {candidate_dict[i]}%')
                        x += 1
            return temp
mlc = ML()
label_con = 4
dim = 1500 # vector dimension
word_idx = pre.word_idx

train_data, train_label = mlc.preprocessing_ml_data(label_con, word_idx,dim=dim)
X_train, X_val, y_train, y_val = train_test_split(train_data,train_label, test_size=0.2,shuffle=True)
model = mlc.ml(label_con,train_data,train_label) # 모델이 리턴됨.

# 5.2
class User_Kmeans(Kmeans_Visualization):
    def __init__(self, user_input):
        self.top10_token = sum(pre.top10_token,[])
        user_input_token = okt.nouns(user_input)
        if len(self.top10_token) < 15:
            self.top10_token.append(' '.join(user_input_token))
            self.top10_token_user = self.top10_token
        else:
            self.top10_token.pop()
            self.top10_token.append(' '.join(user_input_token))
            self.top10_token_user = self.top10_token
        
        self.x = self.Doc2Vec(self.top10_token)
        clusterable_embedding = self.umap()
        self.clusterable_embedding = clusterable_embedding

# 5.3
class SearchPromise():
    def __init__(self):
        self.df = pre.promise140
        self.word_dict = self.word_vectorized()
        self.df_reshaped_vector = self.promise_vectorized()

    def dict_promise(self):
        dict_df = {value: self.df['기호'][idx] for idx, value in zip(self.df.index, self.df['정책 공약'].values)}

        return dict_df

    def word_vectorized(self):
        tokens = []
        for text in self.df['정책 공약_tokens'].values.tolist():
            token = [word for word in text.split()]
            tokens.append(token)

        model = Word2Vec(tokens, min_count=1, window=3, workers=3, seed=1)

        word_dict = {}
        for vocab in model.wv.index_to_key:
            word_dict[vocab] = model.wv[vocab]

        return word_dict

    def promise_vectorized(self):
        dict_vector = {}
        df_reshaped_vector = self.df.copy()
        for idx in df_reshaped_vector.index:
            list_vector = []
            for word in self.df['정책 공약_tokens'][idx].split():
                if word in self.word_dict.keys():
                    list_vector.append(self.word_dict[word])
            dict_vector[df_reshaped_vector['정책 공약'][idx]] = np.sum(list_vector, axis=0).tolist()

        df_reshaped_vector['vector'] = df_reshaped_vector['정책 공약'].map(dict_vector)

        return df_reshaped_vector

    def user_input_to_vector(self, user_input):
        tokenized_input = okt.nouns(user_input)

        list_vector = []

        for word in tokenized_input:
            if word in self.word_dict.keys():
                list_vector.append(self.word_dict[word])
        if len(list_vector) != 0:
            user_vector = (np.sum(list_vector, axis=0) / len(list_vector)).tolist()
        else:
            user_vector = 0

        return user_vector

    def find_similar_promise(self, user_input):
        user_vector = self.user_input_to_vector(user_input)
        if user_vector == 0:
            result = '없음'
        else:
            similarity = {}

            for idx in self.df_reshaped_vector.index:
                sim = cosine_similarity(np.array(user_vector).reshape(1, -1),
                                        np.array([float(i) if i != '.' else float('0.0') for i in
                                                  str(self.df_reshaped_vector.loc[idx, 'vector'])[1:-1].split(
                                                      ', ')]).reshape(1, -1))
                similarity[self.df_reshaped_vector['정책 공약'][idx]] = float(sim)
            similarity = {key: value for key, value in
                          sorted(similarity.items(), key=lambda item: item[1], reverse=True)}
            rating = [str(key) for key, value in sorted(similarity.items(), key=lambda item: item[1], reverse=True)]
            top = rating[:3]
            result = {}

            for i in top:
                result[i] = str(abs(round(similarity[i] * 100, 2))) + '%'

        return result

    def show_similar_promise(self, user_input):
        result = self.find_similar_promise(user_input)
        dict_df = self.dict_promise()
        if result == '없음':
            st.warning('관심사를 다시 입력해주세요.')
        else:
            st.success('검색이 완료되었습니다.')
            for i in result.keys():
                st.write(f'[{dict_df[i]}] {i} : {result[i]}')







