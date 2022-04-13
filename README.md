# dacon 제 20대 대선 후보 정책•공약 시각화 경진대회
- 해당 Repository는 __제 20대 대선 후보 정책•공약 시각화 경진대회__ 제출 자료입니다.
- http://share.streamlit.io/trailblazer-yoo/streamlit_promise/promise.py 로 들어가시면 해당 자료를 streamlit을 통해 확인하실 수 있습니다.

## 사용 방법
<img src="https://user-images.githubusercontent.com/97590480/163096295-2a30e113-9171-4d68-9e32-94c8ee5df191.png">

1. `git clone https://github.com/Trailblazer-Yoo/stl_promise_Visualization`을 __명령창(cmd) 혹은 터미널__ 에 입력해주시면 모든 디렉토리 및 파일이 다운로드 됩니다.
<img width = "50%" src="https://user-images.githubusercontent.com/97590480/163096415-031d24ed-2a17-42b0-bb80-4b14c8ab34e8.png">

2. `env`폴더에서 윈도우 및 mac m1 운영체제에서 __가상환경__ 을 만들어 필요한 라이브러리를 다운받을 수 있도록 __배치 파일__ 을 제공합니다. 사용방법은 다음과 같습니다.

### 1. 윈도우
<img src="https://user-images.githubusercontent.com/97590480/163096474-fd8c39ae-2756-4e81-a3af-7c1db04d345b.png">

1. `Window_virtualenvironment` 디렉토리로 들어가시면 두개 파일이 보이실텐데 그 중 __environment_Windowss.bat__ 을 실행시킵니다.
2. `semi_proj`라는 __python 3.9__ 버전의 가상환경이 만들어지고 자동으로 라이브러리를 다운받습니다.
> 만약 필요한 라이브러리가 다운로드가 안된다면 requirements.txt 에서 해당 라이브러리의 버전을 확인하여 설치하거나 최신 버전을 설치해주시기 바랍니다.
> konlpy는 __`jpype`와 자바__ 가 필요로 합니다. 만약 설치가 안되신다면   
> `https://velog.io/@soo-im/konlpy-설치-에러-해결책-아나콘다-JPYPE`로 들어가셔서 설치해주시기 바랍니다.
3. 라이브러리를 다운받은 후에 __streamlit 로컬 서버__ 창이 실행됩니다. 해당 창에서 다양한 시각화 결과를 확인할 수 있습니다.
   - 실행시키는 방법은 cmd창을 여시고 `promise.py`파일이 들어있는 폴더(여기서는 streamlit_frontendservice입니다)로 들어가서셔 다음 명령어를 실행시켜주시면 됩니다.
   - `streamlit run promise.py`
   - 맨 위에 링크는 로컬이 아닌 streamlit 서버에서 제공하는 url로 로컬에서 실행한 것과 동일합니다.
<img width = "50%" src="https://user-images.githubusercontent.com/97590480/163096631-baf8f862-769f-4c0c-98bb-5ed86127c50d.png">

### 2. m1
<img src="https://user-images.githubusercontent.com/97590480/163097020-24787a7c-8690-493a-a4f0-59a511c1bf04.png">

1. `M1_virtualenvironment` 디렉토리로 들어가시면 3개의 배치파일과 `requirements.txt` 그리고 `제일먼저_이것부터_터미널에_실행해주세요.txt`가 보이실겁니다.
2. 제일 먼저 빨간색 1번인 `제일먼저_이것부터_터미널에_실행해주세요.txt`를 여시고 해당 텍스트를 복사합니다.
<img width = "50%" src="https://user-images.githubusercontent.com/97590480/163097022-9c7c4e6d-548d-4701-a72f-468269a554ec.png">

3. 그리고 해당 디렉토리에서 터미널을 여시고 붙여넣고 엔터를 치시면 됩니다.
4. m1은 `tensorflow`를 설치하기 위해 `xcode`와 `miniforge`를 필요로 합니다.
5. 따라서 __설치가 되어있으시다면__ `xcode_m1`과 `miniforge_m1`을 실행하지 않으셔도 됩니다.
> 만약 __설치가 되어있지 않다면__ 스크롤을 내려주시고 3번에서 설명드리겠습니다.
1. 마지막으로 `virtualenvironment_m1`을 실행시켜주시면 miniforge에 `semi_proj` 가상환경을 설치하여 라이브러리를 설치합니다.
> gensim wordcloud konlpy 라이브러리는 conda로 설치하지 않고 pip으로 설치됩니다. 따라서 해당 라이브러리가 제대로 설치되지 않을경우 따로 설치해주시기 바랍니다.