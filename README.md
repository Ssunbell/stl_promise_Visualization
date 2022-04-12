# dacon 제 20대 대선 후보 정책•공약 시각화 경진대회
- 해당 Repository는 제 20대 대선 후보 정책•공약 시각화 경진대회 제출 자료입니다.

## 사용 방법
1. `git clone https://github.com/Trailblazer-Yoo/stl_promise_Visualization`을 명령창 혹은 터미널에 입력해주시면 모든 디렉토리 및 파일이 다운로드 됩니다.
2. `env`폴더에서 윈도우 및 m1 운영체제에서 가상환경을 만들어 필요한 라이브러리를 다운받을 수 있도록 배치 파일을 제공합니다. 사용방법은 다음과 같습니다.

### 1. 윈도우
1. `Window_virtualenvironment` 디렉토리로 들어가시면 두개 파일이 보이실텐데 그 중 environment_Windowss.bat을 실행시킵니다.
2. `semi_proj`라는 python 3.9 버전의 가상환경이 만들어지고 자동으로 라이브러리를 다운받습니다.
> 만약 필요한 라이브러리가 다운로드가 안된다면 requirements.txt 에서 해당 라이브러리의 버전을 확인하여 설치하거나 최신 버전을 설치해주시기 바랍니다.
> konlpy는 `jpype`와 자바가 필요로 합니다. 만약 설치가 안되신다면 해당 라이브러리를 구글링을 통해 따로 설치해주시기 바랍니다.
3. 라이브러리를 다운받은 후에 streamlit 로컬 서버 창이 실행됩니다. 해당 창에서 다양한 시각화 결과를 확인할 수 있습니다.
> 해당 로컬은 `http://share.streamlit.io/trailblazer-yoo/streamlit_promise/promise.py`에서 동일하게 확인할 수 있습니다.

### 2. m1
1. `M1_virtualenvironment` 디렉토리로 들어가시면 3개의 배치파일과 `requirements.txt`가 보이실겁니다.
2. m1은 `tensorflow`를 설치하기 위해 `xcode`와 `miniforge`를 필요로 합니다.
3. 따라서 설치가 되어있으시다면 `xcode_m1`과 `miniforge_m1`을 실행하지 않으셔도 됩니다.
> 만약 필요하시다면 `xcode_m1`부터 실행해주시고 `miniforge_m1`을 설치하시기 바랍니다.
> miniforge는 맥에 anaconda가 설치되어 있다면 충돌을 일으킬 수 있기 때문에 anaconda를 삭제하시는 걸 권장드립니다.
4. 마지막으로 `virtualenvironment_m1`을 실행시켜주시면 miniforge에 `semi_proj` 가상환경을 설치하여 라이브러리를 설치합니다.
> gensim wordcloud konlpy 라이브러리는 conda로 설치하지 않고 pip으로 설치됩니다. 따라서 해당 라이브러리가 제대로 설치되지 않을경우 따로 설치해주시기 바랍니다.