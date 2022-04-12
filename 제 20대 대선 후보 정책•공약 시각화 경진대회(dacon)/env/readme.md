# m1 기준 가상환경에서 텐서플로우 2.8.0 을 설치하는 방법입니다.

## 설치하기 전에 Xcode와 miniforge가 설치되어 있어야 합니다.
관련 명령어만 첨부하도록 하겠습니다.
```
xcode-select --install ## xcode 설치

chmod +x ~/Downloads/Miniforge3-MacOSX-arm64.sh
sh ~/Downloads/Miniforge3-MacOSX-arm64.sh
source ~/miniforge3/bin/activate
```
1. 터미널을 키고 아래에 해당하는 명령어를 입력합니다.

```
conda create -n 환경이름 python=3.9
conda activate 환경이름
```

> 위의 명령어를 입력했다면 3.9버전의 파이썬 가상환경을 만들었고, 해당 가상환경으로 들어왔습니다.
2. 텐서플로우를 설치해줍니다. 여기서는 m1기준이기 때문에 설치 방법이 다를 수 있습니다.
```
conda install -c apple tensorflow-deps==2.8.0
python -m pip install tensorflow-macos==2.8.0
python -m pip install tensorflow-metal==0.4.0
```

3. 여기서 원하는 라이브러리를 설치해줍니다. 관련 라이브러리는 requirements.txt에 버전과 함께 첨부해두었습니다.

```
conda install [numpy 등등]
```