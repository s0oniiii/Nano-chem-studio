# Nano-Chem Studio

SEM 이미지 입자 분석을 위한 Streamlit 웹 애플리케이션입니다.

## 기능

- **SEM 입자 분석**: SEM 이미지를 업로드하여 입자 개수를 자동으로 분석합니다.
- **3D 화학 랩**: (구현 예정)

## 설치 방법

1. 필요한 라이브러리 설치:
```bash
pip install -r requirements.txt
```

## 실행 방법

```bash
streamlit run app.py
```

브라우저에서 자동으로 열리며, 기본 주소는 `http://localhost:8501`입니다.

## 사용 방법

1. 왼쪽 사이드바에서 "SEM 입자 분석" 메뉴를 선택합니다.
2. "SEM 이미지를 업로드하세요" 섹션에서 JPG 또는 PNG 형식의 이미지를 업로드합니다.
3. 업로드된 이미지가 자동으로 처리되어 입자 개수가 표시됩니다.
