# Automotive News Classification Prompt

[2025 자동차 데이터 분석 경진대회: GPT를 활용한 뉴스클리핑 만들기](https://dacon.io/competitions/official/236577/overview/description)의 1차 예선 과제를 로컬 CSV로 실험하기 위한 코드입니다. 뉴스 제목과 본문을 GPT-4o mini에 전달해 자동차 관련 뉴스가 아니면 `0`, 관련 뉴스이면 `1`로 분류하고 정답 레이블과 비교합니다.

대회는 자동차산업 오픈 생태계 플랫폼의 뉴스 데이터를 활용해 뉴스클리핑 AI Agent의 핵심 모듈을 개발하는 과제였습니다. 1차 예선은 자동차 관련 뉴스 이진 분류를 위한 system prompt를 평가하고, 이후 단계는 뉴스클리핑 AI Agent 시스템 제안과 구축으로 확장됐습니다.

이 저장소는 뉴스클리핑 AI Agent의 구성 요소 중 **뉴스 분류 system prompt를 반복 평가하는 이진 분류 모듈**을 담고 있습니다.

## 구현 방식

1. `system_prompt.txt`에서 앞 3,000자를 읽습니다.
2. CSV의 `title`과 `content`를 대회 입력 형식과 비슷한 user prompt로 만듭니다.
3. OpenAI Chat Completions API를 호출합니다.
4. 응답에서 처음 등장하는 `0` 또는 `1`을 예측값으로 사용합니다.
5. 숫자가 전혀 없으면 보수적으로 `0`을 반환합니다.
6. 정답 `label`과 비교해 Accuracy와 confusion matrix를 출력합니다.
7. 오분류 행을 화면에 표시하고 선택적으로 CSV로 저장합니다.

API 설정은 다음과 같습니다.

| 항목 | 값 |
| --- | --- |
| 기본 모델 | `gpt-4o-mini` |
| Temperature | `0.4` |
| Max tokens | `4` |
| 제목 최대 길이 | 300자 |
| 본문 최대 길이 | 4,000자 |
| System prompt 사용 길이 | 앞 3,000자 |

## 실행 환경 설치

Python 3.10 이상을 사용합니다.

```bash
git clone https://github.com/ChoiSeongyong/Automotive-Competition.git
cd Automotive-Competition

python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

저장소 루트에 `.env` 파일을 만들고 API 설정을 입력합니다.

```dotenv
OPENAI_API_KEY=your_api_key
OPENAI_MODEL=gpt-4o-mini
OPENAI_BASE_URL=https://api.openai.com/v1
```

`.env`는 Git에서 제외되며 API 키를 저장소에 커밋하지 않습니다.

## 실행 방법

입력 CSV에는 최소한 `title`, `content`, `label` 컬럼이 있어야 합니다. `label`은 자동차 관련 없음 `0`, 관련 있음 `1`입니다.

```bash
python3 runner.py --csv data/samples.csv
```

오분류 결과를 별도 파일로 저장하려면:

```bash
python3 runner.py \
  --csv data/samples.csv \
  --save-errors misclassified.csv \
  --max-show 20
```

출력에는 전체 샘플 수, 정답 수, 오류 수, Accuracy, confusion matrix와 오분류 예시가 포함됩니다.

## 주요 파일

| 경로 | 역할 |
| --- | --- |
| `runner.py` | 프롬프트 로드, API 호출, 0/1 파싱과 평가를 수행하는 CLI |
| `system_prompt.txt` | 자동차 관련성 판정 기준과 few-shot 예시 |
| `data/samples.csv` | 프롬프트 점검용 레이블 CSV |
| `data/samples2.csv` | 추가 평가 샘플 |
| `data/extended_samples.csv` | 확장된 평가 샘플 |
| `requirements.txt` | Python 패키지 목록 |

## 프롬프트 설계 기준

System prompt는 차량 판매·생산·부품·EV 충전·자율주행·모빌리티 서비스를 자동차 관련 뉴스로 분류하고, 자동차 적용이 핵심이 아닌 에너지·IT·반도체·로봇·항공 기사는 제외하도록 구성했습니다. 배터리, 관세, 기업명처럼 문맥에 따라 의미가 달라지는 사례는 별도 우선순위와 few-shot 예시로 구분했습니다.
