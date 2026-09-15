import os
import re
import argparse
import httpx
import pandas as pd
from dotenv import load_dotenv

# -----------------------------
# 0) 환경 변수 로드 (.env)
# -----------------------------
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

if not API_KEY:
    raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")


# -----------------------------
# 1) 시스템 프롬프트 읽기 (최대 3000자)
# -----------------------------
def load_system_prompt(path: str = "system_prompt.txt", limit: int = 3000) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()[:limit]


SYSTEM_PROMPT = load_system_prompt()


# -----------------------------
# 2) 유저 프롬프트 빌더
#    - 대회 형식과 유사하게 제목/본문을 전달
#    - 너무 긴 본문은 잘라서 보냄(임의 컷)
# -----------------------------
def build_user_prompt(title: str, content: str) -> str:
    return f"[기사]\n제목: {title[:300]}\n\n내용: {content[:4000]}"


# -----------------------------
# 3) GPT API 호출
#    - temperature=0.0: 결정적 출력 유도
#    - max_tokens=4: 0/1 한 글자만 기대
# -----------------------------
def gpt_predict(title: str, content: str) -> int:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(title, content)},
        ],
        "temperature": 0.4,
        "max_tokens": 4,
    }

    # httpx로 Chat Completions 엔드포인트 호출
    with httpx.Client(
        base_url=BASE_URL,
        timeout=60,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
    ) as client:
        resp = client.post("/chat/completions", json=payload)
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()

    # 첫 번째로 등장하는 '0' 또는 '1'을 예측값으로 사용
    m = re.search(r"[01]", text)
    if not m:
        # 숫자 외 텍스트가 나온 경우를 대비한 보수적 처리(여기선 0으로 둡니다)
        return 0
    return int(m.group(0))


def evaluate(csv_path: str, save_errors: str | None = None, max_show: int = 20):
    # CSV에는 반드시 label 컬럼이 있어야 함 (정답)
    df = pd.read_csv(csv_path)

    # title+content만 GPT에 전달하여 예측
    preds = []
    for _, row in df.iterrows():
        pred = gpt_predict(row["title"], row["content"])
        preds.append(pred)

    df["pred"] = preds
    df["correct"] = df["pred"].astype(int) == df["label"].astype(int)

    # 기본 정확도
    accuracy = df["correct"].mean()
    total = len(df)
    num_correct = int(df["correct"].sum())
    num_errors = total - num_correct

    # 혼동 행렬(간단)
    import numpy as np

    y_true = df["label"].astype(int).to_numpy()
    y_pred = df["pred"].astype(int).to_numpy()

    def cm_val(t, p):
        return int(np.sum((y_true == t) & (y_pred == p)))

    tp = cm_val(1, 1)
    tn = cm_val(0, 0)
    fp = cm_val(0, 1)
    fn = cm_val(1, 0)

    print(f"Total: {total} | Correct: {num_correct} | Errors: {num_errors}")
    print(f"Accuracy: {accuracy:.4f}")
    print("Confusion Matrix (rows=true, cols=pred)  |  pred=0   pred=1")
    print(f"true=0: {tn:6d} {fp:8d}")
    print(f"true=1: {fn:6d} {tp:8d}")

    # 오분류만 별도 추출
    err_df = df.loc[~df["correct"], ["title", "content", "label", "pred"]].copy()
    # 사람이 보기 좋게 기사 번호(1-based) 열 추가
    err_df.insert(0, "row_num", err_df.index + 1)

    if num_errors > 0:
        print("\n=== Misclassified rows (up to max_show) ===")
        # 터미널 프린트: 제목은 길면 잘라서
        for _, r in err_df.head(max_show).iterrows():
            title_snippet = (
                (r["title"][:80] + "…") if len(str(r["title"])) > 80 else r["title"]
            )
            print(
                f"[#{int(r['row_num']):>3}] label={int(r['label'])} pred={int(r['pred'])} | {title_snippet}"
            )

        if save_errors:
            # 오분류 CSV 저장
            err_df.to_csv(save_errors, index=False, encoding="utf-8-sig")
            print(f"\n오분류 {num_errors}건을 CSV로 저장했습니다: {save_errors}")
    else:
        print("\n오분류 없음! ✅")

    return df  # 필요 시 호출부에서 추가 분석 가능


# -----------------------------
# 5) CLI
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate GPT predictions vs labels (accuracy + error breakdown)"
    )
    parser.add_argument(
        "--csv", required=True, help="평가할 CSV 경로 (title,content,label 포함)"
    )
    parser.add_argument(
        "--save-errors", default=None, help="오분류만 저장할 CSV 경로(옵션)"
    )
    parser.add_argument(
        "--max-show",
        type=int,
        default=20,
        help="터미널에 표시할 오분류 최대 건수(기본 20)",
    )
    args = parser.parse_args()

    evaluate(args.csv, save_errors=args.save_errors, max_show=args.max_show)


# # -----------------------------
# # 4) CSV 평가 (정확도만 출력)
# # -----------------------------
# def evaluate(csv_path: str):
#     # CSV에는 반드시 label 컬럼이 있어야 함 (정답)
#     df = pd.read_csv(csv_path)

#     # title+content만 GPT에 전달하여 예측
#     preds = []
#     for _, row in df.iterrows():
#         pred = gpt_predict(row["title"], row["content"])
#         preds.append(pred)

#     df["pred"] = preds

#     # 정확도 = (pred == label)의 평균
#     accuracy = (df["pred"].astype(int) == df["label"].astype(int)).mean()
#     print(f"Accuracy: {accuracy:.4f}")


# # -----------------------------
# # 5) CLI
# # -----------------------------
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="Evaluate GPT predictions vs labels (accuracy only)"
#     )
#     parser.add_argument(
#         "--csv", required=True, help="평가할 CSV 경로 (id,title,content,label 포함)"
#     )
#     args = parser.parse_args()
#     evaluate(args.csv)
