"""정상범위(NORMAL_RANGES)를 벗어난 실제 값을 확인하는 스크립트

사용자가 직접 원본 파일 경로를 인자로 넘겨 실행해야 함

실행:
    python -m module1_preprocessing.scripts.show_outlier_values <원본_엑셀_파일_경로>
"""

import sys
from pathlib import Path

from module1_preprocessing.validation import check_ranges, format_range_violations
from module1_preprocessing.preprocessing import clean_schema, load_raw


def main():
    if len(sys.argv) < 2:
        print("사용법: python show_outlier_values.py <원본 엑셀 파일 경로>")
        sys.exit(1)

    input_path = Path(sys.argv[1])

    df = load_raw(input_path)
    df = clean_schema(df)

    print('데이터 정상범위 이탈 현황:')
    print(format_range_violations(df, check_ranges(df)))


if __name__ == "__main__":
    main()
