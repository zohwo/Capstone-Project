import pandas as pd
import numpy as np
from pathlib import Path

def read_data_file(file_path):
    """
    엑셀 또는 CSV 파일을 읽어 DataFrame으로 반환하는 함수
    파일이 존재하지 않거나 읽을 수 없는 경우 None을 반환
    """
    try:
        ext = Path(file_path).suffix.lower()

        if ext in [".xlsx", ".xls"]:
            df = pd.read_excel(file_path)
        elif ext == ".csv":
            df = pd.read_csv(file_path)
        else:
            print(f"지원하지 않는 파일 형식입니다: {ext}")
            return None

        return df

    except Exception as e:
        print(f"파일을 읽을 수 없습니다. : {e}")
        return None

def convert_data(df):
    """
    DataFrame의 데이터를 datetime으로 변환하는 함수
    변환할 수 없는 값은 NaN으로 처리
    """
    try:
        df["분석일자"] = pd.to_datetime(df["분석일자"], format="mixed", errors='coerce')
        df["분석일자"] = df["분석일자"].dt.strftime('%Y-%m-%d') 
        return df
    except Exception as e:
        print(f"데이터 변환 중 오류 발생: {e}")
        return None


if __name__ == "__main__":
    # 엑셀 파일 경로 설정
    excel_file_path = Path("data/raw/영풍그룹_금속회수율최적화_RawMaterialData_학생배포용.xlsx")

    # CSV 파일 경로 설정
    #csv_file_path = Path("data/raw/영풍그룹_금속회수율최적화_RawMaterialData_학생배포용.csv")

    # 엑셀 파일 읽기
    df = read_data_file(excel_file_path)

    #df = read_csv_file(csv_file_path)
    if df is not None:
        print("파일을 성공적으로 읽었습니다.")
        print(df.head())
    else:
        print("파일을 읽는 데 실패했습니다.")

    # 데이터 변환
    if df is not None:
        converted_df = convert_data(df)
        if converted_df is not None:
            print("데이터 변환이 성공적으로 완료되었습니다.")
            print(converted_df.head())
        else:
            print("데이터 변환에 실패했습니다.")

    print(df.head(20))

    print(df.shape)
    print(df.info())