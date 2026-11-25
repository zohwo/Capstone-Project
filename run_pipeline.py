"""Module 1 전처리·Module 2 모델 학습을 순서대로 실행하고, feed 요청이 있으면 Module 3 추천까지 이어서 실행함"""
import argparse
from pathlib import Path

from module1_preprocessing.pipeline import run as run_module1
from module2_prediction.pipeline import run as run_module2
from module3_optimization.pipeline import run as run_module3


def run_pipeline(raw_input_path, cleaned_output_path='data/processed/cleaned_features.csv',
                  model_output_dir='module2_prediction/reports/final', feed_request_path=None,
                  recommendation_output_dir='module3_optimization/reports/recommendation'):
    """Module 1 → Module 2 → (feed_request_path가 있으면) Module 3 순서로 호출함"""
    run_module1(raw_input_path, cleaned_output_path)
    pre_impute_path = Path(cleaned_output_path).parent / 'pre_impute_data.csv'
    module2_results = run_module2(data_path=pre_impute_path, output_dir=model_output_dir)

    module3_recommendation = None
    if feed_request_path is not None:
        module3_recommendation = run_module3(feed_request_path, model_dir=model_output_dir,
                                              output_dir=recommendation_output_dir)
    return dict(module2_results=module2_results, module3_recommendation=module3_recommendation)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-input', required=True, help='Module 1 원본 입력 경로(xlsx/xls/csv)')
    parser.add_argument('--cleaned-output', default='data/processed/cleaned_features.csv')
    parser.add_argument('--model-output-dir', default='module2_prediction/reports/final')
    parser.add_argument('--feed-request', default=None, help='지정하면 Module 3 추천까지 이어서 실행함')
    parser.add_argument('--recommendation-output-dir', default='module3_optimization/reports/recommendation')
    args = parser.parse_args()

    result = run_pipeline(args.raw_input, args.cleaned_output, args.model_output_dir,
                           args.feed_request, args.recommendation_output_dir)
    print('Module 2 결과:', result['module2_results'])
    if result['module3_recommendation'] is not None:
        print('Module 3 추천:', result['module3_recommendation'])


if __name__ == '__main__':
    main()
