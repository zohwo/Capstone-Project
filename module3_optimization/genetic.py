"""실수형 유전 알고리즘으로 제한 범위 안의 공정조건 탐색함"""
import numpy as np


def genetic_search(objective, bounds, seed, generations=60, population_size=36):
    bounds = np.asarray(bounds, dtype=float)
    rng = np.random.default_rng(seed)
    dimensions = len(bounds)
    population = rng.random((population_size, dimensions))
    history = []
    for generation in range(generations + 1):
        controls = bounds[:, 0] + population * (bounds[:, 1] - bounds[:, 0])
        scores = np.asarray(objective(controls), dtype=float)
        order = np.argsort(scores)
        history.append(float(scores[order[0]]))
        if generation == generations:
            break
        # 상위 2개체를 보존하고, 크기 3의 토너먼트 선택·변수별 산술 교차·가우시안 돌연변이로 자식을 생성함
        children = []
        while len(children) < population_size - 2:
            parents = []
            for _ in range(2):
                contestants = rng.integers(population_size, size=3)
                parents.append(population[contestants[np.argmin(scores[contestants])]])
            alpha = rng.random(dimensions)
            child = alpha * parents[0] + (1 - alpha) * parents[1]
            mutation = rng.random(dimensions) < 1 / dimensions
            child += mutation * rng.normal(0, 0.1, dimensions)
            children.append(np.clip(child, 0, 1))
        population = np.vstack([population[order[:2]], children])
    return dict(algorithm='ga', seed=int(seed), converged=None,
                message='설정한 세대 수 완료 — 수렴 판정은 별도로 수행하지 않음',
                iterations=generations, population_size=population_size,
                evaluations=population_size * (generations + 1),
                best_score_history=history)
