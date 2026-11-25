# 조성변수 보존 예외의 문헌 근거

검토일: 2026-11-17

- 적용 설정: `module1_preprocessing/config.py`의 `REFERENCE_EXCEPTION_RANGES`
- 아래 값은 원료의 원소 질량분율 wt%이며 침출액 농도나 회수율이 아님
- 문헌의 보고값을 경계로 정상범위까지 연결한 구간은 프로젝트의 보존 기준임 — 논문이 해당 구간 전체를 정상범위로 인증한 것은 아님
- 정상범위 이탈값 중 예외에 해당하는 값은 원값과 플래그를 유지하고, 기준 밖 값은 결측으로 변환함
- 기준 밖이라는 이유만으로 측정 오류가 입증되는 것은 아니며 원본을 수정하지 않음

## 설정과 출처 대응

| 변수 | 데이터사전 정상범위 | 보존 예외 | 근거 보고값 및 위치 |
|---|---|---|---|
| Ni | 8–30 | 5.9 ≤ Ni < 8, 30 < Ni ≤ 38.2 | [1] Table 3 NMC 행 5.9 / [6] Table 2 산업 블랙매스 38.2 |
| Co | 2–15 | 없음 | 정상범위 유지 |
| Li | 2–6 | 1.81 ≤ Li < 2, 6 < Li ≤ 6.76 | [7] 인증값 1.81 / [8] Table 1 6.76 |
| Mn | 5–16 | 없음 | [2] Table 1 15.95는 정상범위 안이므로 상한 확대 근거로 사용하지 않음 |
| Cu | 0.3–6 | 0.05 ≤ Cu < 0.3, 6 < Cu ≤ 6.08 | [3] Table A1 0.05 / [8] Table 1 6.08 |
| Fe | 0.2–6 | 0.02 ≤ Fe < 0.2 | [4] §3.1 0.02 ± 0.005의 중심값 사용 |
| Al | 0.5–7 | 0.13 ≤ Al < 0.5, 7 < Al ≤ 7.46 | [9] Table 1 0.13 ± 0.00 / [5] 인증값 7.46 |

## 사용자 제공 자료 5건

### [1] Ni 하한

Wiszniewski et al. (2024), *Evaluating refractory material performance in pyrometallurgical recycling of lithium-ion batteries under a reducing atmosphere*, Ceramics International 50, 43683–43698

- [출판사 원문](https://www.sciencedirect.com/science/article/pii/S0272884224036782) · [DOI](https://doi.org/10.1016/j.ceramint.2024.08.220) · [공개 PDF](https://www.k1-met.com/fileadmin/user_upload/Modul_FuLIBatteR/Publikationen/20241011_FuLIBatteR_Publikation_1-s2.0-S0272884224036782-main.pdf)
- Table 3, NMC 행의 Ni 5.9 wt% 확인함
- 표 제목: “Chemical composition of different black mass samples in wt.%”
- NMC계 폐전지 블랙매스의 조성 참고값으로 사용하며 다른 행의 합성 시료와 구분함

### [2] Mn 상한 검토 자료 — 예외 확대에 미사용

*Recovery of graphite from industrial lithium-ion battery black mass* (2025), RSC Sustainability 3, 264–274

- [원문](https://pubs.rsc.org/su/article/3/1/264/872757/Recovery-of-graphite-from-industrial-lithium-ion) · [DOI](https://doi.org/10.1039/D4SU00427B)
- Table 1의 Black mass 행 Mn 15.95 wt% 확인함
- 산업 혼합 블랙매스의 값이며 회수된 흑연 제품의 조성과 구분함
- 16 wt% 이하이므로 Mn 16.02·16.99 보존 근거로 적용하지 않음
- 같은 표의 Cu 8.08 등은 현재 채택한 예외 경계와 다름 — 이번 작업에서 설정 범위를 추가 확대하지 않음

### [3] Cu 하한

Dittmer et al. (2026), *Experimental and environmental impact of early-stage lithium recovery in lithium-ion battery recycling*, Waste Management 211, 115283

- [공개 원문 PDF](https://publications.rwth-aachen.de/record/1024070/files/1024070.pdf) · [DOI](https://doi.org/10.1016/j.wasman.2025.115283)
- Table A1 “Composition of the used black mass”의 Cu 0.05 wt% 확인함
- NMC111 전지에서 열처리·파쇄·체분급한 블랙매스의 조성임
- 같은 표의 Fe 0.00을 정량된 절대 영점으로 해석하거나 Fe 하한 확대 근거로 사용하지 않음

### [4] Fe 하한

Aliyev et al., *Fluoride removal from NMC black mass leachates during Lithium-Ion battery recycling via aluminum sulfate precipitation*, Waste Management, article 115136

- [출판사 원문](https://www.sciencedirect.com/science/article/abs/pii/S0956053X25005471) · [DOI](https://doi.org/10.1016/j.wasman.2025.115136)
- **§3.1 Black mass and PLS composition**의 Fe 0.02 ± 0.005 wt% 확인함
- NMC 블랙매스를 왕수 분해 후 ICP-OES로 분석한 원료 조성임
- §2.1은 원료 소개이며 해당 정량값의 정확한 인용 위치는 §3.1임
- 오차 범위를 더해 보존 구간을 확장하지 않고 중심값 0.02 사용함

### [5] Al 상한

OREAS, 인증표준물질 AC18.10664

- [공식 인증값](https://www.oreas.com/crm/ac18-10664/)
- NMC622/811계 혼합 블랙매스의 Al 7.46 wt% 인증값 확인함
- 논문 실측 사례가 아닌 인증표준물질 자료로 구분함
- 인증 불확도 대신 인증값 자체를 경계로 사용함

## 나머지 경계값의 보완 자료

제공된 5건만으로 Ni 상한·Li 하한/상한·Cu 상한·Al 하한을 모두 설명할 수 없어 아래 자료를 함께 연결함

### [6] Ni 상한

Siame et al. (2026), *Understanding the Thermal Behavior of Black Mass during Recycling of Spent Lithium-Ion Batteries through Its Individual Components*, ACS Sustainable Chemistry & Engineering 14(4), 1952–1963

- [DOI 및 원문](https://doi.org/10.1021/acssuschemeng.5c10344)
- Table 2 “Chemical Composition of Industrial Black Mass”의 Ni 38.2 wt% 사용함
- 산업 블랙매스의 값이며 Table 1의 합성 조성과 구분함

### [7] Li 하한

OREAS, 인증표준물질 AC18.10665

- [공식 인증값](https://www.oreas.com/crm/ac18-10665/)
- NMC622계 블랙매스의 Li 1.81 wt% 인증값 사용함

### [8] Li·Cu 상한

Jung et al. (2024), *Feasibility Study of Methanesulfonic Acid (MSA), an Alternative Lixiviant to Improve Conventional Sulfuric Acid Leaching of NCM Black Mass*, Resources Recycling 33(1), 58–68

- [원문](https://j-kirr.or.kr/articles/xml/M2AP/) · [DOI](https://doi.org/10.7844/kirr.2024.33.1.58)
- Table 1 “Elemental composition of NCM black mass sample”의 Li 6.76, Cu 6.08 wt% 사용함
- 침출 전 NCM계 분말의 벌크 ICP-OES 조성임 — Table 2의 SEM-EDS 조성과 구분함
- 다른 양극 화학계 혼입 가능성이 언급된 시료임

### [9] Al 하한

Saleem, Buvik, Knuutila & Bandyopadhyay (2024), *Recovery of lithium from oxalic acid leachate produced from black mass of spent electric vehicle Li-ion batteries*, Chemical Engineering Journal Advances 20, 100648

- [DOI](https://doi.org/10.1016/j.ceja.2024.100648) · [대학 저장소](https://hdl.handle.net/11250/3156540) · [원문 PDF 검색 색인](https://ntnuopen.ntnu.no/ntnu-xmlui/bitstream/handle/11250/3156540/Saleem_Recovery%2Bof%2Blithium%2Bfrom%2Boxalic%2Bacid%2Bleachate%2Bproduced%2Bfrom%2Bblack%2Bmass.pdf?isAllowed=y&sequence=1)
- Table 1 “Elemental quantification (wt.) of black mass.”의 Al 0.13 ± 0.00 wt% 확인함
- 원문은 조성에 근거해 NMC111계일 가능성을 설명함 — 원료 화학계를 확정한 것으로 서술하지 않음
- 대학 저장소의 이전 PDF 주소는 새 저장소로 이동됨 현재 확인은 해당 원문 PDF의 검색 색인 표 및 저장소 서지정보를 이용함
