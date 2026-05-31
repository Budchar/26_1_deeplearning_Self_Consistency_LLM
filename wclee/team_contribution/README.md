# 팀 레포 기여 패키지 — wclee

이 폴더를 `26_1_deeplearning_Self_Consistency_LLM/` 루트에 그대로 복사하면 됩니다.

## 복사 구조

```
26_1_deeplearning_Self_Consistency_LLM/
├── wclee/                          ← 이 폴더 통째로 복사
│   ├── README.md
│   ├── code/
│   ├── docs/
│   ├── results/
│   └── plots/
└── 참고자료/
    └── REFERENCES.md               ← REFERENCES_추가분.md 내용 append
```

## 복사 명령 (팀 레포 클론 후)

```bash
# 팀 레포 클론
git clone https://github.com/Budchar/26_1_deeplearning_Self_Consistency_LLM.git
cd 26_1_deeplearning_Self_Consistency_LLM

# wclee 폴더 복사
cp -r /path/to/team_contribution/wclee .

# REFERENCES 추가
cat /path/to/team_contribution/REFERENCES_추가분.md >> 참고자료/REFERENCES.md

# 커밋
git add wclee/ 참고자료/REFERENCES.md
git commit -m "Add wclee: layer mechanism analysis (Exp06/11/12, Mistral family)"
git push
```
