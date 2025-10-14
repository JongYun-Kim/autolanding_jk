### (이미 gym 지웠다면 생략 가능)
```bash
pip uninstall -y gym
```

##3 유지/확인
```bash
pip install "gymnasium==1.1.1" "numpy==2.2.2"
```

### Hydra 최신 + submitit 플러그인
```bash
pip install "hydra-core==1.3.2" "hydra-submitit-launcher==1.2.0"
```

### 디테일, 절차
- `compat.py` 추가
- 데코레이터 gym.Env 상속 대상에 추가 (`BaseAviary`)
- Hydra 데코레이터 변경
  - `@hydra.main(version_base="1.3", config_path="cfgs", config_name="config_gimbal_curriculum")`
- 가능한경우 전부 reset에서 `**kwargs`를 전달
- 전부 gym -> gymnasium or `import gymnasium as gym`
- 
