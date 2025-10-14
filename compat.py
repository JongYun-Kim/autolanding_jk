# compat.py
import types, inspect

def to_legacy_iface(env):
    if getattr(env, "_legacy_iface_patched", False):
        return env
    env._legacy_iface_patched = True

    orig_reset = env.reset       # 바운드 메서드
    orig_step  = env.step

    def _call_reset_tolerant(seed, options, **kwargs):
        """하위 reset이 seed/options를 못 받아도 안전하게 호출"""
        # 1) space 시딩(가능하면)
        if seed is not None:
            for sp_name in ("action_space", "observation_space"):
                sp = getattr(env, sp_name, None)
                if hasattr(sp, "seed"):
                    try:
                        sp.seed(seed)
                    except Exception:
                        pass

        # 2) seed/options가 이미 제거된 kwargs를 가정하고 호출
        try:
            return orig_reset(**kwargs)   # 위치 인자 쓰지 않음
        except TypeError:
            # 방어적 재시도: 혹시 남아있다면 한 번 더 제거
            k2 = dict(kwargs)
            k2.pop("seed", None)
            k2.pop("options", None)
            return orig_reset(**k2)

    def reset_compat(self, *args, **kwargs):
        # 상층(Gymnasium)에서 내려오는 seed/options를 추출
        seed    = kwargs.get("seed", None)
        options = kwargs.get("options", None)

        # ★ 충돌 방지: kwargs에서 seed/options 제거
        clean_kwargs = dict(kwargs)
        clean_kwargs.pop("seed", None)
        clean_kwargs.pop("options", None)

        out = _call_reset_tolerant(seed, options, **clean_kwargs)

        # Gymnasium (obs, info) → 레거시 obs
        if isinstance(out, tuple) and len(out) >= 1:
            return out[0]
        return out

    def step_compat(self, action):
        out = orig_step(action)
        if isinstance(out, tuple):
            if len(out) == 5:
                o, r, term, trunc, info = out
                return o, r, bool(term or trunc), info
            if len(out) == 4:
                return out
        raise RuntimeError(f"Unexpected step() return: {out}")

    def seed_compat(self, seed=None):
        try:
            env.reset(seed=seed)
        except TypeError:
            env.reset()
        for sp_name in ("action_space", "observation_space"):
            sp = getattr(env, sp_name, None)
            if hasattr(sp, "seed"):
                sp.seed(seed)
        return [seed]

    env.reset = types.MethodType(reset_compat, env)
    env.step  = types.MethodType(step_compat, env)
    if not hasattr(env, "seed") or inspect.isfunction(getattr(env, "seed")):
        env.seed = types.MethodType(seed_compat, env)
    return env

def legacy_iface_class(cls):
    orig_init = cls.__init__
    def __init__(self, *a, **kw):
        orig_init(self, *a, **kw)
        to_legacy_iface(self)
    cls.__init__ = __init__
    return cls
