from __future__ import annotations
import os
import time
import math
import json
import re
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable, Dict, Any, Union

from timeloop_utils import (
    update_arch_yaml,
    run_timeloop_mapper,
    run_timeloop_model,
    parse_stats,
)
from mappers import run_lemon, run_gamma


# --------- 设计空间解析与构造 ---------
def parse_design_space(space_desc: Union[dict, str, None]) -> dict:
    """
    解析用户自定义设计空间描述:
    支持:
      1) 直接 dict
      2) JSON 字符串
      3) YAML 字符串
      4) 表达式行: KEY=512..8192 step 256
      5) 列表行: KEY=[1,2,3]
    返回 dict: {"GB_DEPTHS":[...], "PE_COUNTS":[...], ...}
    """
    if space_desc is None:
        return {}
    if isinstance(space_desc, dict):
        return space_desc

    text = str(space_desc).strip()
    # 尝试 JSON
    try:
        return json.loads(text)
    except Exception:
        pass
    # 尝试 YAML
    try:
        import yaml
        y = yaml.safe_load(text)
        if isinstance(y, dict):
            return y
    except Exception:
        pass

    result: Dict[str, List[int]] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # 范围表达式
        m = re.match(r"(\w+)\s*[:=]\s*(\d+)\s*\.\.(\d+)\s*step\s*(\d+)", line, re.IGNORECASE)
        if m:
            k, lo, hi, step = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
            if hi < lo or step <= 0:
                continue
            result[k] = list(range(lo, hi + 1, step))
            continue
        # 列表
        m2 = re.match(r"(\w+)\s*[:=]\s*\[([^\]]+)\]", line)
        if m2:
            k, arr = m2.group(1), m2.group(2)
            vals = []
            for v in arr.split(","):
                v = v.strip()
                if v.isdigit():
                    vals.append(int(v))
            if vals:
                result[k] = vals
    return result


def build_design_space(user_space: dict | None = None) -> dict:
    """
    基于用户空间覆盖默认空间。
    默认：
      GB_DEPTHS: 1024..4096 step 64
      PE_COUNTS: 8..24 step 2
      PE_INPUT_DEPTHS: 4096..16384 step 128
      PE_WEIGHT_DEPTHS: 2048..8192 step 128
      PE_ACCU_DEPTHS: 64..256 step 8
    """
    GB_DEPTHS = [1024 + 64 * i for i in range(49)]
    PE_COUNTS = [8 + 2 * i for i in range(9)]
    PE_INPUT_DEPTHS = [4096 + 128 * i for i in range(97)]
    PE_WEIGHT_DEPTHS = [2048 + 128 * i for i in range(49)]
    PE_ACCU_DEPTHS = [64 + 8 * i for i in range(25)]

    if user_space:
        GB_DEPTHS = user_space.get("GB_DEPTHS", GB_DEPTHS)
        PE_COUNTS = user_space.get("PE_COUNTS", PE_COUNTS)
        PE_INPUT_DEPTHS = user_space.get("PE_INPUT_DEPTHS", PE_INPUT_DEPTHS)
        PE_WEIGHT_DEPTHS = user_space.get("PE_WEIGHT_DEPTHS", PE_WEIGHT_DEPTHS)
        PE_ACCU_DEPTHS = user_space.get("PE_ACCU_DEPTHS", PE_ACCU_DEPTHS)

    bounds = [
        (0, len(GB_DEPTHS) - 1),
        (0, len(PE_INPUT_DEPTHS) - 1),
        (0, len(PE_WEIGHT_DEPTHS) - 1),
        (0, len(PE_ACCU_DEPTHS) - 1),
        (0, len(PE_COUNTS) - 1),
    ]
    return dict(
        GB_DEPTHS=GB_DEPTHS,
        PE_COUNTS=PE_COUNTS,
        PE_INPUT_DEPTHS=PE_INPUT_DEPTHS,
        PE_WEIGHT_DEPTHS=PE_WEIGHT_DEPTHS,
        PE_ACCU_DEPTHS=PE_ACCU_DEPTHS,
        SPACE_BOUNDS=bounds,
    )


def clamp_index(val: float, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(round(val))))


def idx_to_params(space: dict, idxs: List[int]) -> Tuple[int, int, int, int, int]:
    gb = space["GB_DEPTHS"][idxs[0]]
    pe_in = space["PE_INPUT_DEPTHS"][idxs[1]]
    pe_w = space["PE_WEIGHT_DEPTHS"][idxs[2]]
    pe_a = space["PE_ACCU_DEPTHS"][idxs[3]]
    pe_cnt = space["PE_COUNTS"][idxs[4]]
    return gb, pe_in, pe_w, pe_a, pe_cnt


@dataclass
class OptimizerConfig:
    algorithm: str            # "GA" | "AOA" | "BO"
    mapper: str               # "timeloop" | "lemon" | "gamma"
    epochs: int = 50
    pop_size: int = 50
    trials: int = 100
    seed: Optional[int] = None
    # Mapper 相关
    mapper_yaml: Optional[str] = None
    lemon_mapspace_yaml: Optional[str] = None
    lemon_bin: str = "lemon"
    gamma_cmd: Optional[str] = None
    gamma_workdir: Optional[str] = None


def auto_select_config(
    mapper_yaml: Optional[str],
    lemon_mapspace_yaml: Optional[str],
    lemon_bin: str,
    gamma_cmd: Optional[str],
    gamma_workdir: Optional[str],
    budget_hint: int,
) -> OptimizerConfig:
    if budget_hint <= 150:
        algo = "BO"
        trials = budget_hint
        epochs, pop = 0, 0
    else:
        algo = "GA"
        trials = 0
        epochs = min(100, max(30, budget_hint // 2))
        pop = min(100, max(30, budget_hint // 2))

    if mapper_yaml and os.path.exists(mapper_yaml):
        mapper = "timeloop"
    elif lemon_mapspace_yaml and os.path.exists(lemon_mapspace_yaml):
        mapper = "lemon"
    elif gamma_cmd and gamma_workdir and os.path.exists(gamma_workdir):
        mapper = "gamma"
    else:
        mapper = "timeloop"

    return OptimizerConfig(
        algorithm=algo,
        mapper=mapper,
        epochs=epochs or 50,
        pop_size=pop or 50,
        trials=trials or 100,
        seed=None,
        mapper_yaml=mapper_yaml,
        lemon_mapspace_yaml=lemon_mapspace_yaml,
        lemon_bin=lemon_bin,
        gamma_cmd=gamma_cmd,
        gamma_workdir=gamma_workdir,
    )


class BlackBoxObjective:
    def __init__(
        self,
        arch_yaml: str,
        prob_yaml: str,
        components_glob: str,
        work_root: str,
        cfg: OptimizerConfig,
        space: dict,
    ):
        self.arch_yaml = arch_yaml
        self.prob_yaml = prob_yaml
        self.components_glob = components_glob
        self.work_root = work_root
        self.cfg = cfg
        self.space = space
        os.makedirs(self.work_root, exist_ok=True)

    def __call__(self, idxs: List[float]) -> float:
        bnds = self.space["SPACE_BOUNDS"]
        idxs = [
            clamp_index(idxs[0], *bnds[0]),
            clamp_index(idxs[1], *bnds[1]),
            clamp_index(idxs[2], *bnds[2]),
            clamp_index(idxs[3], *bnds[3]),
            clamp_index(idxs[4], *bnds[4]),
        ]
        gb, pe_in, pe_w, pe_a, pe_cnt = idx_to_params(self.space, idxs)

        try:
            update_arch_yaml(self.arch_yaml, gb, pe_in, pe_w, pe_a, pe_cnt)
        except Exception as e:
            print(f"[X] update_arch_yaml error: {e}")
            return 1e12

        stamp = int(time.time() * 1000)
        run_dir = os.path.join(self.work_root, f"eval_{stamp}")
        os.makedirs(run_dir, exist_ok=True)

        try:
            if self.cfg.mapper == "timeloop":
                stats_path = run_timeloop_mapper(
                    self.arch_yaml,
                    self.components_glob,
                    self.prob_yaml,
                    self.cfg.mapper_yaml or "",
                    output_dir=run_dir,
                )
            elif self.cfg.mapper == "lemon":
                stats_path, map_yaml = run_lemon(
                    self.arch_yaml,
                    self.cfg.lemon_mapspace_yaml or "",
                    self.prob_yaml,
                    output_dir=run_dir,
                    lemon_bin=self.cfg.lemon_bin,
                )
                if stats_path is None:
                    if not map_yaml:
                        print("[X] LEMON produced no map.yaml")
                        return 1e12
                    stats_path = run_timeloop_model(
                        self.arch_yaml,
                        self.components_glob,
                        self.prob_yaml,
                        map_yaml,
                        output_dir=run_dir,
                    )
            elif self.cfg.mapper == "gamma":
                stats_path, map_yaml = run_gamma(
                    gamma_cmd=self.cfg.gamma_cmd or "bash run_gamma_timeloop.sh",
                    work_dir=self.cfg.gamma_workdir or os.getcwd(),
                    output_dir=run_dir,
                )
                if stats_path is None:
                    if not map_yaml:
                        print("[X] GAMMA produced no map.yaml/stats")
                        return 1e12
                    stats_path = run_timeloop_model(
                        self.arch_yaml,
                        self.components_glob,
                        self.prob_yaml,
                        map_yaml,
                        output_dir=run_dir,
                    )
            else:
                print(f"[X] Unknown mapper: {self.cfg.mapper}")
                return 1e12

            metrics = parse_stats(stats_path)
            edp = metrics.get("EDP")
            print(f"[✓] Params gb={gb}, pe_in={pe_in}, pe_w={pe_w}, pe_a={pe_a}, pe_cnt={pe_cnt} -> EDP={edp}")
            if edp is None or not math.isfinite(edp):
                return 1e12
            return float(edp)
        except Exception as e:
            print(f"[X] BlackBox eval failed: {e}")
            return 1e12


def _bounds(space: dict):
    return [b for b in space["SPACE_BOUNDS"]]


def run_with_ga(obj: Callable[[List[float]], float], cfg: OptimizerConfig, space: dict) -> Tuple[List[int], float]:
    from mealpy import FloatVar, GA
    b = _bounds(space)
    bounds = FloatVar(lb=[x[0] for x in b], ub=[x[1] for x in b], name="indices")
    model = GA.BaseGA(epoch=cfg.epochs, pop_size=cfg.pop_size, pc=0.9, pm=0.1, seed=cfg.seed)
    best = model.solve({"obj_func": obj, "bounds": bounds, "minmax": "min"})
    idxs = [int(round(i)) for i in best.solution]
    return idxs, float(best.target.fitness)


def run_with_aoa(obj: Callable[[List[float]], float], cfg: OptimizerConfig, space: dict) -> Tuple[List[int], float]:
    from mealpy import FloatVar, AOA
    b = _bounds(space)
    bounds = FloatVar(lb=[x[0] for x in b], ub=[x[1] for x in b], name="indices")
    model = AOA.OriginalAOA(
        epoch=cfg.epochs, pop_size=cfg.pop_size, alpha=5, miu=0.5, moa_min=0.2, moa_max=0.9, seed=cfg.seed
    )
    best = model.solve({"obj_func": obj, "bounds": bounds, "minmax": "min"})
    idxs = [int(round(i)) for i in best.solution]
    return idxs, float(best.target.fitness)


def run_with_bo(obj: Callable[[List[float]], float], cfg: OptimizerConfig, space: dict) -> Tuple[List[int], float]:
    import optuna
    b = _bounds(space)

    def objective(trial: optuna.Trial) -> float:
        idx = [
            trial.suggest_int("GB", *b[0]),
            trial.suggest_int("PEIn", *b[1]),
            trial.suggest_int("PEW", *b[2]),
            trial.suggest_int("PEA", *b[3]),
            trial.suggest_int("PECount", *b[4]),
        ]
        return obj(idx)

    sampler = optuna.samplers.TPESampler(seed=cfg.seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=cfg.trials, show_progress_bar=False)
    best_idx = [study.best_trial.params[k] for k in ["GB", "PEIn", "PEW", "PEA", "PECount"]]
    return best_idx, float(study.best_value)


def run_optimization(
    arch_yaml: str,
    prob_yaml: str,
    components_glob: str,
    workdir: str,
    cfg: OptimizerConfig,
    design_space: dict | str | None = None,
) -> Dict[str, Any]:
    os.makedirs(workdir, exist_ok=True)
    user_space = parse_design_space(design_space)
    space = build_design_space(user_space)

    obj = BlackBoxObjective(
        arch_yaml=arch_yaml,
        prob_yaml=prob_yaml,
        components_glob=components_glob,
        work_root=workdir,
        cfg=cfg,
        space=space,
    )

    if cfg.algorithm == "GA":
        best_idx, best_edp = run_with_ga(obj, cfg, space)
    elif cfg.algorithm == "AOA":
        best_idx, best_edp = run_with_aoa(obj, cfg, space)
    elif cfg.algorithm == "BO":
        best_idx, best_edp = run_with_bo(obj, cfg, space)
    else:
        raise ValueError(f"Unknown algorithm: {cfg.algorithm}")

    gb, pe_in, pe_w, pe_a, pe_cnt = idx_to_params(space, best_idx)
    result = {
        "best_index": best_idx,
        "best_params": {
            "GB": gb,
            "PEIn": pe_in,
            "PEW": pe_w,
            "PEA": pe_a,
            "PECount": pe_cnt,
        },
        "best_edp": best_edp,
        "workdir": os.path.abspath(workdir),
        "config": cfg.__dict__,
        "design_space_effective_sizes": {
            "GB_DEPTHS": len(space["GB_DEPTHS"]),
            "PE_INPUT_DEPTHS": len(space["PE_INPUT_DEPTHS"]),
            "PE_WEIGHT_DEPTHS": len(space["PE_WEIGHT_DEPTHS"]),
            "PE_ACCU_DEPTHS": len(space["PE_ACCU_DEPTHS"]),
            "PE_COUNTS": len(space["PE_COUNTS"]),
        },
        "design_space_user_override": user_space or {},
    }
    with open(os.path.join(workdir, "summary.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result