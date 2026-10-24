import pydantic
import pytest
import yaml

from turboml.config import load_config, require


def test_default_config_loads():
    cfg = load_config()
    assert cfg.design_point.mach == 0.0 and cfg.design_point.alt_m == 0.0
    assert {p.name for p in cfg.off_design_points} == {"idle", "cruise_M05_3km"}
    assert cfg.bounds["imp_blade_count"].integer
    # Nothing un-researched may sneak in as a number.
    assert cfg.constraints.max_tit_K is None


def test_extends_deep_merges(tmp_path):
    (tmp_path / "base.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "base",
                "design_point": {"name": "dp", "mach": 0.0, "alt_m": 0.0},
                "constraints": {"thrust_target_N": 100.0, "max_tit_K": 1200.0},
            }
        )
    )
    (tmp_path / "child.yaml").write_text(
        "extends: base.yaml\nname: child\nconstraints:\n  thrust_target_N: 200.0\n"
    )
    cfg = load_config(tmp_path / "child.yaml")
    assert cfg.name == "child"
    assert cfg.constraints.thrust_target_N == 200.0
    assert cfg.constraints.max_tit_K == 1200.0  # inherited


def test_unknown_keys_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("name: x\ndesign_point: {name: dp, mach: 0, alt_m: 0}\ntypo_key: 1\n")
    with pytest.raises(pydantic.ValidationError):
        load_config(p)


def test_require_refuses_placeholders():
    with pytest.raises(ValueError, match="not set"):
        require(None, "constraints.max_tit_K")
    assert require(1200.0, "x") == 1200.0


def test_active_bounds_have_sources_and_tip_speed_constraint():
    cfg = load_config()
    active = {k: b for k, b in cfg.bounds.items() if b.is_set()}
    assert {"cycle_pressure_ratio", "air_mass_flow_kg_s", "shaft_speed_rpm", "tit_K"} <= set(active)
    for name, b in active.items():
        assert b.source, f"bound {name} has no source"
        assert b.lower < b.upper
    assert cfg.constraints.max_impeller_tip_speed_m_s == 500.0
