"""
Configuration management for experiment protocol

Reads CONFIG.yaml and provides locked parameters to prevent drift.
Logs config snapshot in each week's report_meta.json for auditability.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


_CONFIG_CACHE = None


def load_config() -> dict[str, Any]:
    """
    Load configuration from CONFIG.yaml
    
    Returns
    -------
    dict
        Configuration dictionary
    """
    global _CONFIG_CACHE
    
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE
    
    config_path = Path("CONFIG.yaml")
    if not config_path.exists():
        raise FileNotFoundError(
            "CONFIG.yaml not found. This file defines the experiment protocol."
        )
    
    with open(config_path) as f:
        _CONFIG_CACHE = yaml.safe_load(f)
    
    return _CONFIG_CACHE


def get_experiment_name() -> str:
    """Get experiment name"""
    config = load_config()
    return config["experiment"]["name"]


def get_experiment_version() -> str:
    """Get experiment version"""
    config = load_config()
    return config["experiment"]["version"]


def get_max_clusters_per_symbol() -> int:
    """Get max clusters per symbol"""
    config = load_config()
    return config["clustering"]["max_clusters_per_symbol"]


def get_recap_threshold() -> float:
    """Get PRICE_ACTION_RECAP threshold for low-info detection"""
    config = load_config()
    return config["signals"]["recap_pct_threshold"]


def get_min_clusters_threshold() -> int:
    """Get minimum clusters threshold"""
    config = load_config()
    return config["signals"]["min_clusters_threshold"]


def get_basket_size() -> int:
    """Get basket size (top N)"""
    config = load_config()
    return config["basket"]["size"]


def get_sector_cap() -> int | None:
    """Get sector concentration cap"""
    config = load_config()
    return config["basket"]["sector_cap"]


def get_skip_rules_enabled() -> bool:
    """Check if skip rules are enabled"""
    config = load_config()
    return config["skip_rules"]["enabled"]


def get_entry_timing() -> str:
    """Get entry timing description"""
    config = load_config()
    return config["execution"]["entry_timing"]


def get_exit_timing() -> str:
    """Get exit timing description"""
    config = load_config()
    return config["execution"]["exit_timing"]


def get_benchmark() -> str:
    """Get benchmark symbol"""
    config = load_config()
    return config["performance"]["benchmark"]


def get_config_snapshot() -> dict[str, Any]:
    """
    Get config snapshot for logging in report_meta.json
    
    Returns
    -------
    dict
        Config snapshot with key parameters
    """
    config = load_config()
    
    return {
        "experiment_name": config["experiment"]["name"],
        "experiment_version": config["experiment"]["version"],
        "max_clusters_per_symbol": config["clustering"]["max_clusters_per_symbol"],
        "recap_pct_threshold": config["signals"]["recap_pct_threshold"],
        "min_clusters_threshold": config["signals"]["min_clusters_threshold"],
        "basket_size": config["basket"]["size"],
        "basket_weighting": config["basket"]["weighting_method"],
        "sector_cap": config["basket"]["sector_cap"],
        "skip_rules_enabled": config["skip_rules"]["enabled"],
        "entry_timing": config["execution"]["entry_timing"],
        "exit_timing": config["execution"]["exit_timing"],
        "benchmark": config["performance"]["benchmark"],
        "neutralize_recap": config["signals"]["neutralize_price_action_recap"],
        "exclude_spy": config["signals"]["exclude_spy_from_ranking"],
    }


def validate_config():
    """
    Validate configuration is complete and consistent
    
    Raises
    ------
    ValueError
        If configuration is invalid
    """
    config = load_config()
    
    # Check required sections
    required_sections = [
        "experiment", "data_sources", "clustering", "signals",
        "basket", "skip_rules", "execution", "performance"
    ]
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")
    
    # Validate basket size
    basket_size = config["basket"]["size"]
    if basket_size <= 0 or basket_size > 100:
        raise ValueError(f"Invalid basket size: {basket_size} (must be 1-100)")
    
    # Validate thresholds
    recap_threshold = config["signals"]["recap_pct_threshold"]
    if not 0 <= recap_threshold <= 1:
        raise ValueError(f"Invalid recap_pct_threshold: {recap_threshold} (must be 0-1)")
    
    print("✓ Configuration validated")
    return True


if __name__ == "__main__":
    # Validate and display config
    validate_config()
    
    config = load_config()
    print("\n" + "="*70)
    print("EXPERIMENT CONFIGURATION")
    print("="*70)
    print(f"Name: {config['experiment']['name']}")
    print(f"Version: {config['experiment']['version']}")
    print(f"Description: {config['experiment']['description']}")
    print("\nKey Parameters:")
    print(f"  Max clusters/symbol: {get_max_clusters_per_symbol()}")
    print(f"  Recap threshold: {get_recap_threshold():.0%}")
    print(f"  Basket size: {get_basket_size()}")
    print(f"  Entry/Exit: {get_entry_timing()} → {get_exit_timing()}")
    print(f"  Benchmark: {get_benchmark()}")
    print(f"  Skip rules: {'ENABLED' if get_skip_rules_enabled() else 'DISABLED'}")
    print("="*70)
