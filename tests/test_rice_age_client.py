import os
import time
import json
import pytest
from unittest.mock import patch, MagicMock
from io import BytesIO

from src.rice_age_client import RiceAgeClient, DEFAULT_FALLBACK_IP, TOTAL_TIMEOUT_BUDGET


def test_rice_age_client_init(tmp_path):
    uploads_dir = str(tmp_path / "uploads")
    cache_file = str(tmp_path / "cache.json")
    client = RiceAgeClient(uploads_dir=uploads_dir, cache_file=cache_file)
    assert client.uploads_dir == uploads_dir
    assert client.cache_file == cache_file
    assert client.mdns_host == "rice.local"
    assert os.path.exists(cache_file), "Cache file should be created upon initialization"


def test_parse_dominant_age():
    res, bracket = RiceAgeClient.parse_dominant_age(["98% 6-8 Months", "2% 8-10 Months"])
    assert res == "6-8 Months (98%)"
    assert bracket == "6-8 Months"

    res_empty, bracket_empty = RiceAgeClient.parse_dominant_age([])
    assert res_empty == "Standard Age"
    assert bracket_empty == "Standard"


def test_parse_dominant_age_unsorted():
    """Dominant age should be selected by highest percentage, regardless of array ordering."""
    # 98% is second, 2% is first
    res, bracket = RiceAgeClient.parse_dominant_age(["2% 8-10 Months", "98% 6-8 Months"])
    assert res == "6-8 Months (98%)"
    assert bracket == "6-8 Months"

    # Multi-element array with highest percentage in the middle
    res3, bracket3 = RiceAgeClient.parse_dominant_age(["15% 0-2 Months", "75% 4-6 Months", "10% 8-10 Months"])
    assert res3 == "4-6 Months (75%)"
    assert bracket3 == "4-6 Months"


def test_derive_freshness_rating():
    assert "Fresh" in RiceAgeClient.derive_freshness_rating("2-4 Months")
    assert "Sound" in RiceAgeClient.derive_freshness_rating("6-8 Months")
    assert "Aged" in RiceAgeClient.derive_freshness_rating("10-12 Months")
    assert "Aged" in RiceAgeClient.derive_freshness_rating(">10 Months")


def test_normalize_results_without_download(tmp_path):
    uploads_dir = str(tmp_path / "uploads")
    cache_file = str(tmp_path / "cache.json")
    client = RiceAgeClient(uploads_dir=uploads_dir, cache_file=cache_file)

    raw_data = {
        "serial": "Sample-0042",
        "status": "Mixed Ages Detected",
        "ages": ["5% 8-10 Months", "95% 6-8 Months"],  # Unsorted
        "colors": ["Yellow", "Yellow Orange"],
        "image": "scan_test123.jpg",
        "progress": 100,
        "live_step": "Analysis Complete",
        "timestamp": 1795091911
    }

    norm = client.normalize_results(raw_data, "rice.local", download_image=False)
    assert norm["connected"] is True
    assert norm["serial"] == "Sample-0042"
    assert norm["primary_age"] == "6-8 Months (95%)"
    assert norm["dominant_age_bracket"] == "6-8 Months"
    assert norm["dominant_color"] == "Yellow"
    assert norm["freshness_rating"] == "Sound Commercial Grain (Medium Storage Age)"
    assert norm["image_filename"] is None


def test_offline_resilience(tmp_path):
    uploads_dir = str(tmp_path / "uploads")
    cache_file = str(tmp_path / "cache.json")
    client = RiceAgeClient(
        mdns_host="nonexistent-device.invalid",
        port=5999,
        uploads_dir=uploads_dir,
        cache_file=cache_file,
        timeout=0.2
    )
    # Ensure fallback doesn't point to live IP during offline test
    client._cached_ip = "127.0.0.1"

    with patch("src.rice_age_client.DEFAULT_FALLBACK_IP", "127.0.0.2"):
        res = client.fetch_latest_results(download_image=False, force=True)
        assert res["connected"] is False
        assert res["status"] == "Age Tester Standby"
        assert res["primary_age"] == "Not Available"
        assert res["error"] is not None


def test_offline_strict_timeout_budget(tmp_path):
    """
    Critical requirement: CM5 optical scanning must NEVER be hung or delayed (> 2.5s)
    even when multiple fallback candidate IPs are probed and unresponsive.
    """
    uploads_dir = str(tmp_path / "uploads")
    cache_file = str(tmp_path / "cache.json")
    client = RiceAgeClient(
        mdns_host="unreachable-rice-host-404.local",
        uploads_dir=uploads_dir,
        cache_file=cache_file,
        timeout=2.0
    )
    # Simulate 4 unreachable candidate IPs (e.g. from obsolete subnets)
    client.get_candidate_hosts = lambda: [
        "unreachable-rice-host-404.local",
        "10.254.254.1",
        "10.254.254.2",
        "10.254.254.3"
    ]

    t0 = time.time()
    res = client.fetch_latest_results(download_image=False, force=True)
    elapsed = time.time() - t0

    assert elapsed <= 2.5, f"Elapsed {elapsed:.2f}s exceeds strict 2.5s limit"
    assert res["connected"] is False


def test_offline_cooldown_prevents_scan_delay(tmp_path):
    """
    When confirmed offline, subsequent routine polls should return immediately (< 0.05s)
    without re-probing all dead candidates on every frame.
    """
    uploads_dir = str(tmp_path / "uploads")
    cache_file = str(tmp_path / "cache.json")
    client = RiceAgeClient(
        mdns_host="unreachable-rice-host-404.local",
        uploads_dir=uploads_dir,
        cache_file=cache_file,
        timeout=0.5
    )
    client.get_candidate_hosts = lambda: ["10.254.254.1"]

    # First call probes and records offline
    client.fetch_latest_results(download_image=False, force=True)

    # Second routine call (force=False) must return via cooldown instantly
    t0 = time.time()
    res2 = client.fetch_latest_results(download_image=False, force=False)
    elapsed2 = time.time() - t0

    assert elapsed2 < 0.05, f"Cooldown failed to provide instant response: took {elapsed2:.3f}s"
    assert res2["connected"] is False


def test_cache_ip_persistence(tmp_path):
    uploads_dir = str(tmp_path / "uploads")
    cache_file = str(tmp_path / "device_cache.json")
    client = RiceAgeClient(uploads_dir=uploads_dir, cache_file=cache_file)
    client._save_cached_ip("192.168.100.55")

    # Reinstantiate
    client2 = RiceAgeClient(uploads_dir=uploads_dir, cache_file=cache_file)
    assert client2._cached_ip == "192.168.100.55"
