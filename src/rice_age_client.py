"""
Dynamic Network Client for Rice Age Chemical Tester (Raspberry Pi 4 / rice.local)
Handles dynamic mDNS host resolution, IP caching, timeout protection,
reaction photo caching, and data normalization for unified inspection.
"""

import os
import sys
import time
import json
import socket
import re
import urllib.request
import urllib.error
import concurrent.futures
from typing import Dict, Any, Optional, List, Tuple


DEFAULT_MDNS_HOST = "rice.local"
DEFAULT_PORT = 5000
DEFAULT_TIMEOUT = 1.5  # Fast per-candidate timeout
TOTAL_TIMEOUT_BUDGET = 2.2  # Strict total budget; never stalls CM5 optical scan (<= 2.5s)
DEFAULT_FALLBACK_IP = "192.168.23.176"
OFFLINE_COOLDOWN_SEC = 4.0  # Throttle re-probing when known offline to eliminate UI latency

# Raspberry Pi Foundation MAC OUI prefixes (used to prioritize candidates in ARP)
RPI_MAC_PREFIXES = ("e4:5f:01", "dc:a6:32", "b8:27:eb", "28:cd:c1", "d8:3a:dd")


class RiceAgeClient:
    """
    Fault-tolerant client communicating with the Rice Age Chemical Tester.
    Supports dynamic mDNS resolution with local IP cache fallback across changing subnets.
    Strictly guarantees that querying offline devices will never block CM5 scanning (>2.5s).
    """

    def __init__(
        self,
        mdns_host: str = DEFAULT_MDNS_HOST,
        port: int = DEFAULT_PORT,
        uploads_dir: Optional[str] = None,
        cache_file: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT
    ):
        self.mdns_host = mdns_host
        self.port = port
        self.timeout = timeout
        
        # Determine uploads directory
        if uploads_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            uploads_dir = os.path.join(base_dir, "final", "static", "uploads")
        self.uploads_dir = uploads_dir
        os.makedirs(self.uploads_dir, exist_ok=True)

        # Cache file to remember last known good IP when subnets shift
        if cache_file is None:
            cache_file = os.path.join(self.uploads_dir, "rice_tester_device_cache.json")
        self.cache_file = cache_file

        loaded_ip = self._load_cached_ip()
        self._cached_ip = loaded_ip or DEFAULT_FALLBACK_IP
        if not os.path.exists(self.cache_file):
            self._save_cached_ip(self._cached_ip, hostname=self.mdns_host)

        self._last_normalized: Optional[Dict[str, Any]] = None
        self._last_sync_time: float = 0.0
        self._last_offline_probe_time: float = 0.0

    # -------------------------------------------------------------------------
    # Dynamic Host & IP Resolution
    # -------------------------------------------------------------------------
    def _load_cached_ip(self) -> Optional[str]:
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r") as f:
                    data = json.load(f)
                    return data.get("last_known_ip")
            except Exception:
                pass
        return None

    def _save_cached_ip(self, ip: str, hostname: Optional[str] = None) -> None:
        self._cached_ip = ip
        try:
            payload = {
                "last_known_ip": ip,
                "hostname": hostname or self.mdns_host,
                "port": self.port,
                "updated_at": time.time(),
                "updated_at_str": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            with open(self.cache_file, "w") as f:
                json.dump(payload, f, indent=2)
        except Exception:
            pass

    def _get_local_host_ip(self) -> Optional[str]:
        """
        Determines the local host's active IP address on the current network.
        Does NOT require an active internet connection to 8.8.8.8.
        """
        # Method 1: Connect UDP socket to local broadcast
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            s.connect(("255.255.255.255", 9))
            ip = s.getsockname()[0]
            s.close()
            if ip and not ip.startswith("127."):
                return ip
        except Exception:
            pass

        # Method 2: Inspect default route on Linux
        if os.path.exists("/proc/net/route"):
            try:
                with open("/proc/net/route", "r") as f:
                    for line in f.readlines()[1:]:
                        fields = line.strip().split()
                        if len(fields) >= 8 and fields[1] == "00000000":  # default route
                            iface = fields[0]
                            # Try querying interface address via SIOCGIFADDR
                            import fcntl
                            import struct
                            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                            ip_bytes = fcntl.ioctl(
                                s.fileno(),
                                0x8915,  # SIOCGIFADDR
                                struct.pack('256s', iface[:15].encode('utf-8'))
                            )[20:24]
                            s.close()
                            return socket.inet_ntoa(ip_bytes)
            except Exception:
                pass

        # Method 3: Standard getaddrinfo fallback
        try:
            host_name = socket.gethostname()
            addr_infos = socket.getaddrinfo(host_name, None, socket.AF_INET)
            for info in addr_infos:
                ip = info[4][0]
                if ip and not ip.startswith("127."):
                    return ip
        except Exception:
            pass

        return None

    def _get_local_subnet_candidates(self) -> List[str]:
        """
        Discovers candidate IPs on the same local subnet as the host,
        prioritizing devices with Raspberry Pi MAC OUIs from the ARP table.
        """
        host_ip = self._get_local_host_ip()
        if not host_ip:
            return []

        parts = host_ip.split(".")
        if len(parts) != 4:
            return []

        prefix = f"{parts[0]}.{parts[1]}.{parts[2]}."
        rpi_candidates = []
        other_candidates = []

        # Read Linux kernel ARP cache
        if os.path.exists("/proc/net/arp"):
            try:
                with open("/proc/net/arp", "r") as f:
                    for line in f.readlines()[1:]:
                        cols = line.split()
                        if len(cols) >= 4:
                            arp_ip = cols[0]
                            arp_mac = cols[3].lower()
                            if arp_ip.startswith(prefix) and arp_ip != host_ip:
                                if any(arp_mac.startswith(p) for p in RPI_MAC_PREFIXES):
                                    rpi_candidates.append(arp_ip)
                                else:
                                    other_candidates.append(arp_ip)
            except Exception:
                pass

        return rpi_candidates + other_candidates

    def get_candidate_hosts(self) -> List[str]:
        """
        Returns ordered list of candidate hosts to probe:
          1. mDNS hostname (rice.local)
          2. Cached IP (if on the current subnet)
          3. Subnet candidates (prioritizing Raspberry Pi MACs)
          4. Cached IP / fallback default (if on a different subnet)
        """
        host_ip = self._get_local_host_ip()
        current_prefix = ".".join(host_ip.split(".")[:3]) + "." if host_ip else None

        hosts = []
        # Always prioritize cached IP first if available
        if self._cached_ip:
            hosts.append(self._cached_ip)

        if self.mdns_host not in hosts:
            hosts.append(self.mdns_host)

        # Add discovered subnet candidates from ARP
        for cand in self._get_local_subnet_candidates():
            if cand not in hosts:
                hosts.append(cand)

        # Add cached IP or default fallback as last resorts
        for fallback in [self._cached_ip, DEFAULT_FALLBACK_IP]:
            if fallback and fallback not in hosts:
                hosts.append(fallback)

        return hosts

    # -------------------------------------------------------------------------
    # Image Proxy & Caching
    # -------------------------------------------------------------------------
    def cache_reaction_image(
        self,
        remote_image_name: str,
        preferred_host: str,
        sample_id: str,
        remaining_budget: float = 1.0
    ) -> Optional[str]:
        """
        Downloads the Logitech C920e chemical reaction image from the active host
        and saves it locally to CM5 static uploads directory.
        """
        if not remote_image_name:
            return None

        clean_name = os.path.basename(remote_image_name)
        local_filename = f"rice_age_{sample_id}_{clean_name}"
        local_filepath = os.path.join(self.uploads_dir, local_filename)

        if os.path.exists(local_filepath) and os.path.getsize(local_filepath) > 1000:
            return local_filename

        # Download from preferred host first
        hosts_to_try = [preferred_host]
        if self._cached_ip and self._cached_ip != preferred_host:
            hosts_to_try.append(self._cached_ip)

        req_timeout = max(0.5, min(self.timeout, remaining_budget))
        for host in hosts_to_try:
            img_url = f"http://{host}:{self.port}/static/uploads/{clean_name}"
            try:
                req = urllib.request.Request(img_url, headers={"User-Agent": "RiceAgeClient/2.0"})
                with urllib.request.urlopen(req, timeout=req_timeout) as resp:
                    if resp.status == 200:
                        content = resp.read()
                        if len(content) > 100:
                            with open(local_filepath, "wb") as f:
                                f.write(content)
                            return local_filename
            except Exception:
                continue

        return None

    # -------------------------------------------------------------------------
    # Data Normalization & Freshness Classification
    # -------------------------------------------------------------------------
    @staticmethod
    def parse_dominant_age(ages: List[str]) -> Tuple[str, str]:
        """
        Extracts dominant age bracket and formatted string from ages array.
        Robustly parses percentage values across all entries to identify the true dominant bracket.
        e.g. ['2% 8-10 Months', '98% 6-8 Months'] -> ('6-8 Months (98%)', '6-8 Months')
        """
        if not ages or not isinstance(ages, list):
            return "Standard Age", "Standard"

        best_pct = -1.0
        best_bracket = "Standard"
        best_str = ages[0]

        for entry in ages:
            if not isinstance(entry, str):
                continue
            # Match pattern like "98% 6-8 Months" or "6-8 Months (98%)"
            pct_match = re.search(r"(\d+(?:\.\d+)?)\s*%", entry)
            if pct_match:
                pct_val = float(pct_match.group(1))
                # Extract bracket text
                bracket_text = re.sub(r"\d+(?:\.\d+)?\s*%", "", entry).strip(" ()\t-:")
                if pct_val > best_pct:
                    best_pct = pct_val
                    best_bracket = bracket_text if bracket_text else "Standard"
                    best_str = f"{best_bracket} ({pct_val:.0f}%)" if pct_val.is_integer() else f"{best_bracket} ({pct_val}%)"

        if best_pct >= 0:
            return best_str, best_bracket

        # Fallback if no percentage pattern matched
        first_entry = str(ages[0])
        return first_entry, first_entry

    @staticmethod
    def derive_freshness_rating(bracket: str) -> str:
        """
        Categorizes grain milling and storage age into quality classification ratings.
        """
        b_lower = bracket.lower()
        if any(x in b_lower for x in ["0-2", "2-4", "4-6", "< 6", "<6"]):
            return "Fresh Commercial Grain (High Milling Freshness)"
        elif any(x in b_lower for x in ["6-8", "8-10"]):
            return "Sound Commercial Grain (Medium Storage Age)"
        elif any(x in b_lower for x in ["10-12", "> 10", ">10", "> 12", ">12", "aged", "old"]):
            return "Aged Grain (Long-Term Storage)"
        return "Standard Commercial Storage Age"

    def normalize_results(
        self,
        raw_data: Dict[str, Any],
        host_target: str,
        download_image: bool = True,
        remaining_budget: float = 1.0
    ) -> Dict[str, Any]:
        """
        Normalizes raw JSON from RPi4 into unified structure for CM5 dashboard & PDF report.
        """
        serial = raw_data.get("serial") or "Sample-0000"
        status = raw_data.get("status") or "Analysis Complete"
        ages = raw_data.get("ages") or []
        colors = raw_data.get("colors") or []
        remote_img = raw_data.get("image") or ""
        timestamp = raw_data.get("timestamp") or int(time.time())

        primary_age_str, dominant_bracket = self.parse_dominant_age(ages)
        dominant_color = colors[0] if (colors and len(colors) > 0) else "Yellow"
        freshness_rating = self.derive_freshness_rating(dominant_bracket)

        local_img_filename = None
        if download_image and remote_img:
            local_img_filename = self.cache_reaction_image(
                remote_img, host_target, serial, remaining_budget=remaining_budget
            )

        ts_formatted = time.strftime("%d/%m/%Y %H:%M:%S", time.localtime(timestamp)) if timestamp else ""

        normalized = {
            "connected": True,
            "host": f"{host_target}:{self.port}",
            "resolved_ip": self._cached_ip,
            "serial": serial,
            "status": status,
            "primary_age": primary_age_str,
            "dominant_age_bracket": dominant_bracket,
            "dominant_color": dominant_color,
            "ages": ages,
            "colors": colors,
            "freshness_rating": freshness_rating,
            "image_filename": local_img_filename,
            "image_url": f"/static/uploads/{local_img_filename}" if local_img_filename else None,
            "remote_image_name": remote_img,
            "progress": raw_data.get("progress", 100),
            "live_step": raw_data.get("live_step", "Analysis Complete"),
            "timestamp": timestamp,
            "timestamp_str": ts_formatted,
            "last_sync_time": time.time(),
            "error": None
        }
        return normalized

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def fetch_latest_results(self, download_image: bool = True, force: bool = False) -> Dict[str, Any]:
        """
        Fetches the latest chemical age results from rice.local with dynamic fallback.
        Enforces a strict total timeout budget (<= 2.2s) across all probed candidates
        so CM5 optical scanning and user interactions are NEVER blocked or hung.
        Uses an offline cooldown to eliminate repeated latency when the tester is off.
        """
        now = time.time()
        # Return recent cached normalized response if within 2.0s
        if not force and self._last_normalized and (now - self._last_sync_time < 2.0):
            return self._last_normalized

        # If previously confirmed offline, obey offline cooldown to prevent UI lag
        if not force and (self._last_normalized is None or not self._last_normalized.get("connected", False)):
            if (now - self._last_offline_probe_time) < OFFLINE_COOLDOWN_SEC:
                if self._last_normalized:
                    return self._last_normalized
                return self._build_offline_response("Age Tester Standby (Cooldown)")

        start_time = time.time()
        candidates = self.get_candidate_hosts()
        last_error = None

        for host in candidates:
            # Check remaining total budget
            time_spent = time.time() - start_time
            remaining_budget = TOTAL_TIMEOUT_BUDGET - time_spent
            if remaining_budget <= 0.15:
                break

            iter_timeout = min(self.timeout, remaining_budget)

            # Fast TCP socket pre-check for IP targets (avoids HTTP multi-second hang on dead hosts)
            is_ip = re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", host) is not None
            target_ip = host if is_ip else None

            if not is_ip:
                try:
                    # Quick name resolution bounded by budget (prevents Windows Winsock 12s lockup)
                    dns_timeout = min(0.4, remaining_budget)
                    ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                    try:
                        target_ip = ex.submit(socket.gethostbyname, host).result(timeout=dns_timeout)
                    finally:
                        ex.shutdown(wait=False, cancel_futures=True)
                except Exception as e:
                    last_error = e
                    continue

            # Probe TCP port with fast 0.35s timeout
            sock_timeout = min(0.35, iter_timeout)
            try:
                sock = socket.create_connection((target_ip, self.port), timeout=sock_timeout)
                sock.close()
            except Exception as e:
                last_error = e
                continue

            # HTTP GET on responsive host
            url = f"http://{host}:{self.port}/api/results"
            http_timeout = min(1.0, max(0.5, remaining_budget - (time.time() - start_time)))
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "RiceAgeClient/2.0"})
                with urllib.request.urlopen(req, timeout=http_timeout) as resp:
                    if resp.status == 200:
                        raw_json = json.loads(resp.read().decode("utf-8"))
                        if isinstance(raw_json, dict) and ("ages" in raw_json or "serial" in raw_json or "live_step" in raw_json):
                            # Save successful IP & hostname to cache
                            if target_ip:
                                self._save_cached_ip(target_ip, hostname=host)

                            img_budget = max(0.3, TOTAL_TIMEOUT_BUDGET - (time.time() - start_time))
                            normalized = self.normalize_results(
                                raw_json, host, download_image=download_image, remaining_budget=img_budget
                            )
                            self._last_normalized = normalized
                            self._last_sync_time = time.time()
                            return normalized
            except Exception as e:
                last_error = e
                continue

        # If all candidates failed:
        self._last_offline_probe_time = time.time()

        if self._last_normalized and self._last_normalized.get("connected"):
            copy_res = self._last_normalized.copy()
            copy_res["connected"] = False
            copy_res["error"] = str(last_error) if last_error else "Host unreachable"
            copy_res["status"] = "Age Tester Offline (Cached Data)"
            self._last_normalized = copy_res
            return copy_res

        offline_resp = self._build_offline_response(
            str(last_error) if last_error else "Could not reach rice.local or fallback IPs"
        )
        self._last_normalized = offline_resp
        return offline_resp

    def get_status(self) -> Dict[str, Any]:
        """
        Lightweight status query. Returns current connection health and latest test data.
        """
        return self.fetch_latest_results(download_image=True, force=False)

    # -------------------------------------------------------------------------
    # Unified Remote Orchestration & Control
    # -------------------------------------------------------------------------
    def _get_active_host(self) -> Optional[str]:
        """Fast resolver for the active reachable tester IP/hostname."""
        # 1. Fast probe cached working IP first (< 250ms)
        if self._cached_ip:
            try:
                sock = socket.create_connection((self._cached_ip, self.port), timeout=0.25)
                sock.close()
                return self._cached_ip
            except Exception:
                pass

        # 2. Probe candidate list
        candidates = self.get_candidate_hosts()
        for host in candidates:
            if host == self._cached_ip:
                continue
            is_ip = re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", host) is not None
            target_ip = host if is_ip else None
            if not is_ip:
                try:
                    target_ip = socket.gethostbyname(host)
                except Exception:
                    continue
            try:
                sock = socket.create_connection((target_ip, self.port), timeout=0.35)
                sock.close()
                if target_ip:
                    self._save_cached_ip(target_ip, hostname=host)
                return target_ip or host
            except Exception:
                continue
        return None

    def set_active_session(
        self,
        sample_id: str,
        batch_no: str,
        lot_no: str,
        commodity: str = "Raw Rice"
    ) -> bool:
        """Pushes sample metadata to RPi 4 so both devices share identical batch context."""
        host = self._get_active_host()
        if not host:
            return False
        url = f"http://{host}:{self.port}/api/session/set"
        payload = json.dumps({
            "sample_id": sample_id,
            "batch_no": batch_no,
            "lot_no": lot_no,
            "commodity": commodity
        }).encode("utf-8")
        try:
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json", "User-Agent": "RiceAgeClient/2.0"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                return resp.status == 200
        except Exception:
            return False

    def trigger_remote_test(
        self,
        sample_id: Optional[str] = None,
        batch_no: Optional[str] = None,
        lot_no: Optional[str] = None,
        commodity: Optional[str] = None
    ) -> Dict[str, Any]:
        """Instructs RPi 4 to start chemical reagent reaction sequence."""
        host = self._get_active_host()
        if not host:
            return {"status": "error", "message": "Rice Age Tester unreachable"}

        url = f"http://{host}:{self.port}/api/test/start"
        data: Dict[str, Any] = {}
        if sample_id: data["sample_id"] = sample_id
        if batch_no: data["batch_no"] = batch_no
        if lot_no: data["lot_no"] = lot_no
        if commodity: data["commodity"] = commodity

        payload = json.dumps(data).encode("utf-8")
        try:
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json", "User-Agent": "RiceAgeClient/2.0"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as he:
            try:
                return json.loads(he.read().decode("utf-8"))
            except Exception:
                return {"status": "error", "code": he.code, "message": str(he)}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def abort_remote_test(self) -> bool:
        """Sends immediate abort command to RPi 4 to stop spraying/timers."""
        host = self._get_active_host()
        if not host:
            return False
        url = f"http://{host}:{self.port}/api/test/abort"
        try:
            req = urllib.request.Request(
                url,
                data=b"{}",
                headers={"Content-Type": "application/json", "User-Agent": "RiceAgeClient/2.0"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                return resp.status == 200
        except Exception:
            return False

    def get_detailed_status(self) -> Dict[str, Any]:
        """Fetches real-time step progress percentage and remaining timer from RPi 4."""
        host = self._get_active_host()
        if not host:
            return {"state": "OFFLINE", "live_step": "Device Offline", "progress": 0, "connected": False}
        url = f"http://{host}:{self.port}/api/test/status"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "RiceAgeClient/2.0"})
            with urllib.request.urlopen(req, timeout=1.2) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                data["connected"] = True
                data["host"] = f"{host}:{self.port}"
                return data
        except Exception as e:
            return {"state": "OFFLINE", "live_step": "Unreachable", "progress": 0, "connected": False, "error": str(e)}

    def get_chamber_preview(self) -> Optional[bytes]:
        """Fetches a real-time JPEG thumbnail from RPi 4 camera to verify dish placement."""
        host = self._get_active_host()
        if not host:
            return None
        url = f"http://{host}:{self.port}/api/camera/preview"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "RiceAgeClient/2.0"})
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                if resp.status == 200:
                    return resp.read()
        except Exception:
            pass
        return None

    def _build_offline_response(self, error_msg: str) -> Dict[str, Any]:
        return {
            "connected": False,
            "host": f"{self.mdns_host}:{self.port}",
            "resolved_ip": self._cached_ip,
            "serial": "N/A",
            "status": "Age Tester Standby",
            "primary_age": "Not Available",
            "dominant_age_bracket": "N/A",
            "dominant_color": "N/A",
            "ages": [],
            "colors": [],
            "freshness_rating": "Unverified (Age Tester Standby)",
            "image_filename": None,
            "image_url": None,
            "remote_image_name": None,
            "progress": 0,
            "live_step": "Tester Standby",
            "timestamp": 0,
            "timestamp_str": "",
            "last_sync_time": time.time(),
            "error": error_msg
        }
