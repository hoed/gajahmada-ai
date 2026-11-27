import ipaddress
import socket
from typing import Optional, Set, Tuple
from urllib.parse import urlparse

BLOCKED_PORTS: Set[int] = {
    21, 22, 23, 25, 53, 110, 111, 135, 139, 143, 389, 445,
    1433, 1521, 2049, 2375, 2379, 3306, 3389, 5432, 5984,
    6379, 7000, 7001, 8500, 9042, 9200, 9300, 11211, 27017,
}

BLOCKED_HOSTNAMES: Set[str] = {
    "metadata.google.internal",
    "metadata",
    "instance-data",
    "169.254.169.254",
    "100.100.100.200",
}


def _unwrap_ip(ip_obj: ipaddress._BaseAddress) -> ipaddress._BaseAddress:
    mapped = getattr(ip_obj, "ipv4_mapped", None)
    return mapped or ip_obj


def _ip_block_reason(ip_obj: ipaddress._BaseAddress, allow_local: bool) -> Optional[str]:
    """Return a rejection reason, or None if this address may be contacted."""
    ip_obj = _unwrap_ip(ip_obj)
    if str(ip_obj) in ("169.254.169.254", "100.100.100.200"):
        return f"Cloud metadata IP blocked: {ip_obj}"
    if ip_obj.is_link_local or ip_obj.is_multicast or ip_obj.is_unspecified:
        return f"Link-local / cloud metadata IP blocked: {ip_obj}"
    # Reserved ranges (except private/loopback, which follow allow_local) are never peers.
    if ip_obj.is_reserved and not ip_obj.is_private and not ip_obj.is_loopback:
        return f"Reserved IP blocked: {ip_obj}"
    if not allow_local and (ip_obj.is_loopback or ip_obj.is_private or ip_obj.is_reserved):
        return f"Private/loopback IP {ip_obj} blocked in strict external-only mode"
    return None


def _resolved_ips(hostname: str) -> Tuple[bool, list]:
    """Resolve a hostname to concrete addresses. Literal IPs are not looked up."""
    try:
        return True, [_unwrap_ip(ipaddress.ip_address(hostname))]
    except ValueError:
        pass
    try:
        infos = socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
    except socket.gaierror:
        return False, []
    found = []
    for info in infos:
        sockaddr = info[4] if len(info) > 4 else None
        if not sockaddr:
            continue
        try:
            found.append(_unwrap_ip(ipaddress.ip_address(sockaddr[0])))
        except ValueError:
            continue
    return True, found


def validate_peer_url(
    url: str,
    allow_local: bool = True,
    allowed_hosts: Optional[Set[str]] = None,
) -> Tuple[bool, str]:
    """
    Validate a peer URL against SSRF vectors.
    - Blocks non-HTTP/HTTPS schemes.
    - Blocks embedded credentials (user:pass@host).
    - Blocks cloud metadata & link-local IP addresses (169.254.0.0/16, fe80::/10).
    - Blocks sensitive internal infrastructure ports (PostgreSQL 5432, Redis 6379, SSH 22, etc.).
    - Blocks loopback/private IPs when allow_local=False.
    """
    if not url or not isinstance(url, str):
        return False, "Empty or non-string URL"

    clean = url.strip().rstrip("/")
    if len(clean) > 512:
        return False, "URL exceeds maximum length"

    parsed = urlparse(clean)
    if parsed.scheme not in ("http", "https"):
        return False, f"Disallowed URL scheme: {parsed.scheme}"

    if not parsed.netloc or not parsed.hostname:
        return False, "Missing hostname in URL"

    if parsed.username or parsed.password:
        return False, "Embedded credentials in URL are forbidden"

    hostname = parsed.hostname.lower().strip()
    if hostname in BLOCKED_HOSTNAMES:
        return False, f"Blocked metadata/internal hostname: {hostname}"

    if allowed_hosts and hostname not in allowed_hosts:
        return False, f"Hostname {hostname} not in peer allowlist"

    try:
        port = parsed.port
    except ValueError:
        return False, "Invalid port number"

    if port is not None:
        if port <= 0 or port > 65535:
            return False, f"Port out of range: {port}"
        if port in BLOCKED_PORTS:
            return False, f"Port {port} is restricted (internal service port)"

    # Inspect every address the name resolves to. A public name that points at
    # metadata, link-local, or (when disallowed) private space is rejected.
    # Unresolvable names fail closed so a later lookup cannot slip past.
    resolved, ips = _resolved_ips(hostname)
    if not resolved or not ips:
        return False, f"Hostname did not resolve: {hostname}"
    for ip_obj in ips:
        reason = _ip_block_reason(ip_obj, allow_local=allow_local)
        if reason:
            return False, reason

    if not allow_local and hostname in ("localhost", "localhost.localdomain", "ip6-localhost"):
        return False, "Localhost blocked in strict external-only mode"

    return True, clean
