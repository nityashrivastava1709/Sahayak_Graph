"""
net_check.py  -  finds out WHERE a connection to Neo4j breaks, in four stages:

    1. URI   is the address well-formed?
    2. DNS   does the host name resolve?          (a paused Neo4j Aura instance does NOT resolve)
    3. TCP   can port 7687 be reached?            (firewalls, VPNs and some Wi-Fi networks block it)
    4. TLS   does the encrypted handshake work?   (antivirus HTTPS scanning / proxies can break it)

The Neo4j driver reports all of these as the same vague "Unable to retrieve routing information";
this module tells you which one it is and what to do. No login is attempted here.
"""
import socket
import ssl
from urllib.parse import urlparse


def _result(stage, ok, detail, advice=""):
    return {"stage": stage, "ok": ok, "detail": detail, "advice": advice}


def diagnose(uri: str, timeout: float = 8.0) -> list:
    """Return a list of stage results (each: stage, ok, detail, advice). Stops at the first failure."""
    results = []
    parsed = urlparse(uri)
    host, port = parsed.hostname, parsed.port or 7687
    encrypted = parsed.scheme.endswith("+s")

    if not host:
        return [_result("URI", False, f"'{uri}' has no host name",
                        "NEO4J_URI should look like neo4j+s://xxxxxxxx.databases.neo4j.io or bolt://localhost:7687")]
    results.append(_result("URI", True, f"{parsed.scheme}://{host}:{port}"))

    # ---- 2. DNS
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as e:
        is_aura = host.endswith(".databases.neo4j.io")
        advice = ("The host name does not exist in DNS. For Neo4j Aura this almost always means the instance is "
                  "PAUSED (Free instances pause after 72 hours without use, and a paused instance's name stops "
                  "resolving). Open https://console.neo4j.io, find the instance, press the Resume (play) button, wait "
                  "a few minutes until it says Running, then try again. Other causes: a typo in NEO4J_URI, the "
                  "instance was deleted (Free instances are deleted after 30 days paused), or no internet connection."
                  if is_aura else
                  "The host name does not resolve. Check NEO4J_URI for typos and that you are online.")
        return results + [_result("DNS", False, f"cannot resolve '{host}' ({e})", advice)]
    addresses = []
    for info in infos:
        if info[4][0] not in addresses:
            addresses.append(info[4][0])
    results.append(_result("DNS", True, f"{host} -> {', '.join(addresses[:3])}"))

    # ---- 3. TCP
    connected, errors = None, []
    for info in infos[:4]:
        try:
            with socket.socket(info[0], socket.SOCK_STREAM) as s:
                s.settimeout(timeout)
                s.connect(info[4])
            connected = info
            break
        except OSError as e:
            errors.append(f"{info[4][0]}: {e}")
    if connected is None:
        return results + [_result(
            "TCP", False, f"could not connect to port {port} ({'; '.join(errors)})",
            f"The name resolves but port {port} cannot be reached. Likely a firewall, VPN, proxy or restricted "
            "Wi-Fi (college / office networks often block it). Try: turn off any VPN, use a phone hotspot, or "
            f"another network. Check by hand in PowerShell:  Test-NetConnection {host} -Port {port}   "
            "(TcpTestSucceeded must be True). If the instance was only just resumed, wait 2-3 minutes and retry.")]
    results.append(_result("TCP", True, f"port {port} is reachable"))

    # ---- 4. TLS
    if encrypted:
        try:
            ctx = ssl.create_default_context()
            with socket.create_connection(connected[4][:2], timeout=timeout) as raw:
                with ctx.wrap_socket(raw, server_hostname=host):
                    pass
            results.append(_result("TLS", True, "encrypted handshake succeeded, certificate trusted"))
        except ssl.SSLCertVerificationError as e:
            return results + [_result(
                "TLS", False, f"certificate was not trusted ({e.verify_message})",
                "Something between you and Neo4j is replacing the security certificate, typically antivirus "
                "'HTTPS/SSL scanning' (Kaspersky, ESET, Avast, Bitdefender ...) or a company/college proxy. Try a "
                "different network such as a phone hotspot, or turn off HTTPS scanning for this test. Also run "
                "Windows Update so the root certificates are current.")]
        except (ssl.SSLError, OSError) as e:
            return results + [_result(
                "TLS", False, f"handshake failed ({type(e).__name__}: {e})",
                "The server answered but the encrypted handshake did not complete. If the instance was just "
                "resumed, wait a few minutes and retry. Otherwise a firewall or proxy may be interfering; try "
                "another network.")]
    return results
