import time
import random
import textwrap

_state = {"used": None}  # remembered between calls so memory drifts instead of jumping

PROCESS_NAMES = [
    r"...5n1h2txyewy\TextInputHost.exe",
    r"...yb3d8bbwe\WindowsTerminal.exe",
    r"...8bbwe\PhoneExperienceHost.exe",
    r"...ntrolPanel\SystemSettings.exe",
    r"....0.4341.60\msedgewebview2.exe",
    r"...indows\System32\ShellHost.exe",
    r"...2txyewy\CrossDeviceResume.exe",
    r"C:\Windows\explorer.exe",
    r"...y\StartMenuExperienceHost.exe",
    r"...App_cw5n1h2txyewy\LockApp.exe",
    r"...em32\ApplicationFrameHost.exe",
    r"..._cw5n1h2txyewy\SearchHost.exe",
]


def _next_memory(total, step=25):
    """Small random walk, kept between 30% and 75% of total memory."""
    lo, hi = int(total * 0.3), int(total * 0.75)
    if _state["used"] is None:
        _state["used"] = int(total * 0.38)  # starting point scales with total
    used = _state["used"] + random.randint(-step, step)
    _state["used"] = max(lo, min(hi, used))
    return _state["used"]


def random_processes(min_n=8, max_n=15):
    """Random subset of processes with unique, ascending PIDs."""
    max_n = min(max_n, len(PROCESS_NAMES))   # never ask for more than we have
    min_n = min(min_n, max_n)
    n = random.randint(min_n, max_n)
    names = random.sample(PROCESS_NAMES, n)
    pids = sorted(random.sample(range(1000, 30000), n))
    rows = [
        f"    |    0   N/A  N/A{pid:>16}    C+G   {name:<38}N/A      |"
        for pid, name in zip(pids, names)
    ]
    return "\n".join(rows)


def fake_nsmi(dr_v="581.15", gpu_name="NVIDIA GeForce GTX 1050",
              total_mib=2048, used_mib=None):
    t = time.localtime()
    date = f"{time.strftime('%a %b', t)} {t.tm_mday:>2} {time.strftime('%H:%M:%S %Y', t)}"

    header = f" NVIDIA-SMI {dr_v}                 Driver Version: {dr_v}         CUDA Version: 13.0"
    header = "|" + header.ljust(89) + "|"

    name = f"{gpu_name:<29.29}"

    # fixed value if given, otherwise let it drift
    used = used_mib if used_mib is not None else _next_memory(total_mib)
    mem = f" {used:>7}MiB / {total_mib:>6}MiB "  # always 24 chars wide

    procs = random_processes()

    default = rf"""
    {date}
    +-----------------------------------------------------------------------------------------+
    {header}
    +-----------------------------------------+------------------------+----------------------+
    | GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
    |                                         |                        |               MIG M. |
    |=========================================+========================+======================|
    |   0  {name}WDDM  |   00000000:01:00.0  On |                  N/A |
    | 36%   51C    P0            N/A  /  N/A  |{mem}|      0%      Default |
    |                                         |                        |                  N/A |
    +-----------------------------------------+------------------------+----------------------+

    +-----------------------------------------------------------------------------------------+
    | Processes:                                                                              |
    |  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
    |        ID   ID                                                               Usage      |
    |=========================================================================================|
{procs}
    +-----------------------------------------------------------------------------------------+
    """
    return textwrap.dedent(default)