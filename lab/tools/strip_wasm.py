"""Remove custom sections (DWARF debug info, the name section, the source-map URL)
from a WebAssembly binary. Custom sections never affect execution; in the
@mujoco/mujoco 3.12 build they are 5.8 MB of the 10.1 MB file.

Usage: python tools/strip_wasm.py vendor/mujoco/mujoco.wasm
"""

import sys
from pathlib import Path


def leb128(buf, i):
    result = shift = 0
    while True:
        byte = buf[i]
        i += 1
        result |= (byte & 0x7F) << shift
        shift += 7
        if byte < 0x80:
            return result, i


def strip(buf):
    if buf[:4] != b"\0asm":
        raise ValueError("not a WebAssembly binary")
    out, i, dropped = bytearray(buf[:8]), 8, 0
    while i < len(buf):
        section_id = buf[i]
        size, body = leb128(buf, i + 1)
        end = body + size
        if section_id == 0:
            dropped += end - i
        else:
            out += buf[i:end]
        i = end
    return bytes(out), dropped


if __name__ == "__main__":
    path = Path(sys.argv[1])
    data, dropped = strip(path.read_bytes())
    path.write_bytes(data)
    print(f"{path}: removed {dropped} bytes of custom sections, {len(data)} bytes remain")
