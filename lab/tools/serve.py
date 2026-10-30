"""Static development server for Live Lab.

Python's built-in http.server sends no cache headers, so browsers reuse ES
modules heuristically and an edited file can silently keep running its old
version. This sends `Cache-Control: no-store` so every reload runs what is on
disk.

Usage: python tools/serve.py [port] [directory]
(directory defaults to lab/; pass the repository root to serve the whole site)
"""

import functools
import http.server
import pathlib
import sys


class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    # Development only: lets a page save a capture (e.g. the hero poster rendered
    # from the live canvas) with fetch(url, {method: 'PUT'}). Only under /_captures/.
    def do_PUT(self):
        name = pathlib.PurePosixPath(self.path.split("?")[0])
        if name.parent != pathlib.PurePosixPath("/_captures") or not name.name or name.name.startswith("."):
            self.send_error(403)
            return
        out = pathlib.Path(self.directory) / "_captures" / name.name
        out.parent.mkdir(exist_ok=True)
        out.write_bytes(self.rfile.read(int(self.headers["Content-Length"])))
        self.send_response(201)
        self.end_headers()


port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
root = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else pathlib.Path(__file__).parent.parent).resolve()
handler = functools.partial(NoCacheHandler, directory=str(root))
print(f"Serving {root} on http://127.0.0.1:{port}")
http.server.ThreadingHTTPServer(("127.0.0.1", port), handler).serve_forever()
