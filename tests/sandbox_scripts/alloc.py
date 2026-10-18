import time

chunks = []
for _ in range(60):  # up to ~600 MB in 10 MB steps, touched so RSS grows
    chunks.append(bytearray(b"\x01") * (10 * 1024 * 1024))
    time.sleep(0.05)
time.sleep(30)
