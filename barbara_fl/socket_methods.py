import socket


def receiver(conn: socket.socket, msg_len) -> bytes:
    chunks = []
    bytes_recd = 0
    while bytes_recd < msg_len:
        chunk = conn.recv(min(msg_len - bytes_recd, 1024))
        if chunk == b"":
            raise RuntimeError("socket connection broken")
        chunks.append(chunk)
        bytes_recd += len(chunk)
    return b"".join(chunks)
