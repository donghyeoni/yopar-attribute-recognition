from __future__ import annotations

import json
import threading
import time
from collections import OrderedDict

import server_config as cfg


class MQListener(threading.Thread):
    def __init__(self, sync):
        super().__init__(daemon=True)
        self.sync = sync
        self._stopev = threading.Event()
        self._seen = OrderedDict()
        self.connected = False
        self.received = 0
        self.acked = 0
        self.requeued = 0
        self.duplicates = 0
        self.last_error = ""

    def stop(self):
        self._stopev.set()

    def _dup(self, command_id):
        if not command_id:
            return False
        if command_id in self._seen:
            return True
        self._seen[command_id] = True
        while len(self._seen) > cfg.MQ_COMMAND_MEMORY:
            self._seen.popitem(last=False)
        return False

    def _handle(self, body):
        try:
            msg = json.loads(body.decode("utf-8", "replace"))
        except Exception:
            print("[mq] 파싱 불가 메시지 — 버린다(ACK)", flush=True)
            return True
        cid = msg.get("commandId")
        etype = msg.get("eventType")
        case_id = msg.get("caseId")
        self.received += 1
        if self._dup(cid):
            self.duplicates += 1
            print(f"[mq] 중복 commandId={cid} — 무시(ACK)", flush=True)
            return True
        print(f"[mq] {etype} case={case_id} commandId={cid} -> 동기화", flush=True)
        ok = self.sync.refresh_and_wait(cfg.MQ_SYNC_TIMEOUT)
        if not ok:
            print(f"[mq] 동기화 실패 — ACK 안 함(재전달 유도) commandId={cid}",
                  flush=True)
            self._seen.pop(cid, None)
        return ok

    def run(self):
        if not cfg.MQ_ENABLED:
            print("[mq] MQ_ENABLED=False — 주기 폴링만으로 동기화한다", flush=True)
            return
        if not cfg.MQ_USER or not cfg.MQ_PASS:
            print("[mq] ⚠ MQ 계정이 비어 있다. 주기 폴링만으로 동기화한다.\n"
                  "[mq]   YOPAR_MQ_USER / YOPAR_MQ_PASS 를 설정하면 즉시 반영된다.",
                  flush=True)
            return
        try:
            import pika
        except ImportError:
            print("[mq] pika 없음 — 주기 폴링만으로 동기화한다 "
                  "(pip3 install --user pika)", flush=True)
            return

        params = pika.ConnectionParameters(
            host=cfg.MQ_HOST, port=cfg.MQ_PORT, virtual_host=cfg.MQ_VHOST,
            credentials=pika.PlainCredentials(cfg.MQ_USER, cfg.MQ_PASS),
            heartbeat=30, blocked_connection_timeout=30,
            connection_attempts=1, socket_timeout=10.0)
        bi = 0
        while not self._stopev.is_set():
            conn = None
            try:
                conn = pika.BlockingConnection(params)
                ch = conn.channel()
                ch.basic_qos(prefetch_count=1)
                self.connected = True
                bi = 0
                print(f"[mq] 연결됨 {cfg.MQ_HOST}:{cfg.MQ_PORT}{cfg.MQ_VHOST} "
                      f"queue={cfg.MQ_QUEUE}", flush=True)
                for method, _props, body in ch.consume(
                        cfg.MQ_QUEUE, inactivity_timeout=1.0):
                    if self._stopev.is_set():
                        break
                    if method is None:
                        continue
                    if self._handle(body):
                        ch.basic_ack(method.delivery_tag)
                        self.acked += 1
                    else:
                        ch.basic_nack(method.delivery_tag, requeue=True)
                        self.requeued += 1
                        time.sleep(2.0)
                try:
                    ch.cancel()
                except Exception:
                    pass
            except Exception as e:
                self.last_error = type(e).__name__
                msg = str(e)[:120]
                print(f"[mq] 연결 끊김/실패 ({self.last_error}: {msg}) — 재시도",
                      flush=True)
            finally:
                self.connected = False
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:
                        pass
            if self._stopev.is_set():
                break
            d = cfg.MQ_RECONNECT_DELAYS[min(bi, len(cfg.MQ_RECONNECT_DELAYS) - 1)]
            bi += 1
            self._stopev.wait(d)
        print("[mq] 종료", flush=True)
