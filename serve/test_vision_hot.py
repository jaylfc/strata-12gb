"""The hot vision encoder (opt-in, "vision": {"hot": true} with "vram_elastic": true): no encoder at start, the elastic
expert cache shrinks before the first image's encoder starts, a second image inside the idle window reuses it, the
idle timer stops it and grows the cache back, failures fall back (or answer 503) without leaving the cache shrunk,
and the timer never races a request."""
import contextlib
import io
import os
import stat
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

from serve.frontend import ChatTemplate
from serve.server import (ByteTokenizer, EngineDied, Service, Vision, VisionUnavailable, VISION_HOT_ENCODER_MIB,
                          vision_cpu_config, vision_hot_config)
from serve.test_lifecycle import ResidentEngine

TEMPLATE = Path(__file__).parent / "chat_template.jinja"
HOT = {"exe": "x", "mmproj": "m", "model": "g", "gpu": True, "max_tokens": 1024, "hot": True}


class ElasticEngine(ResidentEngine):
    """A running engine with an elastic cache; every VRAM command goes into the shared event list."""
    def __init__(self, tok, events):
        super().__init__(tok)
        self.events, self.fail = events, None
        self.restart()

    def vram(self, reserve_mib, timeout=120.0):
        self.events.append(("vram", reserve_mib))
        if self.fail is not None:
            raise self.fail
        r = 700 if reserve_mib is None else reserve_mib
        return {"reserve_mib": r, "expert_slots": 4000 - r, "vram_free_mib": r}


class FakeVision:
    """The encoder process stand-in: start / stop / encode go into the same event list as the engine's commands."""
    def __init__(self, d, events, name="gpu"):
        self.dir, self.events, self.name = Path(d), events, name
        self.up, self.cache, self.fail_start, self.unload_delay = False, {}, None, 0.0

    def alive(self):
        return self.up

    def restart(self):
        self.events.append(("start", self.name))
        if self.fail_start:
            raise RuntimeError(self.fail_start)
        self.up = True

    def unload(self):
        time.sleep(self.unload_delay)
        self.events.append(("stop", self.name))
        self.up = False

    def close(self):
        self.up = False

    def cached(self, source):
        return source, source.encode(), self.cache.get(source)

    def encode(self, source, data=None):
        if source in self.cache:
            return self.cache[source]
        if not self.up:
            raise ValueError("the image could not be read: the vision encoder stopped")
        self.events.append(("encode", self.name, source))
        out = self.dir / f"{self.name}-{len(self.cache)}.sve"
        out.write_bytes(b"rows")
        self.cache[source] = (out, 3)
        return self.cache[source]


def image_msg(src):
    return [{"role": "user", "content": [{"type": "text", "text": "what is it?"}, {"type": "image", "source": src}]}]


class HotBase(unittest.TestCase):
    def make(self, cpu=False, cfg=None):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.events = []
        tok = ByteTokenizer()
        self.engine = ElasticEngine(tok, self.events)
        self.vision = FakeVision(self.tmp.name, self.events)
        self.svc = Service(self.engine, tok, ChatTemplate(TEMPLATE), vision=self.vision)
        self.svc.vision_hot = vision_hot_config(cfg or {"vram_elastic": True, "vision": dict(HOT), "args": []})
        if cpu:
            self.svc.vision_cpu = FakeVision(self.tmp.name, self.events, "cpu")
        return self.svc

    def ask(self, src):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.svc.load()
            self.svc.prepare(image_msg(src), None, {})
        return out.getvalue()


class Config(unittest.TestCase):
    def test_off_unless_asked(self):
        self.assertIsNone(vision_hot_config({"vision": {"gpu": True}}))
        self.assertIsNone(vision_hot_config({}))

    def test_needs_vram_elastic_and_one_gpu(self):
        with self.assertRaisesRegex(ValueError, "vram_elastic"):
            vision_hot_config({"vision": dict(HOT)})
        with self.assertRaisesRegex(ValueError, "one GPU"):
            vision_hot_config({"vram_elastic": True, "gpu": [0, 1], "vision": dict(HOT)})
        with self.assertRaisesRegex(ValueError, "reserve_mib"):
            vision_hot_config({"vram_elastic": True, "vision": {**HOT, "reserve_mib": "lots"}})
        with self.assertRaisesRegex(ValueError, "idle_unload_s"):
            vision_hot_config({"vram_elastic": True, "vision": {**HOT, "idle_unload_s": 0}})

    def test_defaults_and_start_reserve(self):
        h = vision_hot_config({"vram_elastic": True, "vision": dict(HOT),
                               "args": ["--vision", "--vram-reserve-mib", "900"]})
        self.assertEqual((h["idle_unload_s"], h["reserve_mib"], h["start_reserve_mib"], h["cpu_fallback"]),
                         (120.0, "auto", 900, False))
        self.assertEqual(vision_hot_config({"vram_elastic": True, "vision": dict(HOT)})["start_reserve_mib"], 700)

    def test_cpu_fallback_config(self):
        c = vision_cpu_config({**HOT, "cuda_device": 1, "threads": 8}, True)
        self.assertNotIn("gpu", c)
        self.assertNotIn("hot", c)
        self.assertEqual((c["max_tokens"], c["threads"]), (300, 8))
        self.assertEqual(vision_cpu_config(HOT, {"max_tokens": 512})["max_tokens"], 512)


class StartWithoutEncoder(HotBase):
    def test_no_encoder_and_no_resize_at_start_or_for_text(self):
        svc = self.make()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            svc.load()                                   # a text request: the stopped encoder is not "down"
            svc.prepare([{"role": "user", "content": "hi"}], None, {})
        self.assertEqual(self.events, [])
        self.assertFalse(self.vision.alive())
        self.assertFalse(svc.vision_shrunk)
        self.assertEqual(svc.vision_idle_unload(0.0), "idle")

    def test_real_vision_constructed_without_a_process(self):
        v = Vision({"exe": "/nonexistent/strata-vision", "mmproj": "m", "model": "g", "gpu": True}, start=False)
        self.assertIsNone(v.proc)
        self.assertFalse(v.alive())
        v.close()                                        # nothing to close
        v.unload()
        self.assertFalse(v.alive())


class ColdThenWarm(HotBase):
    def test_first_image_shrinks_then_starts_then_encodes(self):
        svc = self.make()
        log = self.ask("a.png")
        self.assertEqual(self.events, [("vram", 700 + VISION_HOT_ENCODER_MIB), ("start", "gpu"),
                                       ("encode", "gpu", "a.png")])
        self.assertTrue(svc.vision_shrunk)
        self.assertRegex(log, r"\[strata\] vision hot: shrink \d+ ms, start \d+ ms, encode \d+ ms")
        m = svc.metrics()["vision_hot"]
        self.assertEqual((m["requests"], m["cold_starts"], m["encoder_alive"], m["cache_shrunk"]), (1, 1, True, True))
        self.assertTrue(m["last"]["cold"])
        for k in ("shrink_ms", "start_ms", "encode_ms"):
            self.assertIn(k, m["last"])

    def test_second_image_in_the_window_reuses_the_encoder(self):
        svc = self.make()
        self.ask("a.png")
        self.events.clear()
        log = self.ask("b.png")
        self.assertEqual(self.events, [("encode", "gpu", "b.png")])     # no shrink, no start
        self.assertIn("(encoder warm)", log)
        self.ask("a.png")                                               # cached: not even an encode
        self.assertEqual(self.events, [("encode", "gpu", "b.png")])
        self.assertEqual(svc.vision_hot_stats["cold_starts"], 1)

    def test_a_cached_image_needs_no_encoder(self):
        svc = self.make()
        self.ask("a.png")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(svc.vision_idle_unload(0.0), "unloaded")
        self.events.clear()
        self.ask("a.png")
        self.assertEqual(self.events, [])                              # the encoding on disk is enough
        self.assertFalse(svc.vision_shrunk)

    def test_reserve_follows_a_posted_reserve_and_the_config(self):
        svc = self.make()
        svc.vram_reserve = 3000
        self.ask("a.png")
        self.assertEqual(self.events[0], ("vram", 3000 + VISION_HOT_ENCODER_MIB))
        svc = self.make(cfg={"vram_elastic": True, "vision": {**HOT, "reserve_mib": 2500}})
        self.ask("a.png")
        self.assertEqual(self.events[0], ("vram", 2500))

    def test_posted_reserve_while_hot_keeps_the_encoders_room(self):
        svc = self.make()
        self.ask("a.png")
        self.events.clear()
        with contextlib.redirect_stdout(io.StringIO()):
            svc.vram(None)
            svc.vram(5000)
        self.assertEqual(self.events, [("vram", 700 + VISION_HOT_ENCODER_MIB), ("vram", 5000 + VISION_HOT_ENCODER_MIB)])
        self.assertEqual(svc.vram_reserve, 5000)
        self.events.clear()
        svc.last_image_at = time.time() - 1000
        with contextlib.redirect_stdout(io.StringIO()):
            svc.vision_idle_unload(120)
        self.assertEqual(self.events, [("stop", "gpu"), ("vram", 5000)])   # back to the posted reserve


class IdleTimer(HotBase):
    def test_unloads_and_grows_after_the_idle_window_only(self):
        svc = self.make()
        self.ask("a.png")
        self.events.clear()
        self.assertEqual(svc.vision_idle_unload(120), "busy")          # the image was just now
        self.assertEqual(self.events, [])
        svc.last_image_at = time.time() - 121
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(svc.vision_idle_unload(120), "unloaded")
        self.assertEqual(self.events, [("stop", "gpu"), ("vram", None)])
        self.assertFalse(self.vision.alive())
        self.assertFalse(svc.vision_shrunk)
        self.assertIn("encoder stopped after 120 s", out.getvalue())
        self.assertEqual(svc.vision_idle_unload(120), "idle")
        self.events.clear()
        self.ask("c.png")                                               # cold again: shrink, start
        self.assertEqual([e[0] for e in self.events], ["vram", "start", "encode"])

    def test_the_timer_thread(self):
        svc = self.make(cfg={"vram_elastic": True, "vision": {**HOT, "idle_unload_s": 1}})
        self.ask("a.png")
        with contextlib.redirect_stdout(io.StringIO()):
            svc.start_vision_hot()
            deadline = time.time() + 10
            while self.vision.alive() and time.time() < deadline:
                time.sleep(0.05)
            time.sleep(0.1)
        self.assertFalse(self.vision.alive())
        self.assertFalse(svc.vision_shrunk)
        self.assertEqual(self.events[-2:], [("stop", "gpu"), ("vram", None)])

    def test_a_grow_refused_is_tried_again(self):
        svc = self.make()
        self.ask("a.png")
        svc.last_image_at = time.time() - 1000
        self.engine.fail = ValueError("VRAM: a prompt's loan is still out")
        with contextlib.redirect_stdout(io.StringIO()):
            svc.vision_idle_unload(120)
        self.assertTrue(svc.vision_shrunk)                              # remembered, not forgotten
        self.engine.fail = None
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(svc.vision_idle_unload(120), "unloaded")
        self.assertFalse(svc.vision_shrunk)

    def test_engine_unload_resets_the_shrink(self):
        svc = self.make()
        self.ask("a.png")
        svc.last_request_at = time.time() - 1000
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(svc.unload(), "unloaded")
        self.assertFalse(svc.vision_shrunk)
        self.assertFalse(self.vision.alive())


class Failures(HotBase):
    def test_shrink_refused_falls_back_to_the_cpu_encoder(self):
        svc = self.make(cpu=True)
        self.engine.fail = ValueError("VRAM: a prompt's loan is still out")
        log = self.ask("a.png")
        # the shrink failed -> grow back attempted (refused too) -> CPU encoder -> the grow tried again at the end
        self.assertEqual([e[:2] for e in self.events], [("vram", 2100), ("vram", None), ("start", "cpu"),
                                                        ("encode", "cpu"), ("vram", None)])
        self.assertFalse(self.vision.alive())
        self.assertIn("using the CPU encoder", log)
        self.assertEqual(svc.vision_hot_stats["fallbacks"], 1)
        self.engine.fail = None                                         # the grow is retried by the timer
        svc.last_image_at = time.time() - 1000
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(svc.vision_idle_unload(120), "unloaded")
        self.assertFalse(svc.vision_shrunk)
        self.assertFalse(svc.vision_cpu.alive())

    def test_shrink_timeout_without_fallback_is_a_clear_503(self):
        svc = self.make()
        self.engine.fail = EngineDied("the engine did not answer the VRAM command")
        with self.assertRaisesRegex(VisionUnavailable, "could not start.*did not shrink"):
            self.ask("a.png")
        self.assertFalse(self.vision.alive())
        self.assertEqual(self.events[0], ("vram", 2100))
        self.assertNotIn(("start", "gpu"), self.events)                 # no encoder into VRAM that is not there

    def test_encoder_start_failure_grows_back(self):
        svc = self.make()
        self.vision.fail_start = "the vision encoder did not start: CUDA out of memory"
        with self.assertRaises(VisionUnavailable):
            self.ask("a.png")
        self.assertEqual(self.events, [("vram", 2100), ("start", "gpu"), ("stop", "gpu"), ("vram", None)])
        self.assertFalse(svc.vision_shrunk)
        self.vision.fail_start = None                                   # the next image tries again
        self.events.clear()
        self.ask("a.png")
        self.assertEqual([e[0] for e in self.events], ["vram", "start", "encode"])

    def test_encoder_dying_mid_encode_grows_back(self):
        svc = self.make()
        self.ask("a.png")
        self.vision.up = False                                          # it crashed between requests
        self.events.clear()
        self.ask("b.png")                                               # cold again: shrink (no-op), start, encode
        self.assertEqual([e[0] for e in self.events], ["vram", "start", "encode"])
        orig = self.vision.encode

        def crash(source, data=None):
            self.vision.up = False
            raise ValueError("the image could not be read: the vision encoder stopped")
        self.vision.encode = crash
        self.events.clear()
        with self.assertRaises(ValueError):
            self.ask("z.png")
        self.vision.encode = orig
        self.assertFalse(svc.vision_shrunk)                             # grown back at once
        self.assertEqual(self.events[-1], ("vram", None))


class Concurrency(HotBase):
    def test_no_unload_while_a_request_holds_the_fifo(self):
        svc = self.make()
        self.ask("a.png")
        svc.last_image_at = time.time() - 1000
        with svc.fifo:                                                  # a request is running
            self.assertEqual(svc.vision_idle_unload(120), "busy")
        with svc.status_lock:
            svc.status["queued"] = 1                                    # or one waits for its turn
        self.assertEqual(svc.vision_idle_unload(120), "busy")
        with svc.status_lock:
            svc.status["queued"] = 0
        self.assertTrue(self.vision.alive())

    def test_a_request_during_the_unload_waits_and_starts_cold(self):
        svc = self.make()
        self.ask("a.png")
        svc.last_image_at = time.time() - 1000
        self.vision.unload_delay = 0.3                                  # the stop takes a while
        self.events.clear()
        results = {}

        def unload():
            with contextlib.redirect_stdout(io.StringIO()):
                results["unload"] = svc.vision_idle_unload(120)
        t = threading.Thread(target=unload)
        t.start()
        time.sleep(0.1)                                                 # inside the unload, holding the fifo
        try:
            svc.prepare(image_msg("b.png"), None, {})
            results["request"] = "ok"
        except Exception as e:                                          # pragma: no cover - the failure we test
            results["request"] = repr(e)
        t.join()
        self.assertEqual(results, {"unload": "unloaded", "request": "ok"})
        self.assertEqual(self.events, [("stop", "gpu"), ("vram", None), ("vram", 2100), ("start", "gpu"),
                                       ("encode", "gpu", "b.png")])
        self.assertTrue(self.vision.alive())
        self.assertTrue(svc.vision_shrunk)

    def test_many_cycles_end_balanced(self):
        svc = self.make()
        for i in range(20):
            self.ask(f"img{i}.png")
            svc.last_image_at = time.time() - 1000
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(svc.vision_idle_unload(120), "unloaded")
        shrinks = sum(1 for e in self.events if e[0] == "vram" and e[1] is not None)
        grows = sum(1 for e in self.events if e == ("vram", None))
        self.assertEqual((shrinks, grows), (20, 20))
        self.assertFalse(svc.vision_shrunk)


@unittest.skipIf(os.name == "nt", "a shell script stands in for strata-vision")
class RealProcess(unittest.TestCase):
    """The real Vision class with a stand-in strata-vision (prints READY, answers ENC): start=False, restart(),
    encode, unload, restart again - the process side of the hot cycle."""
    def test_lazy_start_encode_unload_restart(self):
        with tempfile.TemporaryDirectory() as d:
            exe = Path(d) / "fake-vision"
            exe.write_text(f"#!{sys.executable}\n"
                           "import sys\n"
                           "print('READY', flush=True)\n"
                           "for line in sys.stdin:\n"
                           "    p = line.split()\n"
                           "    if p and p[0] == 'QUIT': break\n"
                           "    if p and p[0] == 'ENC':\n"
                           "        open(p[2], 'wb').write(b'rows')\n"
                           "        print('OK 7', flush=True)\n")
            exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
            v = Vision({"exe": str(exe), "mmproj": "m", "model": "g", "gpu": True}, start=False)
            self.assertFalse(v.alive())
            img = Path(d) / "a.png"
            img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 32)
            key, data, hit = v.cached(str(img))
            self.assertIsNone(hit)
            v.restart()
            self.assertTrue(v.alive())
            path, n = v.encode(str(img), data=data)
            self.assertEqual(n, 7)
            self.assertIsNotNone(v.cached(str(img))[2])
            v.unload()
            self.assertFalse(v.alive())
            v.restart()
            self.assertTrue(v.alive())
            v.close()


if __name__ == "__main__":
    unittest.main()
