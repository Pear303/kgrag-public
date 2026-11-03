"""后台任务调度：进程内线程池执行 build pipeline，互斥 + 进度记录。

单机单用户场景：同刻最多 1 个 build（threading.Lock），新任务排队。
任务记录写 JSON 文件，前端轮询读取。
"""
import os
import json
import uuid
import threading
import traceback

from config import PROJECT_ROOT
from backend.build_pipeline import build_pipeline
from backend.manual_registry import ManualRegistry

TASK_DIR = os.path.join(PROJECT_ROOT, ".sisyphus", "registry", "tasks")

_exec_lock = threading.Lock()  # 同刻最多 1 个 build


def _task_path(task_id: str) -> str:
    return os.path.join(TASK_DIR, f"{task_id}.json")


class TaskManager:
    def __init__(self, registry: ManualRegistry = None):
        self.registry = registry or ManualRegistry()

    def _write(self, task: dict) -> None:
        os.makedirs(TASK_DIR, exist_ok=True)
        tmp = _task_path(task["task_id"]) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(task, f, ensure_ascii=False, indent=2)
        os.replace(tmp, _task_path(task["task_id"]))

    def _read(self, task_id: str) -> dict:
        with open(_task_path(task_id), "r", encoding="utf-8") as f:
            return json.load(f)

    def _run(self, task_id: str, manual_id: str, source_manual: str, md_path: str,
             use_calibrated_prompt: bool = False):
        task = self._read(task_id)
        task["status"] = "running"
        task["started_at"] = _now()
        self._write(task)
        try:
            self.registry.update_manual(manual_id, status="processing", task_id=task_id)
            with _exec_lock:
                def progress(step, pct, message):
                    t = self._read(task_id)
                    t.update(step=step, pct=pct, message=message)
                    self._write(t)

                extraction_prompt = None
                if use_calibrated_prompt:
                    # C1/D：只用已过目的冻结校准件；缺失或损坏在此显式失败
                    from backend.prompt_calibrator import get_adapted_prompt
                    extraction_prompt = get_adapted_prompt(md_path)
                result = build_pipeline(manual_id, source_manual, md_path,
                                        progress_cb=progress,
                                        extraction_prompt=extraction_prompt)
            task = self._read(task_id)
            task["status"] = "done"
            task["pct"] = 100
            task["finished_at"] = _now()
            task["result"] = result
            self._write(task)
            self.registry.update_manual(manual_id, status="ready", build=result)
        except Exception as e:
            task = self._read(task_id)
            task["status"] = "failed"
            task["error"] = f"{e}\n{traceback.format_exc()}"
            task["finished_at"] = _now()
            self._write(task)
            self.registry.update_manual(manual_id, status="failed", error=str(e))

    def submit(self, manual_id: str, source_manual: str, md_path: str,
               use_calibrated_prompt: bool = False) -> str:
        """提交构建任务（use_calibrated_prompt=True 时抽取用冻结校准件，C1/D）。"""
        task_id = uuid.uuid4().hex[:12]
        task = {
            "task_id": task_id,
            "kind": "build",
            "manual_id": manual_id,
            "source_manual": source_manual,
            "use_calibrated_prompt": use_calibrated_prompt,
            "status": "queued",
            "step": 0,
            "pct": 0,
            "message": "排队中",
            "error": None,
            "started_at": None,
            "finished_at": None,
        }
        self._write(task)
        t = threading.Thread(target=self._run,
                             args=(task_id, manual_id, source_manual, md_path,
                                   use_calibrated_prompt),
                             daemon=True)
        t.start()
        return task_id

    def _run_calibrate(self, task_id: str, manual_id: str, md_path: str,
                       force: bool = False):
        """校准任务：产出/复用冻结校准件，落盘后停在 calibrated 等人工确认（C1）。"""
        task = self._read(task_id)
        task["status"] = "running"
        task["started_at"] = _now()
        self._write(task)
        try:
            from backend.prompt_calibrator import (
                TOKEN_BUDGET, calibration_json_path, run_calibration,
            )
            self.registry.update_manual(manual_id, status="calibrating", task_id=task_id)

            def progress(pct, message):
                t = self._read(task_id)
                t.update(step=1, pct=int(pct), message=message)
                self._write(t)

            progress(10, "提示词校准：解析手册与 token 预算")
            rec = run_calibration(md_path, budget=TOKEN_BUDGET, force=force)
            progress(90, "校准完成，等待人工确认")
            task = self._read(task_id)
            task.update(
                status="calibrated" if not rec.get("reused") else "calibrated_reused",
                pct=95,
                message=f"校准完成（{rec['mode']}），变更 {len(rec.get('updates', {}))} 处，"
                        f"等待人工过目后确认构建",
                calibration={
                    "path": calibration_json_path(md_path),
                    "mode": rec.get("mode"),
                    "manual_chars": rec.get("manual_chars"),
                    "selected_chars": rec.get("selected_chars"),
                    "n_windows": rec.get("n_windows"),
                    "adapted_prompt_sha8": rec.get("adapted_prompt_sha8"),
                    "updates": rec.get("updates", {}),
                    "changes": rec.get("changes", []),
                },
                finished_at=_now(),
            )
            self._write(task)
        except Exception as e:
            task = self._read(task_id)
            task["status"] = "failed"
            task["error"] = f"{e}\n{traceback.format_exc()}"
            task["finished_at"] = _now()
            self._write(task)
            self.registry.update_manual(manual_id, status="failed", error=str(e))

    def submit_calibrate(self, manual_id: str, md_path: str,
                         force: bool = False) -> str:
        """提交提示词校准任务（C1 第一步：只校准落盘，不建图）。"""
        task_id = uuid.uuid4().hex[:12]
        task = {
            "task_id": task_id,
            "kind": "calibrate",
            "manual_id": manual_id,
            "source_manual": os.path.basename(md_path),
            "force": force,
            "status": "queued",
            "step": 0,
            "pct": 0,
            "message": "校准排队中",
            "error": None,
            "started_at": None,
            "finished_at": None,
        }
        self._write(task)
        t = threading.Thread(target=self._run_calibrate,
                             args=(task_id, manual_id, md_path, force),
                             daemon=True)
        t.start()
        return task_id

    def get_task(self, task_id: str) -> dict:
        if not os.path.exists(_task_path(task_id)):
            return None
        return self._read(task_id)

    def list_tasks(self) -> list:
        if not os.path.isdir(TASK_DIR):
            return []
        out = []
        for fn in os.listdir(TASK_DIR):
            if fn.endswith(".json"):
                with open(os.path.join(TASK_DIR, fn), "r", encoding="utf-8") as f:
                    out.append(json.load(f))
        out.sort(key=lambda t: t.get("started_at") or "", reverse=True)
        return out

    def wait(self, task_id: str, timeout: float = 60.0) -> dict:
        import time
        deadline = time.time() + timeout
        while time.time() < deadline:
            t = self.get_task(task_id)
            if t and t["status"] in ("done", "failed"):
                return t
            time.sleep(0.2)
        return self.get_task(task_id)


def _now() -> str:
    from datetime import datetime
    return datetime.now().isoformat(timespec="seconds")
