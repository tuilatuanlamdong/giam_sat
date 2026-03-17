import os
import json
from datetime import datetime

try:
    from pymongo import MongoClient
except Exception:
    MongoClient = None


JSON_LOG_PATH = "events_log.json"


class EventLogger:
    def __init__(
        self,
        json_path=JSON_LOG_PATH,
        mongo_enabled=False,
        mongo_uri="mongodb://localhost:27017/",
        mongo_db="giamsat_ai",
        mongo_collection="events"
    ):
        self.json_path = json_path
        self.mongo_enabled = mongo_enabled
        self.mongo_uri = mongo_uri
        self.mongo_db = mongo_db
        self.mongo_collection = mongo_collection

        self.mongo_client = None
        self.mongo_col = None

        self._init_json()
        self._init_mongo()

    def _init_json(self):
        if not os.path.exists(self.json_path):
            with open(self.json_path, "w", encoding="utf-8") as f:
                json.dump([], f, ensure_ascii=False, indent=2)

    def _init_mongo(self):
        if not self.mongo_enabled:
            print("[LOGGER] MongoDB logging dang tat.")
            return

        if MongoClient is None:
            print("[LOGGER] Chua cai pymongo, bo qua MongoDB.")
            return

        try:
            self.mongo_client = MongoClient(self.mongo_uri, serverSelectionTimeoutMS=5000)
            self.mongo_client.admin.command("ping")
            db = self.mongo_client[self.mongo_db]
            self.mongo_col = db[self.mongo_collection]
            print("[LOGGER] MongoDB connected.")
        except Exception as ex:
            print("[LOGGER] Khong ket noi duoc MongoDB:", repr(ex))
            self.mongo_client = None
            self.mongo_col = None

    def _build_event(
        self,
        event_type,
        cam_id,
        person_id=None,
        person_name="Unknown",
        extra=None
    ):
        now = datetime.now()
        return {
            "event_type": event_type,
            "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "weekday": now.strftime("%A"),
            "cam_id": cam_id,
            "person_id": person_id,
            "person_name": person_name,
            "extra": extra or {}
        }

    def log_event(
        self,
        event_type,
        cam_id,
        person_id=None,
        person_name="Unknown",
        extra=None
    ):
        event = self._build_event(
            event_type=event_type,
            cam_id=cam_id,
            person_id=person_id,
            person_name=person_name,
            extra=extra
        )

        self._write_json(event)
        self._write_mongo(event)

        print(
            f"[LOG] {event['timestamp']} | {event_type} | "
            f"cam_id={cam_id} | person_id={person_id} | name={person_name}"
        )

    def _write_json(self, event):
        try:
            data = []
            if os.path.exists(self.json_path):
                with open(self.json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

            data.append(event)

            with open(self.json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as ex:
            print("[LOGGER] Loi ghi JSON:", ex)

    def _write_mongo(self, event):
        if self.mongo_col is None:
            print("[LOGGER] mongo_col is None, khong ghi duoc MongoDB")
            return

        try:
            result = self.mongo_col.insert_one(event)
            print("[MONGO_INSERT_OK]", result.inserted_id)
        except Exception as ex:
            print("[LOGGER] Loi ghi MongoDB:", repr(ex))
