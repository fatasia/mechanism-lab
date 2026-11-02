"""认证主体与表单填报执行人分开记录；本模块不依赖 Web。"""

import json
from contextvars import ContextVar

from phm_runtime.observability import redact

actor = ContextVar("audit_actor", default="local-script")


def audit(db, action, entity, detail):
    db.execute(
        "INSERT INTO audit(action,entity,detail) VALUES(?,?,?)",
        (
            action,
            str(entity),
            json.dumps(
                redact({**detail, "authenticated_actor": actor.get()}), ensure_ascii=False
            ),
        ),
    )
