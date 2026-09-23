// godot-lite: a synchronous WorkerThreadPool. A task runs to completion
// inside add_native_task; the returned id is already complete, so
// is_task_completed() is true and waiting is a no-op. The guest is
// single-threaded (rule 4: state machines, not waits).
#pragma once

#include "core/object/object.h"
#include "core/string/ustring.h"

#include <cstdint>

namespace gdl {

class WorkerThreadPool : public Object {
	GDCLASS(WorkerThreadPool, Object);
	int64_t last_task_id = 0;

public:
	typedef int64_t TaskID;
	enum {
		INVALID_TASK_ID = -1
	};

	static WorkerThreadPool *get_singleton();

	TaskID add_native_task(void (*p_func)(void *), void *p_userdata, bool p_high_priority = false, const String &p_description = String()) {
		p_func(p_userdata);
		return ++last_task_id;
	}
	bool is_task_completed(TaskID p_task_id) const { return p_task_id > 0 && p_task_id <= last_task_id; }
	Error wait_for_task_completion(TaskID p_task_id) {
		return is_task_completed(p_task_id) ? OK : ERR_INVALID_PARAMETER;
	}
};

} // namespace gdl
