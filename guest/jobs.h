// jobs -- a stage queue the host advances one process frame at a time.
//
// AGENTS.md rule 4: never sync() in the frame that submit()s. A job is a
// queue of stages; the host calls tick() once per process frame and tick()
// runs stages in order until the queue is empty or the job's solver has a
// GPU submit in flight. So every submit ends the tick by construction, and
// the stage that reads the result back runs on a later frame, when the fence
// is known to be done. A CPU backend is never pending, so its stages run
// back to back (up to a per-tick cap), except those marked `yield`, which
// end the tick on every backend (the bench uses them so a CPU substep and a
// GPU substep are timed over the same number of frames).
//
// Stages are std::function<void()> closures over the job's own members. A
// running stage may queue more stages with next(): they run right after it,
// before anything queued earlier (in call order), which is how a job adds
// work only when a result asks for it (the gradcheck's h-sweep).
#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <string>
#include <vector>

namespace jobs {

class StageQueue {
public:
	using Fn = std::function<void()>;

	// Append a stage at the end of the queue.
	void push(Fn fn, bool yield = false);
	// From inside a running stage: run `fn` after the current stage, before
	// the rest of the queue. Several calls keep their order.
	void next(Fn fn, bool yield = false);
	// Run stages until the queue is empty, `stop()` is true after a stage (a
	// submit is in flight: it ends the tick), a yielding stage ran, or
	// `max_stages` ran. Returns the number of stages run.
	int tick(const std::function<bool()> &stop, int max_stages = 256);

	bool empty() const { return q_.empty(); }
	size_t done() const { return done_; }
	size_t total() const { return total_; }

private:
	struct Stage {
		Fn fn;
		bool yield = false;
	};
	std::deque<Stage> q_;
	std::vector<Stage> inserts_;
	bool running_ = false;
	size_t done_ = 0, total_ = 0;
};

// A job: a stage queue, a way to ask whether its GPU work is in flight, and
// the final verdict. The host's clock comes in with every tick (the guest
// clock is not a clock: AGENTS.md), so a job that times itself reads now_us.
class Job {
public:
	virtual ~Job() = default;
	// A GPU submit is in flight: the tick must end, and the job must not be
	// destroyed in this frame.
	virtual bool pending() const = 0;
	// Wait out the in-flight submit (called on a later tick than the one that
	// submitted, only when the verdict is in and pending() is still true).
	virtual void drain() {}

	// One host frame: run stages, then report "RUNNING done/total" or, once
	// the queue is drained and nothing is in flight, the verdict.
	std::string tick(int64_t host_us);
	// The verdict is in and nothing is in flight: safe to destroy.
	bool finished() const { return finished_ && !pending(); }
	const std::string &result() const { return result_; }

protected:
	// The verdict: "PASS <head>" or "FAIL <head>", then the detail lines.
	void finish(bool pass, const std::string &head, const std::string &detail);

	StageQueue q;
	int64_t now_us = 0;
	int max_stages_per_tick = 256;

private:
	bool finished_ = false;
	std::string result_;
};

} // namespace jobs
