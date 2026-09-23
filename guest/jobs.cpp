#include "jobs.h"

namespace jobs {

void StageQueue::push(Fn fn, bool yield) {
	q_.push_back(Stage{ std::move(fn), yield });
	++total_;
}

void StageQueue::next(Fn fn, bool yield) {
	if (!running_) {
		// Outside a stage "next" is simply the front.
		q_.push_front(Stage{ std::move(fn), yield });
		++total_;
		return;
	}
	inserts_.push_back(Stage{ std::move(fn), yield });
	++total_;
}

int StageQueue::tick(const std::function<bool()> &stop, int max_stages) {
	int ran = 0;
	// No check before the first stage: a submit from the previous tick is what
	// the first stage of this one reads back (a frame later, as rule 4 wants).
	while (!q_.empty() && ran < max_stages) {
		Stage s = std::move(q_.front());
		q_.pop_front();
		running_ = true;
		s.fn();
		running_ = false;
		if (!inserts_.empty()) {
			q_.insert(q_.begin(), inserts_.begin(), inserts_.end());
			inserts_.clear();
		}
		++done_;
		++ran;
		if (s.yield || stop()) {
			break;
		}
	}
	return ran;
}

std::string Job::tick(int64_t host_us) {
	now_us = host_us;
	if (finished_) {
		// The verdict came with a submit still in flight: wait it out now,
		// a frame later, so the job can be destroyed.
		if (pending()) {
			drain();
		}
	} else {
		// A verdict mid-queue (an early FAIL) stops the queue too.
		q.tick([this]() { return pending() || finished_; }, max_stages_per_tick);
		if (!finished_ && q.empty() && !pending()) {
			finish(false, "job drained its queue without a verdict", "");
		}
	}
	if (finished_ && !pending()) {
		return result_;
	}
	return "RUNNING " + std::to_string(q.done()) + "/" + std::to_string(q.total());
}

void Job::finish(bool pass, const std::string &head, const std::string &detail) {
	finished_ = true;
	result_ = std::string(pass ? "PASS " : "FAIL ") + head;
	if (!detail.empty()) {
		result_ += "\n" + detail;
		if (result_.back() == '\n') {
			result_.pop_back();
		}
	}
}

} // namespace jobs
