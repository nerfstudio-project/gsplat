import os
import sys

from citygs.train.scheduler import Job, run_queue


def _job(block_id, cost, tmp_path, rc=0):
    return Job(
        block_id=block_id,
        cost=cost,
        cmd=[sys.executable, "-c", f"import sys; sys.exit({rc})"],
        log_path=os.path.join(str(tmp_path), f"logs/block_{block_id}.txt"),
    )


def test_run_queue_all_succeed(tmp_path):
    jobs = [_job(i, cost=10 - i, tmp_path=tmp_path) for i in range(6)]
    result = run_queue(jobs, gpus=[0, 1], max_retries=0)
    assert sorted(result.succeeded) == list(range(6))
    assert result.failed == []
    for job in jobs:
        assert os.path.isfile(job.log_path)


def test_run_queue_reports_failures(tmp_path):
    jobs = [
        _job(0, 5, tmp_path, rc=0),
        _job(1, 4, tmp_path, rc=3),
        _job(2, 3, tmp_path, rc=0),
    ]
    result = run_queue(jobs, gpus=[0], max_retries=1)
    assert sorted(result.succeeded) == [0, 2]
    assert result.failed == [1]
    # Retried once: two attempt headers in the log.
    with open(jobs[1].log_path) as f:
        assert f.read().count("attempt") == 2


def test_run_queue_longest_first(tmp_path):
    order_file = os.path.join(str(tmp_path), "order.txt")
    jobs = [
        Job(
            block_id=i,
            cost=cost,
            cmd=[
                sys.executable,
                "-c",
                f"open(r'{order_file}', 'a').write('{i} ')",
            ],
            log_path=os.path.join(str(tmp_path), f"logs/b{i}.txt"),
        )
        for i, cost in [(0, 1.0), (1, 100.0), (2, 10.0)]
    ]
    result = run_queue(jobs, gpus=[0], max_retries=0)  # single worker: strict order
    assert sorted(result.succeeded) == [0, 1, 2]
    with open(order_file) as f:
        assert f.read().split() == ["1", "2", "0"]


def test_gpu_pinning(tmp_path):
    out = os.path.join(str(tmp_path), "gpu.txt")
    jobs = [
        Job(
            block_id=0,
            cost=1.0,
            cmd=[
                sys.executable,
                "-c",
                f"import os; open(r'{out}', 'w').write(os.environ['CUDA_VISIBLE_DEVICES'])",
            ],
            log_path=os.path.join(str(tmp_path), "logs/b0.txt"),
        )
    ]
    run_queue(jobs, gpus=[5], max_retries=0)
    with open(out) as f:
        assert f.read() == "5"
